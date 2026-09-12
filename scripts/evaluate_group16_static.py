#!/usr/bin/env python3
"""Fit and validate fixed layer/group -> Indexer assignments.

This intentionally loads only the all-64 utility labels. Router queries and
coarse-context features are ignored, keeping full-data static evaluation small.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SPLITS = {"train": 0, "validation": 1, "test": 2}


@dataclass
class StaticDataset:
    utility: torch.Tensor  # [contexts, 8, 64], indexed by global Indexer ID
    layer_ids: torch.Tensor  # [contexts], compact layer indices
    layers: torch.Tensor  # [num_layers], real model layer IDs
    split_ids: torch.Tensor
    source_ids: list[str]
    source_hashes: list[str]
    datasets: list[str]
    lengths: list[str]

    def split_mask(self, split: str) -> torch.Tensor:
        return self.split_ids == SPLITS[split]


def _manifest(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate request_id: {request_id}")
        split = str(row["split"])
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        source_id = str(row["source_id"])
        source_hash = str(row.get("source_hash", source_id))
        source_splits[source_id].add(split)
        hash_splits[source_hash].add(split)
        result[request_id] = row
    if not result:
        raise ValueError(f"no successful samples in {path}")
    for name, groups in (
        ("source_id", source_splits),
        ("source_hash", hash_splits),
    ):
        overlap = {key: value for key, value in groups.items() if len(value) > 1}
        if overlap:
            raise ValueError(f"{name} appears in multiple splits: {next(iter(overlap.items()))}")
    return result


def load_static_dataset(probe_dir: Path, manifest_path: Path) -> StaticDataset:
    """Stream probe shards and retain only Group16 utility labels."""
    metadata = _manifest(manifest_path)
    contexts: dict[tuple[str, int], dict[int, torch.Tensor]] = defaultdict(dict)
    candidate_orders: dict[tuple[str, int], torch.Tensor] = {}

    shards = sorted(probe_dir.glob("probe_shard_L*_rank*_s*.pt"))
    if not shards:
        raise ValueError(f"no probe shards in {probe_dir}")
    for shard in shards:
        for row in torch.load(shard, map_location="cpu", weights_only=False):
            request_id = str(row["request_id"])
            if request_id not in metadata:
                raise ValueError(f"probe request missing from manifest: {request_id}")
            meta = metadata[request_id]
            if (
                str(row["prefix_hash"]) != str(meta["prefix_hash"])
                or int(row["query_token_id"]) != int(meta["query_token_id"])
                or int(row["seq_len"]) != int(meta["expected_seq_len"])
            ):
                raise ValueError(f"probe/manifest identity mismatch: {request_id}")
            layer, rank = int(row["layer"]), int(row["tp_rank"])
            if rank not in range(8):
                raise ValueError(f"invalid TP rank {rank}")
            key = (request_id, layer)
            if rank in contexts[key]:
                raise ValueError(f"duplicate rank for context {key}: {rank}")
            ids = row["candidate_head_ids"].long()
            if ids.shape != (64,) or not torch.equal(
                ids.sort().values, torch.arange(64)
            ):
                raise ValueError("candidate IDs must contain 0..63 exactly once")
            previous = candidate_orders.setdefault(key, ids)
            if not torch.equal(previous, ids):
                raise ValueError(f"candidate order differs across ranks: {key}")
            mass = row["attention_mass_matrix"].float()
            if (
                mass.shape != (64, 16)
                or not torch.isfinite(mass).all()
                or mass.min() < 0
                or mass.max() > 1.0001
            ):
                raise ValueError("attention_mass_matrix must be finite [64,16] in [0,1]")
            by_global_id = torch.empty(64)
            by_global_id.scatter_(0, ids, mass.mean(-1))
            contexts[key][rank] = by_global_id

    layers = sorted({layer for _, layer in contexts})
    layer_to_id = {layer: index for index, layer in enumerate(layers)}
    expected = {(request_id, layer) for request_id in metadata for layer in layers}
    if set(contexts) != expected:
        missing = sorted(expected - set(contexts))
        extra = sorted(set(contexts) - expected)
        raise ValueError(f"incomplete request/layer coverage: missing={missing[:2]} extra={extra[:2]}")

    utility, layer_ids, split_ids = [], [], []
    source_ids, source_hashes, datasets, lengths = [], [], [], []
    for request_id in sorted(metadata):
        meta = metadata[request_id]
        for layer in layers:
            ranks = contexts[(request_id, layer)]
            if sorted(ranks) != list(range(8)):
                raise ValueError(f"context lacks TP ranks 0..7: {(request_id, layer)}")
            utility.append(torch.stack([ranks[rank] for rank in range(8)]))
            layer_ids.append(layer_to_id[layer])
            split_ids.append(SPLITS[str(meta["split"])])
            source_ids.append(str(meta["source_id"]))
            source_hashes.append(str(meta.get("source_hash", meta["source_id"])))
            datasets.append(str(meta.get("dataset", "unknown")))
            lengths.append(str(meta.get("length", "unknown")))
    return StaticDataset(
        utility=torch.stack(utility),
        layer_ids=torch.tensor(layer_ids),
        layers=torch.tensor(layers),
        split_ids=torch.tensor(split_ids),
        source_ids=source_ids,
        source_hashes=source_hashes,
        datasets=datasets,
        lengths=lengths,
    )


def fit_static(
    data: StaticDataset,
    mask: torch.Tensor,
    weighting: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean utility and selected global Indexer, both [layer,8,(64)]."""
    if weighting not in {"context", "source", "dataset"}:
        raise ValueError(f"unknown weighting: {weighting}")
    scores = torch.empty(len(data.layers), 8, 64)
    for layer_id in range(len(data.layers)):
        rows = (mask & (data.layer_ids == layer_id)).nonzero().flatten().tolist()
        if not rows:
            raise ValueError(f"layer {int(data.layers[layer_id])} has no calibration rows")
        if weighting == "context":
            scores[layer_id] = data.utility[rows].mean(0)
            continue
        source_groups: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            source_groups[data.source_ids[row]].append(row)
        source_means = {
            source: data.utility[indices].mean(0)
            for source, indices in source_groups.items()
        }
        if weighting == "source":
            scores[layer_id] = torch.stack(list(source_means.values())).mean(0)
            continue
        dataset_groups: dict[str, list[torch.Tensor]] = defaultdict(list)
        for source, value in source_means.items():
            representative = source_groups[source][0]
            dataset_groups[data.datasets[representative]].append(value)
        scores[layer_id] = torch.stack(
            [
                torch.stack(values).mean(0)
                for values in dataset_groups.values()
            ]
        ).mean(0)
    return scores, scores.argmax(-1)


def evaluate_static(
    data: StaticDataset,
    mask: torch.Tensor,
    assignment: torch.Tensor,
) -> dict:
    rows = mask.nonzero().flatten()
    if rows.numel() == 0:
        raise ValueError("evaluation split is empty")
    utility = data.utility[rows]
    choices = assignment[data.layer_ids[rows]]
    selected = utility.gather(-1, choices.unsqueeze(-1)).squeeze(-1).mean(-1)
    oracle = utility.max(-1).values.mean(-1)

    def summarize(indices: list[int]) -> dict:
        picked = selected[indices].numpy()
        best = oracle[indices].numpy()
        return {
            "contexts": len(indices),
            "static_utility": float(picked.mean()),
            "all64_oracle": float(best.mean()),
            "regret": float((best - picked).mean()),
        }

    result = {"overall": summarize(list(range(rows.numel())))}
    for field, labels in (("dataset", data.datasets), ("length", data.lengths)):
        groups: dict[str, list[int]] = defaultdict(list)
        for offset, row in enumerate(rows.tolist()):
            groups[labels[row]].append(offset)
        result[f"by_{field}"] = {
            name: summarize(indices) for name, indices in sorted(groups.items())
        }
    return result


def _sample_sources(
    data: StaticDataset,
    train_rows: list[int],
    count: int,
    rng: random.Random,
) -> set[str]:
    """Round-robin datasets so small calibration subsets remain broad."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in train_rows:
        source = data.source_ids[row]
        if source not in groups[data.datasets[row]]:
            groups[data.datasets[row]].append(source)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[str] = []
    names = list(groups)
    rng.shuffle(names)
    while len(selected) < count:
        progressed = False
        for name in names:
            if groups[name]:
                selected.append(groups[name].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return set(selected)


def scaling_curve(
    data: StaticDataset,
    weighting: str,
    sizes: list[int],
    repeats: int,
    seed: int,
) -> list[dict]:
    train_mask = data.split_mask("train")
    validation_mask = data.split_mask("validation")
    train_rows = train_mask.nonzero().flatten().tolist()
    sources = sorted({data.source_ids[row] for row in train_rows})
    _, full_assignment = fit_static(data, train_mask, weighting)
    curve = []
    for requested in sizes:
        size = min(requested, len(sources))
        trials = []
        for repeat in range(repeats if size < len(sources) else 1):
            chosen = _sample_sources(
                data, train_rows, size, random.Random(seed + 1009 * size + repeat)
            )
            mask = torch.tensor(
                [
                    bool(train_mask[row]) and data.source_ids[row] in chosen
                    for row in range(len(data.source_ids))
                ]
            )
            _, assignment = fit_static(data, mask, weighting)
            metrics = evaluate_static(data, validation_mask, assignment)["overall"]
            trials.append(
                {
                    **metrics,
                    "mapping_agreement": float(
                        (assignment == full_assignment).float().mean()
                    ),
                }
            )
        curve.append(
            {
                "sources": size,
                "repeats": len(trials),
                **{
                    f"{name}_{stat}": float(value)
                    for name in (
                        "static_utility",
                        "regret",
                        "mapping_agreement",
                    )
                    for stat, value in (
                        ("mean", np.mean([trial[name] for trial in trials])),
                        ("std", np.std([trial[name] for trial in trials])),
                    )
                },
            }
        )
    return curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-probe-dir", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--test-probe-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--sizes", default="4,8,16,32,64,128,256,512")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-config-out", type=Path)
    parser.add_argument(
        "--runtime-weighting",
        choices=("context", "source", "dataset"),
        default="context",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_static_dataset(
        args.calibration_probe_dir, args.calibration_manifest
    )
    test = load_static_dataset(args.test_probe_dir, args.test_manifest)
    if calibration.layers.tolist() != test.layers.tolist():
        raise ValueError("calibration and test layer sets differ")
    if set(calibration.source_ids) & set(test.source_ids):
        raise ValueError("calibration/test source_id overlap")
    if set(calibration.source_hashes) & set(test.source_hashes):
        raise ValueError("calibration/test source content overlap")
    sizes = sorted({int(value) for value in args.sizes.split(",") if int(value) > 0})
    report = {
        "calibration_contexts": len(calibration.utility),
        "test_contexts": len(test.utility),
        "weightings": {},
    }
    assignments: dict[str, torch.Tensor] = {}
    for weighting in ("context", "source", "dataset"):
        scores, assignment = fit_static(
            calibration, calibration.split_mask("train"), weighting
        )
        assignments[weighting] = assignment
        report["weightings"][weighting] = {
            "validation": evaluate_static(
                calibration, calibration.split_mask("validation"), assignment
            ),
            "test": evaluate_static(test, test.split_mask("test"), assignment),
            "scaling_curve": scaling_curve(
                calibration, weighting, sizes, args.repeats, args.seed
            ),
            "assignment": assignment.tolist(),
            "mean_top1_top2_gap": float(
                scores.topk(2, dim=-1).values.diff(dim=-1).abs().mean()
            ),
            "mean_unique_indexers_per_layer": float(
                torch.tensor(
                    [row.unique().numel() for row in assignment]
                ).float().mean()
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.runtime_config_out is not None:
        assignment = assignments[args.runtime_weighting]
        runtime_config = {
            "budget": 1,
            "assignment_mode": "static",
            "head_selection_mode": "static",
            "mla_heads_per_group": 16,
            "min_seq_len": 4096,
            # Must match the all-64 teacher candidate policy.
            "misa_chunk_size": 512,
            "misa_prune_keep_fraction": 0.6,
            "static_assignment_by_layer": {
                str(int(layer)): assignment[index].tolist()
                for index, layer in enumerate(calibration.layers)
            },
        }
        args.runtime_config_out.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_config_out.write_text(
            json.dumps(runtime_config, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
