#!/usr/bin/env python3
"""Validate MISA set selection, assignment quality, and TP score parity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from misa_router_dataset import RouterDataset, load_router_dataset, load_sample_metadata
from sglang.srt.layers.attention.nsa.assignment_router_model import (
    MISAAssignmentRouter,
)


def load_model(path: Path, device: torch.device) -> MISAAssignmentRouter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("mla_heads_per_group") != 16:
        raise ValueError("group16 validation requires a fresh 16-head checkpoint")
    model = MISAAssignmentRouter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device=device, dtype=torch.float32).eval()


def score_candidate_subset(
    model: MISAAssignmentRouter,
    inputs: tuple[torch.Tensor, ...],
    selected_slots: torch.Tensor,
) -> torch.Tensor:
    """Score selected candidates using the same feature normalization as runtime.

    Candidate standardization depends on the selected set, so scoring all heads
    and slicing their scores afterward can change the selected winner.
    """
    return model(
        inputs[0],
        *(value.index_select(1, selected_slots) for value in inputs[1:-1]),
        inputs[-1],
    )


def greedy_set(utility: torch.Tensor, budget: int) -> torch.Tensor:
    """Greedily maximize mean group utility for one [groups, candidates] matrix."""
    covered = torch.full_like(utility[:, 0], float("-inf"))
    available = torch.ones(utility.shape[1], dtype=torch.bool, device=utility.device)
    selected = []
    for _ in range(budget):
        value = torch.maximum(
            covered.unsqueeze(1), utility
        ).mean(dim=0)
        value = value.masked_fill(~available, float("-inf"))
        choice = value.argmax()
        selected.append(choice)
        available[choice] = False
        covered = torch.maximum(covered, utility[:, choice])
    return torch.stack(selected)


def rank_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    return (x_rank * y_rank).sum() / (
        x_rank.square().sum().sqrt() * y_rank.square().sum().sqrt()
    ).clamp_min(1e-12)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(tensor.quantile(0.1)),
        "p50": float(tensor.quantile(0.5)),
        "p90": float(tensor.quantile(0.9)),
        "count": int(tensor.numel()),
    }


def load_attention_ceilings(
    probe_dir: Path,
) -> dict[tuple[int, int], dict[str, float]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(probe_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = payload if isinstance(payload, list) else [payload]
        for row in records:
            grouped[(int(row["sample_id"]), int(row["layer"]))].append(row)

    ceilings = {}
    for key, rows in grouped.items():
        if len(rows) != 8:
            continue
        official = torch.empty(128)
        exact = torch.empty(128)
        for row in rows:
            head_ids = row["global_mla_head_ids"].long()
            official[head_ids] = row["official_shared_mass"].float()
            exact[head_ids] = row["exact_mla_top720_mass"].float()
        ceilings[key] = {
            "official_shared_top2048": float(
                official.mean()
            ),
            "exact_mla_top720": float(exact.mean()),
        }
    return ceilings


def selected_indices(
    data: RouterDataset,
    metadata: dict[int, dict],
    *,
    dataset: str | None,
    split: str | None,
) -> list[int]:
    indices = []
    for index in range(len(data)):
        sample_id = int(data.sample_ids[index])
        meta = metadata.get(sample_id, {})
        if dataset is not None and str(meta.get("dataset", "")) != dataset:
            continue
        if split is not None and str(meta.get("split", "")) != split:
            continue
        if split is not None and not meta:
            # Fall back to the dataset-encoded split when metadata is sparse.
            if int(data.split_ids[index]) != {"train": 0, "validation": 1, "test": 2}[split]:
                continue
        indices.append(index)
    return indices


@torch.no_grad()
def validate(
    data: RouterDataset,
    model: MISAAssignmentRouter,
    device: torch.device,
    budget: int,
    ceilings: dict[tuple[int, int], dict[str, float]],
    metadata: dict[int, dict],
    indices: list[int],
) -> dict:
    if data.candidate_head_ids.shape[1] != 64:
        raise ValueError("full-head validation requires exactly 64 candidates")
    if not indices:
        raise ValueError("no contexts remain after dataset/split filtering")

    metric_names = (
        "misa_set_oracle",
        "greedy_set_oracle",
        "all64_oracle",
        "learned_assignment",
        "misa_greedy_overlap",
        "importance_utility_spearman",
        "official_shared_top2048",
        "exact_mla_top720",
        "indexer_candidate_gap",
        "misa_set_regret",
        "assignment_regret",
        "signed_misa_learned_assignment",
    )
    selector_names = (
        "importance",
        "signed_importance",
        "gate",
        "context_mean",
        "context_std",
        "context_max",
        "context_positive_fraction",
    )
    totals: dict[str, list[float]] = {name: [] for name in metric_names}
    for name in selector_names:
        totals[f"selector_top8:{name}"] = []
        totals[f"selector_bottom8:{name}"] = []

    grouped: dict[str, dict[str, dict[str, list[float]]]] = {
        "by_length": defaultdict(lambda: defaultdict(list)),
        "by_task": defaultdict(lambda: defaultdict(list)),
        "by_layer": defaultdict(lambda: defaultdict(list)),
        "by_decode_position": defaultdict(lambda: defaultdict(list)),
    }
    group_core = (
        "all64_oracle",
        "misa_set_oracle",
        "learned_assignment",
        "misa_set_regret",
        "assignment_regret",
        "exact_mla_top720",
        "official_shared_top2048",
        "indexer_candidate_gap",
        "misa_greedy_overlap",
    )

    max_tp_score_error = 0.0
    candidate_permutation_failures = 0

    for index in indices:
        batch = data.batch(torch.tensor([index]), device)
        utility = batch.utility[0].float()
        heads = batch.candidate_head_ids[0]
        importance = batch.candidate_importance[0].float()
        if not torch.equal(heads.sort().values, torch.arange(64, device=device)):
            candidate_permutation_failures += 1

        # Production selector uses topk(sorted=False). Recover the deployed
        # MISA set from importance rather than assuming slots [0:budget].
        misa_slots = importance.topk(
            budget, largest=True, sorted=False
        ).indices
        selected_greedy = greedy_set(utility, budget)
        misa_oracle = utility[:, misa_slots].max(dim=-1).values
        greedy_oracle = utility[:, selected_greedy].max(dim=-1).values

        score = score_candidate_subset(model, batch.model_inputs(), misa_slots)[0]
        learned_slots = score.argmax(dim=-1)
        learned = utility[:, misa_slots].gather(
            -1, learned_slots.unsqueeze(-1)
        ).squeeze(-1)

        tp_scores = []
        for rank in range(8):
            start = rank
            tp_inputs = (
                batch.q_group[:, start : start + 1],
                *batch.model_inputs()[1:],
            )
            tp_scores.append(score_candidate_subset(model, tp_inputs, misa_slots))
        tp_score = torch.cat(tp_scores, dim=1)[0]
        max_tp_score_error = max(
            max_tp_score_error,
            float((score - tp_score).abs().max()),
        )

        misa_heads = set(heads[misa_slots].tolist())
        greedy_heads = set(heads[selected_greedy].tolist())
        candidate_value = utility.mean(dim=0)
        selector_values = {
            "importance": importance,
            "signed_importance": (
                importance * batch.candidate_gate[0].float().sign()
            ),
            "gate": batch.candidate_gate[0].float(),
            "context_mean": batch.candidate_context_stats[0, :, 0].float(),
            "context_std": batch.candidate_context_stats[0, :, 1].float(),
            "context_max": batch.candidate_context_stats[0, :, 2].float(),
            "context_positive_fraction": (
                batch.candidate_context_stats[0, :, 3].float()
            ),
        }
        values = {
            "misa_set_oracle": float(misa_oracle.mean()),
            "greedy_set_oracle": float(greedy_oracle.mean()),
            "all64_oracle": float(utility.max(dim=-1).values.mean()),
            "learned_assignment": float(learned.mean()),
            "misa_greedy_overlap": len(misa_heads & greedy_heads) / budget,
            "importance_utility_spearman": float(
                rank_correlation(importance, candidate_value)
            ),
        }
        for name, selector_value in selector_values.items():
            top_slots = selector_value.topk(
                budget, largest=True, sorted=False
            ).indices
            bottom_slots = selector_value.topk(
                budget, largest=False, sorted=False
            ).indices
            values[f"selector_top8:{name}"] = float(
                utility[:, top_slots].max(dim=-1).values.mean()
            )
            values[f"selector_bottom8:{name}"] = float(
                utility[:, bottom_slots].max(dim=-1).values.mean()
            )
            if name == "signed_importance":
                signed_score = score_candidate_subset(
                    model, batch.model_inputs(), top_slots
                )[0]
                signed_choice = signed_score.argmax(dim=-1)
                values["signed_misa_learned_assignment"] = float(
                    utility[:, top_slots]
                    .gather(-1, signed_choice.unsqueeze(-1))
                    .squeeze(-1)
                    .mean()
                )
        key = (
            int(data.sample_ids[index]),
            int(data.layers[data.layer_ids[index]]),
        )
        ceiling = ceilings[key]
        values.update(ceiling)
        values.update(
            {
                "indexer_candidate_gap": (
                    ceiling["exact_mla_top720"] - values["all64_oracle"]
                ),
                "misa_set_regret": (
                    values["all64_oracle"] - values["misa_set_oracle"]
                ),
                "assignment_regret": (
                    values["misa_set_oracle"] - values["learned_assignment"]
                ),
            }
        )
        for name, value in values.items():
            totals[name].append(float(value))

        meta = metadata.get(int(data.sample_ids[index]), {})
        source_id = str(meta.get("source_id") or "")
        length = str(meta.get("length") or "unknown")
        if length == "unknown" and source_id.startswith("ruler:"):
            parts = source_id.split(":")
            if len(parts) >= 3 and parts[2] in {"32k", "128k", "4k", "8k", "16k", "64k"}:
                length = parts[2]
        task = str(meta.get("task") or "unknown")
        if task == "unknown" and source_id.startswith("ruler:") and len(source_id.split(":")) >= 2:
            task = source_id.split(":")[1]
        decode_position = str(meta.get("decode_position", "unknown"))
        layer = str(int(data.layers[data.layer_ids[index]]))
        for bucket_name, bucket_key in (
            ("by_length", length),
            ("by_task", task),
            ("by_layer", layer),
            ("by_decode_position", decode_position),
        ):
            for metric in group_core:
                grouped[bucket_name][bucket_key][metric].append(values[metric])

    report = {
        "contexts": len(indices),
        "budget": budget,
        "metrics": {
            name: summarize(values) for name, values in totals.items()
        },
        "contracts": {
            "candidate_permutation_failures": candidate_permutation_failures,
            "max_full_vs_tp_score_error": max_tp_score_error,
            "tp_pair_mapping_numerically_equivalent": (
                max_tp_score_error <= 1e-5
            ),
        },
        "groups": {
            group_name: {
                key: {metric: summarize(vals) for metric, vals in metrics.items()}
                for key, metrics in sorted(buckets.items(), key=lambda item: item[0])
            }
            for group_name, buckets in grouped.items()
        },
        "interpretation": {
            "set_selection_regret": "all64_oracle - misa_set_oracle",
            "assignment_regret": "misa_set_oracle - learned_assignment",
            "candidate_ceiling_gap": "exact_mla_top720 - all64_oracle",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default=None,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    data = load_router_dataset(
        args.probe_dir,
        args.sample_manifest,
        expected_ranks=8,
        queries_per_prompt=1,
    )
    metadata = load_sample_metadata(args.sample_manifest)
    indices = selected_indices(
        data,
        metadata,
        dataset=args.dataset,
        split=args.split,
    )
    model = load_model(args.checkpoint, device)
    ceilings = load_attention_ceilings(args.probe_dir)
    report = validate(
        data,
        model,
        device,
        args.budget,
        ceilings,
        metadata,
        indices,
    )
    report["filters"] = {"dataset": args.dataset, "split": args.split}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
