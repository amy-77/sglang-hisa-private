#!/usr/bin/env python3
"""Compare pooled-mean MISA head selectors using only their strongest chunks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from misa_router_dataset import load_router_dataset, load_sample_metadata


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    values = values.to(torch.float64)
    return {
        "mean": float(values.mean()),
        "p10": float(values.quantile(0.10)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "count": int(values.numel()),
    }


def load_variants(
    probe_dir: Path,
) -> dict[tuple[int, int], tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    contexts = {}
    for path in sorted(probe_dir.glob("probe_shard_L*_rank0_s*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = payload if isinstance(payload, list) else [payload]
        for row in records:
            variants = row.get("importance_variants") or {}
            if not variants:
                raise ValueError(f"{path} does not contain importance_variants")
            key = (int(row["sample_id"]), int(row["layer"]))
            contexts[key] = (
                row["candidate_head_ids"].long(),
                {name: value.float() for name, value in variants.items()},
            )
    if not contexts:
        raise RuntimeError(f"no rank-0 probe variants found in {probe_dir}")
    return contexts


def metadata_groups(meta: dict, layer: int) -> dict[str, str]:
    source_id = str(meta.get("source_id") or "")
    length = str(meta.get("length") or "unknown")
    task = str(meta.get("task") or "unknown")
    if source_id.startswith("ruler:"):
        parts = source_id.split(":")
        if length == "unknown" and len(parts) >= 3:
            length = parts[2]
        if task == "unknown" and len(parts) >= 2:
            task = parts[1]
    return {
        "by_length": length,
        "by_task": task,
        "by_layer": str(layer),
        "by_decode_position": str(meta.get("decode_position", "unknown")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = load_router_dataset(
        args.probe_dir,
        args.sample_manifest,
        expected_ranks=8,
        queries_per_prompt=1,
    )
    metadata = load_sample_metadata(args.sample_manifest)
    raw_variants = load_variants(args.probe_dir)

    variant_rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    baseline_errors = []
    context_groups = []
    for index in range(len(data)):
        sample_id = int(data.sample_ids[index])
        layer = int(data.layers[data.layer_ids[index]])
        heads, variants = raw_variants[(sample_id, layer)]
        if not torch.equal(heads, data.candidate_head_ids[index]):
            raise ValueError(
                f"candidate order mismatch for sample={sample_id}, layer={layer}"
            )
        for name, value in variants.items():
            variant_rows[name].append(value)
        baseline_errors.append(
            (
                variants["baseline_all"]
                - data.candidate_importance[index].float()
            ).abs().max()
        )
        context_groups.append(metadata_groups(metadata.get(sample_id, {}), layer))

    scores = {
        name: torch.stack(rows)
        for name, rows in sorted(variant_rows.items())
    }
    utility = data.utility.float()
    all64_oracle = utility.max(dim=-1).values.mean(dim=-1)
    baseline_slots = scores["baseline_all"].topk(
        args.budget, dim=-1, largest=True, sorted=False
    ).indices

    metrics: dict[str, dict[str, torch.Tensor]] = {}
    for name, score in scores.items():
        slots = score.topk(
            args.budget, dim=-1, largest=True, sorted=False
        ).indices
        chosen = utility.gather(
            2, slots.unsqueeze(1).expand(-1, utility.shape[1], -1)
        )
        set_oracle = chosen.max(dim=-1).values.mean(dim=-1)
        selected_heads = data.candidate_head_ids.gather(1, slots)
        baseline_heads = data.candidate_head_ids.gather(1, baseline_slots)
        overlap = (
            selected_heads.unsqueeze(-1)
            .eq(baseline_heads.unsqueeze(1))
            .any(dim=-1)
            .sum(dim=-1)
            .float()
            / args.budget
        )
        metrics[name] = {
            "set_oracle": set_oracle,
            "set_regret": all64_oracle - set_oracle,
            "delta_vs_baseline": set_oracle
            - metrics.get("baseline_all", {}).get(
                "set_oracle", torch.zeros_like(set_oracle)
            ),
            "head_overlap_vs_baseline": overlap,
        }

    baseline_oracle = metrics["baseline_all"]["set_oracle"]
    for values in metrics.values():
        values["delta_vs_baseline"] = values["set_oracle"] - baseline_oracle

    grouped: dict[
        str, dict[str, dict[str, dict[str, list[torch.Tensor]]]]
    ] = {
        name: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for name in ("by_length", "by_task", "by_layer", "by_decode_position")
    }
    for index, groups in enumerate(context_groups):
        for group_name, group_key in groups.items():
            for variant, variant_metrics in metrics.items():
                for metric, values in variant_metrics.items():
                    grouped[group_name][group_key][variant][metric].append(
                        values[index]
                    )

    aggregate = {
        variant: {
            metric: summarize(values)
            for metric, values in variant_metrics.items()
        }
        for variant, variant_metrics in metrics.items()
    }
    best = max(
        aggregate,
        key=lambda name: float(aggregate[name]["set_oracle"]["mean"]),
    )
    report = {
        "contexts": len(data),
        "budget": args.budget,
        "pooling_summary": "chunk_mean",
        "selector_formula": (
            "gate_h * sum(Top-L or Top-fraction ReLU(q_h @ chunk_mean))"
        ),
        "metrics": aggregate,
        "best_mean_set_oracle": {
            "variant": best,
            **aggregate[best]["set_oracle"],
        },
        "contracts": {
            "max_recomputed_baseline_importance_error": float(
                torch.stack(baseline_errors).max()
            ),
            "all_contexts_have_variants": len(raw_variants) == len(data),
        },
        "groups": {
            group_name: {
                group_key: {
                    variant: {
                        metric: summarize(torch.stack(values))
                        for metric, values in variant_metrics.items()
                    }
                    for variant, variant_metrics in variants.items()
                }
                for group_key, variants in sorted(groups.items())
            }
            for group_name, groups in grouped.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
