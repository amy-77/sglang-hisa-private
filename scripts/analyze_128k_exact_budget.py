#!/usr/bin/env python3
"""Aggregate exact per-MLA-head K needed to match official DSA Top-2048 mass."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BUDGETS = (64, 128, 256, 512, 720, 1024, 1536, 2048)


def describe(values) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def summarize(rows: list[dict]) -> dict:
    shared = np.asarray([row["shared_mass"] for row in rows])
    minimum_k = np.asarray([row["minimum_k"] for row in rows])
    result = {
        "observations": len(rows),
        "official_shared_top2048_mass": describe(shared),
        "exact_minimum_k_to_match": describe(minimum_k),
        "mean_k_as_fraction_of_2048": float(minimum_k.mean() / 2048),
        "fraction_matching_by_budget": {
            str(k): float(np.mean(minimum_k <= k)) for k in BUDGETS
        },
        "exact_topk_mass": {},
        "exact_topk_minus_official_mass": {},
    }
    for k in BUDGETS:
        mass = np.asarray([row[f"mass_{k}"] for row in rows])
        result["exact_topk_mass"][str(k)] = describe(mass)
        result["exact_topk_minus_official_mass"][str(k)] = describe(mass - shared)
    return result


def compact(summary: dict) -> dict:
    return {
        "observations": summary["observations"],
        "official_mass": summary["official_shared_top2048_mass"],
        "minimum_k": summary["exact_minimum_k_to_match"],
        "fraction_matching_by_budget": summary["fraction_matching_by_budget"],
        "exact_mass_mean": {
            k: summary["exact_topk_mass"][k]["mean"] for k in map(str, BUDGETS)
        },
        "exact_minus_official_mass_mean": {
            k: summary["exact_topk_minus_official_mass"][k]["mean"]
            for k in map(str, BUDGETS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample_meta = {}
    for prompt in manifest["records"]:
        for sample_id in range(
            int(prompt["probe_sample_start"]),
            int(prompt["probe_sample_end"]) + 1,
        ):
            sample_meta[sample_id] = {
                "dataset": prompt["dataset"],
                "task": prompt["task"],
                "prompt_id": int(prompt["prompt_id"]),
            }

    rows = []
    for path in sorted(args.results.glob("sample*_L*_rank*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        sample_id = int(shard["sample_id"])
        if sample_id not in sample_meta:
            continue
        budget = shard["exact_mla_budget_to_match_official_mass"]
        shared = shard["official_shared_top2048"]["mla_attention_mass_per_head"]
        minimum_k = budget["minimum_k_per_head"]
        mass_by_k = budget["mass_by_exact_topk_budget_per_head"]
        for local_id, global_head in enumerate(shard["global_mla_head_ids"]):
            row = {
                "sample_id": sample_id,
                "layer": int(shard["layer"]),
                "mla_head": int(global_head),
                "seq_len": int(shard["seq_len"]),
                **sample_meta[sample_id],
                "shared_mass": float(shared[local_id]),
                "minimum_k": int(minimum_k[local_id]),
            }
            for k in BUDGETS:
                row[f"mass_{k}"] = float(mass_by_k[str(k)][local_id])
            rows.append(row)
    if not rows:
        raise SystemExit("No matching probe shards found")
    keys = {(row["sample_id"], row["layer"], row["mla_head"]) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate sample/layer/MLA-head observations found")
    observations_per_sample_layer = Counter(
        (row["sample_id"], row["layer"]) for row in rows
    )
    incomplete = {
        key: count
        for key, count in observations_per_sample_layer.items()
        if count != 128
    }
    if incomplete:
        raise RuntimeError(
            f"expected 128 MLA heads per sample/layer, got {incomplete}"
        )
    observed_layers = {row["layer"] for row in rows}
    expected_sample_layers = {
        (sample_id, layer)
        for sample_id in sample_meta
        for layer in observed_layers
    }
    missing_sample_layers = expected_sample_layers - set(
        observations_per_sample_layer
    )
    if missing_sample_layers:
        raise RuntimeError(
            f"missing sample/layer observations: {sorted(missing_sample_layers)}"
        )

    by_dataset = defaultdict(list)
    by_layer = defaultdict(list)
    by_layer_head = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
        by_layer[row["layer"]].append(row)
        by_layer_head[(row["layer"], row["mla_head"])].append(row)

    report = {
        "definition": (
            "For each sampled query/layer/MLA head, sort exact dense MLA "
            "attention probabilities and choose the smallest K whose cumulative "
            "mass is >= the mass retained by official shared DSA Top-2048."
        ),
        "sequence_length": describe([row["seq_len"] for row in rows]),
        "samples": len({row["sample_id"] for row in rows}),
        "prompts": len({row["prompt_id"] for row in rows}),
        "layers": sorted(by_layer),
        "overall": summarize(rows),
        "by_dataset": {
            name: summarize(group) for name, group in sorted(by_dataset.items())
        },
        "by_layer": {
            str(layer): summarize(group) for layer, group in sorted(by_layer.items())
        },
        "per_layer_mla_head": {
            f"L{layer:02d}_H{head:03d}": summarize(group)
            for (layer, head), group in sorted(by_layer_head.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    compact_report = {
        "definition": report["definition"],
        "sequence_length": report["sequence_length"],
        "samples": report["samples"],
        "prompts": report["prompts"],
        "overall": compact(report["overall"]),
        "by_dataset": {
            name: compact(summary)
            for name, summary in report["by_dataset"].items()
        },
        "by_layer": {
            layer: compact(summary) for layer, summary in report["by_layer"].items()
        },
    }
    static_head_budgets = {
        key: int(np.ceil(summary["exact_minimum_k_to_match"]["p95"]))
        for key, summary in report["per_layer_mla_head"].items()
    }
    compact_report["static_per_layer_head_budget_p95"] = {
        "definition": (
            "A separate fixed K for every (layer, MLA head), chosen as the "
            "95th percentile of its 128 sampled-query minimum K values."
        ),
        "all_layer_heads": describe(list(static_head_budgets.values())),
        "mean_fraction_of_2048": float(
            np.mean(list(static_head_budgets.values())) / 2048
        ),
        "achieved_matching_fraction": float(
            np.mean(
                [
                    row["minimum_k"]
                    <= static_head_budgets[
                        f"L{row['layer']:02d}_H{row['mla_head']:03d}"
                    ]
                    for row in rows
                ]
            )
        ),
        "by_layer": {},
    }
    for layer in sorted(by_layer):
        prefix = f"L{layer:02d}_"
        layer_budgets = {
            key: value
            for key, value in static_head_budgets.items()
            if key.startswith(prefix)
        }
        compact_report["static_per_layer_head_budget_p95"]["by_layer"][
            str(layer)
        ] = {
            "budget": describe(list(layer_budgets.values())),
            "highest_budget_heads": sorted(
                layer_budgets.items(), key=lambda item: item[1], reverse=True
            )[:8],
        }
    (args.output_dir / "compact_summary.json").write_text(
        json.dumps(compact_report, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "per_head_budget.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "layer",
                "mla_head",
                "observations",
                "official_mass_mean",
                "minimum_k_mean",
                "minimum_k_p50",
                "minimum_k_p90",
                "minimum_k_p95",
                "minimum_k_p99",
                "minimum_k_max",
                "fraction_k_le_720",
                "fraction_k_le_1024",
            ]
        )
        for (layer, head), group in sorted(by_layer_head.items()):
            summary = summarize(group)
            mass = summary["official_shared_top2048_mass"]
            k = summary["exact_minimum_k_to_match"]
            fractions = summary["fraction_matching_by_budget"]
            writer.writerow(
                [
                    layer,
                    head,
                    len(group),
                    mass["mean"],
                    k["mean"],
                    k["p50"],
                    k["p90"],
                    k["p95"],
                    k["p99"],
                    k["max"],
                    fractions["720"],
                    fractions["1024"],
                ]
            )
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
