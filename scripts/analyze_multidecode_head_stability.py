#!/usr/bin/env python3
"""Measure whether a unique-head set can be reused across decode queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_128k_unique_head_oracle import (
    greedy_set,
    load_probe,
    mean_ci,
    pair_utility,
    score_dynamic_assignment,
)


def score_fixed_assignment(
    mass: np.ndarray,
    selected: list[int],
    reference_mass: np.ndarray,
) -> np.ndarray:
    """Use one pair-to-head mapping learned from a reference query/mass."""
    ref_pair = pair_utility(reference_mass[None, ...])[0, selected]
    choice = ref_pair.argmax(0)
    test = mass[:, selected].reshape(mass.shape[0], len(selected), 64, 2)
    gathered = np.take_along_axis(
        test, choice[None, None, :, None], axis=1
    )[:, 0]
    return gathered.mean(axis=(1, 2))


def set_jaccard(a: list[int], b: list[int]) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--queries-per-prompt", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mass, samples, layers, _, metadata = load_probe(
        args.probe_dir, args.manifest, args.expected_ranks
    )
    sample_to_index = {sample: i for i, sample in enumerate(samples)}
    groups: dict[int, list[int]] = {}
    for sample in samples:
        group = sample // args.queries_per_prompt
        groups.setdefault(group, []).append(sample_to_index[sample])
    groups = {
        group: indices
        for group, indices in groups.items()
        if len(indices) == args.queries_per_prompt
    }
    if not groups:
        raise RuntimeError("no complete multi-query prompt groups")

    result: dict[str, dict] = {}
    for budget in args.budgets:
        metric: dict[str, list[float]] = {
            "dynamic_set_dynamic_assignment": [],
            "first_query_set_dynamic_assignment": [],
            "first_query_set_first_query_assignment": [],
            "prompt_oracle_set_dynamic_assignment": [],
            "prompt_oracle_set_prompt_assignment": [],
            "within_prompt_set_jaccard_to_first": [],
            "within_prompt_dynamic_set_union_size": [],
        }
        per_layer: dict[str, dict] = {}
        by_position: dict[int, dict[str, list[float]]] = {
            qi: {
                "dynamic_set_dynamic_assignment": [],
                "first_query_set_dynamic_assignment": [],
                "first_query_set_first_query_assignment": [],
            }
            for qi in range(args.queries_per_prompt)
        }
        for li, layer in enumerate(layers):
            layer_metric = {key: [] for key in metric}
            for indices in groups.values():
                prompt_mass = mass[indices, li]
                utility = pair_utility(prompt_mass)
                dynamic_sets = [greedy_set(u, budget) for u in utility]
                first_set = dynamic_sets[0]
                prompt_set = greedy_set(utility.mean(0), budget)

                for qi, selected in enumerate(dynamic_sets):
                    value = score_dynamic_assignment(
                        prompt_mass[qi : qi + 1], selected, True
                    )
                    layer_metric["dynamic_set_dynamic_assignment"].extend(value)
                    by_position[qi]["dynamic_set_dynamic_assignment"].extend(value)
                first_dynamic = score_dynamic_assignment(prompt_mass, first_set, True)
                first_fixed = score_fixed_assignment(
                    prompt_mass, first_set, prompt_mass[0]
                )
                layer_metric["first_query_set_dynamic_assignment"].extend(first_dynamic)
                layer_metric["first_query_set_first_query_assignment"].extend(first_fixed)
                for qi in range(args.queries_per_prompt):
                    by_position[qi]["first_query_set_dynamic_assignment"].append(
                        float(first_dynamic[qi])
                    )
                    by_position[qi]["first_query_set_first_query_assignment"].append(
                        float(first_fixed[qi])
                    )
                layer_metric["prompt_oracle_set_dynamic_assignment"].extend(
                    score_dynamic_assignment(prompt_mass, prompt_set, True)
                )
                layer_metric["prompt_oracle_set_prompt_assignment"].extend(
                    score_fixed_assignment(
                        prompt_mass, prompt_set, prompt_mass.mean(0)
                    )
                )
                layer_metric["within_prompt_set_jaccard_to_first"].extend(
                    set_jaccard(first_set, selected)
                    for selected in dynamic_sets[1:]
                )
                layer_metric["within_prompt_dynamic_set_union_size"].append(
                    float(len(set().union(*(set(x) for x in dynamic_sets))))
                )
            per_layer[str(layer)] = {
                name: mean_ci(values) for name, values in layer_metric.items()
            }
            for name, values in layer_metric.items():
                metric[name].extend(values)
        result[str(budget)] = {
            "aggregate": {name: mean_ci(values) for name, values in metric.items()},
            "per_layer": per_layer,
            "per_query_position": {
                str(qi): {name: mean_ci(values) for name, values in metrics.items()}
                for qi, metrics in by_position.items()
            },
        }

    output = {
        "metadata": {
            **metadata,
            "queries_per_prompt": args.queries_per_prompt,
            "complete_prompt_groups": sorted(groups),
            "layers": layers,
        },
        "budgets": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# Multi-decode unique-head stability",
        "",
        f"Complete prompt groups: {len(groups)}; queries/group: {args.queries_per_prompt}",
        "",
        "| M | Dynamic oracle | First set + dynamic assign | First set + first assign | Prompt-set oracle | Prompt-set + prompt assign | Jaccard to first | Union over queries |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, values in result.items():
        m = values["aggregate"]
        lines.append(
            f"| {budget} | {m['dynamic_set_dynamic_assignment']['mean']:.4f} | "
            f"{m['first_query_set_dynamic_assignment']['mean']:.4f} | "
            f"{m['first_query_set_first_query_assignment']['mean']:.4f} | "
            f"{m['prompt_oracle_set_dynamic_assignment']['mean']:.4f} | "
            f"{m['prompt_oracle_set_prompt_assignment']['mean']:.4f} | "
            f"{m['within_prompt_set_jaccard_to_first']['mean']:.3f} | "
            f"{m['within_prompt_dynamic_set_union_size']['mean']:.2f} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
