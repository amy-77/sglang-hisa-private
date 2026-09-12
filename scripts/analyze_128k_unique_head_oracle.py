#!/usr/bin/env python3
"""Analyze Router + Global Unique-Head Budget oracles from 128K probe shards."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def greedy_set(utility: np.ndarray, budget: int) -> list[int]:
    """Greedy facility-location selection for utility [indexer, consumer]."""
    n_indexer = utility.shape[0]
    budget = min(budget, n_indexer)
    selected: list[int] = []
    best = np.zeros(utility.shape[1], dtype=np.float64)
    remaining = set(range(n_indexer))
    for _ in range(budget):
        candidate = max(
            remaining,
            key=lambda i: np.maximum(best, utility[i]).mean(),
        )
        selected.append(candidate)
        remaining.remove(candidate)
        best = np.maximum(best, utility[candidate])
    return selected


def pair_utility(mass: np.ndarray) -> np.ndarray:
    """[sample,indexer,128] -> [sample,indexer,64]."""
    return mass.reshape(*mass.shape[:-1], 64, 2).mean(-1)


def score_dynamic_assignment(
    mass: np.ndarray, selected: list[int], pair_route: bool
) -> np.ndarray:
    """Per-sample score after choosing the best selected head per consumer."""
    sub = mass[:, selected]
    if not pair_route:
        return sub.max(1).mean(-1)
    util = pair_utility(sub)
    choice = util.argmax(1)  # [sample,pair], indices within selected
    gathered = np.take_along_axis(
        sub.reshape(sub.shape[0], sub.shape[1], 64, 2),
        choice[:, None, :, None],
        axis=1,
    )[:, 0]
    return gathered.mean(axis=(1, 2))


def score_static_assignment(
    train_mass: np.ndarray,
    test_mass: np.ndarray,
    selected: list[int],
    pair_route: bool,
) -> np.ndarray:
    train_mean = train_mass.mean(0)[selected]
    test_sub = test_mass[:, selected]
    if not pair_route:
        choice = train_mean.argmax(0)
        gathered = np.take_along_axis(
            test_sub, choice[None, None, :], axis=1
        )[:, 0]
        return gathered.mean(-1)
    train_pair = train_mean.reshape(len(selected), 64, 2).mean(-1)
    choice = train_pair.argmax(0)
    gathered = np.take_along_axis(
        test_sub.reshape(test_sub.shape[0], len(selected), 64, 2),
        choice[None, None, :, None],
        axis=1,
    )[:, 0]
    return gathered.mean(axis=(1, 2))


def mean_ci(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def jaccard(sets: list[list[int]]) -> float:
    if len(sets) < 2:
        return 1.0
    vals = []
    for a, b in itertools.combinations(sets, 2):
        sa, sb = set(a), set(b)
        vals.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(vals))


def load_probe(
    probe_dir: Path, manifest_path: Path | None, expected_ranks: int
) -> tuple[np.ndarray, list[int], list[int], dict, dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(probe_dir.glob("sample*_L*_rank*.json")):
        record = json.loads(path.read_text())
        groups[(int(record["sample_id"]), int(record["layer"]))].append(record)
    if not groups:
        raise RuntimeError(f"no probe JSON files in {probe_dir}")

    samples = sorted({key[0] for key in groups})
    layers = sorted({key[1] for key in groups})
    complete_samples = [
        sample
        for sample in samples
        if all(len(groups.get((sample, layer), [])) == expected_ranks for layer in layers)
    ]
    if not complete_samples:
        raise RuntimeError("no sample has complete layer/rank shards")

    mass = np.full((len(complete_samples), len(layers), 64, 128), np.nan)
    scalar: dict[str, list[float]] = defaultdict(list)
    seq_lens: dict[str, int] = {}
    for si, sample in enumerate(complete_samples):
        for li, layer in enumerate(layers):
            records = groups[(sample, layer)]
            ranks = {int(r["tp_rank"]) for r in records}
            if len(records) != expected_ranks or len(ranks) != expected_ranks:
                raise RuntimeError(
                    f"sample {sample} layer {layer}: got ranks {sorted(ranks)}"
                )
            for record in records:
                heads = np.asarray(record["global_mla_head_ids"], dtype=np.int64)
                shard = np.asarray(record["attention_mass_matrix"], dtype=np.float64)
                # Chained indexing preserves [indexer, local_mla] axis order;
                # NumPy's mixed advanced indexing would move `heads` to front.
                mass[si, li][:, heads] = shard
                seq_lens[f"{sample}:{layer}:{record['tp_rank']}"] = int(
                    record["seq_len"]
                )
                scalar[f"official:L{layer}"].append(
                    record["official_shared_top2048"]["mla_attention_mass"]["mean"]
                )
                scalar[f"exact:L{layer}"].append(
                    record["exact_mla_top720_mass"]["mean"]
                )
                scalar[f"fixed:L{layer}"].append(
                    record["fixed_1to2_top720"]["mla_attention_mass"]["mean"]
                )
    if np.isnan(mass).any():
        raise RuntimeError("assembled utility tensor contains missing MLA-head columns")

    manifest = {}
    if manifest_path is not None and manifest_path.exists():
        raw = json.loads(manifest_path.read_text())
        manifest = {
            int(r["sample_id"]): r
            for r in raw.get("records", [])
            if int(r["sample_id"]) in complete_samples
        }
    metadata = {
        "complete_samples": complete_samples,
        "discarded_samples": sorted(set(samples) - set(complete_samples)),
        "seq_len_min": min(seq_lens.values()),
        "seq_len_max": max(seq_lens.values()),
        "tasks": [manifest.get(s, {}).get("task", "unknown") for s in complete_samples],
    }
    return mass, complete_samples, layers, scalar, metadata


def analyze_route(
    mass: np.ndarray,
    layers: list[int],
    budgets: list[int],
    pair_route: bool,
    sample_groups: np.ndarray | None = None,
) -> tuple[dict, dict]:
    route_name = "pair" if pair_route else "mla_head"
    n_samples, n_layers = mass.shape[:2]
    if sample_groups is not None:
        unique_groups = np.unique(sample_groups)
        first = set(unique_groups[::2].tolist())
        first_mask = np.asarray([group in first for group in sample_groups])
        folds = [
            (np.flatnonzero(first_mask), np.flatnonzero(~first_mask)),
            (np.flatnonzero(~first_mask), np.flatnonzero(first_mask)),
        ]
    elif n_samples < 2:
        folds = [(np.arange(n_samples), np.arange(n_samples))]
    else:
        even = np.arange(0, n_samples, 2)
        odd = np.arange(1, n_samples, 2)
        folds = [(even, odd), (odd, even)]

    output: dict[str, dict] = {}
    selection_detail: dict[str, dict] = {}
    for budget in budgets:
        dynamic_scores: list[float] = []
        cv_static_set_dynamic_scores: list[float] = []
        cv_static_assignment_scores: list[float] = []
        dynamic_sets_by_layer: dict[int, list[list[int]]] = defaultdict(list)
        full_static_by_layer: dict[int, list[int]] = {}
        full_static_assignment_by_layer: dict[int, list[int]] = {}
        cv_sets_by_layer: dict[int, list[list[int]]] = defaultdict(list)

        for li, layer in enumerate(layers):
            layer_mass = mass[:, li]
            consumer = pair_utility(layer_mass) if pair_route else layer_mass
            for sample in range(n_samples):
                selected = greedy_set(consumer[sample], budget)
                dynamic_sets_by_layer[layer].append(selected)
                dynamic_scores.extend(
                    score_dynamic_assignment(
                        layer_mass[sample : sample + 1], selected, pair_route
                    ).tolist()
                )
            full_static_by_layer[layer] = greedy_set(consumer.mean(0), budget)
            full_selected = full_static_by_layer[layer]
            full_mean = layer_mass.mean(0)[full_selected]
            if pair_route:
                full_choice = pair_utility(full_mean).argmax(0)
            else:
                full_choice = full_mean.argmax(0)
            # Store global indexer-head ids rather than set-relative slots so
            # the artifact stays unambiguous if a consumer reorders the set.
            full_static_assignment_by_layer[layer] = [
                int(full_selected[int(slot)]) for slot in full_choice
            ]
            for train_idx, test_idx in folds:
                if len(test_idx) == 0:
                    continue
                selected = greedy_set(consumer[train_idx].mean(0), budget)
                cv_sets_by_layer[layer].append(selected)
                cv_static_set_dynamic_scores.extend(
                    score_dynamic_assignment(layer_mass[test_idx], selected, pair_route)
                )
                cv_static_assignment_scores.extend(
                    score_static_assignment(
                        layer_mass[train_idx],
                        layer_mass[test_idx],
                        selected,
                        pair_route,
                    )
                )

        output[str(budget)] = {
            "dynamic_set_dynamic_assignment_oracle": mean_ci(dynamic_scores),
            "cv_static_set_dynamic_assignment": mean_ci(
                cv_static_set_dynamic_scores
            ),
            "cv_static_set_static_assignment": mean_ci(cv_static_assignment_scores),
            "dynamic_set_jaccard_mean": float(
                np.mean([jaccard(v) for v in dynamic_sets_by_layer.values()])
            ),
        }
        frequency = Counter(
            h
            for layer_sets in dynamic_sets_by_layer.values()
            for selected in layer_sets
            for h in selected
        )
        selection_detail[str(budget)] = {
            "route": route_name,
            "full_data_static_set_by_layer": {
                str(layer): selected
                for layer, selected in full_static_by_layer.items()
            },
            "full_data_static_assignment_by_layer": {
                str(layer): assignment
                for layer, assignment in full_static_assignment_by_layer.items()
            },
            "cv_sets_by_layer": {
                str(layer): selected for layer, selected in cv_sets_by_layer.items()
            },
            "most_frequent_dynamic_heads": frequency.most_common(16),
        }
    return output, selection_detail


def baseline_metrics(mass: np.ndarray, layers: list[int], scalar: dict) -> dict:
    n_samples, n_layers, _, n_heads = mass.shape
    fixed = []
    unrestricted_head = []
    unrestricted_pair = []
    for li in range(n_layers):
        layer_mass = mass[:, li]
        fixed_index = np.arange(n_heads) // 2
        fixed.extend(layer_mass[:, fixed_index, np.arange(n_heads)].mean(-1))
        unrestricted_head.extend(layer_mass.max(1).mean(-1))
        unrestricted_pair.extend(pair_utility(layer_mass).max(1).mean(-1))
    official = list(
        itertools.chain.from_iterable(scalar[f"official:L{layer}"] for layer in layers)
    )
    exact = list(
        itertools.chain.from_iterable(scalar[f"exact:L{layer}"] for layer in layers)
    )
    fixed_dump = list(
        itertools.chain.from_iterable(scalar[f"fixed:L{layer}"] for layer in layers)
    )
    return {
        "fixed_1to2_top720_reassembled": mean_ci(fixed),
        "fixed_1to2_top720_dump": mean_ci(fixed_dump),
        "unrestricted_per_mla_head_top720": mean_ci(unrestricted_head),
        "unrestricted_per_pair_top720": mean_ci(unrestricted_pair),
        "official_shared_top2048": mean_ci(official),
        "exact_mla_top720": mean_ci(exact),
    }


def add_ratios(summary: dict) -> None:
    for route in ("pair", "mla_head"):
        denom_key = (
            "unrestricted_per_pair_top720"
            if route == "pair"
            else "unrestricted_per_mla_head_top720"
        )
        denom = summary["baselines"][denom_key]["mean"]
        for values in summary["routes"][route].values():
            for metric in (
                "dynamic_set_dynamic_assignment_oracle",
                "cv_static_set_dynamic_assignment",
                "cv_static_set_static_assignment",
            ):
                values[metric]["fraction_of_unrestricted"] = (
                    values[metric]["mean"] / denom
                )


def write_report(summary: dict, path: Path) -> None:
    b = summary["baselines"]
    lines = [
        "# Real 128K Router + Global Unique-Head Budget Oracle",
        "",
        f"Samples: {summary['metadata']['n_samples']}; layers: "
        + ", ".join(map(str, summary["layers"])),
        f"Sequence length observed: {summary['metadata']['seq_len_min']}–"
        f"{summary['metadata']['seq_len_max']}",
        "",
        "## Baselines",
        "",
        f"- Fixed 1:2 Top-720 mass: {b['fixed_1to2_top720_reassembled']['mean']:.4f}",
        f"- Unrestricted per-pair Top-720 mass: {b['unrestricted_per_pair_top720']['mean']:.4f}",
        f"- Unrestricted per-MLA-head Top-720 mass: {b['unrestricted_per_mla_head_top720']['mean']:.4f}",
        f"- Official shared Top-2048 mass: {b['official_shared_top2048']['mean']:.4f}",
        f"- Exact MLA Top-720 mass ceiling: {b['exact_mla_top720']['mean']:.4f}",
        "",
        "## Pair router (one indexer head per MLA pair)",
        "",
        "| Unique budget | Dynamic set oracle | Static set + dynamic assignment (CV) | Static set + static assignment (CV) | Static/dynamic ÷ unrestricted | Dynamic-set Jaccard |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, value in summary["routes"]["pair"].items():
        dynamic = value["dynamic_set_dynamic_assignment_oracle"]["mean"]
        static_dynamic = value["cv_static_set_dynamic_assignment"]["mean"]
        static_static = value["cv_static_set_static_assignment"]["mean"]
        ratio = value["cv_static_set_dynamic_assignment"][
            "fraction_of_unrestricted"
        ]
        lines.append(
            f"| {budget} | {dynamic:.4f} | {static_dynamic:.4f} | "
            f"{static_static:.4f} | {ratio:.3%} | "
            f"{value['dynamic_set_jaccard_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## System budget",
            "",
            "Assuming int32 Top-720 indices, storing all 64 candidate lists costs "
            "184,320 bytes per query/layer. A budget of M stores "
            "M × 720 × 4 bytes and ideally reduces logical indexer scans by 64/M.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--queries-per-prompt", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mass, samples, layers, scalar, metadata = load_probe(
        args.probe_dir, args.manifest, args.expected_ranks
    )
    sample_groups = np.asarray(samples) // args.queries_per_prompt
    pair, pair_detail = analyze_route(
        mass, layers, args.budgets, True, sample_groups
    )
    head, head_detail = analyze_route(
        mass, layers, args.budgets, False, sample_groups
    )
    per_layer = {}
    for layer_index, layer in enumerate(layers):
        layer_mass = mass[:, layer_index : layer_index + 1]
        layer_pair, _ = analyze_route(
            layer_mass, [layer], args.budgets, True, sample_groups
        )
        layer_head, _ = analyze_route(
            layer_mass, [layer], args.budgets, False, sample_groups
        )
        per_layer[str(layer)] = {
            "baselines": baseline_metrics(layer_mass, [layer], scalar),
            "routes": {"pair": layer_pair, "mla_head": layer_head},
        }
    summary = {
        "metadata": {
            **metadata,
            "n_samples": len(samples),
            "samples": samples,
            "utility_shape": list(mass.shape),
            "queries_per_prompt": args.queries_per_prompt,
            "n_prompt_groups": int(np.unique(sample_groups).size),
        },
        "layers": layers,
        "budgets": args.budgets,
        "baselines": baseline_metrics(mass, layers, scalar),
        "routes": {"pair": pair, "mla_head": head},
        "per_layer": per_layer,
        "selection_detail": {"pair": pair_detail, "mla_head": head_detail},
        "system": {
            str(m): {
                "logical_scan_reduction": 64 / m,
                "top720_indices_bytes": m * 720 * 4,
                "fraction_of_64_head_indices": m / 64,
            }
            for m in args.budgets
        },
    }
    add_ratios(summary)
    for layer_summary in summary["per_layer"].values():
        for route, denominator_key in (
            ("pair", "unrestricted_per_pair_top720"),
            ("mla_head", "unrestricted_per_mla_head_top720"),
        ):
            denominator = layer_summary["baselines"][denominator_key]["mean"]
            for values in layer_summary["routes"][route].values():
                for metric in (
                    "dynamic_set_dynamic_assignment_oracle",
                    "cv_static_set_dynamic_assignment",
                    "cv_static_set_static_assignment",
                ):
                    values[metric]["fraction_of_unrestricted"] = (
                        values[metric]["mean"] / denominator
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = args.out.with_suffix(".md")
    write_report(summary, report)
    print(f"wrote {args.out} and {report}")


if __name__ == "__main__":
    main()
