#!/usr/bin/env python3
"""Measure exact Top-K token-index overlap among adjacent MLA heads."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


SHARD_RE = re.compile(r"probe_shard_L(?P<layer>\d+)_rank(?P<rank>\d+)_s.*\.pt")


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "count": int(array.size),
    }


def load_lengths(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[int(row["sample_id"])] = str(row.get("length") or "unknown")
    return result


def pair_metrics(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heads = ids.shape[0]
    intersection = np.eye(heads, dtype=np.float64)
    jaccard = np.eye(heads, dtype=np.float64)
    sets = [set(row.tolist()) for row in ids]
    for left, right in combinations(range(heads), 2):
        common = len(sets[left] & sets[right])
        intersection[left, right] = intersection[right, left] = common / ids.shape[1]
        union = len(sets[left] | sets[right])
        jaccard[left, right] = jaccard[right, left] = common / union
    return intersection, jaccard


def grouped_pair_mean(matrix: np.ndarray, partition: np.ndarray) -> float:
    values = []
    for group in partition.reshape(-1, 8):
        values.extend(matrix[left, right] for left, right in combinations(group, 2))
    return float(np.mean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=720)
    parser.add_argument("--random-partitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = load_lengths(args.sample_manifest)
    contiguous = np.arange(16)
    random_partitions = [
        np.random.default_rng(args.seed + index).permutation(16)
        for index in range(args.random_partitions)
    ]
    values: dict[str, list[float]] = defaultdict(list)
    by_length: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_layer: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    random_intersection = [[] for _ in random_partitions]
    random_jaccard = [[] for _ in random_partitions]
    rank_contexts = 0

    paths = sorted(args.probe_dir.glob("probe_shard_L*_rank*_s*.pt"))
    if not paths:
        raise RuntimeError(f"no probe shards found in {args.probe_dir}")
    for path in paths:
        match = SHARD_RE.fullmatch(path.name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        for row in torch.load(path, map_location="cpu", weights_only=False):
            ids_tensor = row.get("exact_mla_top720_ids")
            if ids_tensor is None:
                raise ValueError(f"{path} does not contain exact_mla_top720_ids")
            ids = ids_tensor.cpu().numpy().astype(np.int64)
            if ids.shape != (16, args.topk):
                raise ValueError(
                    f"expected exact IDs [16,{args.topk}], got {ids.shape}"
                )
            intersection, jaccard = pair_metrics(ids)
            adjacent_intersection = grouped_pair_mean(intersection, contiguous)
            adjacent_jaccard = grouped_pair_mean(jaccard, contiguous)
            group_union_ratios = []
            group_common_ratios = []
            group_reuse_fractions = []
            for group in contiguous.reshape(2, 8):
                sets = [set(ids[head].tolist()) for head in group]
                union_size = len(set.union(*sets))
                common_size = len(set.intersection(*sets))
                group_union_ratios.append(union_size / args.topk)
                group_common_ratios.append(common_size / args.topk)
                group_reuse_fractions.append(
                    1.0 - union_size / (len(group) * args.topk)
                )
            record = {
                "pairwise_intersection_fraction": adjacent_intersection,
                "pairwise_jaccard": adjacent_jaccard,
                "union_size_over_topk": float(np.mean(group_union_ratios)),
                "all8_common_fraction": float(np.mean(group_common_ratios)),
                "index_reuse_fraction": float(np.mean(group_reuse_fractions)),
            }
            length = lengths[int(row["sample_id"])]
            for name, value in record.items():
                values[name].append(value)
                by_length[length][name].append(value)
                by_layer[layer][name].append(value)
            for index, partition in enumerate(random_partitions):
                random_intersection[index].append(
                    grouped_pair_mean(intersection, partition)
                )
                random_jaccard[index].append(grouped_pair_mean(jaccard, partition))
            rank_contexts += 1

    random_intersection_means = [
        float(np.mean(partition_values))
        for partition_values in random_intersection
    ]
    random_jaccard_means = [
        float(np.mean(partition_values)) for partition_values in random_jaccard
    ]
    adjacent_intersection_mean = float(np.mean(values["pairwise_intersection_fraction"]))
    adjacent_jaccard_mean = float(np.mean(values["pairwise_jaccard"]))
    report = {
        "definition": {
            "heads": "two contiguous 8-MLA-head groups within each TP-local 16-head rank",
            "indices": f"each MLA head's exact attention Top-{args.topk} token IDs",
            "pairwise_intersection_fraction": f"|A intersect B| / {args.topk}",
            "pairwise_jaccard": "|A intersect B| / |A union B|",
            "union_size_over_topk": f"|union of 8 sets| / {args.topk}",
            "all8_common_fraction": f"|intersection of all 8 sets| / {args.topk}",
            "index_reuse_fraction": (
                f"1 - |union of 8 sets| / (8 * {args.topk}); "
                "0 means disjoint and 0.875 means identical"
            ),
        },
        "rank_contexts": rank_contexts,
        "adjacent_group_instances": rank_contexts * 2,
        "overall": {name: summarize(metric) for name, metric in values.items()},
        "random_balanced_local_group8": {
            "partition_count": args.random_partitions,
            "scope": (
                "Each fixed permutation partitions the same TP-local 16 heads "
                "into two groups of 8 and is reused across all samples."
            ),
            "pairwise_intersection_fraction": summarize(random_intersection_means),
            "pairwise_jaccard": summarize(random_jaccard_means),
            "adjacent_minus_random_mean_intersection_fraction": (
                adjacent_intersection_mean - float(np.mean(random_intersection_means))
            ),
            "adjacent_minus_random_mean_jaccard": (
                adjacent_jaccard_mean - float(np.mean(random_jaccard_means))
            ),
            "adjacent_intersection_percentile": (
                100.0
                * float(
                    np.mean(
                        np.asarray(random_intersection_means)
                        <= adjacent_intersection_mean
                    )
                )
            ),
        },
        "by_length": {
            length: {name: summarize(metric) for name, metric in metrics.items()}
            for length, metrics in sorted(by_length.items())
        },
        "by_layer": {
            str(layer): {name: summarize(metric) for name, metric in metrics.items()}
            for layer, metrics in sorted(by_layer.items())
        },
        "limitations": [
            "This measures exact MLA attention Top-K set overlap, not overlap among Indexer-generated candidate lists.",
            "Overlap alone does not measure retained attention mass or output error.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
