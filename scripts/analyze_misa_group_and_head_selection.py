#!/usr/bin/env python3
"""Analyze MISA head-set selection and fixed/shared MLA-head grouping oracles."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


SHARD_RE = re.compile(
    r"probe_shard_L(?P<layer>\d+)_rank(?P<rank>\d+)_"
    r"s(?P<start>\d+)-(?P<end>\d+)\.pt$"
)
REFERENCE_MEANS = {
    "actual_m8_contiguous_2": 0.6503818498288103,
    "actual_m8_contiguous_16": 0.6450113780447397,
    "all64_contiguous_2": 0.7060432015794399,
    "all64_contiguous_16": 0.696628657550226,
}


def parse_args() -> argparse.Namespace:
    base = Path("/DATA/disk0/qyl/data/ruler_router_decomposition_20260909")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=base / "probe")
    parser.add_argument("--sample-manifest", type=Path, default=base / "samples.jsonl")
    parser.add_argument(
        "--out", type=Path, default=base / "reports/group_and_head_selection.json"
    )
    parser.add_argument(
        "--log", type=Path, default=base / "reports/group_and_head_selection.log"
    )
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--random-head-sets", type=int, default=32)
    parser.add_argument("--random-partitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--reference-tolerance", type=float, default=2e-7)
    return parser.parse_args()


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, mode="w"), logging.StreamHandler()],
    )


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    metadata = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_ids = row.get("probe_sample_ids") or [row["sample_id"]]
            for sample_id in sample_ids:
                metadata[int(sample_id)] = row
    return metadata


def discover_shard_groups(probe_dir: Path) -> list[tuple[tuple[int, int, int], list[Path]]]:
    groups: dict[tuple[int, int, int], dict[int, Path]] = defaultdict(dict)
    for path in probe_dir.glob("probe_shard_L*_rank*_s*.pt"):
        match = SHARD_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (
            int(match.group("layer")),
            int(match.group("start")),
            int(match.group("end")),
        )
        rank = int(match.group("rank"))
        if rank in groups[key]:
            raise ValueError(f"duplicate rank {rank} for shard group {key}")
        groups[key][rank] = path
    result = []
    for key, by_rank in sorted(groups.items()):
        if sorted(by_rank) != list(range(8)):
            raise ValueError(f"shard group {key} does not contain TP ranks 0..7")
        result.append((key, [by_rank[rank] for rank in range(8)]))
    if not result:
        raise RuntimeError(f"no probe shards found in {probe_dir}")
    return result


def summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "count": 0}
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "count": int(array.size),
    }


def contiguous_oracle(mass: np.ndarray, candidates: np.ndarray, size: int) -> float:
    groups = mass.reshape(64, 128 // size, size).mean(axis=2)
    return float(groups[candidates].max(axis=0).mean())


def partition_oracle(
    mass: np.ndarray, candidates: np.ndarray, partition: np.ndarray
) -> float:
    groups = mass[:, partition].reshape(64, 8, 16).mean(axis=2)
    return float(groups[candidates].max(axis=0).mean())


def align_context(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    mass = np.empty((64, 128), dtype=np.float32)
    importance_by_rank = []
    for rank, row in enumerate(records):
        if int(row["tp_rank"]) != rank:
            raise ValueError("rank record order mismatch")
        global_heads = row["global_mla_head_ids"].cpu().numpy().astype(np.int64)
        expected_heads = np.arange(rank * 16, (rank + 1) * 16)
        if not np.array_equal(global_heads, expected_heads):
            raise ValueError(
                f"rank {rank} has non-contiguous or reordered global MLA heads"
            )
        candidate_ids = row["candidate_head_ids"].cpu().numpy().astype(np.int64)
        if (
            candidate_ids.shape != (64,)
            or np.unique(candidate_ids).size != 64
            or candidate_ids.min() != 0
            or candidate_ids.max() != 63
        ):
            raise ValueError("candidate_head_ids must be a permutation of 0..63")
        local_mass = row["attention_mass_matrix"].float().cpu().numpy()
        if local_mass.shape != (64, 16):
            raise ValueError("attention_mass_matrix must have shape [64, 16]")
        mass[candidate_ids[:, None], global_heads[None, :]] = local_mass
        local_importance = np.empty(64, dtype=np.float32)
        local_importance[candidate_ids] = (
            row["candidate_importance"].float().cpu().numpy()
        )
        importance_by_rank.append(local_importance)
    importance = importance_by_rank[0]
    for rank_importance in importance_by_rank[1:]:
        if not np.allclose(importance, rank_importance, rtol=0.0, atol=1e-6):
            raise ValueError("candidate_importance differs across TP ranks")
    return mass, importance


def main() -> None:
    args = parse_args()
    if args.budget != 8:
        raise ValueError("this report's strict Group16 upper-bound argument requires M=8")
    if args.random_head_sets < 32 or args.random_partitions < 16:
        raise ValueError("require at least 32 random head sets and 16 partitions")
    setup_logging(args.log)
    metadata = load_manifest(args.sample_manifest)
    shard_groups = discover_shard_groups(args.probe_dir)
    logging.info("found %d complete shard groups", len(shard_groups))

    rng_sets = np.random.default_rng(args.seed)
    random_sets = np.stack(
        [np.sort(rng_sets.choice(64, args.budget, replace=False))
         for _ in range(args.random_head_sets)]
    )
    partition_seeds = [args.seed + 100_000 + index for index in range(args.random_partitions)]
    random_partitions = np.stack(
        [np.random.default_rng(seed).permutation(128) for seed in partition_seeds]
    )

    grouping_sizes = (1, 2, 4, 8, 16, 32, 64, 128)
    metrics: dict[str, list[float]] = defaultdict(list)
    by_length: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    random_set_context = np.empty((args.random_head_sets, 0), dtype=np.float32)
    random_partition_actual = np.empty((args.random_partitions, 0), dtype=np.float32)
    random_partition_all64 = np.empty((args.random_partitions, 0), dtype=np.float32)
    random_set_columns = []
    random_partition_actual_columns = []
    random_partition_all64_columns = []
    contexts = 0

    for group_index, (shard_key, paths) in enumerate(shard_groups):
        rank_payloads = [
            torch.load(path, map_location="cpu", weights_only=False) for path in paths
        ]
        lengths = {len(payload) for payload in rank_payloads}
        if len(lengths) != 1:
            raise ValueError(f"rank payload lengths differ for {shard_key}")
        for rows in zip(*rank_payloads):
            records = list(rows)
            sample_id = int(records[0]["sample_id"])
            layer = int(records[0]["layer"])
            if any(
                int(row["sample_id"]) != sample_id or int(row["layer"]) != layer
                for row in records
            ):
                raise ValueError(f"context order mismatch in shard group {shard_key}")
            if sample_id not in metadata:
                raise ValueError(f"sample {sample_id} absent from manifest")
            mass, _ = align_context(records)
            # Match the deployed/probe selector exactly: top-k is evaluated in
            # the saved candidate order, then slots are mapped to global IDs.
            # Reordering scores by global ID first can change zero-score ties.
            saved_candidate_ids = records[0]["candidate_head_ids"].long()
            saved_topk_slots = records[0]["candidate_importance"].float().topk(
                args.budget, largest=True, sorted=False
            ).indices
            actual = saved_candidate_ids[saved_topk_slots].cpu().numpy()
            all64 = np.arange(64)

            context_values = {}
            for candidate_name, candidates in (("actual_m8", actual), ("all64", all64)):
                per_head = contiguous_oracle(mass, candidates, 1)
                for size in grouping_sizes:
                    value = contiguous_oracle(mass, candidates, size)
                    name = f"{candidate_name}_contiguous_{size}"
                    metrics[name].append(value)
                    context_values[name] = value
                    regret_name = f"{candidate_name}_grouping_regret_{size}"
                    regret = per_head - value
                    metrics[regret_name].append(regret)
                    context_values[regret_name] = regret

            group16 = mass.reshape(64, 8, 16).mean(axis=2)
            all64_winners = group16.argmax(axis=0)
            optimal_set = np.unique(all64_winners)
            overlap = np.intersect1d(actual, optimal_set).size
            metrics["optimal_group16_unique_indexers"].append(float(optimal_set.size))
            metrics["actual_m8_optimal_set_overlap_count"].append(float(overlap))
            metrics["actual_m8_optimal_set_recall"].append(
                float(overlap / optimal_set.size)
            )
            metrics["actual_m8_optimal_set_precision"].append(
                float(overlap / args.budget)
            )
            head_selection_regret = (
                context_values["all64_contiguous_16"]
                - context_values["actual_m8_contiguous_16"]
            )
            metrics["head_selection_regret"].append(head_selection_regret)
            context_values["head_selection_regret"] = head_selection_regret

            random_set_columns.append(
                np.asarray(
                    [contiguous_oracle(mass, candidate_set, 16)
                     for candidate_set in random_sets],
                    dtype=np.float32,
                )
            )
            random_partition_actual_columns.append(
                np.asarray(
                    [partition_oracle(mass, actual, partition)
                     for partition in random_partitions],
                    dtype=np.float32,
                )
            )
            random_partition_all64_columns.append(
                np.asarray(
                    [partition_oracle(mass, all64, partition)
                     for partition in random_partitions],
                    dtype=np.float32,
                )
            )

            length = str(metadata[sample_id].get("length", "unknown"))
            for name in (
                "actual_m8_contiguous_16",
                "all64_contiguous_16",
                "head_selection_regret",
                "actual_m8_grouping_regret_16",
            ):
                by_length[length][name].append(context_values[name])
            contexts += 1
        if (group_index + 1) % 50 == 0 or group_index + 1 == len(shard_groups):
            logging.info(
                "processed %d/%d shard groups (%d contexts)",
                group_index + 1,
                len(shard_groups),
                contexts,
            )

    random_set_context = np.stack(random_set_columns, axis=1)
    random_partition_actual = np.stack(random_partition_actual_columns, axis=1)
    random_partition_all64 = np.stack(random_partition_all64_columns, axis=1)

    random_set_means = random_set_context.mean(axis=1, dtype=np.float64)
    actual_group16_mean = summary(metrics["actual_m8_contiguous_16"])["mean"]
    random_set_mean = float(random_set_means.mean())
    actual_uplift = float(actual_group16_mean - random_set_mean)

    partition_report = {}
    for name, values, contiguous_key in (
        ("actual_m8", random_partition_actual, "actual_m8_contiguous_16"),
        ("all64", random_partition_all64, "all64_contiguous_16"),
    ):
        partition_means = values.mean(axis=1, dtype=np.float64)
        contiguous_mean = float(summary(metrics[contiguous_key])["mean"])
        percentile = float(100.0 * np.mean(partition_means <= contiguous_mean))
        partition_report[name] = {
            "random_partition_mean_oracle_distribution": summary(partition_means),
            "contiguous_group16_mean": contiguous_mean,
            "contiguous_minus_random_partition_mean": float(
                contiguous_mean - partition_means.mean()
            ),
            "contiguous_percentile_among_random_partitions": percentile,
        }

    core_keys = (
        "actual_m8_contiguous_16",
        "all64_contiguous_16",
        "head_selection_regret",
        "actual_m8_grouping_regret_16",
    )
    report = {
        "experiment": {
            "probe_dir": str(args.probe_dir),
            "sample_manifest": str(args.sample_manifest),
            "contexts": contexts,
            "candidate_indexer_ids": "global IDs 0..63, aligned via candidate_head_ids",
            "mla_head_ids": "global IDs 0..127, assembled from TP ranks 0..7",
            "budget": args.budget,
            "seed": args.seed,
        },
        "definitions": {
            "actual_m8": (
                "Per-context top-8 by saved candidate_importance in saved candidate "
                "order, then mapped through candidate_head_ids to global Indexer IDs. "
                "Keeping saved order preserves torch.topk tie behavior."
            ),
            "head_selection_loss": (
                "all64 Group16 oracle - actual-M8 Group16 oracle."
            ),
            "grouping_shared_candidate_loss": (
                "per-head oracle - same-candidate-range grouped oracle."
            ),
            "learning_loss_not_measured": (
                "router-vs-M8-oracle is assignment/router learning loss; no router "
                "was trained or evaluated in this experiment."
            ),
        },
        "head_set_quality": {
            "actual_m8_group16_oracle": summary(metrics["actual_m8_contiguous_16"]),
            "all64_group16_oracle_strict_budget8_upper_bound": summary(
                metrics["all64_contiguous_16"]
            ),
            "head_selection_regret": summary(metrics["head_selection_regret"]),
            "optimal_group16_unique_indexers": summary(
                metrics["optimal_group16_unique_indexers"]
            ),
            "actual_m8_vs_optimal_set": {
                "overlap_count": summary(
                    metrics["actual_m8_optimal_set_overlap_count"]
                ),
                "recall": summary(metrics["actual_m8_optimal_set_recall"]),
                "precision": summary(metrics["actual_m8_optimal_set_precision"]),
            },
            "random_m8_baseline": {
                "sets": random_sets.tolist(),
                "set_count": args.random_head_sets,
                "mean_group16_oracle_distribution_across_fixed_sets": summary(
                    random_set_means
                ),
                "actual_m8_minus_random_mean": actual_uplift,
                "actual_m8_relative_improvement_fraction": (
                    float(actual_uplift / random_set_mean)
                ),
            },
        },
        "grouping_quality": {
            candidate_name: {
                str(size): {
                    "oracle": summary(
                        metrics[f"{candidate_name}_contiguous_{size}"]
                    ),
                    "regret_vs_per_head": summary(
                        metrics[f"{candidate_name}_grouping_regret_{size}"]
                    ),
                    "deployability": (
                        "current contiguous layout"
                        if size <= 16
                        else "diagnostic only; crosses current TP rank boundaries"
                    ),
                }
                for size in grouping_sizes
            }
            for candidate_name in ("actual_m8", "all64")
        },
        "random_balanced_group16_partitions": {
            "partition_count": args.random_partitions,
            "partition_seeds": partition_seeds,
            "scope": (
                "Each fixed global partition is reused for every context. These "
                "cross-TP partitions are diagnostic and not directly deployable."
            ),
            **partition_report,
        },
        "by_length": {
            length: {key: summary(values[key]) for key in core_keys}
            for length, values in sorted(by_length.items())
        },
        "consistency_checks": {},
        "limitations": [
            (
                "The probe has candidate-level attention mass, not complete token "
                "attention vectors, so exact Group16 shared-token Top720 cannot be "
                "computed or claimed."
            ),
            (
                "exact_mla_top720_mass is an independent Top720 result for each MLA "
                "head and is intentionally not used as Group16 shared-token Top720."
            ),
            (
                "All metrics are oracle analyses on the fixed probe candidates; "
                "there is no router training or learned-router evaluation."
            ),
            (
                "Random balanced Group16 partitions and contiguous sizes above 16 "
                "cross TP boundaries and are diagnostic, not current deployable layouts."
            ),
        ],
    }

    checks = {}
    all_pass = True
    for name, expected in REFERENCE_MEANS.items():
        observed = float(summary(metrics[name])["mean"])
        error = abs(observed - expected)
        passed = error <= args.reference_tolerance
        checks[name] = {
            "observed": observed,
            "expected": expected,
            "absolute_error": error,
            "tolerance": args.reference_tolerance,
            "passed": passed,
        }
        all_pass &= passed
    checks["all_passed"] = all_pass
    report["consistency_checks"] = checks
    if not all_pass:
        raise RuntimeError(
            "reference consistency checks failed; refusing to write a misleading report: "
            + json.dumps(checks)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("wrote %s", args.out)
    logging.info("reference consistency checks passed")


if __name__ == "__main__":
    main()
