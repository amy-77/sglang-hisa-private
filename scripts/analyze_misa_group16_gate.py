#!/usr/bin/env python3
"""Gate Group16 training on a newly collected MISA candidate policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch  # noqa: E402

from misa_router_dataset import load_router_dataset  # noqa: E402
from misa_router_trainer import _static_assignment  # noqa: E402


def summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.double().flatten()
    return {
        "mean": float(values.mean()),
        "p10": float(values.quantile(0.10)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "count": int(values.numel()),
    }


def evaluate_split(data, split: str, dataset: str | None = None) -> dict:
    mask = data.split_mask(split)
    if dataset is not None:
        dataset_id = data.dataset_names.index(dataset)
        mask &= data.dataset_ids == dataset_id
    indices = mask.nonzero().flatten()
    if indices.numel() == 0:
        return {}

    utility = data.utility[indices].float()
    oracle_per_context = utility.max(dim=-1).values.mean(dim=-1)
    static_head_utility = _static_assignment(data)
    static_scores = torch.stack(
        [
            static_head_utility[int(layer), :, heads]
            for layer, heads in zip(
                data.layer_ids[indices],
                data.candidate_head_ids[indices],
            )
        ]
    )
    static_choices = static_scores.argmax(dim=-1, keepdim=True)
    static_per_context = utility.gather(-1, static_choices).squeeze(-1).mean(-1)
    return {
        "group16_candidate_oracle": summary(oracle_per_context),
        "static_assignment": summary(static_per_context),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--probe-sample-stride", type=int, default=1)
    parser.add_argument("--min-seq-len", type=int, default=4096)
    parser.add_argument("--old-control", type=Path)
    parser.add_argument("--oracle-tolerance", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_router_dataset(
        args.probe_dir,
        args.sample_manifest,
        expected_ranks=8,
        queries_per_prompt=1,
        min_seq_len=args.min_seq_len,
        probe_sample_stride=args.probe_sample_stride,
    )
    report = {
        "policy": {
            "misa_chunk_size": 256,
            "misa_prune_topk": 8,
            "head_importance": "signed_gate_times_top8_positive_coarse_affinity_sum",
            "fine_chunk_size": 16,
            "dense_chunks": 52,
            "dense_token_quota": 8,
            "sparse_chunks": 76,
            "sparse_token_quota": 4,
            "candidate_tokens": 720,
        },
        "contexts": len(data),
        "prompt_groups": int(data.prompt_ids.unique().numel()),
        "probe_sample_stride": args.probe_sample_stride,
        "validation": {
            "all": evaluate_split(data, "validation"),
            "ruler": evaluate_split(data, "validation", "ruler"),
        },
    }

    if args.old_control is not None:
        old = json.loads(args.old_control.read_text(encoding="utf-8"))
        comparisons = {}
        passes = []
        for name, old_key in (("all", "validation_all"), ("ruler", "validation_ruler")):
            new_mean = report["validation"][name]["group16_candidate_oracle"]["mean"]
            old_mean = float(old[old_key]["group16_oracle_mass"])
            delta = new_mean - old_mean
            passed = delta + args.oracle_tolerance >= 0
            comparisons[name] = {
                "old_group16_oracle": old_mean,
                "new_group16_oracle": new_mean,
                "delta": delta,
                "passes": passed,
            }
            passes.append(passed)
        report["comparison"] = comparisons
        report["gate"] = {
            "criterion": (
                "new Group16 candidate oracle must not be below the matched old "
                "candidate oracle on both all-validation and RULER-validation"
            ),
            "oracle_tolerance": args.oracle_tolerance,
            "train_new_group16_router": all(passes),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
