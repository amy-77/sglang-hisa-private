#!/usr/bin/env python3
"""Train and evaluate the MISA candidate-to-TP-local 16-head assignment router."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch  # noqa: E402

from misa_router_dataset import load_router_dataset  # noqa: E402
from misa_router_trainer import (  # noqa: E402
    TrainConfig,
    evaluate_router,
    save_checkpoint,
    train_router,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--queries-per-prompt", type=int, default=1)
    parser.add_argument(
        "--probe-sample-stride",
        type=int,
        default=1,
        help="Keep every Nth probe record and map raw sample id N*j to j.",
    )
    parser.add_argument("--min-seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--top1-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_907)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        **{field.name: getattr(args, field.name) for field in fields(TrainConfig)}
    )
    data = load_router_dataset(
        args.probe_dir,
        args.sample_manifest,
        expected_ranks=args.expected_ranks,
        queries_per_prompt=args.queries_per_prompt,
        min_seq_len=args.min_seq_len,
        probe_sample_stride=args.probe_sample_stride,
    )
    device = torch.device(args.device)
    model, history = train_router(data, config, device)
    report = {
        "contexts": len(data),
        "assignment_unit": "tp_local",
        "mla_heads_per_group": 16,
        "attention_tp_size": 8,
        "initialization": "random",
        "probe_dir": str(args.probe_dir.resolve()),
        "prompt_groups": int(data.prompt_ids.unique().numel()),
        "datasets": data.dataset_names,
        "config": asdict(config),
        "history": history,
        "validation": evaluate_router(
            model, data, "validation", args.eval_batch_size, device
        ),
        "test": evaluate_router(
            model, data, "test", args.eval_batch_size, device
        ),
    }
    save_checkpoint(args.checkpoint, model, data, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.checkpoint} and {args.out}")


if __name__ == "__main__":
    main()
