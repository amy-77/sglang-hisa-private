#!/usr/bin/env python3
"""Build and validate an SGLang offline-router runtime configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--oracle-summary",
        type=Path,
        help="Required only for fixed-set/static-assignment ablations.",
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--assignment-mode",
        choices=("learned", "static"),
        default="learned",
        help="Use the trained per-query scorer or the full-data static pair map.",
    )
    parser.add_argument(
        "--head-selection-mode",
        choices=("misa", "static"),
        help=(
            "Select M heads per query from pooled keys (misa), or retain the "
            "offline fixed head set (static). Defaults to misa with learned "
            "assignment and static with static assignment."
        ),
    )
    parser.add_argument(
        "--misa-chunk-size",
        type=int,
        default=256,
        help="Number of historical tokens per mean-pooled MISA routing chunk.",
    )
    parser.add_argument(
        "--misa-prune-keep-fraction",
        type=float,
        default=None,
        help="Fraction of top MISA chunks retained for chunk16 scoring.",
    )
    parser.add_argument("--misa-prune-topk", type=int, default=None,
                        help="Fixed coarse count; defaults to 8 unless a legacy fraction is set.")
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=4096,
        help="Minimum visible context length before sparse routing is enabled.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Optional deployed layer subset; defaults to every checkpoint layer.",
    )
    parser.add_argument(
        "--description", default="MISA TP-local 16-Head Assignment Router"
    )
    args = parser.parse_args()

    head_selection_mode = args.head_selection_mode
    if head_selection_mode is None:
        head_selection_mode = "misa" if args.assignment_mode == "learned" else "static"
    if head_selection_mode == "misa" and args.assignment_mode == "static":
        parser.error("--head-selection-mode misa requires --assignment-mode learned")
    if args.misa_chunk_size <= 0 or args.misa_chunk_size % 16:
        parser.error("--misa-chunk-size must be a positive multiple of 16")
    if args.misa_prune_keep_fraction is not None and not 0 < args.misa_prune_keep_fraction <= 1:
        parser.error("--misa-prune-keep-fraction must be in (0, 1]")
    if args.misa_prune_topk is not None and args.misa_prune_topk <= 0:
        parser.error("--misa-prune-topk must be positive")
    if args.misa_prune_topk is not None and args.misa_prune_keep_fraction is not None:
        parser.error("choose either --misa-prune-topk or --misa-prune-keep-fraction")
    if args.misa_prune_topk is None and args.misa_prune_keep_fraction is None:
        args.misa_prune_topk = 8
    if args.min_seq_len < 720:
        parser.error("--min-seq-len must be at least 720")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    group_size = int(checkpoint.get("mla_heads_per_group", 2))
    if args.assignment_mode == "learned" and (
        group_size != 16 or checkpoint.get("assignment_unit") != "tp_local"
        or checkpoint.get("attention_tp_size") != 8
    ):
        parser.error("learned group16 configs require a freshly trained TP-local 16-head checkpoint; pair checkpoints are not accepted")
    checkpoint_layers = [int(layer) for layer in checkpoint["layers"]]
    requested_layers = checkpoint_layers if args.layers is None else args.layers
    missing_checkpoint = sorted(set(requested_layers) - set(checkpoint_layers))
    if missing_checkpoint:
        parser.error(f"layers absent from checkpoint: {missing_checkpoint}")

    summary = None
    if args.oracle_summary is not None:
        summary = json.loads(args.oracle_summary.read_text())
    if head_selection_mode == "misa":
        # Values are coverage markers only; runtime MISA replaces them per query.
        selected = {
            str(layer): list(range(args.budget)) for layer in requested_layers
        }
    else:
        if summary is None:
            parser.error("--oracle-summary is required for static head selection")
        try:
            all_sets = summary["selection_detail"]["pair"][str(args.budget)][
                "full_data_static_set_by_layer"
            ]
        except KeyError as exc:
            parser.error(f"oracle summary has no pair budget M={args.budget}: {exc}")
        selected = {str(layer): all_sets[str(layer)] for layer in requested_layers}
    for layer, heads in selected.items():
        if len(heads) != args.budget or len(set(heads)) != args.budget:
            parser.error(f"invalid M={args.budget} set for layer {layer}: {heads}")

    static_assignment = None
    if args.assignment_mode == "static":
        assert summary is not None
        try:
            all_assignments = summary["selection_detail"]["pair"][
                str(args.budget)
            ]["full_data_static_assignment_by_layer"]
        except KeyError as exc:
            parser.error(
                "oracle summary has no full-data static pair assignment; "
                f"rerun analyze_128k_unique_head_oracle.py: {exc}"
            )
        static_assignment = {
            str(layer): all_assignments[str(layer)] for layer in requested_layers
        }
        for layer, assignment in static_assignment.items():
            if len(assignment) != 64:
                parser.error(
                    f"layer {layer} static assignment has {len(assignment)} pairs, "
                    "expected 64"
                )
            outside = sorted(set(assignment) - set(selected[layer]))
            if outside:
                parser.error(
                    f"layer {layer} static assignment uses heads outside its "
                    f"M={args.budget} set: {outside}"
                )

    payload = {
        "format_version": 1,
        "description": args.description,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "budget": args.budget,
        "assignment_mode": args.assignment_mode,
        "mla_heads_per_group": group_size,
        "attention_tp_size": checkpoint.get("attention_tp_size"),
        "head_selection_mode": head_selection_mode,
        "misa_chunk_size": args.misa_chunk_size,
        "misa_prune_keep_fraction": args.misa_prune_keep_fraction,
        "misa_prune_topk": args.misa_prune_topk,
        "min_seq_len": args.min_seq_len,
        "selected_heads_by_layer": selected,
        "coverage": {
            "trained_layers": len(checkpoint_layers),
            "deployed_layers": len(requested_layers),
            "model_layers_expected": 61,
            "is_full_layer_coverage": set(requested_layers) == set(range(61)),
        },
        "training_config": checkpoint.get("training_config"),
        "train_samples": checkpoint.get("train_samples"),
    }
    if args.oracle_summary is not None:
        payload["oracle_summary"] = str(args.oracle_summary.resolve())
        payload["oracle_summary_sha256"] = sha256(args.oracle_summary)
    if static_assignment is not None:
        payload["static_assignment_by_layer"] = static_assignment
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.out}: M={args.budget}, "
        f"selection={head_selection_mode}, chunk={args.misa_chunk_size}, "
        f"coarse_topk={args.misa_prune_topk}, keep_fraction={args.misa_prune_keep_fraction}, "
        f"min_seq_len={args.min_seq_len}, "
        f"layers={len(requested_layers)}/{payload['coverage']['model_layers_expected']}"
    )


if __name__ == "__main__":
    main()
