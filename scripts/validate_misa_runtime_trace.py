#!/usr/bin/env python3
"""Replay traced runtime routing and verify scores, slots, TP, and output indices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from sglang.srt.layers.attention.nsa.assignment_router_model import (
    MISAAssignmentRouter,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MISAAssignmentRouter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=device, dtype=torch.float32).eval()
    layer_to_id = {
        int(layer): index for index, layer in enumerate(checkpoint["layers"])
    }

    traces = []
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(args.trace_dir.glob("route_L*_rank*.pt")):
        row = torch.load(path, map_location="cpu", weights_only=False)
        traces.append(row)
        by_layer[int(row["layer_id"])].append(row)
    if not traces:
        raise RuntimeError(f"no runtime traces found in {args.trace_dir}")

    max_score_error = 0.0
    score_choice_mismatches = 0
    output_gather_mismatches = 0
    rank_candidate_head_mismatches = 0
    rank_candidate_token_mismatches = 0
    shape_failures = 0

    for row in traces:
        layer_id = int(row["layer_id"])
        batch_size = row["q_group"].shape[0]

        def move(name: str, *, fp32: bool = False) -> torch.Tensor:
            value = row[name].to(device)
            return value.float() if fp32 else value

        score = model(
            move("q_group", fp32=True),
            move("candidate_head_ids").long(),
            move("candidate_q_indexer", fp32=True),
            move("candidate_context_summary", fp32=True),
            move("candidate_gate", fp32=True),
            move("candidate_importance", fp32=True),
            move("candidate_context_stats", fp32=True),
            torch.full(
                (batch_size,),
                layer_to_id[layer_id],
                dtype=torch.long,
                device=device,
            ),
        ).cpu()
        max_score_error = max(
            max_score_error, float((score - row["score"]).abs().max())
        )
        choice = score.argmax(dim=-1)
        score_choice_mismatches += int((choice != row["choice"]).sum())

        candidates = row["candidates"]
        gather = choice.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
        expected_output = (
            candidates.unsqueeze(1)
            .expand(-1, choice.shape[1], -1, -1)
            .gather(2, gather.unsqueeze(2))
            .squeeze(2)
            .contiguous()
        )
        group_size = int(row.get("mla_heads_per_group", 2))
        if group_size == 16:
            expected_output = expected_output.repeat_interleave(8, dim=1)
        output_gather_mismatches += int((expected_output != row["output"]).sum())
        if (
            row["q_group"].shape[1] != (1 if group_size == 16 else 8)
            or candidates.shape[1] != 8
            or row["output"].shape[1] != 8
        ):
            shape_failures += 1

    for rows in by_layer.values():
        rows.sort(key=lambda row: int(row["tp_rank"]))
        reference = rows[0]
        for row in rows[1:]:
            rank_candidate_head_mismatches += int(
                (row["candidate_head_ids"] != reference["candidate_head_ids"]).sum()
            )
            rank_candidate_token_mismatches += int(
                (row["candidates"] != reference["candidates"]).sum()
            )

    tp_rank_coverage_valid = all(
        sorted(int(row["tp_rank"]) for row in rows) == list(range(8))
        for rows in by_layer.values()
    )
    report = {
        "trace_records": len(traces),
        "tp_rank_coverage_valid": tp_rank_coverage_valid,
        "layers": len(by_layer),
        "ranks_per_layer": sorted({len(rows) for rows in by_layer.values()}),
        "max_score_replay_error": max_score_error,
        "score_choice_mismatches": score_choice_mismatches,
        "output_gather_mismatches": output_gather_mismatches,
        "rank_candidate_head_mismatches": rank_candidate_head_mismatches,
        "rank_candidate_token_mismatches": rank_candidate_token_mismatches,
        "shape_failures": shape_failures,
        "runtime_mapping_valid": tp_rank_coverage_valid and all(
            value == 0
            for value in (
                score_choice_mismatches,
                output_gather_mismatches,
                rank_candidate_head_mismatches,
                rank_candidate_token_mismatches,
                shape_failures,
            )
        )
        and max_score_error <= 1e-6,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
