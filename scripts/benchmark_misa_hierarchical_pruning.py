#!/usr/bin/env python3
"""Benchmark unpruned vs hierarchical MISA per-head Top-720 decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.layers.attention.nsa import per_head_paged


def elapsed_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def build_cache(
    batch_size: int, sequence_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pages_per_request = (
        sequence_length + per_head_paged.PAGE - 1
    ) // per_head_paged.PAGE
    total_pages = batch_size * pages_per_request
    key_bytes = torch.randint(
        0,
        256,
        (total_pages, per_head_paged.PAGE * per_head_paged.DIM),
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.rand(
        total_pages, per_head_paged.PAGE, dtype=torch.float32, device=device
    )
    kv = torch.cat(
        [key_bytes, scales.view(torch.uint8).view(total_pages, -1)], dim=1
    ).contiguous()
    chunk_sum = torch.randn(
        total_pages * per_head_paged.CHUNKS_PER_PAGE,
        per_head_paged.DIM,
        dtype=torch.float32,
        device=device,
    )
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=device
    ).view(batch_size, pages_per_request)
    return kv, chunk_sum, block_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk-heads", type=int, default=8)
    parser.add_argument("--coarse-block-size", type=int, default=256)
    parser.add_argument("--keep-fraction", type=float, default=None)
    parser.add_argument("--coarse-topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.coarse_topk is not None and args.keep_fraction is not None:
        parser.error("choose --coarse-topk or --keep-fraction")
    if args.coarse_topk is None and args.keep_fraction is None:
        args.coarse_topk = 8
    device = torch.device(args.device)
    kv, chunk_sum, block_table = build_cache(
        args.batch_size, args.sequence_length, device
    )
    q = torch.randn(
        args.batch_size,
        args.heads,
        per_head_paged.DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(args.batch_size, args.heads, device=device)
    row_batch = torch.arange(args.batch_size, dtype=torch.int32, device=device)
    row_len = torch.full(
        (args.batch_size,),
        args.sequence_length,
        dtype=torch.int32,
        device=device,
    )

    def route():
        return per_head_paged.misa_topk_heads_paged(
            q,
            weights,
            chunk_sum,
            block_table,
            row_batch,
            row_len,
            args.topk_heads,
            pooling_block_size=args.coarse_block_size,
            prune_keep_fraction=args.keep_fraction,
            prune_topk=args.coarse_topk,
        )

    head_ids, coarse_blocks = route()
    common = dict(
        q=q,
        head_range=None,
        kv_cache=kv,
        chunk_sum=chunk_sum,
        block_tables=block_table,
        row_batch=row_batch,
        row_len=row_len,
        topk=per_head_paged.OUTK,
        head_ids=head_ids,
    )

    unpruned_ms = elapsed_ms(
        lambda: per_head_paged.per_head_topk_paged(**common),
        args.warmup,
        args.iterations,
    )
    hierarchical_ms = elapsed_ms(
        lambda: per_head_paged.per_head_topk_paged(
            **common,
            coarse_block_ids=coarse_blocks,
            coarse_block_size=args.coarse_block_size,
            mean_ranked_chunks=args.coarse_topk is not None,
        ),
        args.warmup,
        args.iterations,
    )
    router_ms = elapsed_ms(route, args.warmup, args.iterations)
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "heads": args.heads,
        "topk_heads": args.topk_heads,
        "coarse_block_size": args.coarse_block_size,
        "keep_fraction": args.keep_fraction,
        "coarse_topk": args.coarse_topk,
        "misa_router_ms": router_ms,
        "unpruned_scan_ms": unpruned_ms,
        "hierarchical_scan_ms": hierarchical_ms,
        "scan_speedup": unpruned_ms / hierarchical_ms,
        "unpruned_total_ms": router_ms + unpruned_ms,
        "hierarchical_total_ms": router_ms + hierarchical_ms,
        "total_speedup": (router_ms + unpruned_ms)
        / (router_ms + hierarchical_ms),
    }
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
