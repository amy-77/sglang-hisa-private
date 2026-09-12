#!/usr/bin/env python3
"""Benchmark current M=8 pair decode against one TP-local 16-head group.

The benchmark uses production MISA/coarse-pruned Top-720 retrieval operators and
the real FA3 or FlashMLA sparse attention operator.  It reports both full-path
component sums and an attention-only control where all eight pairs receive the
exact same indices as the 16-head group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.nsa import per_head_paged as php


MLA_QK = 576
MLA_V = 512
MLA_ROPE = MLA_QK - MLA_V
LOCAL_MLA_HEADS = 16
PAIR_GROUPS = 8
PAD_K = php.pad_index_width(php.OUTK)
SM_SCALE = MLA_QK**-0.5


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
    return float(start.elapsed_time(end) / iterations)


def build_indexer_cache(batch: int, seq_len: int, device: torch.device):
    pages_per_request = (seq_len + php.PAGE - 1) // php.PAGE
    total_pages = batch * pages_per_request
    key_bytes = torch.randint(
        0,
        256,
        (total_pages, php.PAGE * php.DIM),
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.rand(total_pages, php.PAGE, device=device) * 0.5 + 0.1
    kv = torch.cat(
        [key_bytes, scales.view(torch.uint8).view(total_pages, -1)], dim=1
    ).contiguous()
    chunk_sum = torch.randn(
        total_pages * php.CHUNKS_PER_PAGE,
        php.DIM,
        dtype=torch.float32,
        device=device,
    )
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=device
    ).view(batch, pages_per_request)
    return kv, chunk_sum, block_table


def pad_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[-1] == PAD_K:
        return indices
    if indices.shape[-1] > PAD_K:
        return indices[..., :PAD_K].contiguous()
    padding = indices.new_full(
        (*indices.shape[:-1], PAD_K - indices.shape[-1]), -1
    )
    return torch.cat([indices, padding], dim=-1)


def grouped_rows(q: torch.Tensor, groups: int) -> torch.Tensor:
    batch, heads, dim = q.shape
    return q.contiguous().view(batch * groups, heads // groups, dim)


def fa3_attention(
    q_mla: torch.Tensor, kv_mla: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache

    batch, groups, width = indices.shape
    q_rows = grouped_rows(q_mla, groups)
    rows, heads, _ = q_rows.shape
    q_nope, q_rope = q_rows.split([MLA_V, MLA_ROPE], dim=-1)
    index_rows = indices.reshape(rows, width)
    lengths = (index_rows >= 0).sum(-1, dtype=torch.int32)
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q_mla.device)
    cu_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=q_mla.device),
            lengths.cumsum(0),
        ]
    )
    value_cache = kv_mla[:, :, :MLA_V].view(-1, 1, 1, MLA_V)
    rope_cache = kv_mla[:, :, MLA_V:].view(-1, 1, 1, MLA_ROPE)
    output = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=rope_cache,
        v_cache=value_cache,
        qv=q_nope,
        page_table=index_rows,
        cache_seqlens=lengths,
        cu_seqlens_q=cu_q,
        cu_seqlens_k_new=cu_k,
        max_seqlen_q=1,
        softmax_scale=SM_SCALE,
        causal=True,
        softcap=0.0,
        return_softmax_lse=False,
        num_splits=1,
    )
    return output.view(batch, heads * groups, MLA_V)


def flashmla_attention(
    q_mla: torch.Tensor, kv_mla: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    batch, groups, width = indices.shape
    q_rows = grouped_rows(q_mla, groups)
    rows, heads, _ = q_rows.shape
    q_input = q_rows.new_zeros((rows, 64, MLA_QK))
    q_input[:, :heads] = q_rows
    output, _, _ = flash_mla_sparse_fwd(
        q=q_input,
        kv=kv_mla,
        indices=indices.reshape(rows, 1, width),
        sm_scale=SM_SCALE,
        d_v=MLA_V,
    )
    return output[:, :heads].reshape(batch, heads * groups, MLA_V)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--backend", choices=("fa3", "flashmla"), default="fa3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    php._get_ext()

    batch = args.batch_size
    seq_len = args.sequence_length
    kv_indexer, chunk_sum, block_table = build_indexer_cache(
        batch, seq_len, device
    )
    q_indexer = torch.randn(
        batch, 64, php.DIM, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    gates = torch.randn(batch, 64, dtype=torch.float32, device=device)
    row_batch = torch.arange(batch, dtype=torch.int32, device=device)
    row_len = torch.full(
        (batch,), seq_len, dtype=torch.int32, device=device
    )
    q_mla = torch.randn(
        batch,
        LOCAL_MLA_HEADS,
        MLA_QK,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_mla = torch.randn(
        seq_len, 1, MLA_QK, dtype=torch.bfloat16, device=device
    )

    # Representative route-before-retrieval MLP. Group/layer embeddings are
    # additions into the 64-wide hidden state and do not change kernel count.
    router_w1 = torch.randn(
        64, LOCAL_MLA_HEADS * MLA_QK, dtype=torch.bfloat16, device=device
    ) / (LOCAL_MLA_HEADS * MLA_QK) ** 0.5
    router_w2 = torch.randn(64, 64, dtype=torch.bfloat16, device=device) / 8

    def group_router():
        group_q = F.layer_norm(
            q_mla.reshape(batch, -1), (LOCAL_MLA_HEADS * MLA_QK,)
        )
        return F.linear(F.gelu(F.linear(group_q, router_w1)), router_w2)

    chosen_heads = group_router().argmax(-1)
    gather_q = chosen_heads.view(batch, 1, 1).expand(-1, 1, php.DIM)
    q_group_indexer = q_indexer.gather(1, gather_q)
    gate_group = gates.gather(1, chosen_heads.view(batch, 1))

    def current_route():
        return php.misa_topk_heads_paged(
            q_indexer,
            gates,
            chunk_sum,
            block_table,
            row_batch,
            row_len,
            PAIR_GROUPS,
            pooling_block_size=512,
            prune_keep_fraction=0.60,
        )

    current_heads, current_blocks = current_route()

    def current_retrieval():
        return php.per_head_topk_paged(
            q_indexer,
            head_range=None,
            kv_cache=kv_indexer,
            chunk_sum=chunk_sum,
            block_tables=block_table,
            row_batch=row_batch,
            row_len=row_len,
            topk=php.OUTK,
            head_ids=current_heads,
            coarse_block_ids=current_blocks,
            coarse_block_size=512,
        )

    def group_coarse_route():
        return php.misa_topk_heads_paged(
            q_group_indexer,
            gate_group,
            chunk_sum,
            block_table,
            row_batch,
            row_len,
            1,
            pooling_block_size=512,
            prune_keep_fraction=0.60,
        )

    group_local_heads, group_blocks = group_coarse_route()

    def group_retrieval():
        return php.per_head_topk_paged(
            q_group_indexer,
            head_range=None,
            kv_cache=kv_indexer,
            chunk_sum=chunk_sum,
            block_tables=block_table,
            row_batch=row_batch,
            row_len=row_len,
            topk=php.OUTK,
            head_ids=group_local_heads,
            coarse_block_ids=group_blocks,
            coarse_block_size=512,
        )

    current_indices = pad_indices(current_retrieval())
    group_indices = pad_indices(group_retrieval())
    repeated_group_indices = group_indices.expand(-1, PAIR_GROUPS, -1).contiguous()
    attention = fa3_attention if args.backend == "fa3" else flashmla_attention

    pair_same_output = attention(q_mla, kv_mla, repeated_group_indices)
    group_output = attention(q_mla, kv_mla, group_indices)
    max_same_index_error = float(
        (pair_same_output.float() - group_output.float()).abs().max()
    )

    timings = {
        "current_misa_64_to_8": elapsed_ms(
            current_route, args.warmup, args.iterations
        ),
        "current_retrieve_8": elapsed_ms(
            current_retrieval, args.warmup, args.iterations
        ),
        "current_pair_attention": elapsed_ms(
            lambda: attention(q_mla, kv_mla, current_indices),
            args.warmup,
            args.iterations,
        ),
        "group_router_mlp": elapsed_ms(
            group_router, args.warmup, args.iterations
        ),
        "group_coarse_route_1": elapsed_ms(
            group_coarse_route, args.warmup, args.iterations
        ),
        "group_retrieve_1": elapsed_ms(
            group_retrieval, args.warmup, args.iterations
        ),
        "pair_attention_same_indices": elapsed_ms(
            lambda: attention(q_mla, kv_mla, repeated_group_indices),
            args.warmup,
            args.iterations,
        ),
        "group_attention_same_indices": elapsed_ms(
            lambda: attention(q_mla, kv_mla, group_indices),
            args.warmup,
            args.iterations,
        ),
    }
    current_total = (
        timings["current_misa_64_to_8"]
        + timings["current_retrieve_8"]
        + timings["current_pair_attention"]
    )
    group_total = (
        timings["group_router_mlp"]
        + timings["group_coarse_route_1"]
        + timings["group_retrieve_1"]
        + timings["group_attention_same_indices"]
    )
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "config": {
            "batch_size": batch,
            "sequence_length": seq_len,
            "backend": args.backend,
            "mla_heads_per_tp_group": LOCAL_MLA_HEADS,
            "current_pair_groups": PAIR_GROUPS,
            "valid_topk": php.OUTK,
            "physical_index_width": PAD_K,
            "coarse_block_size": 512,
            "coarse_keep_fraction": 0.60,
        },
        "latency_ms": timings,
        "totals_ms": {
            "current_m8_pair_expanded": current_total,
            "tp16_group_shared": group_total,
            "saved": current_total - group_total,
            "speedup": current_total / group_total,
        },
        "attention_only_control": {
            "max_output_error_same_indices": max_same_index_error,
            "pair_expanded_ms": timings["pair_attention_same_indices"],
            "tp16_shared_ms": timings["group_attention_same_indices"],
            "speedup": timings["pair_attention_same_indices"]
            / timings["group_attention_same_indices"],
        },
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
