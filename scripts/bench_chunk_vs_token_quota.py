#!/usr/bin/env python3
"""Does intra-chunk Top-720 beat attending all 128 selected chunks?

After coarse chunk ranking, two policies:

  A. Whole-chunk attention: keep every token in Top-128 chunks (2048).
  B. Quota: current fused kernel keeps 52*8 + 76*4 = 720 tokens.

B must pay extra indexer QK + scattered Top-K, then save work in sparse MLA.
This script times those pieces separately with the production operators:

  - stage 0: q @ chunk_sum   (cuBLAS, same for A and B)
  - B only:  chunk16_quota_paged  (stage 1 radix + stage 2 fp8 QK + quota)
  - A only:  torch.topk on coarse scores, expand 128 chunks -> 2048 ids
  - both:    sgl_kernel.flash_mla_sparse_fwd at K=768 (padded 720) vs K=2048

Default is decode-like: one query row per request, G indexer heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.layers.attention.nsa import per_head_paged as php


CHUNK = php.CHUNK
SEL = php.SEL
OUTK = php.OUTK
PAD_K = php.pad_index_width(OUTK)  # 768
FULL_K = SEL * CHUNK  # 2048
MLA_QK = 576
MLA_V = 512
SM_SCALE = 576**-0.5


def elapsed_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def build_indexer_cache(batch: int, seq_len: int, device: torch.device):
    pages_per = (seq_len + php.PAGE - 1) // php.PAGE
    total_pages = batch * pages_per
    key_bytes = torch.randint(
        0, 256, (total_pages, php.PAGE * php.DIM), dtype=torch.uint8, device=device
    )
    scales = torch.rand(total_pages, php.PAGE, device=device) * 0.5 + 0.1
    kv = torch.cat(
        [key_bytes, scales.view(torch.uint8).view(total_pages, -1)], dim=1
    ).contiguous()
    chunk_sum = torch.randn(
        total_pages * php.CHUNKS_PER_PAGE, php.DIM, device=device
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, pages_per
    )
    return kv, chunk_sum, block_table


def pad_indices(idx: torch.Tensor, width: int) -> torch.Tensor:
    if idx.shape[-1] >= width:
        return idx[..., :width].contiguous()
    pad = idx.new_full((*idx.shape[:-1], width - idx.shape[-1]), -1)
    return torch.cat([idx, pad], dim=-1)


def expand_top_chunks(
    scores: torch.Tensor, row_len: torch.Tensor, n_chunks: int
) -> torch.Tensor:
    """Top-SEL coarse chunks -> all 16 tokens, request-relative, -1 padded."""
    m, c = scores.shape
    k = min(SEL, c)
    chunk_ids = scores.topk(k, dim=-1, largest=True, sorted=False).indices
    # pin sink / tail like the kernel
    chunk_ids[:, 0] = 0
    tail = ((row_len.to(torch.int64) + CHUNK - 1) // CHUNK - 1).clamp(min=0)
    chunk_ids[:, -1] = tail
    off = torch.arange(CHUNK, device=scores.device)
    tokens = (chunk_ids.unsqueeze(-1) * CHUNK + off).reshape(m, k * CHUNK)
    valid = tokens < row_len.view(-1, 1)
    return torch.where(valid, tokens.to(torch.int32), tokens.new_full((), -1))


def run_quota(q, kv_idx, chunk_sum, bt, row_batch, row_len, heads: int):
    return php.per_head_topk_paged(
        q,
        head_range=(0, heads),
        kv_cache=kv_idx,
        chunk_sum=chunk_sum,
        block_tables=bt,
        row_batch=row_batch,
        row_len=row_len,
        topk=max(OUTK, PAD_K),
        segments=None,
    )


def coarse_scores(q_bf, chunk_sum, bt, row_batch, row_len, heads: int):
    n_chunks = php._pad8(int((int(row_len.max().item()) + CHUNK - 1) // CHUNK))
    pages = bt[row_batch.to(torch.int64)]
    stat = chunk_sum[php._chunk_ids(pages, n_chunks)].to(torch.bfloat16)
    scores = torch.bmm(q_bf, stat.transpose(1, 2))  # [b, G, C]
    return scores, n_chunks


def flashmla_sparse(q_mla: torch.Tensor, kv_mla: torch.Tensor, indices: torch.Tensor):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    # Production per-head path: 2 MLA heads per indexer head, pad q heads to 64.
    n, groups, topk = indices.shape
    q_rows = q_mla.contiguous().view(n * groups, q_mla.shape[1] // groups, MLA_QK)
    h = q_rows.shape[1]
    pad_h = 64
    if h % pad_h:
        q_in = q_rows.new_zeros((q_rows.shape[0], pad_h, MLA_QK))
        q_in[:, :h] = q_rows
    else:
        q_in = q_rows
    idx = indices.reshape(n * groups, 1, topk)
    o, _, _ = flash_mla_sparse_fwd(
        q=q_in, kv=kv_mla, indices=idx, sm_scale=SM_SCALE, d_v=MLA_V
    )
    return o[:, :h]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--heads", type=int, default=8, help="Indexer heads / MISA budget")
    p.add_argument("--mla-heads", type=int, default=16, help="Must be 2 * heads")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--skip-attn", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.mla_heads != 2 * args.heads:
        raise SystemExit("--mla-heads must be 2 * --heads (pair contract)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device(args.device)
    php._get_ext()
    bsz, seq, g = args.batch_size, args.seq_len, args.heads

    kv_idx, chunk_sum, bt = build_indexer_cache(bsz, seq, device)
    q_idx = torch.randn(bsz, g, php.DIM, dtype=torch.bfloat16, device=device)
    row_batch = torch.arange(bsz, dtype=torch.int32, device=device)
    row_len = torch.full((bsz,), seq, dtype=torch.int32, device=device)
    q_bf = q_idx.to(torch.bfloat16).contiguous()

    q_mla = torch.randn(bsz, args.mla_heads, MLA_QK, dtype=torch.bfloat16, device=device)
    kv_mla = torch.randn(seq, 1, MLA_QK, dtype=torch.bfloat16, device=device)

    # --- indexer ---
    def stage0():
        return coarse_scores(q_bf, chunk_sum, bt, row_batch, row_len, g)

    scores0, _ = stage0()
    idx_full = expand_top_chunks(
        scores0.reshape(bsz * g, -1),
        row_len.view(bsz, 1).expand(bsz, g).reshape(-1),
        scores0.shape[-1],
    ).view(bsz, g, -1)
    idx_quota = pad_indices(run_quota(q_idx, kv_idx, chunk_sum, bt, row_batch, row_len, g), PAD_K)
    idx_full = pad_indices(idx_full, FULL_K)

    t0 = elapsed_ms(stage0, args.warmup, args.iters)

    def path_a_select():
        scores, _ = stage0()
        return expand_top_chunks(
            scores.reshape(bsz * g, -1),
            row_len.view(bsz, 1).expand(bsz, g).reshape(-1),
            scores.shape[-1],
        )

    def path_b_select():
        return run_quota(q_idx, kv_idx, chunk_sum, bt, row_batch, row_len, g)

    t_a_sel = elapsed_ms(path_a_select, args.warmup, args.iters)
    t_b_sel = elapsed_ms(path_b_select, args.warmup, args.iters)

    report = {
        "config": {
            "batch_size": bsz,
            "seq_len": seq,
            "indexer_heads": g,
            "mla_heads": args.mla_heads,
            "quota_k": OUTK,
            "quota_padded": PAD_K,
            "full_chunk_k": FULL_K,
        },
        "latency_ms": {
            "stage0_coarse_bmm": t0,
            "A_coarse_plus_expand_2048": t_a_sel,
            "B_quota_kernel_720": t_b_sel,
            "B_minus_A_selection_overhead": t_b_sel - t_a_sel,
        },
        "index_stats": {
            "quota_valid": int((idx_quota >= 0).sum().item()),
            "full_valid": int((idx_full >= 0).sum().item()),
        },
    }

    if not args.skip_attn:
        def attn_quota():
            return flashmla_sparse(q_mla, kv_mla, idx_quota)

        def attn_full():
            return flashmla_sparse(q_mla, kv_mla, idx_full)

        t_attn_b = elapsed_ms(attn_quota, args.warmup, args.iters)
        t_attn_a = elapsed_ms(attn_full, args.warmup, args.iters)
        report["latency_ms"].update(
            {
                "A_flashmla_2048": t_attn_a,
                "B_flashmla_768": t_attn_b,
                "A_total_select_plus_attn": t_a_sel + t_attn_a,
                "B_total_select_plus_attn": t_b_sel + t_attn_b,
            }
        )
        report["verdict"] = (
            "quota_720_faster"
            if (t_b_sel + t_attn_b) < (t_a_sel + t_attn_a)
            else "full_chunk_2048_faster"
        )
        report["saved_ms"] = (t_a_sel + t_attn_a) - (t_b_sel + t_attn_b)

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
