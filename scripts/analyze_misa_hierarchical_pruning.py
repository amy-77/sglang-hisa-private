#!/usr/bin/env python3
"""Measure whether MISA block scores preserve fine chunk16 Top-128 choices.

For each sampled query, this script:
1. runs MISA head selection at a configurable coarse block size;
2. computes the baseline Top-128 16-token chunks for the selected M heads;
3. keeps the highest-scoring coarse blocks; and
4. reports how many baseline fine chunks remain covered.

The input is an SGLang Indexer dump containing q_fp8_u8, k_fp8_u8, k_scale,
weights, ks, and ke. Sink and visible-tail blocks are pinned at both levels,
matching the paged per-head selector's intended semantics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


FINE_CHUNK = 16
FINE_TOPK = 128


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    return {
        "mean": float(values.mean()),
        "p01": float(torch.quantile(values, 0.01)),
        "p05": float(torch.quantile(values, 0.05)),
        "p10": float(torch.quantile(values, 0.10)),
        "p50": float(torch.quantile(values, 0.50)),
        "min": float(values.min()),
    }


def make_chunk_sums(
    key: torch.Tensor, token_count: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n_chunks = math.ceil(token_count / chunk_size)
    padded = torch.zeros(
        n_chunks * chunk_size,
        key.shape[-1],
        dtype=key.dtype,
        device=key.device,
    )
    padded[:token_count] = key
    sums = padded.view(n_chunks, chunk_size, -1).float().sum(dim=1)
    counts = torch.full(
        (n_chunks,), chunk_size, dtype=torch.float32, device=key.device
    )
    counts[-1] = token_count - (n_chunks - 1) * chunk_size
    return sums, counts


def pin_and_mask_scores(
    scores: torch.Tensor, visible_count: torch.Tensor
) -> torch.Tensor:
    block_ids = torch.arange(scores.shape[-1], device=scores.device)
    scores = scores.masked_fill(
        block_ids.view(1, 1, -1) >= visible_count.view(-1, 1, 1),
        float("-inf"),
    )
    scores[:, :, 0] = float("inf")
    tail = (visible_count - 1).view(-1, 1, 1).expand(-1, scores.shape[1], 1)
    return scores.scatter(2, tail, float("inf"))


@torch.no_grad()
def analyze_dump(
    path: Path,
    *,
    rows: int,
    block_sizes: list[int],
    keep_fractions: list[float],
    topk_heads: int,
    device: torch.device,
) -> dict:
    record = torch.load(path, map_location="cpu", weights_only=False)
    q_all = record["q_fp8_u8"].view(torch.float8_e4m3fn)
    key_fp8 = record["k_fp8_u8"].view(torch.float8_e4m3fn)
    key = key_fp8.float() * record["k_scale"].float().unsqueeze(-1)
    weights_all = record["weights"].float()
    ke_all = record["ke"].long()

    row_ids = torch.linspace(0, q_all.shape[0] - 1, min(rows, q_all.shape[0])).long()
    q = q_all[row_ids].to(device=device, dtype=torch.bfloat16)
    weights = weights_all[row_ids].to(device)
    visible_tokens = ke_all[row_ids].to(device)
    key = key.to(device=device, dtype=torch.bfloat16)

    fine_sums, _ = make_chunk_sums(key, key.shape[0], FINE_CHUNK)
    visible_fine = torch.div(
        visible_tokens + FINE_CHUNK - 1, FINE_CHUNK, rounding_mode="floor"
    )

    by_block = {}
    for block_size in block_sizes:
        if block_size % FINE_CHUNK:
            raise ValueError(f"block size {block_size} is not divisible by 16")
        coarse_sums, coarse_counts = make_chunk_sums(
            key, key.shape[0], block_size
        )
        coarse_means = coarse_sums / coarse_counts.unsqueeze(-1)
        affinities = torch.matmul(q, coarse_means.to(torch.bfloat16).T).float()
        visible_coarse = torch.div(
            visible_tokens + block_size - 1, block_size, rounding_mode="floor"
        )
        visible_mask = (
            torch.arange(coarse_means.shape[0], device=device).view(1, 1, -1)
            < visible_coarse.view(-1, 1, 1)
        )
        importance = (
            torch.relu(affinities).masked_fill(~visible_mask, 0.0)
            * weights.abs().unsqueeze(-1)
        ).sum(dim=-1)
        selected_heads = torch.topk(
            importance, topk_heads, dim=1, sorted=False
        ).indices

        gather_q = selected_heads.unsqueeze(-1).expand(-1, -1, q.shape[-1])
        selected_q = torch.gather(q, 1, gather_q)
        fine_scores = torch.matmul(
            selected_q, fine_sums.to(torch.bfloat16).T
        ).float()
        fine_scores = pin_and_mask_scores(fine_scores, visible_fine)
        baseline_fine = torch.topk(
            fine_scores, FINE_TOPK, dim=-1, sorted=False
        ).indices

        gather_coarse = selected_heads.unsqueeze(-1).expand(
            -1, -1, affinities.shape[-1]
        )
        selected_coarse_scores = torch.gather(affinities, 1, gather_coarse)
        selected_coarse_scores = pin_and_mask_scores(
            selected_coarse_scores, visible_coarse
        )
        baseline_coarse_ids = baseline_fine // (block_size // FINE_CHUNK)

        fraction_results = {}
        for fraction in keep_fractions:
            keep_count = torch.ceil(visible_coarse.float() * fraction).long()
            keep_count = keep_count.clamp(min=2, max=selected_coarse_scores.shape[-1])
            max_keep = int(keep_count.max())
            kept = torch.topk(
                selected_coarse_scores, max_keep, dim=-1, sorted=True
            ).indices
            valid_rank = (
                torch.arange(max_keep, device=device).view(1, 1, -1)
                < keep_count.view(-1, 1, 1)
            )
            covered = (
                baseline_coarse_ids.unsqueeze(-1) == kept.unsqueeze(-2)
            ) & valid_rank.unsqueeze(-2)
            recall = covered.any(dim=-1).float().mean(dim=-1)

            mean_visible_fine = float(visible_fine.float().mean())
            mean_kept_fine = float(
                (keep_count * (block_size // FINE_CHUNK)).float().mean()
            )
            misa_dot_products = 64.0 * float(visible_coarse.float().mean())
            full_fine_dot_products = topk_heads * mean_visible_fine
            pruned_fine_dot_products = topk_heads * mean_kept_fine
            fraction_results[str(fraction)] = {
                "recall": quantiles(recall.flatten()),
                "mean_kept_coarse_blocks": float(keep_count.float().mean()),
                "mean_candidate_fine_chunks": mean_kept_fine,
                "fine_chunk_fraction": mean_kept_fine / mean_visible_fine,
                "relative_dot_products_vs_unpruned": (
                    misa_dot_products + pruned_fine_dot_products
                )
                / (misa_dot_products + full_fine_dot_products),
            }

        by_block[str(block_size)] = {
            "num_coarse_blocks": coarse_means.shape[0],
            "mean_visible_coarse_blocks": float(visible_coarse.float().mean()),
            "fractions": fraction_results,
        }

    return {
        "file": str(path),
        "layer": int(record["layer_id"]),
        "sequence_tokens": int(key.shape[0]),
        "sampled_queries": int(q.shape[0]),
        "topk_heads": topk_heads,
        "fine_chunk_size": FINE_CHUNK,
        "baseline_fine_topk": FINE_TOPK,
        "results": by_block,
    }


def aggregate(reports: list[dict]) -> dict:
    output = {}
    for block_size in reports[0]["results"]:
        output[block_size] = {}
        fractions = reports[0]["results"][block_size]["fractions"]
        for fraction in fractions:
            # Each dump uses the same number of sampled rows and selected heads.
            metrics = [
                report["results"][block_size]["fractions"][fraction]
                for report in reports
            ]
            output[block_size][fraction] = {
                "mean_recall_across_layers": sum(
                    metric["recall"]["mean"] for metric in metrics
                )
                / len(metrics),
                "worst_layer_p05_recall": min(
                    metric["recall"]["p05"] for metric in metrics
                ),
                "worst_layer_p01_recall": min(
                    metric["recall"]["p01"] for metric in metrics
                ),
                "mean_relative_dot_products_vs_unpruned": sum(
                    metric["relative_dot_products_vs_unpruned"]
                    for metric in metrics
                )
                / len(metrics),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", type=Path, nargs="+")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument(
        "--keep-fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5]
    )
    parser.add_argument("--topk-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(not 0 < fraction <= 1 for fraction in args.keep_fractions):
        parser.error("keep fractions must be in (0, 1]")

    device = torch.device(args.device)
    reports = [
        analyze_dump(
            path,
            rows=args.rows,
            block_sizes=args.block_sizes,
            keep_fractions=args.keep_fractions,
            topk_heads=args.topk_heads,
            device=device,
        )
        for path in args.dumps
    ]
    payload = {
        "method": (
            "Recall of baseline per-selected-head Top-128 chunk16 IDs after "
            "pruning by same-resolution MISA coarse mean scores."
        ),
        "reports": reports,
        "aggregate": aggregate(reports),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
