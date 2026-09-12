from __future__ import annotations

import json

import pytest
import torch

from sglang.srt.layers.attention.nsa import offline_unique_head_router
from sglang.srt.layers.attention.nsa import per_head_paged


def test_experimental_sparse_min_seq_len_defaults_to_safe_boundary(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN", raising=False)
    assert per_head_paged.per_head_min_seq_len() == 4096
    monkeypatch.setenv("SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN", "719")
    with pytest.raises(ValueError, match="at least OUTK=720"):
        per_head_paged.per_head_min_seq_len()


def test_prefill_per_head_defaults_on_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD", raising=False)
    assert per_head_paged.prefill_per_head_enabled()
    monkeypatch.setenv("SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD", "0")
    assert not per_head_paged.prefill_per_head_enabled()


def test_index_width_is_padded_to_flashmla_block() -> None:
    assert per_head_paged.pad_index_width(per_head_paged.OUTK) == 768
    assert per_head_paged.pad_index_width(4095) == 4096
    assert per_head_paged.pad_index_width(4096) == 4096


def test_prefill_dense_boundary_counts_rows_per_request() -> None:
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer

    class ForwardBatch:
        seq_lens_cpu = torch.tensor([6144, 5000, 3000])

    class Metadata:
        @staticmethod
        def get_nsa_extend_len_cpu() -> list[int]:
            # Cached prefixes are respectively 2048, 4500, and 0 tokens.
            return [4096, 500, 3000]

    counts, width = Indexer._early_prefill_rows(
        ForwardBatch(), Metadata(), min_seq_len=4096
    )
    assert counts == [2047, 0, 3000]
    assert width == 4095


def test_misa_topk_heads_uses_logical_pooled_key_means() -> None:
    dim = per_head_paged.DIM
    chunk_sum = torch.zeros(3 * per_head_paged.CHUNKS_PER_PAGE, dim)

    # Request 0 occupies physical page 2.  Its 50 visible tokens form a
    # 32-token [2, 0] block and an 18-token [0, 3] block.
    chunk_sum[8, 0] = 16 * 2
    chunk_sum[9, 0] = 16 * 2
    chunk_sum[10, 1] = 16 * 3
    chunk_sum[11, 1] = 2 * 3

    # Request 1 occupies physical page 0 and has only 20 visible tokens.  The
    # last two physical chunks contain deliberately large stale values, which
    # must be masked before pooling.
    chunk_sum[0, 1] = 16 * 4
    chunk_sum[1, 1] = 4 * 4
    chunk_sum[2:4, 0] = 1000

    q = torch.zeros(2, 4, dim, dtype=torch.bfloat16)
    q[:, 0, 0] = 1
    q[:, 1, 1] = 1
    q[:, 2, 0] = 1
    q[:, 2, 1] = 0.5
    q[:, 3, :2] = -1
    weights = torch.tensor([[2.0, 1.0, 1.0, 100.0], [1.0, 1.0, 1.0, 1.0]])

    selected = per_head_paged.misa_topk_heads_paged(
        q,
        weights,
        chunk_sum,
        torch.tensor([[2, 1], [0, 1]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([50, 20], dtype=torch.int32),
        2,
        pooling_block_size=32,
        rows_per_launch=2,
    )

    assert set(selected[0].tolist()) == {0, 2}
    assert set(selected[1].tolist()) == {1, 2}


def test_misa_fixed_coarse_topk_selects_heads_by_same_region_prefix() -> None:
    dim = per_head_paged.DIM
    chunk_sum = torch.zeros(12, dim)
    # Head 0 wins the legacy all-block sum: 10 * 6 = 60.
    # Head 1 wins the fixed Top-8 sum: 8 * 7 = 56 > 8 * 6 = 48.
    chunk_sum[:10, 0] = 16 * 6
    chunk_sum[:8, 1] = 16 * 7
    q = torch.zeros(1, 2, dim, dtype=torch.bfloat16)
    q[0, 0, 0] = 1
    q[0, 1, 1] = 1
    common = (
        q,
        torch.ones(1, 2),
        chunk_sum,
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([160], dtype=torch.int32),
        1,
    )

    legacy_head = per_head_paged.misa_topk_heads_paged(
        *common, pooling_block_size=16
    )
    selected_head, selected_blocks = per_head_paged.misa_topk_heads_paged(
        *common, pooling_block_size=16, prune_topk=8
    )

    assert legacy_head.tolist() == [[0]]
    assert selected_head.tolist() == [[1]]
    assert set(selected_blocks[0, 0].tolist()) == set(range(8))


def test_misa_reuses_selected_head_affinities_for_coarse_pruning() -> None:
    dim = per_head_paged.DIM
    chunk_sum = torch.zeros(8, dim)
    # Four 32-token blocks score [high, low, medium, tail] for head 0.
    chunk_sum[0:2, 0] = 8
    chunk_sum[2:4, 0] = 1
    chunk_sum[4:6, 0] = 4
    chunk_sum[6:8, 0] = 2
    q = torch.zeros(1, 2, dim, dtype=torch.bfloat16)
    q[0, 0, 0] = 1
    q[0, 1, 0] = -1

    heads, blocks = per_head_paged.misa_topk_heads_paged(
        q,
        torch.ones(1, 2),
        chunk_sum,
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([128], dtype=torch.int32),
        1,
        pooling_block_size=32,
        prune_keep_fraction=0.5,
    )

    assert heads.tolist() == [[0]]
    # Sink and tail are pinned, so 50% of four blocks keeps exactly those two.
    assert set(blocks[0, 0].tolist()) == {0, 3}


def test_misa_mixed_lengths_keep_each_rows_best_block_prefix() -> None:
    dim = per_head_paged.DIM
    chunk_sum = torch.zeros(6 * per_head_paged.CHUNKS_PER_PAGE, dim)
    block_tables = torch.tensor(
        [[0, 1, 2, 3], [4, 5, -1, -1]], dtype=torch.int32
    )
    # Each 32-token block has an increasing affinity. Row 0 has eight blocks;
    # row 1 has four. Both sink and visible tail are pinned by the selector.
    for page in range(6):
        for chunk in range(per_head_paged.CHUNKS_PER_PAGE):
            chunk_sum[page * 4 + chunk, 0] = page * 4 + chunk + 1
    q = torch.zeros(2, 1, dim, dtype=torch.bfloat16)
    q[:, :, 0] = 1

    _, blocks = per_head_paged.misa_topk_heads_paged(
        q,
        torch.ones(2, 1),
        chunk_sum,
        block_tables,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([256, 128], dtype=torch.int32),
        1,
        pooling_block_size=32,
        prune_keep_fraction=0.5,
        rows_per_launch=2,
    )

    assert set(blocks[0, 0].tolist()) == {0, 5, 6, 7}
    assert set(blocks[1, 0, :2].tolist()) == {0, 3}
    assert blocks[1, 0, 2:].tolist() == [-1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_misa_cuda_fp8_matches_direct_pooled_reference() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    lengths = [2050, 1537]
    max_pages = (max(lengths) + per_head_paged.PAGE - 1) // per_head_paged.PAGE
    block_tables = torch.stack(
        [torch.arange(max_pages) * 2, torch.arange(max_pages) * 2 + 1]
    ).to(device=device, dtype=torch.int32)
    keys = torch.randn(2, max(lengths), per_head_paged.DIM, device=device)
    chunk_sum = torch.zeros(
        max_pages * 2 * per_head_paged.CHUNKS_PER_PAGE,
        per_head_paged.DIM,
        device=device,
    )
    for row, length in enumerate(lengths):
        for chunk in range((length + per_head_paged.CHUNK - 1) // per_head_paged.CHUNK):
            physical = int(block_tables[row, chunk // 4]) * 4 + chunk % 4
            start = chunk * per_head_paged.CHUNK
            end = min(start + per_head_paged.CHUNK, length)
            chunk_sum[physical] = keys[row, start:end].sum(dim=0)

    q = torch.randn(2, 64, per_head_paged.DIM, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(2, 64, 1, device=device)
    actual = per_head_paged.misa_topk_heads_paged(
        q,
        weights,
        chunk_sum,
        block_tables,
        torch.arange(2, device=device, dtype=torch.int32),
        torch.tensor(lengths, device=device, dtype=torch.int32),
        8,
        pooling_block_size=1024,
        rows_per_launch=2,
    )

    expected = []
    for row, length in enumerate(lengths):
        pooled = torch.stack(
            [
                keys[row, start : min(start + 1024, length)]
                .sum(dim=0)
                .div(min(1024, length - start))
                for start in range(0, length, 1024)
            ]
        ).to(torch.bfloat16)
        affinity = (q[row].to(torch.bfloat16) @ pooled.T).float()
        importance = (torch.relu(affinity) * weights[row].float()).sum(dim=-1)
        expected.append(torch.topk(importance, 8).indices)
    expected = torch.stack(expected)

    assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hierarchical_pruning_full_retention_matches_unpruned_scan() -> None:
    device = torch.device("cuda")
    torch.manual_seed(9)
    length = 3000
    pages = (length + per_head_paged.PAGE - 1) // per_head_paged.PAGE
    loc = torch.arange(length, device=device)
    key = torch.randn(length, per_head_paged.DIM, device=device)
    scale = key.abs().amax(-1).clamp_min(1e-4) / 448.0
    key_fp8 = (key / scale.unsqueeze(-1)).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )
    key_bytes = torch.zeros(
        pages * per_head_paged.PAGE,
        per_head_paged.DIM,
        dtype=torch.uint8,
        device=device,
    )
    scale_cache = torch.ones(
        pages * per_head_paged.PAGE, dtype=torch.float32, device=device
    )
    key_bytes.view(torch.float8_e4m3fn)[loc] = key_fp8
    scale_cache[loc] = scale
    kv = torch.cat(
        [
            key_bytes.view(pages, -1),
            scale_cache.view(torch.uint8).view(pages, -1),
        ],
        dim=1,
    ).contiguous()
    chunk_sum = torch.zeros(
        pages * per_head_paged.CHUNKS_PER_PAGE,
        per_head_paged.DIM,
        dtype=torch.float32,
        device=device,
    )
    per_head_paged.update_chunk_sum(
        chunk_sum, key_fp8.float() * scale.unsqueeze(-1), loc
    )
    block_table = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    q = torch.randn(1, 4, per_head_paged.DIM, device=device).to(
        torch.float8_e4m3fn
    )
    heads, blocks = per_head_paged.misa_topk_heads_paged(
        q,
        torch.randn(1, 4, device=device),
        chunk_sum,
        block_table,
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.tensor([length], device=device, dtype=torch.int32),
        2,
        pooling_block_size=512,
        prune_keep_fraction=1.0,
    )
    kwargs = dict(
        q=q,
        head_range=None,
        kv_cache=kv,
        chunk_sum=chunk_sum,
        block_tables=block_table,
        row_batch=torch.zeros(1, device=device, dtype=torch.int32),
        row_len=torch.tensor([length], device=device, dtype=torch.int32),
        topk=per_head_paged.OUTK,
        head_ids=heads,
    )
    unpruned = per_head_paged.per_head_topk_paged(**kwargs)
    hierarchical = per_head_paged.per_head_topk_paged(
        **kwargs,
        coarse_block_ids=blocks,
        coarse_block_size=512,
    )

    for head in range(2):
        expected = set(unpruned[0, head].tolist())
        actual = set(hierarchical[0, head].tolist())
        assert len(expected & actual) / len(expected | actual) > 0.99


def test_per_head_scan_gathers_query_dependent_head_ids(monkeypatch) -> None:
    seen_queries: list[torch.Tensor] = []

    class FakeExtension:
        @staticmethod
        def chunk16_quota_paged(
            scores, q, kv, block_table, batch, length, tokens
        ) -> None:
            seen_queries.append(q.clone())

    monkeypatch.setattr(per_head_paged, "_ext", FakeExtension())
    q = torch.zeros(2, 4, per_head_paged.DIM, dtype=torch.bfloat16)
    for row in range(2):
        for head in range(4):
            q[row, head, 0] = row * 10 + head

    result = per_head_paged.per_head_topk_paged(
        q,
        None,
        torch.zeros(2, per_head_paged.PAGE * 132, dtype=torch.uint8),
        torch.zeros(2 * per_head_paged.CHUNKS_PER_PAGE, per_head_paged.DIM),
        torch.tensor([[0], [1]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        per_head_paged.OUTK,
        head_ids=torch.tensor([[3, 1], [0, 2]]),
        pairs_per_launch=4,
    )

    assert result.shape == (2, 2, per_head_paged.OUTK)
    gathered = torch.cat(seen_queries, dim=0)[:, 0].tolist()
    assert gathered == [3.0, 1.0, 10.0, 12.0]


def test_per_head_scan_scores_only_retained_coarse_blocks(monkeypatch) -> None:
    seen_chunk_ids: list[torch.Tensor] = []

    class FakeExtension:
        @staticmethod
        def selected_chunk_scores_paged(
            scores, q, chunk_sum, block_table, batch, length, chunk_ids
        ) -> None:
            seen_chunk_ids.append(chunk_ids.clone())
            scores.zero_()

        @staticmethod
        def chunk16_quota_paged_mapped(
            scores, q, kv, block_table, batch, length, chunk_ids, tokens
        ) -> None:
            tokens[:, 0] = chunk_ids[:, 0] * per_head_paged.CHUNK

    monkeypatch.setattr(per_head_paged, "_ext", FakeExtension())
    q = torch.zeros(1, 4, per_head_paged.DIM, dtype=torch.bfloat16)
    result = per_head_paged.per_head_topk_paged(
        q,
        None,
        torch.zeros(16, per_head_paged.PAGE * 132, dtype=torch.uint8),
        torch.zeros(16 * per_head_paged.CHUNKS_PER_PAGE, per_head_paged.DIM),
        torch.arange(16, dtype=torch.int32).view(1, -1),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1000], dtype=torch.int32),
        per_head_paged.OUTK,
        head_ids=torch.tensor([[3, 1]]),
        coarse_block_ids=torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32),
        coarse_block_size=32,
        pairs_per_launch=2,
    )

    assert seen_chunk_ids[0][0, :4].tolist() == [0, 1, 4, 5]
    assert seen_chunk_ids[0][1, :4].tolist() == [2, 3, 6, 7]
    assert result[0, :, 0].tolist() == [0, 32]


def test_prefill_and_decode_use_identical_coarse_pruned_candidates(
    monkeypatch,
) -> None:
    class FakeExtension:
        @staticmethod
        def selected_chunk_scores_paged(
            scores, q, chunk_sum, block_table, batch, length, chunk_ids
        ) -> None:
            scores.copy_(chunk_ids.to(scores.dtype))

        @staticmethod
        def chunk16_quota_paged_mapped(
            scores, q, kv, block_table, batch, length, chunk_ids, tokens
        ) -> None:
            tokens[:, : chunk_ids.shape[1]] = torch.where(
                chunk_ids >= 0,
                chunk_ids * per_head_paged.CHUNK,
                chunk_ids,
            )

    monkeypatch.setattr(per_head_paged, "_ext", FakeExtension())
    kwargs = dict(
        q=torch.zeros(2, 4, per_head_paged.DIM, dtype=torch.bfloat16),
        head_range=None,
        kv_cache=torch.zeros(16, per_head_paged.PAGE * 132, dtype=torch.uint8),
        chunk_sum=torch.zeros(
            16 * per_head_paged.CHUNKS_PER_PAGE, per_head_paged.DIM
        ),
        block_tables=torch.arange(16, dtype=torch.int32).view(1, -1),
        row_batch=torch.zeros(2, dtype=torch.int32),
        row_len=torch.tensor([4096, 4097], dtype=torch.int32),
        topk=per_head_paged.OUTK,
        head_ids=torch.tensor([[3, 1], [2, 0]]),
        coarse_block_ids=torch.tensor(
            [[[0, 2], [1, 3]], [[2, 4], [3, 5]]], dtype=torch.int32
        ),
        coarse_block_size=32,
        pairs_per_launch=4,
    )

    decode = per_head_paged.per_head_topk_paged(**kwargs)
    prefill = per_head_paged.per_head_topk_paged(
        **kwargs, segments=[(0, 2, 0)]
    )
    assert torch.equal(prefill, decode)


def test_pair_router_restricts_scores_to_each_queries_misa_set(
    monkeypatch, tmp_path
) -> None:
    from sglang.srt.layers import dp_attention

    monkeypatch.setattr(dp_attention, "get_attention_tp_rank", lambda: 0)
    monkeypatch.setattr(dp_attention, "get_attention_tp_size", lambda: 64)

    prior = torch.zeros(1, 64)
    prior[0, 5] = 10
    prior[0, 7] = 5
    checkpoint = {
        "model_class": "OfflineRouter",
        "model_config": {"q_dim": 4, "dynamic_indexer_keys": False},
        "layers": [0],
        "state_dict": {
            "assignment_scorer.q_proj.weight": torch.zeros(1, 4),
            "assignment_scorer.indexer_embedding": torch.zeros(64, 1),
            "assignment_scorer.layer_embedding.weight": torch.zeros(1, 1),
            "assignment_scorer.prior": prior,
            "assignment_scorer.pair_prior": torch.zeros(1, 64, 64),
            "assignment_scorer.gate_scale": torch.zeros(64),
        },
    }
    checkpoint_path = tmp_path / "router.pt"
    torch.save(checkpoint, checkpoint_path)
    config_path = tmp_path / "router.json"
    config_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "budget": 2,
                "assignment_mode": "learned",
                "head_selection_mode": "misa",
                "misa_chunk_size": 1024,
                "selected_heads_by_layer": {"0": [0, 1]},
            }
        )
    )
    monkeypatch.setenv("SGLANG_NSA_OFFLINE_ROUTER_CONFIG", str(config_path))
    offline_unique_head_router.reset_for_tests()

    class ForwardBatch:
        pass

    forward_batch = ForwardBatch()
    offline_unique_head_router.stash_gate(forward_batch, 0, torch.zeros(2, 64))
    offline_unique_head_router.stash_selected_heads(
        forward_batch,
        0,
        torch.tensor([[5, 7], [7, 5]]),
        total_rows=2,
    )
    candidates = torch.tensor(
        [[[100, 101], [200, 201]], [[300, 301], [400, 401]]],
        dtype=torch.int32,
    )
    routed = offline_unique_head_router.route_candidates(
        layer_id=0,
        q_mla=torch.zeros(2, 2, 2),
        candidates=candidates,
        forward_batch=forward_batch,
    )

    assert routed[:, 0].tolist() == [[100, 101], [400, 401]]
    offline_unique_head_router.reset_for_tests()


def test_misa_fixed_top8_mean_regions_mixed_lengths() -> None:
    lengths = [2560, 769, 17]
    pages = 40
    sums = torch.full((len(lengths) * pages * 4, 128), 10000.0)
    tables = torch.arange(len(lengths) * pages).view(len(lengths), pages).flip(1).int()
    q = torch.zeros(len(lengths), 2, 128, dtype=torch.bfloat16)
    q[:, 0, 0] = 1
    q[:, 1, 0] = -1
    for row, length in enumerate(lengths):
        for c in range((length + 15) // 16):
            value = float(c // 16 + 1)
            if c // 16 in (0, 9):
                value = -100.0
            physical = int(tables[row, c // 4]) * 4 + c % 4
            sums[physical].zero_()
            sums[physical, 0] = value * min(16, length - c * 16)
    result = per_head_paged.misa_topk_heads_paged(
        q, torch.ones(len(lengths), 2), sums, tables,
        torch.arange(len(lengths)).int(), torch.tensor(lengths).int(), 2,
        pooling_block_size=256, prune_topk=8, return_metadata=True,
        rows_per_launch=2,
    )
    assert result.coarse_block_ids.shape == (3, 2, 8)
    for row, length in enumerate(lengths):
        count = (length + 255) // 256
        means = torch.arange(1, count + 1).float()
        means[0] = -100
        if count == 10:
            means[9] = -100
        for slot, head in enumerate(result.candidate_head_ids[row].tolist()):
            scores = means if head == 0 else -means
            ids = result.coarse_block_ids[row, slot]
            valid = ids[ids >= 0].long()
            assert valid.numel() == min(count, 8)
            assert valid.unique().numel() == valid.numel()
            assert torch.equal(scores[valid], scores.sort(descending=True).values[:8])
            assert (ids[min(count, 8):] == -1).all()
    positive_slot = result.candidate_head_ids[0].tolist().index(0)
    assert result.coarse_block_ids[0, positive_slot].tolist() == list(range(8, 0, -1))
    assert torch.isfinite(result.candidate_context_summary).all()


@pytest.mark.parametrize("extra", [{"prune_topk": 0}, {"prune_topk": 1.5},
                                  {"prune_topk": 8, "prune_keep_fraction": 0.6}])
def test_misa_rejects_invalid_fixed_region_policy(extra) -> None:
    with pytest.raises(ValueError):
        per_head_paged.misa_topk_heads_paged(
            torch.zeros(1, 1, 128), torch.ones(1, 1), torch.zeros(4, 128),
            torch.zeros(1, 1).int(), torch.zeros(1).int(), torch.ones(1).int(), 1,
            **extra,
        )


@pytest.mark.parametrize("fraction,expected_topk", [(None, 8), (0.6, None)])
def test_misa_runtime_fixed_default_and_explicit_legacy(monkeypatch, tmp_path, fraction, expected_topk):
    config = {"checkpoint": "unused.pt", "budget": 1,
              "selected_heads_by_layer": {"0": [0]}}
    if fraction is not None:
        config["misa_prune_keep_fraction"] = fraction
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SGLANG_NSA_OFFLINE_ROUTER_CONFIG", str(path))
    offline_unique_head_router.reset_for_tests()
    try:
        assert offline_unique_head_router.misa_chunk_size() == 256
        assert offline_unique_head_router.misa_prune_topk() == expected_topk
        assert offline_unique_head_router.misa_prune_keep_fraction() == fraction
    finally:
        offline_unique_head_router.reset_for_tests()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("length", [2560, 2545, 1537, 720])
def test_fixed_mean_top8_cuda_quota_reference(length) -> None:
    device = "cuda"
    pages = (length + 63) // 64
    # Exact FP8 key=1, per-token scale carries a known strictly increasing score.
    # Coarse blocks 0 and 9 deliberately rank below the eight interior blocks.
    loc = torch.arange(length, device=device)
    chunks = loc // 16
    scalar = chunks.float() + 1 + (loc % 16).float() / 32
    scalar = torch.where((chunks // 16 == 0) | (chunks // 16 == 9), -100 + (loc % 16).float() / 32, scalar)
    key_fp8 = torch.zeros(pages * 64, 128, dtype=torch.float8_e4m3fn, device=device)
    key_fp8[:, 0] = 1
    scales = torch.ones(pages * 64, device=device)
    scales[:length] = scalar
    kv = torch.cat([key_fp8.view(torch.uint8).view(pages, -1),
                    scales.view(torch.uint8).view(pages, -1)], dim=1).contiguous()
    sums = torch.zeros(pages * 4, 128, device=device)
    sums[:, 0].index_add_(0, chunks, scalar)
    table = torch.arange(pages, device=device).flip(0).int().view(1, -1)
    # Store the cache in the reverse physical order to exercise page mapping.
    kv = kv.flip(0).contiguous()
    sums = sums.view(pages, 4, 128).flip(0).reshape(-1, 128).contiguous()
    q = torch.zeros(1, 1, 128, device=device, dtype=torch.bfloat16)
    q[:, :, 0] = 1
    lengths = torch.tensor([length], device=device, dtype=torch.int32)
    rows = torch.zeros(1, device=device, dtype=torch.int32)
    heads, blocks = per_head_paged.misa_topk_heads_paged(
        q, torch.ones(1, 1, device=device), sums, table, rows, lengths, 1,
        pooling_block_size=256, prune_topk=8,
    )
    result = per_head_paged.per_head_topk_paged(
        q, None, kv, sums, table, rows, lengths, 768,
        head_ids=heads, coarse_block_ids=blocks, coarse_block_size=256,
        mean_ranked_chunks=True,
    )[0, 0]
    got = result[result >= 0].cpu().tolist()
    assert len(got) == len(set(got))
    assert (result[len(got):] == -1).all()
    if length <= 720:
        assert set(got) == set(range(length))
        return
    candidates = [c for block in blocks[0, 0].tolist() if block >= 0
                  for c in range(block * 16, block * 16 + 16) if c * 16 < length]
    # CPU reference uses rounded mean scores, matching the bf16 score buffer.
    values = scalar.cpu()
    score = {c: float(values[c * 16:min(c * 16 + 16, length)].mean().bfloat16()) for c in candidates}
    ranked = sorted(candidates, key=lambda c: (-score[c], c))
    expected = []
    for rank, c in enumerate(ranked[:128]):
        quota = 8 if rank < 52 else 4
        tokens = list(range(c * 16, min(c * 16 + 16, length)))
        expected.extend(sorted(tokens, key=lambda t: (-float(values[t]), t))[:quota])
    assert set(got) == set(expected)
    assert all(t // 256 in blocks[0, 0].tolist() for t in got)
    if length == 2560:
        assert len(got) == 720
        assert set(blocks[0, 0].tolist()) == set(range(1, 9))
