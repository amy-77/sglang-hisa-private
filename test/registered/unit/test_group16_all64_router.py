"""Contracts for the identified Group16 all-64 collection/training path."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from collect_misa_router_samples import replay_plan  # pyright: ignore[reportMissingImports]
from evaluate_group16_static import (  # pyright: ignore[reportMissingImports]
    StaticDataset,
    evaluate_static,
    fit_static,
)
from train_group16_indexer_router_ablation import (  # pyright: ignore[reportMissingImports]
    MODES,
    Group16IndexerRouter,
    RouterDataset,
    TrainConfig,
    _aggregate_context,
    build_static_assignment,
    checkpoint_payload,
    evaluate,
    load_group16_dataset,
    train_one,
)
from sglang.srt.layers.attention.nsa.assignment_router_model import standardize
from sglang.srt.layers.attention.nsa.headmap_probe import request_info
from sglang.srt.layers.attention.nsa.per_head_paged import (
    MISASelection,
    misa_topk_heads_paged,
)


def _records(request_id: str = "group16-probe-0-4097-deadbeef") -> list[dict]:
    generator = torch.Generator().manual_seed(7)
    common = {
        "sample_id": 0,
        "request_id": request_id,
        "prefix_hash": "deadbeef",
        "query_position": 4096,
        "query_token_id": 17,
        "layer": 0,
        "seq_len": 4097,
        "candidate_head_ids": torch.arange(64),
        "candidate_q_indexer": torch.randn(64, 4, generator=generator),
        "candidate_gate": torch.randn(64, generator=generator),
        "candidate_importance": torch.randn(64, generator=generator),
        "candidate_top8_sum": torch.randn(64, generator=generator),
        "candidate_top8_context_summary": torch.randn(
            64, 4, generator=generator
        ),
    }
    rows = []
    for rank in range(8):
        rows.append(
            {
                **copy.deepcopy(common),
                "tp_rank": rank,
                "global_mla_head_ids": torch.arange(
                    rank * 16, (rank + 1) * 16
                ),
                "q_mla": torch.randn(16, 3, generator=generator),
                "attention_mass_matrix": torch.rand(
                    64, 16, generator=generator
                ),
                "exact_mla_top720_mass": torch.rand(
                    16, generator=generator
                ),
                "group_exact_top720_mass": torch.rand(
                    (), generator=generator
                ),
            }
        )
    return rows


def test_replay_plan_has_stable_identity_and_skips_short_queries() -> None:
    trajectories = [
        {
            "prompt_input_ids": [1] * 4095,
            "output_ids": [2] * 20,
            "source_id": "source",
            "kind": "long",
        }
    ]
    plan = replay_plan(
        trajectories, max_replay_tokens=131070, min_query_seq_len=4096
    )
    assert [row["position"] for row in plan] == [0, 4, 16]
    assert plan[0]["expected_seq_len"] == 4096
    assert plan[0]["request_id"].startswith("group16-probe-0-4096-")
    assert len({row["request_id"] for row in plan}) == len(plan)


def test_replay_plan_adds_first_query_that_crosses_sparse_threshold() -> None:
    trajectories = [
        {
            "prompt_input_ids": [1] * 100,
            "output_ids": [2] * 4096,
            "source_id": "reasoning",
            "dataset": "math",
            "split": "train",
        }
    ]
    plan = replay_plan(
        trajectories, max_replay_tokens=131070, min_query_seq_len=4096
    )
    assert any(row["position"] == 3995 for row in plan)
    assert min(row["expected_seq_len"] for row in plan) == 4096


def test_request_identity_roundtrip() -> None:
    rid = "group16-probe-12-8193-0123456789abcdef"
    info = request_info(SimpleNamespace(rids=[rid]))
    assert info is not None
    assert (info.sample_id, info.expected_seq_len, info.prefix_hash) == (
        12,
        8193,
        "0123456789abcdef",
    )
    assert request_info(SimpleNamespace(rids=["unrelated-request"])) is None


def test_group16_aggregation_is_eight_by_64() -> None:
    aggregated = _aggregate_context(_records())
    assert aggregated["q_group"].shape == (8, 16 * 3)
    assert aggregated["utility"].shape == (8, 64)
    torch.testing.assert_close(
        aggregated["utility"][0],
        _records()[0]["attention_mass_matrix"].mean(-1),
    )


def test_candidate_standardization_is_finite_for_one_static_candidate() -> None:
    normalized = standardize(torch.tensor([[3.5], [-2.0]]), dim=1)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized, torch.zeros_like(normalized))


def test_fraction_pruning_pins_boundaries_but_fixed_topk_is_mean_only() -> None:
    common = dict(
        q=torch.ones(1, 1, 128),
        weights=torch.ones(1, 1),
        chunk_sum=torch.stack(
            [
                torch.full((128,), 1.0),
                torch.full((128,), 10.0),
                torch.full((128,), 9.0),
                torch.full((128,), 0.5),
            ]
        ),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        row_batch=torch.tensor([0], dtype=torch.int32),
        row_len=torch.tensor([64], dtype=torch.int32),
        topk_heads=1,
        pooling_block_size=16,
        return_metadata=True,
    )
    fraction = misa_topk_heads_paged(**common, prune_keep_fraction=0.5)
    fixed = misa_topk_heads_paged(**common, prune_topk=2)
    assert isinstance(fraction, MISASelection)
    assert isinstance(fixed, MISASelection)
    assert set(fraction.coarse_block_ids[0, 0].tolist()) == {0, 3}
    assert set(fixed.coarse_block_ids[0, 0].tolist()) == {1, 2}


def test_static_assignment_is_fixed_per_layer_and_group() -> None:
    utility = torch.zeros(4, 8, 64)
    utility[0, :, 3] = 0.8
    utility[1, :, 5] = 0.7
    utility[2, :, 3] = 0.9
    utility[3, :, 5] = 0.85
    data = StaticDataset(
        utility=utility,
        layer_ids=torch.tensor([0, 1, 0, 1]),
        layers=torch.tensor([0, 1]),
        split_ids=torch.tensor([0, 0, 1, 1]),
        source_ids=["train", "train", "validation", "validation"],
        source_hashes=["train-hash", "train-hash", "validation-hash", "validation-hash"],
        datasets=["ruler"] * 4,
        lengths=["32k"] * 4,
    )
    _, assignment = fit_static(data, data.split_mask("train"), "context")
    assert assignment.shape == (2, 8)
    assert assignment[0].tolist() == [3] * 8
    assert assignment[1].tolist() == [5] * 8
    result = evaluate_static(data, data.split_mask("validation"), assignment)
    assert result["overall"]["static_utility"] == pytest.approx(0.875)
    assert result["overall"]["regret"] == pytest.approx(0.0)


def test_group16_aggregation_rejects_duplicate_candidate_id() -> None:
    records = _records()
    for row in records:
        row["candidate_head_ids"][-1] = 62
    with pytest.raises(ValueError, match="0..63"):
        _aggregate_context(records)


@pytest.mark.parametrize("field", ["q_mla", "candidate_gate", "candidate_q_indexer"])
def test_group16_aggregation_rejects_nonfinite_features(field: str) -> None:
    records = _records()
    records[0][field].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite|differs"):
        _aggregate_context(records)


def test_group16_aggregation_rejects_cross_rank_feature_mismatch() -> None:
    records = _records()
    records[1]["candidate_q_indexer"][0, 0] += 1
    with pytest.raises(ValueError, match="differs across TP ranks"):
        _aggregate_context(records)


def test_loader_rejects_source_split_leakage(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": index,
            "request_id": f"group16-probe-{index}-4097-hash{index}",
            "source_id": "same-source",
            "source_hash": "same-content",
            "split": split,
            "status": "ok",
            "expected_seq_len": 4097,
        }
        for index, split in enumerate(("train", "validation"))
    ]
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="multiple splits"):
        load_group16_dataset(tmp_path, manifest, expected_layers=[0])


def test_strict_loader_rejects_incomplete_tp_ranks(tmp_path: Path) -> None:
    request_id = "group16-probe-0-4097-deadbeef"
    manifest = {
        "sample_id": 0,
        "request_id": request_id,
        "source_id": "source",
        "dataset": "synthetic",
        "split": "train",
        "status": "ok",
        "expected_seq_len": 4097,
        "query_token_id": 17,
    }
    manifest_path = tmp_path / "samples.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    torch.save(
        _records()[:-1],
        tmp_path / "probe_shard_L00_rank0_s00000-00000.pt",
    )
    with pytest.raises(ValueError, match="7 TP records"):
        load_group16_dataset(
            tmp_path, manifest_path, expected_layers=[0]
        )


@pytest.mark.parametrize("mode", MODES)
def test_all_router_modes_score_eight_groups_by_64(mode: str) -> None:
    model = Group16IndexerRouter(
        q_dim=48,
        indexer_dim=4,
        context_dim=4,
        n_layers=2,
        rank=8,
        mode=mode,
    )
    score = model(
        torch.randn(2, 8, 48),
        torch.arange(64).repeat(2, 1),
        torch.randn(2, 64, 4),
        torch.randn(2, 64),
        torch.tensor([0, 1]),
        torch.randn(2, 64),
        torch.randn(2, 64, 4),
    )
    assert score.shape == (2, 8, 64)


@pytest.mark.parametrize("mode", MODES)
def test_all_router_modes_train_one_step(mode: str) -> None:
    generator = torch.Generator().manual_seed(11)
    contexts = 4
    data = RouterDataset(
        q_group=torch.randn(contexts, 8, 48, generator=generator),
        candidate_head_ids=torch.arange(64).repeat(contexts, 1),
        candidate_q_indexer=torch.randn(
            contexts, 64, 4, generator=generator
        ),
        candidate_gate=torch.randn(contexts, 64, generator=generator),
        candidate_importance=torch.randn(
            contexts, 64, generator=generator
        ),
        layer_ids=torch.tensor([0, 1, 0, 1]),
        utility=torch.rand(contexts, 8, 64, generator=generator),
        per_head_exact_top720=torch.rand(
            contexts, 8, generator=generator
        ),
        group_exact_top720=torch.rand(contexts, 8, generator=generator),
        candidate_top8_sum=torch.randn(
            contexts, 64, generator=generator
        ),
        candidate_top8_context_summary=torch.randn(
            contexts, 64, 4, generator=generator
        ),
        sample_ids=torch.arange(contexts),
        source_ids=[f"source-{i}" for i in range(contexts)],
        source_hashes=[f"hash-{i}" for i in range(contexts)],
        split_ids=torch.tensor([0, 0, 1, 2]),
        dataset_ids=torch.zeros(contexts, dtype=torch.long),
        layers=torch.tensor([0, 1]),
        dataset_names=["synthetic"],
    )
    model, history, best_state, best_step, best_validation, optimizer = train_one(
        data,
        TrainConfig(
            steps=1, batch_size=2, rank=8, log_every=1, eval_every=1
        ),
        mode,
        torch.device("cpu"),
        validation_indices=torch.tensor([2]),
    )
    assert model.mode == mode
    assert len(history) == 2
    assert torch.isfinite(torch.tensor(history[0]["loss"]))
    assert best_step == 1
    assert best_state
    assert best_validation["overall"]["learned_utility"]["count"] == 1
    metrics = evaluate(
        model,
        data,
        torch.tensor([2]),
        torch.device("cpu"),
        1,
        static_assignment=build_static_assignment(data),
    )
    assert "static_utility" in metrics["overall"]
    assert "misa_m8_group_oracle" in metrics["overall"]
    last = checkpoint_payload(
        state_dict=model.state_dict(),
        data=data,
        config=TrainConfig(steps=1, rank=8),
        mode=mode,
        step=1,
        kind="last",
        optimizer_state=optimizer,
    )
    assert last["step"] == 1
    assert "optimizer_state_dict" in last
