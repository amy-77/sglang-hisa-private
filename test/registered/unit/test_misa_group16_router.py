"""TP-local 16-head assignment preserves placement and uses a shared target."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from misa_router_dataset import RouterDataset, _aggregate_context, load_router_dataset
from misa_router_trainer import (
    TrainConfig,
    assignment_loss,
    build_model,
    save_checkpoint,
    train_router,
)
from sglang.srt.layers.attention.nsa import offline_unique_head_router as runtime
from sglang.srt.layers.attention.nsa.assignment_router_model import MISAAssignmentRouter


def _probe_records() -> list[dict]:
    generator = torch.Generator().manual_seed(17)
    common = {
        "candidate_head_ids": torch.tensor([3, 9, 5]),
        "candidate_q_indexer": torch.randn(3, 4, generator=generator),
        "candidate_context_summary": torch.randn(3, 4, generator=generator),
        "candidate_gate": torch.tensor([0.1, 0.6, 0.3]),
        "candidate_importance": torch.tensor([0.2, 0.7, 0.4]),
        "candidate_context_stats": torch.randn(3, 4, generator=generator),
    }
    records = []
    for rank in range(8):
        mass = torch.full((3, 16), 0.2)
        # Seven pairs narrowly prefer slot 0, but group mean prefers slot 1.
        mass[0, :14], mass[0, 14:] = 0.51, 0.0
        mass[1, :14], mass[1, 14:] = 0.49, 1.0
        records.append(
            {
                **copy.deepcopy(common),
                "tp_rank": rank,
                "global_mla_head_ids": torch.arange(rank * 16, (rank + 1) * 16),
                "q_mla": torch.randn(16, 3, generator=generator) + rank,
                "attention_mass_matrix": mass,
            }
        )
    return records


def _dataset() -> RouterDataset:
    aggregated = _aggregate_context(_probe_records())
    stacked = {name: value.unsqueeze(0).repeat(2, *([1] * value.ndim))
               for name, value in aggregated.items()}
    return RouterDataset(
        **stacked,
        layer_ids=torch.zeros(2, dtype=torch.long),
        sample_ids=torch.tensor([0, 1]),
        prompt_ids=torch.tensor([0, 1]),
        split_ids=torch.zeros(2, dtype=torch.long),
        dataset_ids=torch.zeros(2, dtype=torch.long),
        layers=torch.tensor([0]),
        dataset_names=["synthetic"],
    )


def test_probe_aggregation_preserves_each_tp_ranks_original_16_heads() -> None:
    records = _probe_records()
    # File arrival order is irrelevant; each rank's internal head order is not.
    aggregated = _aggregate_context(list(reversed(records)))
    assert aggregated["q_group"].shape == (8, 16 * 3)
    assert aggregated["utility"].shape == (8, 3)
    for rank, record in enumerate(records):
        torch.testing.assert_close(
            aggregated["q_group"][rank],
            record["q_mla"].to(torch.bfloat16).flatten(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            aggregated["utility"][rank],
            record["attention_mass_matrix"].mean(-1),
        )
    assert aggregated["utility"].argmax(-1).tolist() == [1] * 8
    pair_choices = records[0]["attention_mass_matrix"].view(3, 8, 2).mean(-1).argmax(0)
    assert pair_choices.bincount(minlength=3).argmax().item() == 0


@pytest.mark.parametrize(
    "invalid",
    ["missing_rank", "duplicate_rank", "cross_rank_head", "duplicate_head",
     "reordered_heads", "wrong_local_head_count", "pair_query", "pair_utility"],
)
def test_probe_aggregation_rejects_incompatible_tp_or_pair_data(invalid: str) -> None:
    records = _probe_records()
    if invalid == "missing_rank":
        records.pop()
    elif invalid == "duplicate_rank":
        records[-1]["tp_rank"] = 6
    elif invalid == "cross_rank_head":
        records[0]["global_mla_head_ids"][15] = 16
    elif invalid == "duplicate_head":
        records[0]["global_mla_head_ids"][15] = 14
    elif invalid == "reordered_heads":
        records[0]["global_mla_head_ids"] = records[0]["global_mla_head_ids"].flip(0)
    elif invalid == "wrong_local_head_count":
        records[0]["global_mla_head_ids"] = torch.arange(8)
    elif invalid == "pair_query":
        records[0]["q_mla"] = torch.zeros(8, 6)
    elif invalid == "pair_utility":
        records[0]["attention_mass_matrix"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="group16"):
        _aggregate_context(records)


def test_group16_training_rejects_other_attention_tp_sizes(tmp_path) -> None:
    with pytest.raises(ValueError, match="expected_ranks=8"):
        load_router_dataset(tmp_path, None, expected_ranks=4)


def test_group16_aggregation_rejects_different_candidate_orders() -> None:
    records = _probe_records()
    records[7]["candidate_head_ids"] = records[7]["candidate_head_ids"].flip(0)
    with pytest.raises(ValueError, match="candidate heads differ"):
        _aggregate_context(records)


def test_group16_model_and_loss_backpropagate_all_local_head_inputs() -> None:
    data = _dataset()
    batch = data.batch(torch.tensor([0, 1]), torch.device("cpu"))
    batch.q_group.requires_grad_()
    model = build_model(data, rank=4)
    score = model(*batch.model_inputs())
    assert score.shape == (2, 8, 3)
    loss, parts = assignment_loss(score, batch.utility, TrainConfig())
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in parts.values())
    loss.backward()
    assert model.q_proj.weight.grad is not None
    assert torch.isfinite(model.q_proj.weight.grad).all()
    assert batch.q_group.grad is not None
    gradients = batch.q_group.grad.reshape(2, 8, 16, 3)
    assert (gradients.abs().sum(-1) > 0).all()


def test_group16_trains_from_scratch_and_checkpoint_roundtrips(
    monkeypatch, tmp_path
) -> None:
    data = _dataset()
    config = TrainConfig(steps=3, batch_size=2, rank=4, log_every=3, seed=123)
    with monkeypatch.context() as patch:
        def forbidden_load(*args, **kwargs):
            raise AssertionError("group16 training must not load pair-router weights")
        patch.setattr(torch, "load", forbidden_load)
        model, history = train_router(data, config, torch.device("cpu"))
    assert history[-1]["step"] == 3
    checkpoint_path = tmp_path / "group16.pt"
    save_checkpoint(checkpoint_path, model, data, config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 3
    assert checkpoint["assignment_unit"] == "tp_local"
    assert checkpoint["mla_heads_per_group"] == 16
    assert checkpoint["attention_tp_size"] == 8
    assert checkpoint["initialization"] == "random"
    assert checkpoint["model_config"]["q_dim"] == 16 * 3
    restored = MISAAssignmentRouter(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    inputs = data.batch(torch.tensor([0]), torch.device("cpu")).model_inputs()
    torch.testing.assert_close(restored(*inputs), model(*inputs), rtol=0, atol=0)


@pytest.fixture
def group16_runtime(monkeypatch, tmp_path):
    from sglang.srt.layers import dp_attention

    monkeypatch.setattr(dp_attention, "get_attention_tp_rank", lambda: 5)
    monkeypatch.setattr(dp_attention, "get_attention_tp_size", lambda: 8)
    data = _dataset()
    checkpoint_path = tmp_path / "group16.pt"
    save_checkpoint(
        checkpoint_path, build_model(data, rank=4), data, TrainConfig(rank=4)
    )
    config_path = tmp_path / "router.json"
    config_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "budget": 3,
        "assignment_mode": "learned",
        "head_selection_mode": "misa",
        "mla_heads_per_group": 16,
        "misa_chunk_size": 256,
        "selected_heads_by_layer": {"0": [3, 5, 9]},
    }))
    monkeypatch.setenv("SGLANG_NSA_OFFLINE_ROUTER_CONFIG", str(config_path))
    runtime.reset_for_tests()
    yield checkpoint_path
    runtime.reset_for_tests()


def _runtime_inputs():
    generator = torch.Generator().manual_seed(29)
    batch = SimpleNamespace()
    heads = torch.tensor([[3, 9, 5], [5, 3, 9]])
    selection = SimpleNamespace(
        candidate_head_ids=heads,
        candidate_context_summary=torch.randn(2, 3, 4, generator=generator),
        candidate_importance=torch.randn(2, 3, generator=generator),
        candidate_context_stats=torch.randn(2, 3, 4, generator=generator),
    )
    runtime.stash_selected_heads(batch, 0, heads, total_rows=2)
    runtime.stash_misa_features(
        batch, 0,
        torch.randn(2, 64, 4, generator=generator),
        torch.randn(2, 64, generator=generator),
        selection, total_rows=2,
    )
    return {
        "layer_id": 0,
        "q_mla": torch.randn(2, 16, 3, generator=generator),
        "candidates": torch.arange(2 * 3 * 7, dtype=torch.int32).reshape(2, 3, 7),
        "forward_batch": batch,
    }


def test_runtime_scores_one_local_group_and_repeats_one_choice(
    group16_runtime, monkeypatch
) -> None:
    calls = []

    def scorer(q_group, *candidate_inputs):
        calls.append(q_group.detach().clone())
        assert candidate_inputs[0].tolist() == [[3, 9, 5], [5, 3, 9]]
        return torch.tensor([[[0.0, 2.0, 1.0]], [[3.0, 2.0, 1.0]]])

    monkeypatch.setattr(runtime, "_model_for", lambda device: scorer)
    inputs = _runtime_inputs()
    original_q = inputs["q_mla"].clone()
    result = runtime.route_candidates(**inputs)
    assert len(calls) == 1
    assert calls[0].shape == (2, 1, 16 * 3)
    torch.testing.assert_close(calls[0], original_q.reshape(2, 1, -1))
    torch.testing.assert_close(inputs["q_mla"], original_q, rtol=0, atol=0)
    expected = torch.stack([inputs["candidates"][0, 1], inputs["candidates"][1, 0]])
    assert result.shape == (2, 8, 7)
    assert result.is_contiguous()
    torch.testing.assert_close(result, expected[:, None, :].expand(-1, 8, -1))


def test_runtime_restores_fresh_group16_checkpoint(group16_runtime) -> None:
    inputs = _runtime_inputs()
    features = inputs["forward_batch"].offline_router_misa_features[0]
    checkpoint = torch.load(group16_runtime, map_location="cpu", weights_only=False)
    model = MISAAssignmentRouter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    score = model(
        inputs["q_mla"].reshape(2, 1, -1),
        features["candidate_head_ids"], features["candidate_q_indexer"],
        features["candidate_context_summary"], features["candidate_gate"],
        features["candidate_importance"], features["candidate_context_stats"],
        torch.zeros(2, dtype=torch.long),
    )
    choice = score.argmax(-1).squeeze(1)
    expected = inputs["candidates"][torch.arange(2), choice]
    result = runtime.route_candidates(**inputs)
    torch.testing.assert_close(result, expected[:, None, :].expand(-1, 8, -1))


def test_static_group16_scans_one_frozen_indexer_per_tp_rank(
    monkeypatch, tmp_path
) -> None:
    from sglang.srt.layers import dp_attention

    monkeypatch.setattr(dp_attention, "get_attention_tp_rank", lambda: 5)
    monkeypatch.setattr(dp_attention, "get_attention_tp_size", lambda: 8)
    assignment = [3, 5, 9, 11, 13, 17, 19, 23]
    config_path = tmp_path / "static_group16.json"
    config_path.write_text(
        json.dumps(
            {
                "budget": 1,
                "assignment_mode": "static",
                "head_selection_mode": "static",
                "mla_heads_per_group": 16,
                "misa_chunk_size": 512,
                "misa_prune_keep_fraction": 0.6,
                "static_assignment_by_layer": {"0": assignment},
            }
        )
    )
    monkeypatch.setenv("SGLANG_NSA_OFFLINE_ROUTER_CONFIG", str(config_path))
    runtime.reset_for_tests()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "static Group16 must not load a router checkpoint"
        ),
    )

    assert runtime.budget() == 1
    assert runtime.candidate_head_ids(0) == (17,)
    candidates = torch.arange(2 * 1 * 7, dtype=torch.int32).reshape(2, 1, 7)
    result = runtime.route_candidates(
        layer_id=0,
        q_mla=torch.zeros(2, 16, 3),
        candidates=candidates,
        forward_batch=SimpleNamespace(),
    )
    assert result.shape == (2, 8, 7)
    assert result.is_contiguous()
    torch.testing.assert_close(result, candidates.expand(-1, 8, -1))
    runtime.reset_for_tests()


@pytest.mark.parametrize("tp_size,local_heads", [(4, 32), (16, 8), (8, 8)])
def test_runtime_rejects_incompatible_attention_tp_or_local_heads(
    group16_runtime, monkeypatch, tp_size, local_heads
) -> None:
    from sglang.srt.layers import dp_attention

    monkeypatch.setattr(dp_attention, "get_attention_tp_size", lambda: tp_size)
    inputs = _runtime_inputs()
    inputs["q_mla"] = torch.zeros(2, local_heads, 3)
    with pytest.raises(ValueError, match="TP|local|16"):
        runtime.route_candidates(**inputs)


@pytest.mark.parametrize("legacy_class", ["MISAAssignmentRouter", "OfflineRouter"])
def test_group16_config_rejects_pair_router_checkpoints(
    group16_runtime, legacy_class
) -> None:
    checkpoint = torch.load(group16_runtime, map_location="cpu", weights_only=False)
    for key in (
        "assignment_unit", "mla_heads_per_group", "attention_tp_size", "initialization"
    ):
        checkpoint.pop(key)
    checkpoint["format_version"] = 2
    checkpoint["model_class"] = legacy_class
    checkpoint["model_config"]["q_dim"] = 2 * 3
    torch.save(checkpoint, group16_runtime)
    with pytest.raises(ValueError, match="group|16|pair|checkpoint"):
        runtime._load_checkpoint()


@pytest.mark.parametrize(
    "metadata_key,bad_value",
    [("mla_heads_per_group", 2), ("attention_tp_size", 4),
     ("assignment_unit", "pair")],
)
def test_group16_checkpoint_rejects_incompatible_group_metadata(
    group16_runtime, metadata_key, bad_value
) -> None:
    checkpoint = torch.load(group16_runtime, map_location="cpu", weights_only=False)
    checkpoint[metadata_key] = bad_value
    torch.save(checkpoint, group16_runtime)
    with pytest.raises(ValueError, match="group|16|TP|tp_local|checkpoint"):
        runtime._load_checkpoint()


def test_validation_scores_actual_subset_before_candidate_normalization() -> None:
    from validate_misa_router import score_candidate_subset

    data = _dataset()
    batch = data.batch(torch.tensor([0]), torch.device("cpu"))
    batch.candidate_importance = torch.tensor([[2.0, 0.0, 1.0]])
    batch.candidate_gate = torch.tensor([[0.0, 1.0, 100.0]])
    model = build_model(data, rank=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.feature_proj.weight[0, 0, 0] = 1.0
        model.feature_proj.weight[0, 0, 1] = 2.0

    selected_slots = torch.tensor([0, 1])
    full_set_score = model(*batch.model_inputs())[:, :, selected_slots]
    runtime_score = score_candidate_subset(
        model, batch.model_inputs(), selected_slots
    )
    # The unselected extreme gate changes full-set scaling and flips the winner.
    assert full_set_score.argmax(-1).tolist() == [[0] * 8]
    assert runtime_score.shape == (1, 8, 2)
    assert runtime_score.argmax(-1).tolist() == [[1] * 8]
