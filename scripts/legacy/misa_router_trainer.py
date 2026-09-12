"""Training and evaluation for the candidate-wise MISA assignment router."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from legacy.misa_router_dataset import RouterDataset
from sglang.srt.layers.attention.nsa.assignment_router_model import (
    MISAAssignmentRouter,
)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    rank: int = 64
    temperature: float = 0.02
    regret_weight: float = 1.0
    top1_weight: float = 0.1
    grad_clip: float = 1.0
    seed: int = 20_260_907
    log_every: int = 100


def build_model(data: RouterDataset, rank: int) -> MISAAssignmentRouter:
    return MISAAssignmentRouter(
        q_dim=data.q_group.shape[-1],
        indexer_dim=data.candidate_q_indexer.shape[-1],
        context_dim=data.candidate_context_summary.shape[-1],
        context_stats_dim=data.candidate_context_stats.shape[-1],
        n_layers=data.layers.numel(),
        rank=rank,
    )


def assignment_loss(
    score: torch.Tensor,
    utility: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    best_utility = utility.max(-1, keepdim=True).values
    soft_target = F.softmax(
        (utility - best_utility) / config.temperature, dim=-1
    )
    log_probability = F.log_softmax(score, dim=-1)
    soft_loss = F.kl_div(
        log_probability, soft_target, reduction="none"
    ).sum(-1).mean()
    expected_regret = (
        log_probability.exp() * (best_utility - utility).clamp_min(0)
    ).sum(-1).mean()
    # Utility ties are common when the visible context is short or candidate
    # lists overlap. A hard argmax would arbitrarily label slot 0 as correct.
    top1_target = (utility == best_utility).to(log_probability.dtype)
    top1_target = top1_target / top1_target.sum(-1, keepdim=True)
    top1_loss = -(top1_target * log_probability).sum(-1).mean()
    loss = (
        soft_loss
        + config.regret_weight * expected_regret
        + config.top1_weight * top1_loss
    )
    parts = {
        "soft_loss": float(soft_loss.detach()),
        "expected_regret": float(expected_regret.detach()),
        "top1_loss": float(top1_loss.detach()),
    }
    return loss, parts


def train_router(
    data: RouterDataset,
    config: TrainConfig,
    device: torch.device,
) -> tuple[MISAAssignmentRouter, list[dict]]:
    # Always construct a fresh model: this path does not load pair-router weights.
    torch.manual_seed(config.seed)
    model = build_model(data, config.rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_indices = data.split_mask("train").nonzero().flatten()
    if train_indices.numel() == 0:
        raise RuntimeError("training split is empty")
    dataset_counts = torch.bincount(
        data.dataset_ids[train_indices],
        minlength=int(data.dataset_ids.max()) + 1,
    ).clamp_min(1)
    sample_weights = 1.0 / dataset_counts[
        data.dataset_ids[train_indices]
    ].float()
    generator = torch.Generator().manual_seed(config.seed + 1)

    history = []
    model.train()
    for step in range(1, config.steps + 1):
        offsets = torch.multinomial(
            sample_weights,
            config.batch_size,
            replacement=True,
            generator=generator,
        )
        batch = data.batch(train_indices[offsets], device)
        optimizer.zero_grad(set_to_none=True)
        score = model(*batch.model_inputs())
        loss, parts = assignment_loss(score, batch.utility, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            record = {"step": step, "loss": float(loss.detach()), **parts}
            history.append(record)
            print(json.dumps(record), flush=True)
    return model.cpu().eval(), history


def _static_assignment(data: RouterDataset) -> torch.Tensor:
    """Mean train utility indexed by [layer, TP-local group, global Indexer head]."""
    total = torch.zeros(data.layers.numel(), data.utility.shape[1], 64)
    count = torch.zeros_like(total)
    for index in data.split_mask("train").nonzero().flatten().tolist():
        layer = int(data.layer_ids[index])
        for slot, head in enumerate(data.candidate_head_ids[index].tolist()):
            total[layer, :, head] += data.utility[index, :, slot]
            count[layer, :, head] += 1
    return total / count.clamp_min(1)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


@torch.no_grad()
def evaluate_router(
    model: MISAAssignmentRouter,
    data: RouterDataset,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    indices = data.split_mask(split).nonzero().flatten()
    if indices.numel() == 0:
        return {}
    model = model.to(device).eval()
    static_utility = _static_assignment(data)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = data.batch(batch_indices, device)
        score = model(*batch.model_inputs())
        learned = batch.utility.gather(
            -1, score.argmax(-1).unsqueeze(-1)
        ).squeeze(-1)
        oracle = batch.utility.max(-1).values
        static_score = torch.stack(
            [
                static_utility[int(layer), :, heads]
                for layer, heads in zip(
                    data.layer_ids[batch_indices],
                    data.candidate_head_ids[batch_indices],
                )
            ]
        ).to(device)
        static = batch.utility.gather(
            -1, static_score.argmax(-1).unsqueeze(-1)
        ).squeeze(-1)

        for offset, context_index in enumerate(batch_indices.tolist()):
            groups = (
                "all",
                f"dataset:{data.dataset_names[int(data.dataset_ids[context_index])]}",
                f"layer:{int(data.layers[data.layer_ids[context_index]])}",
            )
            values = {
                "learned_utility": float(learned[offset].mean()),
                "oracle_utility": float(oracle[offset].mean()),
                "regret": float((oracle[offset] - learned[offset]).mean()),
                "static_utility": float(static[offset].mean()),
                "top1_accuracy": float(
                    (
                        score[offset].argmax(-1)
                        == batch.utility[offset].argmax(-1)
                    )
                    .float()
                    .mean()
                ),
            }
            for group in groups:
                for name, value in values.items():
                    metrics[group][name].append(value)
    return {
        group: {name: _summary(values) for name, values in group_metrics.items()}
        for group, group_metrics in metrics.items()
    }


def save_checkpoint(
    path: Path,
    model: MISAAssignmentRouter,
    data: RouterDataset,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "assignment_unit": "tp_local",
            "mla_heads_per_group": 16,
            "attention_tp_size": 8,
            "initialization": "random",
            "model_class": "MISAAssignmentRouter",
            "model_config": {
                "q_dim": data.q_group.shape[-1],
                "indexer_dim": data.candidate_q_indexer.shape[-1],
                "context_dim": data.candidate_context_summary.shape[-1],
                "context_stats_dim": data.candidate_context_stats.shape[-1],
                "n_layers": data.layers.numel(),
                "rank": config.rank,
                "n_indexer_heads": 64,
            },
            "layers": data.layers.tolist(),
            "training_config": asdict(config),
            "state_dict": model.state_dict(),
        },
        path,
    )
