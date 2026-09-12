#!/usr/bin/env python3
"""Train query-only and coarse-context Group16-to-Indexer routers."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sglang.srt.layers.attention.nsa.assignment_router_model import (  # noqa: E402
    LayerwiseLinear,
    standardize,
)


MODES = (
    "query_gate",
    "query_gate_group",
    "query_gate_layer_group",
    "coarse_context",
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
    seed: int = 20_260_912
    log_every: int = 100
    eval_every: int = 500


@dataclass
class RouterBatch:
    q_group: torch.Tensor
    candidate_head_ids: torch.Tensor
    candidate_q_indexer: torch.Tensor
    candidate_gate: torch.Tensor
    candidate_importance: torch.Tensor
    layer_ids: torch.Tensor
    utility: torch.Tensor
    per_head_exact_top720: torch.Tensor
    group_exact_top720: torch.Tensor
    candidate_top8_sum: torch.Tensor
    candidate_top8_context_summary: torch.Tensor


@dataclass
class RouterDataset:
    """One row is one request/layer with eight Group16 targets."""

    q_group: torch.Tensor  # [contexts, 8, 16 * mla_dim]
    candidate_head_ids: torch.Tensor  # [contexts, 64]
    candidate_q_indexer: torch.Tensor  # [contexts, 64, indexer_dim]
    candidate_gate: torch.Tensor  # [contexts, 64]
    candidate_importance: torch.Tensor  # [contexts, 64]
    layer_ids: torch.Tensor  # [contexts]
    utility: torch.Tensor  # [contexts, 8, 64]
    per_head_exact_top720: torch.Tensor  # [contexts, 8]
    group_exact_top720: torch.Tensor  # [contexts, 8]
    candidate_top8_sum: torch.Tensor  # [contexts, 64]
    candidate_top8_context_summary: torch.Tensor  # [contexts, 64, indexer_dim]
    sample_ids: torch.Tensor
    source_ids: list[str]
    source_hashes: list[str]
    split_ids: torch.Tensor
    dataset_ids: torch.Tensor
    layers: torch.Tensor
    dataset_names: list[str]

    def __len__(self) -> int:
        return self.q_group.shape[0]

    def split_mask(self, split: str) -> torch.Tensor:
        return self.split_ids == {"train": 0, "validation": 1, "test": 2}[split]

    def batch(self, indices: torch.Tensor, device: torch.device) -> RouterBatch:
        def move(value: torch.Tensor, *, fp32: bool = False) -> torch.Tensor:
            value = value[indices].to(device)
            return value.float() if fp32 else value

        return RouterBatch(
            q_group=move(self.q_group, fp32=True),
            candidate_head_ids=move(self.candidate_head_ids),
            candidate_q_indexer=move(self.candidate_q_indexer, fp32=True),
            candidate_gate=move(self.candidate_gate),
            candidate_importance=move(self.candidate_importance),
            layer_ids=move(self.layer_ids),
            utility=move(self.utility),
            per_head_exact_top720=move(self.per_head_exact_top720),
            group_exact_top720=move(self.group_exact_top720),
            candidate_top8_sum=move(self.candidate_top8_sum),
            candidate_top8_context_summary=move(
                self.candidate_top8_context_summary, fp32=True
            ),
        )


def assignment_loss(
    score: torch.Tensor,
    utility: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Utility-aware 64-way loss over the final candidate dimension."""
    best = utility.max(-1, keepdim=True).values
    target = F.softmax((utility - best) / config.temperature, dim=-1)
    log_probability = F.log_softmax(score, dim=-1)
    soft_loss = F.kl_div(
        log_probability, target, reduction="none"
    ).sum(-1).mean()
    expected_regret = (
        log_probability.exp() * (best - utility).clamp_min(0)
    ).sum(-1).mean()
    tied_best = (utility == best).to(log_probability.dtype)
    tied_best = tied_best / tied_best.sum(-1, keepdim=True)
    top1_loss = -(tied_best * log_probability).sum(-1).mean()
    loss = (
        soft_loss
        + config.regret_weight * expected_regret
        + config.top1_weight * top1_loss
    )
    return loss, {
        "soft_loss": float(soft_loss.detach()),
        "expected_regret": float(expected_regret.detach()),
        "top1_loss": float(top1_loss.detach()),
    }


def _read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_identity(row: dict, *, allow_legacy: bool) -> str:
    request_id = row.get("request_id")
    if request_id is not None:
        return str(request_id)
    if allow_legacy:
        return f"legacy:{int(row['sample_id'])}"
    raise ValueError("probe record is missing request_id; recollect with the all-64 driver")


def _aggregate_context(records: list[dict]) -> dict[str, torch.Tensor]:
    records.sort(key=lambda row: int(row["tp_rank"]))
    if [int(row["tp_rank"]) for row in records] != list(range(8)):
        raise ValueError("each Group16 context requires TP ranks 0..7 exactly once")
    candidate_ids = records[0]["candidate_head_ids"].long()
    expected_ids = torch.arange(64, dtype=candidate_ids.dtype)
    if candidate_ids.shape != (64,) or not torch.equal(
        candidate_ids.sort().values, expected_ids
    ):
        raise ValueError("candidate IDs must cover 0..63 exactly once")
    if any(
        not torch.equal(candidate_ids, row["candidate_head_ids"].long())
        for row in records[1:]
    ):
        raise ValueError("candidate order differs across TP ranks")

    first = records[0]
    shared_shapes = {
        "candidate_q_indexer": (64, first["candidate_q_indexer"].shape[-1]),
        "candidate_gate": (64,),
        "candidate_importance": (64,),
        "candidate_top8_sum": (64,),
        "candidate_top8_context_summary": (
            64,
            first["candidate_top8_context_summary"].shape[-1],
        ),
    }
    for name, shape in shared_shapes.items():
        value = first[name]
        if value.shape != shape or not torch.isfinite(value.float()).all():
            raise ValueError(f"{name} must have finite shape {shape}")
        if any(not torch.equal(value, row[name]) for row in records[1:]):
            raise ValueError(f"{name} differs across TP ranks")

    queries, utilities = [], []
    per_head_exact, group_exact = [], []
    for rank, row in enumerate(records):
        heads = row["global_mla_head_ids"].long()
        if not torch.equal(heads, torch.arange(rank * 16, (rank + 1) * 16)):
            raise ValueError("each TP rank must preserve its contiguous 16 MLA heads")
        query = row["q_mla"]
        if query.ndim != 2 or query.shape[0] != 16:
            raise ValueError("q_mla must have shape [16, mla_dim]")
        if not torch.isfinite(query.float()).all():
            raise ValueError("q_mla must be finite")
        mass = row["attention_mass_matrix"].float()
        if mass.shape != (64, 16):
            raise ValueError("attention_mass_matrix must have shape [64,16]")
        if not torch.isfinite(mass).all() or mass.min() < 0 or mass.max() > 1.0001:
            raise ValueError("candidate attention mass must be finite and within [0,1]")
        queries.append(query.to(torch.bfloat16).flatten())
        utilities.append(mass.mean(-1))
        exact_per_head = row["exact_mla_top720_mass"].float()
        if exact_per_head.shape != (16,) or not torch.isfinite(exact_per_head).all():
            raise ValueError("exact_mla_top720_mass must have finite shape [16]")
        per_head_exact.append(exact_per_head.mean())
        exact_group = row.get("group_exact_top720_mass")
        group_exact.append(
            torch.tensor(float("nan"))
            if exact_group is None
            else torch.as_tensor(exact_group).float()
        )

    return {
        "q_group": torch.stack(queries),
        "candidate_head_ids": candidate_ids,
        "candidate_q_indexer": first["candidate_q_indexer"].to(torch.bfloat16),
        "candidate_gate": first["candidate_gate"].float(),
        "candidate_importance": first["candidate_importance"].float(),
        "candidate_top8_sum": first["candidate_top8_sum"].float(),
        "candidate_top8_context_summary": first[
            "candidate_top8_context_summary"
        ].to(torch.bfloat16),
        "utility": torch.stack(utilities),
        "per_head_exact_top720": torch.stack(per_head_exact),
        "group_exact_top720": torch.stack(group_exact),
    }


def load_group16_dataset(
    probe_dir: Path,
    manifest_path: Path,
    *,
    min_seq_len: int = 4096,
    allow_legacy_identity: bool = False,
    expected_layers: list[int] | None = None,
) -> RouterDataset:
    """Load only complete, identity-checked all-64 Group16 contexts."""
    manifest_rows = _read_manifest(manifest_path)
    metadata: dict[str, dict] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    content_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        if row.get("status") != "ok":
            continue
        request_id = row.get("request_id")
        if request_id is None:
            if not allow_legacy_identity:
                raise ValueError("manifest is missing request_id")
            request_id = f"legacy:{int(row['sample_id'])}"
        if request_id in metadata:
            raise ValueError(f"duplicate request_id in manifest: {request_id}")
        metadata[str(request_id)] = row
        split = str(row["split"])
        source_id = str(row["source_id"])
        source_hash = str(row.get("source_hash", source_id))
        source_splits[source_id].add(split)
        content_splits[source_hash].add(split)
    leaked_sources = {
        source: splits for source, splits in source_splits.items() if len(splits) > 1
    }
    if leaked_sources:
        raise ValueError(
            f"source_id appears in multiple splits: {next(iter(leaked_sources.items()))}"
        )
    leaked_content = {
        digest: splits for digest, splits in content_splits.items() if len(splits) > 1
    }
    if leaked_content:
        raise ValueError(
            f"source content appears in multiple splits: {next(iter(leaked_content.items()))}"
        )

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for shard in sorted(probe_dir.glob("probe_shard_L*_rank*_s*.pt")):
        for row in torch.load(shard, map_location="cpu", weights_only=False):
            identity = _record_identity(row, allow_legacy=allow_legacy_identity)
            grouped[(identity, int(row["layer"]))].append(row)
    if not grouped:
        raise RuntimeError(f"no Group16 probe records in {probe_dir}")

    observed_ids = {identity for identity, _ in grouped}
    missing_manifest = observed_ids - set(metadata)
    if missing_manifest:
        raise ValueError(f"probe request IDs absent from manifest: {sorted(missing_manifest)[:3]}")
    expected_ids = {
        identity
        for identity, row in metadata.items()
        if int(row.get("expected_seq_len", row.get("replay_tokens", 0))) >= min_seq_len
    }
    missing_probe = expected_ids - observed_ids
    if missing_probe:
        raise ValueError(f"manifest requests missing probe data: {sorted(missing_probe)[:3]}")

    layers = sorted({layer for _, layer in grouped})
    if expected_layers is not None and layers != expected_layers:
        missing = sorted(set(expected_layers) - set(layers))
        extra = sorted(set(layers) - set(expected_layers))
        raise ValueError(f"probe layer mismatch: missing={missing}, extra={extra}")
    expected_keys = {(identity, layer) for identity in expected_ids for layer in layers}
    missing_keys = expected_keys - set(grouped)
    if missing_keys:
        raise ValueError(f"incomplete request/layer probe data: {sorted(missing_keys)[:3]}")
    for key, rows in grouped.items():
        if key[0] in expected_ids and len(rows) != 8:
            raise ValueError(f"{key} has {len(rows)} TP records, expected 8")

    layer_to_id = {layer: index for index, layer in enumerate(layers)}
    dataset_names = sorted({str(metadata[i].get("dataset", "unknown")) for i in expected_ids})
    dataset_to_id = {name: index for index, name in enumerate(dataset_names)}
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    source_ids: list[str] = []
    source_hashes: list[str] = []
    for identity in sorted(expected_ids):
        meta = metadata[identity]
        for layer in layers:
            rows = grouped[(identity, layer)]
            expected_len = meta.get("expected_seq_len")
            if expected_len is not None and any(
                int(row["seq_len"]) != int(expected_len) for row in rows
            ):
                raise ValueError(f"{identity} observed an unexpected query length")
            expected_hash = meta.get("prefix_hash")
            if expected_hash is not None and any(
                str(row.get("prefix_hash")) != str(expected_hash) for row in rows
            ):
                raise ValueError(f"{identity} observed an unexpected prefix hash")
            if expected_len is not None and any(
                int(row.get("query_position", -1)) != int(expected_len) - 1
                for row in rows
            ):
                raise ValueError(f"{identity} observed an unexpected query position")
            expected_query_token = meta.get("query_token_id")
            if expected_query_token is not None and any(
                int(row.get("query_token_id", -1)) != int(expected_query_token)
                for row in rows
            ):
                raise ValueError(f"{identity} observed an unexpected query token")
            for name, value in _aggregate_context(rows).items():
                tensors[name].append(value)
            tensors["layer_ids"].append(torch.tensor(layer_to_id[layer]))
            tensors["sample_ids"].append(torch.tensor(int(meta["sample_id"])))
            split = str(meta["split"])
            tensors["split_ids"].append(
                torch.tensor({"train": 0, "validation": 1, "test": 2}[split])
            )
            dataset = str(meta.get("dataset", "unknown"))
            tensors["dataset_ids"].append(torch.tensor(dataset_to_id[dataset]))
            source_ids.append(str(meta["source_id"]))
            source_hashes.append(
                str(meta.get("source_hash", meta["source_id"]))
            )

    stacked = {name: torch.stack(values) for name, values in tensors.items()}
    return RouterDataset(
        **stacked,
        source_ids=source_ids,
        source_hashes=source_hashes,
        layers=torch.tensor(layers),
        dataset_names=dataset_names,
    )


class Group16IndexerRouter(nn.Module):
    def __init__(
        self,
        *,
        q_dim: int,
        indexer_dim: int,
        context_dim: int,
        n_layers: int,
        rank: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.rank = rank
        self.q_proj = LayerwiseLinear(n_layers, q_dim, rank)
        self.indexer_q_proj = LayerwiseLinear(
            n_layers, indexer_dim, rank
        )
        self.indexer_embedding = nn.Parameter(
            torch.randn(n_layers, 64, rank) * 0.02
        )
        self.gate_vector = nn.Parameter(torch.randn(n_layers, rank) * 0.02)
        if mode in ("query_gate_group", "coarse_context"):
            self.group_embedding = nn.Parameter(
                torch.randn(8, rank) * 0.02
            )
        else:
            self.register_parameter("group_embedding", None)
        if mode == "query_gate_layer_group":
            self.layer_group_embedding = nn.Parameter(
                torch.randn(n_layers, 8, rank) * 0.02
            )
        else:
            self.register_parameter("layer_group_embedding", None)
        if mode == "coarse_context":
            self.top8_vector = nn.Parameter(
                torch.randn(n_layers, rank) * 0.02
            )
            self.context_proj = LayerwiseLinear(
                n_layers, context_dim, rank
            )
        else:
            self.register_parameter("top8_vector", None)
            self.context_proj = None

    def forward(
        self,
        q_group: torch.Tensor,
        candidate_head_ids: torch.Tensor,
        candidate_q_indexer: torch.Tensor,
        candidate_gate: torch.Tensor,
        layer_ids: torch.Tensor,
        candidate_top8_sum: torch.Tensor | None = None,
        candidate_top8_context_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        group = self.q_proj(
            F.layer_norm(q_group, (q_group.shape[-1],)), layer_ids
        )
        if self.group_embedding is not None:
            group = group + self.group_embedding.unsqueeze(0)
        if self.layer_group_embedding is not None:
            group = group + self.layer_group_embedding[layer_ids]

        candidate = self.indexer_embedding[
            layer_ids.unsqueeze(1), candidate_head_ids
        ]
        candidate = candidate + self.indexer_q_proj(
            F.layer_norm(
                candidate_q_indexer, (candidate_q_indexer.shape[-1],)
            ),
            layer_ids,
        )
        candidate = candidate + (
            standardize(candidate_gate.float(), 1).unsqueeze(-1)
            * self.gate_vector[layer_ids].unsqueeze(1)
        )
        if self.mode == "coarse_context":
            if (
                candidate_top8_sum is None
                or candidate_top8_context_summary is None
            ):
                raise ValueError(
                    "coarse_context requires Top8Sum and CoarseSummary"
                )
            candidate = candidate + (
                standardize(candidate_top8_sum.float(), 1).unsqueeze(-1)
                * self.top8_vector[layer_ids].unsqueeze(1)
            )
            assert self.context_proj is not None
            candidate = candidate + self.context_proj(
                F.layer_norm(
                    candidate_top8_context_summary,
                    (candidate_top8_context_summary.shape[-1],),
                ),
                layer_ids,
            )
        return torch.einsum("bgr,bmr->bgm", group, candidate) / math.sqrt(
            self.rank
        )


def model_inputs(data: RouterDataset, indices: torch.Tensor, device: torch.device):
    batch = data.batch(indices, device)
    return batch, (
        batch.q_group,
        batch.candidate_head_ids,
        batch.candidate_q_indexer,
        batch.candidate_gate,
        batch.layer_ids,
        batch.candidate_top8_sum,
        batch.candidate_top8_context_summary,
    )


def build_model(
    data: RouterDataset, rank: int, mode: str
) -> Group16IndexerRouter:
    return Group16IndexerRouter(
        q_dim=data.q_group.shape[-1],
        indexer_dim=data.candidate_q_indexer.shape[-1],
        context_dim=data.candidate_top8_context_summary.shape[-1],
        n_layers=data.layers.numel(),
        rank=rank,
        mode=mode,
    )


def train_one(
    data: RouterDataset,
    config: TrainConfig,
    mode: str,
    device: torch.device,
    validation_indices: torch.Tensor | None = None,
    eval_batch_size: int = 64,
) -> tuple[
    Group16IndexerRouter,
    list[dict],
    dict[str, torch.Tensor],
    int,
    dict,
    dict,
]:
    torch.manual_seed(config.seed)
    model = build_model(data, config.rank, mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_indices = data.split_mask("train").nonzero().flatten()
    dataset_counts = torch.bincount(
        data.dataset_ids[train_indices],
        minlength=int(data.dataset_ids.max()) + 1,
    ).clamp_min(1)
    weights = 1.0 / dataset_counts[data.dataset_ids[train_indices]].float()
    generator = torch.Generator().manual_seed(config.seed + 1)
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    best_validation: dict = {}
    model.train()
    for step in range(1, config.steps + 1):
        offsets = torch.multinomial(
            weights, config.batch_size, replacement=True, generator=generator
        )
        batch, inputs = model_inputs(
            data, train_indices[offsets], device
        )
        optimizer.zero_grad(set_to_none=True)
        score = model(*inputs)
        loss, parts = assignment_loss(score, batch.utility, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{mode} produced non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row = {
                "mode": mode,
                "step": step,
                "loss": float(loss.detach()),
                **parts,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if (
            validation_indices is not None
            and validation_indices.numel()
            and (step % config.eval_every == 0 or step == config.steps)
        ):
            validation = evaluate(
                model, data, validation_indices, device, eval_batch_size
            )
            learned = validation["overall"]["learned_utility"]["mean"]
            validation_row = {
                "mode": mode,
                "step": step,
                "validation_learned_utility": learned,
                "validation_regret": validation["overall"]["regret"]["mean"],
            }
            history.append(validation_row)
            print(json.dumps(validation_row), flush=True)
            if (
                best_state is None
                or learned
                > best_validation["overall"]["learned_utility"]["mean"]
            ):
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_step = step
                best_validation = copy.deepcopy(validation)
            model.train()
    if best_state is None:
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        best_step = config.steps
    return (
        model.eval(),
        history,
        best_state,
        best_step,
        best_validation,
        optimizer.state_dict(),
    )


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "count": int(array.size),
    }


def build_static_assignment(data: RouterDataset) -> torch.Tensor:
    """Train-only mean utility indexed by [layer, group, global Indexer]."""
    table = torch.empty(data.layers.numel(), 8, 64)
    train_mask = data.split_mask("train")
    for layer_id in range(data.layers.numel()):
        indices = (train_mask & (data.layer_ids == layer_id)).nonzero().flatten()
        if indices.numel() == 0:
            raise ValueError(f"layer {int(data.layers[layer_id])} has no train rows")
        utility = data.utility[indices]
        ids = data.candidate_head_ids[indices].unsqueeze(1).expand(-1, 8, -1)
        global_utility = torch.empty_like(utility)
        global_utility.scatter_(2, ids, utility)
        table[layer_id] = global_utility.mean(0)
    return table


@torch.no_grad()
def evaluate(
    model: Group16IndexerRouter,
    data: RouterDataset,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    lengths: dict[int, str] | None = None,
    static_assignment: torch.Tensor | None = None,
) -> dict:
    metrics: dict[str, list[float]] = defaultdict(list)
    by_length: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    model.eval()
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
        batch, inputs = model_inputs(data, selected, device)
        score = model(*inputs)
        if not torch.isfinite(score).all():
            raise FloatingPointError("router produced non-finite evaluation scores")
        choice = score.argmax(-1)
        learned = batch.utility.gather(
            -1, choice.unsqueeze(-1)
        ).squeeze(-1)
        oracle = batch.utility.max(-1).values
        chosen_head_ids = batch.candidate_head_ids.gather(1, choice)
        misa_top1_slot = batch.candidate_importance.argmax(-1)
        misa_top1 = batch.utility.gather(
            2,
            misa_top1_slot[:, None, None].expand(-1, 8, 1),
        ).squeeze(-1)
        misa_m8_slots = batch.candidate_importance.topk(8, dim=-1).indices
        misa_m8_oracle = batch.utility.gather(
            2, misa_m8_slots[:, None, :].expand(-1, 8, -1)
        ).max(-1).values
        static = None
        if static_assignment is not None:
            static_global = static_assignment[data.layer_ids[selected]].to(device)
            static_slots = static_global.gather(
                2,
                batch.candidate_head_ids[:, None, :].expand(-1, 8, -1),
            ).argmax(-1)
            static = batch.utility.gather(
                2, static_slots.unsqueeze(-1)
            ).squeeze(-1)
        for offset, data_index in enumerate(selected.tolist()):
            values = {
                "learned_utility": float(learned[offset].mean()),
                "oracle_utility": float(oracle[offset].mean()),
                "regret": float(
                    (oracle[offset] - learned[offset]).mean()
                ),
                "top1_accuracy": float(
                    (learned[offset] == oracle[offset]).float().mean()
                ),
                "unique_selected_indexers": float(
                    chosen_head_ids[offset].unique().numel()
                ),
                "misa_top1_utility": float(misa_top1[offset].mean()),
                "misa_m8_group_oracle": float(
                    misa_m8_oracle[offset].mean()
                ),
            }
            exact_group = batch.group_exact_top720[offset]
            if torch.isfinite(exact_group).all():
                values["group_exact_top720_utility"] = float(
                    exact_group.mean()
                )
                values["group16_constraint_loss"] = float(
                    (
                        batch.per_head_exact_top720[offset] - exact_group
                    ).mean()
                )
                values["candidate_generation_loss"] = float(
                    (exact_group - oracle[offset]).mean()
                )
            if static is not None:
                values["static_utility"] = float(static[offset].mean())
            for name, value in values.items():
                metrics[name].append(value)
            if lengths is not None:
                length = lengths.get(
                    int(data.sample_ids[data_index]), "unknown"
                )
                for name, value in values.items():
                    by_length[length][name].append(value)
    return {
        "overall": {
            name: summarize(values) for name, values in metrics.items()
        },
        "by_length": {
            length: {
                name: summarize(values)
                for name, values in length_metrics.items()
            }
            for length, length_metrics in sorted(by_length.items())
        },
    }


def load_lengths(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row["sample_id"])] = str(
                row.get("length") or "unknown"
            )
    return result


def validate_all64_candidates(data: RouterDataset, name: str) -> None:
    candidate_ids = data.candidate_head_ids
    candidate_count = candidate_ids.shape[1]
    if candidate_count != 64:
        raise ValueError(
            f"{name} data has {candidate_count} candidates; "
            "Group16 Indexer routing requires all-64 supervision"
        )

    expected = torch.arange(64, dtype=candidate_ids.dtype)
    for start in range(0, candidate_ids.shape[0], 8192):
        ids = candidate_ids[start : start + 8192]
        sorted_ids = ids.sort(dim=1).values
        bad_rows = (sorted_ids != expected.unsqueeze(0)).any(dim=1)
        if bad_rows.any():
            offset = int(bad_rows.nonzero()[0])
            row = start + offset
            raise ValueError(
                f"{name} candidate IDs at context {row} do not cover "
                f"0..63 exactly once: {ids[offset].tolist()}"
            )


def checkpoint_payload(
    *,
    state_dict: dict[str, torch.Tensor],
    data: RouterDataset,
    config: TrainConfig,
    mode: str,
    step: int,
    kind: str,
    optimizer_state: dict | None = None,
) -> dict:
    payload = {
        "format_version": 2,
        "model_class": "Group16IndexerRouter",
        "checkpoint_kind": kind,
        "step": step,
        "mode": mode,
        "model_config": {
            "q_dim": data.q_group.shape[-1],
            "indexer_dim": data.candidate_q_indexer.shape[-1],
            "context_dim": data.candidate_top8_context_summary.shape[-1],
            "n_layers": data.layers.numel(),
            "rank": config.rank,
            "mode": mode,
        },
        "layers": data.layers.tolist(),
        "training_config": asdict(config),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in state_dict.items()
        },
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-probe-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-probe-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-identity",
        action="store_true",
        help="Only for reproducing old counter-based probes; never use for new training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        rank=args.rank,
        seed=args.seed,
        eval_every=args.eval_every,
    )
    if config.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    train_data = load_group16_dataset(
        args.train_probe_dir,
        args.train_manifest,
        min_seq_len=4096,
        allow_legacy_identity=args.allow_legacy_identity,
        expected_layers=list(range(61)),
    )
    test_data = load_group16_dataset(
        args.test_probe_dir,
        args.test_manifest,
        min_seq_len=4096,
        allow_legacy_identity=args.allow_legacy_identity,
        expected_layers=list(range(61)),
    )
    for name, data in (("train", train_data), ("test", test_data)):
        if (
            data.candidate_top8_sum is None
            or data.candidate_top8_context_summary is None
        ):
            raise ValueError(f"{name} data is missing coarse router features")
        validate_all64_candidates(data, name)
    if train_data.layers.tolist() != test_data.layers.tolist():
        raise ValueError("train and test layer sets differ")
    train_sources = set(train_data.source_ids)
    test_sources = set(test_data.source_ids)
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError(f"train/test source_id overlap: {sorted(overlap)[:3]}")
    content_overlap = set(train_data.source_hashes) & set(test_data.source_hashes)
    if content_overlap:
        raise ValueError(
            f"train/test source content overlap: {sorted(content_overlap)[:3]}"
        )
    device = torch.device(args.device)
    lengths = load_lengths(args.test_manifest)
    validation_indices = train_data.split_mask("validation").nonzero().flatten()
    test_indices = test_data.split_mask("test").nonzero().flatten()
    if validation_indices.numel() == 0:
        raise ValueError("training manifest has no validation contexts")
    if test_indices.numel() == 0:
        raise ValueError("test manifest has no test contexts")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    static_assignment = build_static_assignment(train_data)
    report = {
        "config": asdict(config),
        "train_contexts": len(train_data),
        "test_contexts": len(test_data),
        "train_candidates": train_data.candidate_head_ids.shape[1],
        "test_candidates": test_data.candidate_head_ids.shape[1],
        "modes": {},
    }
    for mode in MODES:
        (
            model,
            history,
            best_state,
            best_step,
            best_validation,
            optimizer_state,
        ) = train_one(
            train_data,
            config,
            mode,
            device,
            validation_indices,
            args.eval_batch_size,
        )
        last_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        last_validation = evaluate(
            model,
            train_data,
            validation_indices,
            device,
            args.eval_batch_size,
            static_assignment=static_assignment,
        )
        model.load_state_dict(best_state)
        selected_best_validation = evaluate(
            model,
            train_data,
            validation_indices,
            device,
            args.eval_batch_size,
            static_assignment=static_assignment,
        )
        mode_report = {
            "history": history,
            "best_step": best_step,
            "best_validation": selected_best_validation,
            "last_validation": last_validation,
            "heldout_ruler_all64": evaluate(
                model,
                test_data,
                test_indices,
                device,
                args.eval_batch_size,
                lengths,
                static_assignment,
            ),
        }
        mode_dir = args.out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                state_dict=best_state,
                data=train_data,
                config=config,
                mode=mode,
                step=best_step,
                kind="best_validation",
            ),
            mode_dir / "best.pt",
        )
        torch.save(
            checkpoint_payload(
                state_dict=last_state,
                data=train_data,
                config=config,
                mode=mode,
                step=config.steps,
                kind="last",
                optimizer_state=optimizer_state,
            ),
            mode_dir / "last.pt",
        )
        report["modes"][mode] = mode_report
        (args.out_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
