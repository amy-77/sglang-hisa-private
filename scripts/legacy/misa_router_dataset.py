"""Load TP probe shards into explicit MISA assignment-router tensors."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class RouterBatch:
    q_group: torch.Tensor
    candidate_head_ids: torch.Tensor
    candidate_q_indexer: torch.Tensor
    candidate_context_summary: torch.Tensor
    candidate_gate: torch.Tensor
    candidate_importance: torch.Tensor
    candidate_context_stats: torch.Tensor
    layer_ids: torch.Tensor
    utility: torch.Tensor
    candidate_top8_sum: torch.Tensor | None = None
    candidate_top8_context_summary: torch.Tensor | None = None

    def model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.q_group,
            self.candidate_head_ids,
            self.candidate_q_indexer,
            self.candidate_context_summary,
            self.candidate_gate,
            self.candidate_importance,
            self.candidate_context_stats,
            self.layer_ids,
        )


@dataclass
class RouterDataset:
    """All tensors use context as their first dimension."""

    q_group: torch.Tensor  # [contexts, 8, 16 * mla_dim]
    candidate_head_ids: torch.Tensor  # [contexts, M]
    candidate_q_indexer: torch.Tensor  # [contexts, M, indexer_dim]
    candidate_context_summary: torch.Tensor  # [contexts, M, indexer_dim]
    candidate_gate: torch.Tensor  # [contexts, M]
    candidate_importance: torch.Tensor  # [contexts, M]
    candidate_context_stats: torch.Tensor  # [contexts, M, 4]
    layer_ids: torch.Tensor  # [contexts]
    utility: torch.Tensor  # [contexts, 8, M]
    sample_ids: torch.Tensor
    prompt_ids: torch.Tensor
    split_ids: torch.Tensor
    dataset_ids: torch.Tensor
    layers: torch.Tensor
    dataset_names: list[str]
    candidate_top8_sum: torch.Tensor | None = None
    candidate_top8_context_summary: torch.Tensor | None = None

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
            candidate_context_summary=move(
                self.candidate_context_summary, fp32=True
            ),
            candidate_gate=move(self.candidate_gate),
            candidate_importance=move(self.candidate_importance),
            candidate_context_stats=move(self.candidate_context_stats),
            layer_ids=move(self.layer_ids),
            utility=move(self.utility),
            candidate_top8_sum=(
                move(self.candidate_top8_sum)
                if self.candidate_top8_sum is not None
                else None
            ),
            candidate_top8_context_summary=(
                move(self.candidate_top8_context_summary, fp32=True)
                if self.candidate_top8_context_summary is not None
                else None
            ),
        )


def load_sample_metadata(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        payload = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    else:
        decoded = json.loads(text)
        payload = decoded.get("records", decoded)

    by_sample = {}
    for row in payload:
        sample_ids = row.get("probe_sample_ids")
        if sample_ids is None and "probe_sample_start" in row:
            sample_ids = range(
                int(row["probe_sample_start"]),
                int(row["probe_sample_end"]) + 1,
            )
        if sample_ids is None:
            sample_ids = [row.get("sample_id")]
        for sample_id in sample_ids:
            if sample_id is not None:
                by_sample[int(sample_id)] = row
    return by_sample


def _fallback_split(prompt_id: int) -> str:
    bucket = prompt_id % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def _aggregate_context(records: list[dict]) -> dict[str, torch.Tensor]:
    records.sort(key=lambda record: int(record["tp_rank"]))
    if len(records) != 8 or [int(row["tp_rank"]) for row in records] != list(range(8)):
        raise ValueError("group16 requires exactly TP ranks 0..7")
    candidate_head_ids = records[0].get("candidate_head_ids")
    if candidate_head_ids is None:
        raise ValueError("legacy all-head probes cannot train the MISA router")
    candidate_head_ids = candidate_head_ids.long()
    if any(
        not torch.equal(candidate_head_ids, row["candidate_head_ids"].long())
        for row in records[1:]
    ):
        raise ValueError("MISA candidate heads differ across TP ranks")

    # Each record is one physical TP rank. Preserve its existing head order;
    # aggregation is CPU dataset preparation, never a runtime head exchange.
    queries, utilities = [], []
    for row in records:
        rank = int(row["tp_rank"])
        heads = row["global_mla_head_ids"].long()
        expected = torch.arange(rank * 16, (rank + 1) * 16)
        if not torch.equal(heads, expected):
            raise ValueError("group16 requires the original contiguous 16 heads of each TP rank")
        q_local = row["q_mla"]
        mass = row["attention_mass_matrix"]
        if q_local.ndim != 2 or q_local.shape[0] != 16:
            raise ValueError("group16 q_mla must contain 16 local heads")
        if mass.shape != (candidate_head_ids.numel(), 16):
            raise ValueError("group16 attention_mass_matrix must have shape [M, 16]")
        queries.append(q_local.to(torch.bfloat16).flatten())
        # Average the original per-head mass BEFORE deriving a group target.
        # Voting over pair labels is not the objective for a shared candidate.
        utilities.append(mass.float().mean(dim=-1))

    aggregated = {
        "q_group": torch.stack(queries),
        "candidate_head_ids": candidate_head_ids,
        "candidate_q_indexer": records[0]["candidate_q_indexer"].to(
            torch.bfloat16
        ),
        "candidate_context_summary": records[0][
            "candidate_context_summary"
        ].to(torch.bfloat16),
        "candidate_gate": records[0]["candidate_gate"].float(),
        "candidate_importance": records[0]["candidate_importance"].float(),
        "candidate_context_stats": records[0][
            "candidate_context_stats"
        ].float(),
        "utility": torch.stack(utilities),
    }
    if "candidate_top8_sum" in records[0]:
        if any("candidate_top8_sum" not in row for row in records):
            raise ValueError("candidate_top8_sum differs across TP ranks")
        aggregated["candidate_top8_sum"] = records[0][
            "candidate_top8_sum"
        ].float()
    if "candidate_top8_context_summary" in records[0]:
        if any("candidate_top8_context_summary" not in row for row in records):
            raise ValueError(
                "candidate_top8_context_summary differs across TP ranks"
            )
        aggregated["candidate_top8_context_summary"] = records[0][
            "candidate_top8_context_summary"
        ].to(torch.bfloat16)
    return aggregated


def load_router_dataset(
    probe_dir: Path,
    sample_manifest: Path | None,
    expected_ranks: int = 8,
    queries_per_prompt: int = 1,
    min_seq_len: int = 4096,
    probe_sample_stride: int = 1,
) -> RouterDataset:
    if expected_ranks != 8:
        raise ValueError("TP-local group16 training requires expected_ranks=8")
    if probe_sample_stride < 1:
        raise ValueError("probe_sample_stride must be at least 1")
    shards: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(probe_dir.glob("sample*_L*_rank*.pt")):
        row = torch.load(path, map_location="cpu", weights_only=False)
        raw_sample_id = int(row["sample_id"])
        if raw_sample_id % probe_sample_stride:
            continue
        row = dict(row)
        row["sample_id"] = raw_sample_id // probe_sample_stride
        shards[(int(row["sample_id"]), int(row["layer"]))].append(row)
    for path in sorted(probe_dir.glob("probe_shard_L*_rank*_s*.pt")):
        for row in torch.load(path, map_location="cpu", weights_only=False):
            raw_sample_id = int(row["sample_id"])
            if raw_sample_id % probe_sample_stride:
                continue
            row = dict(row)
            row["sample_id"] = raw_sample_id // probe_sample_stride
            shards[(int(row["sample_id"]), int(row["layer"]))].append(row)
    contexts = [
        (key, rows)
        for key, rows in sorted(shards.items())
        if len(rows) == expected_ranks
        and int(rows[0]["seq_len"]) >= min_seq_len
    ]
    if not contexts:
        raise RuntimeError(f"no complete {expected_ranks}-rank contexts in {probe_dir}")

    metadata = load_sample_metadata(sample_manifest)
    if sample_manifest is not None:
        missing = {sample for (sample, _), _ in contexts} - set(metadata)
        if missing:
            raise ValueError(f"probe samples absent from manifest after stride mapping: {sorted(missing)[:10]}")
    layers = sorted({layer for (sample, layer), rows in contexts})
    layer_to_id = {layer: index for index, layer in enumerate(layers)}
    dataset_names = sorted(
        {
            str(metadata.get(sample, {}).get("dataset", "unknown"))
            for (sample, layer), rows in contexts
        }
    )
    dataset_to_id = {name: index for index, name in enumerate(dataset_names)}

    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (sample_id, layer), rows in contexts:
        for name, value in _aggregate_context(rows).items():
            tensors[name].append(value)
        meta = metadata.get(sample_id, {})
        prompt_id = int(meta.get("prompt_id", sample_id // queries_per_prompt))
        split = str(meta.get("split", _fallback_split(prompt_id)))
        dataset = str(meta.get("dataset", "unknown"))
        tensors["sample_ids"].append(torch.tensor(sample_id))
        tensors["prompt_ids"].append(torch.tensor(prompt_id))
        tensors["split_ids"].append(
            torch.tensor({"train": 0, "validation": 1, "test": 2}[split])
        )
        tensors["dataset_ids"].append(torch.tensor(dataset_to_id[dataset]))
        tensors["layer_ids"].append(torch.tensor(layer_to_id[layer]))

    stacked = {name: torch.stack(values) for name, values in tensors.items()}
    return RouterDataset(
        **stacked,
        layers=torch.tensor(layers),
        dataset_names=dataset_names,
    )
