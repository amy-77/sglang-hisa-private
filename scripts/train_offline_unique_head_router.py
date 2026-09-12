#!/usr/bin/env python3
"""Cross-validate a small offline MLA-pair assignment scorer."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_contexts(probe_dir: Path, expected_ranks: int = 8) -> dict[str, torch.Tensor]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(probe_dir.glob("sample*_L*_rank*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        groups[(int(record["sample_id"]), int(record["layer"]))].append(record)

    complete = [key for key, records in sorted(groups.items()) if len(records) == expected_ranks]
    if not complete:
        raise RuntimeError(f"no complete {expected_ranks}-rank contexts under {probe_dir}")
    layers = sorted({layer for _, layer in complete})
    layer_to_id = {layer: i for i, layer in enumerate(layers)}

    q_pairs = []
    indexer_queries = []
    gates = []
    utilities = []
    samples = []
    layer_ids = []
    for sample, layer in complete:
        records = sorted(groups[(sample, layer)], key=lambda x: int(x["tp_rank"]))
        q_mla = torch.empty(128, records[0]["q_mla"].shape[-1], dtype=torch.float32)
        mass = torch.empty(64, 128, dtype=torch.float32)
        for record in records:
            heads = record["global_mla_head_ids"].long()
            q_mla[heads] = record["q_mla"].float()
            mass[:, heads] = record["attention_mass_matrix"].float()
        pair_q = q_mla.reshape(64, 2, -1).reshape(64, -1)
        pair_util = mass.reshape(64, 64, 2).mean(-1).transpose(0, 1).contiguous()
        q_pairs.append(pair_q)
        indexer_queries.append(records[0]["q_indexer_quant"].float())
        gates.append(records[0]["indexer_gate"].float())
        utilities.append(pair_util)
        samples.append(sample)
        layer_ids.append(layer_to_id[layer])

    return {
        "q_pair": torch.stack(q_pairs),
        "q_indexer": torch.stack(indexer_queries),
        "gate": torch.stack(gates),
        "util": torch.stack(utilities),
        "sample": torch.tensor(samples, dtype=torch.long),
        "layer_id": torch.tensor(layer_ids, dtype=torch.long),
        "layers": torch.tensor(layers, dtype=torch.long),
    }


class RouterScorer(nn.Module):
    """One low-rank pair/head utility scorer."""

    def __init__(
        self,
        q_dim: int,
        indexer_dim: int,
        n_layers: int,
        rank: int = 32,
        dynamic_indexer_keys: bool = False,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(q_dim, rank, bias=False)
        self.indexer_embedding = nn.Parameter(torch.randn(64, rank) * 0.02)
        self.indexer_q_proj = (
            nn.Linear(indexer_dim, rank, bias=False)
            if dynamic_indexer_keys
            else None
        )
        self.layer_embedding = nn.Embedding(n_layers, rank)
        self.prior = nn.Parameter(torch.zeros(n_layers, 64))
        # The layer-level prior selects globally useful heads.  A separate
        # pair prior is necessary to represent the strong static
        # layer/pair/head specialization before learning query-dependent
        # residuals.
        self.pair_prior = nn.Parameter(torch.zeros(n_layers, 64, 64))
        self.gate_scale = nn.Parameter(torch.zeros(64))
        nn.init.normal_(self.layer_embedding.weight, std=0.02)

    def forward(
        self,
        q_pair: torch.Tensor,
        q_indexer: torch.Tensor,
        gate: torch.Tensor,
        layer_id: torch.Tensor,
    ) -> torch.Tensor:
        q_pair = F.layer_norm(q_pair, (q_pair.shape[-1],))
        pair_latent = self.q_proj(q_pair) + self.layer_embedding(layer_id)[:, None, :]
        indexer_latent = self.indexer_embedding[None, :, :]
        if self.indexer_q_proj is not None:
            q_indexer = F.layer_norm(q_indexer, (q_indexer.shape[-1],))
            indexer_latent = indexer_latent + self.indexer_q_proj(q_indexer)
        dynamic = torch.einsum("npr,nir->npi", pair_latent, indexer_latent)
        gate = (gate - gate.mean(-1, keepdim=True)) / gate.std(
            -1, keepdim=True
        ).clamp_min(1e-5)
        return (
            dynamic
            + self.prior[layer_id, None, :]
            + self.pair_prior[layer_id]
            + gate[:, None, :] * self.gate_scale[None, None, :]
        )


class OfflineRouter(nn.Module):
    """Assignment scorer used after MISA has selected the candidate head set."""

    def __init__(
        self,
        q_dim: int,
        indexer_dim: int,
        n_layers: int,
        rank: int = 32,
        dynamic_indexer_keys: bool = False,
    ) -> None:
        super().__init__()
        kwargs = dict(
            q_dim=q_dim,
            indexer_dim=indexer_dim,
            n_layers=n_layers,
            rank=rank,
            dynamic_indexer_keys=dynamic_indexer_keys,
        )
        self.assignment_scorer = RouterScorer(**kwargs)

    def forward(
        self,
        q_pair: torch.Tensor,
        q_indexer: torch.Tensor,
        gate: torch.Tensor,
        layer_id: torch.Tensor,
    ) -> torch.Tensor:
        inputs = (q_pair, q_indexer, gate, layer_id)
        return self.assignment_scorer(*inputs)


def listwise(score: torch.Tensor, util: torch.Tensor) -> torch.Tensor:
    target = F.softmax(util, dim=-1)
    return -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()


def router_loss(
    assignment_score: torch.Tensor,
    util: torch.Tensor,
    temperature: float,
    mode: str,
    assignment_heads: torch.Tensor | None,
) -> torch.Tensor:
    scaled_util = util / temperature
    if assignment_heads is not None:
        gather_index = assignment_heads[:, None, :].expand(
            -1, assignment_score.shape[1], -1
        )
        score = assignment_score.gather(-1, gather_index)
        assignment_util = util.gather(-1, gather_index)
    else:
        score = assignment_score
        assignment_util = util
    scaled_assignment_util = assignment_util / temperature
    assignment_listwise = listwise(score, scaled_assignment_util)
    if mode == "listwise":
        pair_loss = assignment_listwise
    elif mode == "hard":
        pair_loss = F.cross_entropy(
            score.reshape(-1, score.shape[-1]),
            assignment_util.argmax(-1).reshape(-1),
        )
    elif mode == "regression":
        target = (assignment_util - assignment_util.mean(-1, keepdim=True)) / assignment_util.std(
            -1, keepdim=True
        ).clamp_min(1e-5)
        prediction = (score - score.mean(-1, keepdim=True)) / score.std(
            -1, keepdim=True
        ).clamp_min(1e-5)
        pair_loss = F.smooth_l1_loss(prediction, target)
    elif mode == "hybrid":
        hard = F.cross_entropy(
            score.reshape(-1, score.shape[-1]),
            assignment_util.argmax(-1).reshape(-1),
        )
        pair_loss = assignment_listwise + 0.5 * hard
    else:
        raise ValueError(f"unknown loss mode: {mode}")
    # The full-listwise auxiliary keeps the scorer useful when the runtime
    # candidate set differs from the assignment set sampled during training.
    assignment_full = listwise(assignment_score, scaled_util)
    return pair_loss + 0.25 * assignment_full


def greedy_set(score: torch.Tensor, budget: int) -> list[int]:
    """Select heads maximizing mean per-pair predicted score."""
    selected: list[int] = []
    best = torch.full((score.shape[0],), -torch.inf)
    remaining = set(range(score.shape[1]))
    for _ in range(min(budget, score.shape[1])):
        head = max(remaining, key=lambda i: float(torch.maximum(best, score[:, i]).mean()))
        selected.append(head)
        remaining.remove(head)
        best = torch.maximum(best, score[:, head])
    return selected


def score_prediction(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    selected: list[int],
    oracle_assignment: bool,
) -> float:
    sel = torch.tensor(selected, dtype=torch.long)
    assignment_score = actual[:, sel] if oracle_assignment else predicted[:, sel]
    choice = assignment_score.argmax(-1)
    pair = torch.arange(actual.shape[0])
    return float(actual[pair, sel[choice]].mean())


def train_fold(
    data: dict[str, torch.Tensor],
    train_mask: torch.Tensor,
    steps: int,
    learning_rate: float,
    rank: int,
    temperature: float,
    seed: int,
    dynamic_indexer_keys: bool,
    loss_mode: str,
    assignment_budget: int,
    batch_size: int,
    device: torch.device,
) -> tuple[OfflineRouter, float]:
    torch.manual_seed(seed)
    model = OfflineRouter(
        q_dim=data["q_pair"].shape[-1],
        indexer_dim=data["q_indexer"].shape[-1],
        n_layers=int(data["layers"].numel()),
        rank=rank,
        dynamic_indexer_keys=dynamic_indexer_keys,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    q = data["q_pair"][train_mask].to(device)
    q_indexer = data["q_indexer"][train_mask].to(device)
    gate = data["gate"][train_mask].to(device)
    layer_id = data["layer_id"][train_mask].to(device)
    util_cpu = data["util"][train_mask]
    util = util_cpu.to(device)
    assignment_heads = None
    if assignment_budget > 0:
        assignment_heads = torch.tensor(
            [greedy_set(value, assignment_budget) for value in util_cpu],
            dtype=torch.long,
        ).to(device)
    model.train()
    loss_value = 0.0
    generator = torch.Generator(device=device).manual_seed(seed + 10000)
    for _ in range(steps):
        if batch_size > 0 and batch_size < q.shape[0]:
            index = torch.randint(
                q.shape[0], (batch_size,), generator=generator, device=device
            )
            batch = (
                q[index],
                q_indexer[index],
                gate[index],
                layer_id[index],
                util[index],
                None if assignment_heads is None else assignment_heads[index],
            )
        else:
            batch = (q, q_indexer, gate, layer_id, util, assignment_heads)
        optimizer.zero_grad(set_to_none=True)
        assignment_score = model(*batch[:4])
        loss = router_loss(
            assignment_score,
            batch[4],
            temperature,
            loss_mode,
            batch[5],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach())
    return model.cpu().eval(), loss_value


def save_router_checkpoint(
    path: Path,
    model: OfflineRouter,
    data: dict[str, torch.Tensor],
    *,
    train_samples: list[int],
    config: dict,
    final_train_loss: float,
) -> None:
    """Save a self-describing router checkpoint for replay or deployment tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_class": "OfflineRouter",
            "model_config": {
                "q_dim": int(data["q_pair"].shape[-1]),
                "indexer_dim": int(data["q_indexer"].shape[-1]),
                "n_layers": int(data["layers"].numel()),
                "rank": int(config["rank"]),
                "dynamic_indexer_keys": bool(config["dynamic_indexer_keys"]),
            },
            "training_config": config,
            "layers": data["layers"].tolist(),
            "train_samples": train_samples,
            "final_train_loss": final_train_loss,
            "state_dict": model.state_dict(),
        },
        path,
    )


@torch.no_grad()
def evaluate_fold(
    model: OfflineRouter,
    data: dict[str, torch.Tensor],
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    budgets: list[int],
) -> dict[int, dict[str, list[float]]]:
    assignment_prediction = model(
        data["q_pair"], data["q_indexer"], data["gate"], data["layer_id"]
    )
    train_view = {
        "util": data["util"][train_mask],
        "layer_id": data["layer_id"][train_mask],
    }
    result = {
        budget: {
            "static_set_router_assignment": [],
            "static_set_static_assignment": [],
            "static_set_oracle_assignment": [],
            "gate_argmax": [],
        }
        for budget in budgets
    }
    static_routes = {}
    for layer_id in set(data["layer_id"].tolist()):
        layer_util = train_view["util"][train_view["layer_id"] == layer_id].mean(0)
        route = greedy_set(layer_util, max(budgets))
        for budget in budgets:
            static_routes[(layer_id, budget)] = route[:budget]

    for idx in test_mask.nonzero(as_tuple=False).flatten().tolist():
        predicted_assignment = assignment_prediction[idx]
        actual = data["util"][idx]
        layer_id = int(data["layer_id"][idx])
        layer_static_score = train_view["util"][
            train_view["layer_id"] == layer_id
        ].mean(0)
        gate_head = int(data["gate"][idx].argmax())
        for budget in budgets:
            static_selected = static_routes[(layer_id, budget)]
            result[budget]["static_set_router_assignment"].append(
                score_prediction(predicted_assignment, actual, static_selected, False)
            )
            result[budget]["static_set_static_assignment"].append(
                score_prediction(layer_static_score, actual, static_selected, False)
            )
            result[budget]["static_set_oracle_assignment"].append(
                score_prediction(predicted_assignment, actual, static_selected, True)
            )
            result[budget]["gate_argmax"].append(float(actual[:, gate_head].mean()))
    return result


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Contexts per optimizer step; 0 keeps the original full-batch training.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Training device. Evaluation and saved checkpoints remain on CPU.",
    )
    parser.add_argument(
        "--loss",
        choices=("listwise", "hard", "regression", "hybrid"),
        default="listwise",
    )
    parser.add_argument(
        "--assignment-budget",
        type=int,
        default=0,
        help="Train assignment inside each query's oracle M-head set; 0 uses all 64 heads.",
    )
    parser.add_argument(
        "--dynamic-indexer-keys",
        action="store_true",
        help="Project the current 64 indexer queries instead of using only static head embeddings.",
    )
    parser.add_argument(
        "--queries-per-prompt",
        type=int,
        default=1,
        help="Group consecutive probe sample ids from the same prompt for CV.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optionally save each CV fold and an all-data final-fit checkpoint.",
    )
    parser.add_argument(
        "--fit-all-steps",
        type=int,
        default=0,
        help="Final-fit steps on all prompt groups; requires --checkpoint-dir.",
    )
    args = parser.parse_args()

    if args.fit_all_steps < 0:
        parser.error("--fit-all-steps must be non-negative")
    if args.fit_all_steps > 0 and args.checkpoint_dir is None:
        parser.error("--fit-all-steps requires --checkpoint-dir")
    if args.batch_size < 0:
        parser.error("--batch-size must be non-negative")

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(device_name)

    torch.set_num_threads(min(16, torch.get_num_threads()))
    data = load_contexts(args.probe_dir, args.expected_ranks)
    n_queries = len(set(data["sample"].tolist()))
    if args.queries_per_prompt > 1:
        data["sample"] = torch.div(
            data["sample"], args.queries_per_prompt, rounding_mode="floor"
        )
    unique_samples = sorted(set(data["sample"].tolist()))
    even_samples = set(unique_samples[::2])
    folds = []
    mask_even = torch.tensor([int(x) in even_samples for x in data["sample"]])
    if bool(mask_even.all()) or bool((~mask_even).all()):
        folds.append((torch.ones_like(mask_even), torch.ones_like(mask_even)))
    else:
        folds.extend([(mask_even, ~mask_even), (~mask_even, mask_even)])

    combined = {
        budget: defaultdict(list) for budget in args.budgets
    }
    training_config = {
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "temperature": args.temperature,
        "queries_per_prompt": args.queries_per_prompt,
        "dynamic_indexer_keys": args.dynamic_indexer_keys,
        "loss": args.loss,
        "assignment_budget": args.assignment_budget,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    fold_records = []
    for fold_id, (train_mask, test_mask) in enumerate(folds):
        model, final_loss = train_fold(
            data,
            train_mask,
            args.steps,
            args.learning_rate,
            args.rank,
            args.temperature,
            seed=1234 + fold_id,
            dynamic_indexer_keys=args.dynamic_indexer_keys,
            loss_mode=args.loss,
            assignment_budget=args.assignment_budget,
            batch_size=args.batch_size,
            device=device,
        )
        result = evaluate_fold(model, data, train_mask, test_mask, args.budgets)
        for budget, metrics in result.items():
            for name, values in metrics.items():
                combined[budget][name].extend(values)
        fold_records.append(
            {
                "fold": fold_id,
                "train_samples": sorted(set(data["sample"][train_mask].tolist())),
                "test_samples": sorted(set(data["sample"][test_mask].tolist())),
                "final_train_loss": final_loss,
            }
        )
        if args.checkpoint_dir is not None:
            save_router_checkpoint(
                args.checkpoint_dir / f"router_fold{fold_id}.pt",
                model,
                data,
                train_samples=sorted(set(data["sample"][train_mask].tolist())),
                config=training_config,
                final_train_loss=final_loss,
            )

    final_fit_record = None
    if args.fit_all_steps > 0:
        final_model, final_loss = train_fold(
            data,
            torch.ones_like(data["sample"], dtype=torch.bool),
            args.fit_all_steps,
            args.learning_rate,
            args.rank,
            args.temperature,
            seed=4321,
            dynamic_indexer_keys=args.dynamic_indexer_keys,
            loss_mode=args.loss,
            assignment_budget=args.assignment_budget,
            batch_size=args.batch_size,
            device=device,
        )
        final_config = dict(training_config)
        final_config["steps"] = args.fit_all_steps
        final_fit_record = {
            "steps": args.fit_all_steps,
            "final_train_loss": final_loss,
            "checkpoint": str(args.checkpoint_dir / "router_all.pt"),
        }
        save_router_checkpoint(
            args.checkpoint_dir / "router_all.pt",
            final_model,
            data,
            train_samples=unique_samples,
            config=final_config,
            final_train_loss=final_loss,
        )

    summary = {
        "n_samples": len(unique_samples),
        "n_queries": n_queries,
        "n_contexts": int(data["q_pair"].shape[0]),
        "layers": data["layers"].tolist(),
        "q_pair_dim": int(data["q_pair"].shape[-1]),
        "config": training_config,
        "folds": fold_records,
        "final_fit": final_fit_record,
        "budgets": {
            str(budget): {name: stats(values) for name, values in metrics.items()}
            for budget, metrics in combined.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Offline assignment-scorer cross-validation",
        "",
        f"Prompt groups: {summary['n_samples']}; queries: {summary['n_queries']}; "
        f"query-layer contexts: {summary['n_contexts']}",
        "",
        "| M | Static set + static assignment | Static set + learned assignment | Static set + oracle assignment | Gate argmax |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget, metrics in summary["budgets"].items():
        lines.append(
            f"| {budget} | {metrics['static_set_static_assignment']['mean']:.4f} | "
            f"{metrics['static_set_router_assignment']['mean']:.4f} | "
            f"{metrics['static_set_oracle_assignment']['mean']:.4f} | "
            f"{metrics['gate_argmax']['mean']:.4f} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
