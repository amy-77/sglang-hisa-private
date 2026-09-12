"""Offline same-budget MLA specialization metrics, using NumPy FP32.

The oracles maximize captured attention *probability mass*, not output accuracy.
Heads are grouped in their original contiguous order; no head permutation occurs.
`compute_specialization_metrics` returns a JSON-serializable report and an array
dictionary suitable for `np.savez_compressed(path, **arrays)`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _integer_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return array.astype(np.int64, copy=False)


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Select largest scores, breaking boundary ties by smallest token index."""
    threshold = np.partition(scores, scores.size - k)[scores.size - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[: k - above.size]
    selected = np.concatenate((above, tied))
    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def _oracle_indices(
    probs: np.ndarray, lengths: np.ndarray, k: int, group_size: int
) -> np.ndarray:
    n, h, _ = probs.shape
    width = min(k, int(lengths.max()))
    indices = np.full((n, h // group_size, width), -1, dtype=np.int64)
    for query, length in enumerate(lengths):
        scores = probs[query, :, :length].reshape(h // group_size, group_size, length)
        scores = scores.mean(axis=1, dtype=np.float32)
        effective_k = min(k, int(length))
        for group in range(h // group_size):
            indices[query, group, :effective_k] = _topk(scores[group], effective_k)
    return indices


def _shared_indices(value, lengths: np.ndarray, k: int) -> np.ndarray:
    indices = _integer_array(value, f"shared_indices_by_k[{k}]")
    width = min(k, int(lengths.max()))
    if indices.ndim != 2 or indices.shape[0] != lengths.size:
        raise ValueError(f"shared indices for K={k} must have shape [N, width]")
    if indices.shape[1] not in (width, k):
        raise ValueError(f"shared indices for K={k} must have width {width} or {k}")
    result = np.full((lengths.size, 1, width), -1, dtype=np.int64)
    for query, length in enumerate(lengths):
        row = indices[query]
        if np.any(row < -1):
            raise ValueError("shared indices may only use -1 as padding")
        valid = row[row != -1]
        if valid.size != min(k, int(length)):
            raise ValueError("shared indices must contain exactly min(K, valid_length) tokens")
        if np.any(valid >= length):
            raise ValueError("shared indices must be valid causal token indices")
        if np.unique(valid).size != valid.size:
            raise ValueError("shared indices must be unique")
        result[query, 0, : valid.size] = valid
    return result


def _captured_mass(probs: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, h, _ = probs.shape
    groups = indices.shape[1]
    group_size = h // groups
    head_mass = np.empty((n, h), dtype=np.float32)
    for query in range(n):
        for group in range(groups):
            selected = indices[query, group]
            selected = selected[selected >= 0]
            start = group * group_size
            head_mass[query, start : start + group_size] = probs[
                query, start : start + group_size
            ][:, selected].sum(axis=-1, dtype=np.float32)
    group_mass = head_mass.reshape(n, groups, group_size).mean(axis=-1, dtype=np.float32)
    return head_mass, group_mass


def _pair_overlap(indices: np.ndarray, effective_k: np.ndarray) -> np.ndarray:
    n, groups, _ = indices.shape
    overlap = np.ones((n, groups, groups), dtype=np.float32)
    for query in range(n):
        selected_sets = [set(row[row >= 0].tolist()) for row in indices[query]]
        for first in range(groups):
            for second in range(first):
                value = len(selected_sets[first] & selected_sets[second]) / int(effective_k[query])
                overlap[query, first, second] = value
                overlap[query, second, first] = value
    return overlap


def _validate_values(values, probs_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n, h, t = probs_shape
    valid_shape = (
        (values.ndim == 3 and values.shape[:2] == (t, h))
        or (values.ndim == 4 and values.shape[:3] == (n, t, h))
    )
    if not valid_shape or values.shape[-1] < 1:
        raise ValueError("values must have shape [T,H,D] or [N,T,H,D] with D > 0")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    return values


def _dense_outputs(probs: np.ndarray, lengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    n, h, _ = probs.shape
    output = np.empty((n, h, values.shape[-1]), dtype=np.float32)
    for query, length in enumerate(lengths):
        query_values = values if values.ndim == 3 else values[query]
        output[query] = np.einsum(
            "ht,thd->hd", probs[query, :, :length], query_values[:length], dtype=np.float32
        )
    return output


def _sparse_outputs(probs: np.ndarray, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    n, h, _ = probs.shape
    group_size = h // indices.shape[1]
    output = np.zeros((n, h, values.shape[-1]), dtype=np.float32)
    for query in range(n):
        query_values = values if values.ndim == 3 else values[query]
        for head in range(h):
            selected = indices[query, head // group_size]
            selected = selected[selected >= 0]
            weights = probs[query, head, selected]
            mass = weights.sum(dtype=np.float32)
            if mass > 0:
                output[query, head] = np.einsum(
                    "t,td->d", weights / mass, query_values[selected, head], dtype=np.float32
                )
    return output


def compute_specialization_metrics(
    attention_probs,
    valid_lengths,
    *,
    ks: Sequence[int] = (720,),
    shared_indices_by_k: Mapping[int, object] | None = None,
    values=None,
    normalization_atol: float = 1e-4,
    relative_l2_epsilon: float = 1e-12,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compare contiguous group2, group16, and all-head mass-optimal TopK.

    `attention_probs` is normalized, nonnegative [N,H,T], with exact zeros at
    token indices >= each query's `valid_lengths`. H must be divisible by 16.
    Selection averages probabilities over heads, never logits. All reductions
    on probabilities, masses, and attention outputs use FP32.

    Optional DSA indices are keyed by requested K and shaped [N,K], or
    [N,min(K,max(valid_lengths))]. Short prefixes require -1 padding and exactly
    min(K,valid_length) distinct valid tokens. Every method uses that same count.
    Tie-breaking prefers lower token indices; oracle indices are score ordered.

    Optional V has shape [T,H,D] or [N,T,H,D]. Sparse outputs renormalize selected
    probabilities. A head with zero selected mass has a defined zero output;
    its occurrence is reported. Relative L2 is per head against the dense output,
    dividing by max(dense L2 norm, relative_l2_epsilon). These are output errors
    of mass-selected sets, not output-optimal oracles.

    Array keys use `k{K}__{policy}__{metric}`, where policies are `group2`,
    `group16`, `allH` (shared MLA mass oracle), and optional `shared_dsa`.
    Indices are [N,groups,width], head_mass [N,H], group_mass [N,groups].
    Gain arrays have the same head/group shapes. Pair overlaps are [N,H/2,H/2],
    normalized by effective K, with query-mean matrices also provided.
    """
    probs = np.asarray(attention_probs, dtype=np.float32)
    if probs.ndim != 3 or any(size < 1 for size in probs.shape):
        raise ValueError("attention_probs must have nonempty shape [N,H,T]")
    n, h, t = probs.shape
    if h % 16:
        raise ValueError("H must be divisible by 16 for contiguous group2/group16")
    lengths = _integer_array(valid_lengths, "valid_lengths")
    if lengths.shape != (n,) or np.any((lengths < 1) | (lengths > t)):
        raise ValueError("valid_lengths must have shape [N] with lengths in [1,T]")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("attention_probs must be finite and nonnegative")
    if not np.isfinite(normalization_atol) or normalization_atol <= 0:
        raise ValueError("normalization_atol must be finite and positive")
    if not np.isfinite(relative_l2_epsilon) or relative_l2_epsilon <= 0:
        raise ValueError("relative_l2_epsilon must be finite and positive")
    for query, length in enumerate(lengths):
        if np.any(probs[query, :, length:] != 0):
            raise ValueError("attention_probs must be causal: padded/future probabilities must be zero")
        mass = probs[query, :, :length].sum(axis=-1, dtype=np.float32)
        if not np.allclose(mass, 1.0, atol=normalization_atol, rtol=0):
            raise ValueError("attention_probs must be normalized over each head's valid prefix")
    requested_ks = tuple(ks)
    if not requested_ks or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0
        for k in requested_ks
    ):
        raise ValueError("ks must contain positive integers")
    requested_ks = tuple(int(k) for k in requested_ks)
    if len(set(requested_ks)) != len(requested_ks):
        raise ValueError("ks must be unique")
    shared = {} if shared_indices_by_k is None else shared_indices_by_k
    if any(k not in requested_ks for k in shared):
        raise ValueError("shared_indices_by_k contains a K absent from ks")

    arrays = {"valid_lengths": lengths.copy()}
    dense = None
    if values is not None:
        values = _validate_values(values, probs.shape)
        dense = _dense_outputs(probs, lengths, values)
        arrays["dense_output"] = dense
    report = {
        "num_queries": n,
        "num_heads": h,
        "token_width": t,
        "dtype": "float32",
        "oracle_objective": "maximize mean attention probability mass within each group",
        "head_grouping": "original contiguous head order",
        "allH_definition": f"shared MLA mass oracle from mean probability over all {h} heads",
        "short_prefix_policy": "effective_k = min(requested_k, valid_length)",
        "relative_l2_definition": None if dense is None else {
            "formula": "norm(renormalized_sparse_output-dense_output)/max(norm(dense_output),epsilon)",
            "epsilon": float(relative_l2_epsilon),
            "zero_selected_mass_output": "zero vector",
            "selection_objective": "attention mass; not output error",
        },
        "by_k": {},
    }
    for k in requested_ks:
        prefix = f"k{k}__"
        effective_k = np.minimum(lengths, k)
        arrays[prefix + "effective_k"] = effective_k
        selections = {
            name: _oracle_indices(probs, lengths, k, group_size)
            for name, group_size in (("group2", 2), ("group16", 16), ("allH", h))
        }
        if k in shared:
            selections["shared_dsa"] = _shared_indices(shared[k], lengths, k)
        masses = {name: _captured_mass(probs, indices) for name, indices in selections.items()}
        policies = {}
        for name, indices in selections.items():
            policy_prefix = prefix + name + "__"
            head_mass, group_mass = masses[name]
            arrays[policy_prefix + "indices"] = indices
            arrays[policy_prefix + "head_mass"] = head_mass
            arrays[policy_prefix + "group_mass"] = group_mass
            group_count = indices.shape[1]
            summary = {
                "heads_per_group": h // group_count,
                "group_count": group_count,
                "mean_mass": float(head_mass.mean(dtype=np.float32)),
                "per_query_mean_mass": head_mass.mean(axis=1, dtype=np.float32).tolist(),
            }
            for reference, label in (("allH", "shared_mla"), ("shared_dsa", "shared_dsa")):
                if reference not in masses:
                    continue
                gain = head_mass - masses[reference][0]
                group_gain = gain.reshape(n, group_count, h // group_count).mean(
                    axis=-1, dtype=np.float32
                )
                arrays[policy_prefix + "head_gain_vs_" + label] = gain
                arrays[policy_prefix + "group_gain_vs_" + label] = group_gain
                summary["mean_gain_vs_" + label] = float(gain.mean(dtype=np.float32))
            if dense is not None:
                sparse = _sparse_outputs(probs, indices, values)
                numerator = np.linalg.norm(sparse - dense, axis=-1)
                denominator = np.maximum(np.linalg.norm(dense, axis=-1), relative_l2_epsilon)
                relative_l2 = (numerator / denominator).astype(np.float32)
                arrays[policy_prefix + "sparse_output"] = sparse
                arrays[policy_prefix + "relative_l2"] = relative_l2
                summary["mean_relative_l2"] = float(relative_l2.mean(dtype=np.float32))
                summary["zero_selected_mass_heads"] = int(np.count_nonzero(head_mass == 0))
            policies[name] = summary
        overlap = _pair_overlap(selections["group2"], effective_k)
        arrays[prefix + "pair_overlap"] = overlap
        arrays[prefix + "pair_overlap_mean"] = overlap.mean(axis=0, dtype=np.float32)
        off_diagonal = ~np.eye(h // 2, dtype=bool)
        report["by_k"][str(k)] = {
            "requested_k": k,
            "effective_k": effective_k.tolist(),
            "short_prefix_queries": int(np.count_nonzero(lengths < k)),
            "policies": policies,
            "pair_overlap_mean_off_diagonal": float(overlap[:, off_diagonal].mean(dtype=np.float32)),
            "pair_overlap_denominator": "per-query effective_k",
        }
    report["arrays"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }
    return report, arrays
