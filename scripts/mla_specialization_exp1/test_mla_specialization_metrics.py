"""Numerical counterexamples and contracts for the offline mass metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mla_specialization_metrics import compute_specialization_metrics


class SpecializationMetricsTests(unittest.TestCase):
    def test_mean_probabilities_and_mean_logits_select_different_tokens(self):
        logits = np.array([[10, 9, -100], [-100, 0, 2]], dtype=np.float32)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        probs = np.tile(probabilities, (8, 1))[None]
        report, arrays = compute_specialization_metrics(probs, [3], ks=(1,))
        self.assertEqual(logits.mean(axis=0).argmax(), 1)
        self.assertTrue(np.all(arrays["k1__group2__indices"] == 2))
        self.assertEqual(arrays["k1__allH__indices"].item(), 2)
        self.assertGreater(report["by_k"]["1"]["policies"]["group2"]["mean_mass"], 0.4)

    def test_identical_heads_have_zero_specialization_gain(self):
        probs = np.tile(np.array([0.1, 0.2, 0.3, 0.4], np.float32), (2, 32, 1))
        report, arrays = compute_specialization_metrics(
            probs, [4, 4], ks=(2,), shared_indices_by_k={2: [[3, 2], [3, 2]]}
        )
        for policy in ("group2", "group16", "allH"):
            np.testing.assert_allclose(arrays[f"k2__{policy}__head_gain_vs_shared_mla"], 0)
            np.testing.assert_allclose(arrays[f"k2__{policy}__head_gain_vs_shared_dsa"], 0)
        np.testing.assert_allclose(arrays["k2__pair_overlap"], 1)
        json.dumps(report, allow_nan=False)

    def test_independent_mass_oracle_dominates_shared_for_each_group_mean(self):
        rng = np.random.default_rng(17)
        probs = rng.random((3, 32, 17), dtype=np.float32)
        lengths = [17, 12, 5]
        for query, length in enumerate(lengths):
            probs[query, :, length:] = 0
        probs /= probs.sum(axis=-1, keepdims=True)
        _, arrays = compute_specialization_metrics(probs, lengths, ks=(3, 9, 30))
        for k in (3, 9, 30):
            for policy in ("group2", "group16"):
                gain = arrays[f"k{k}__{policy}__group_gain_vs_shared_mla"]
                self.assertGreaterEqual(float(gain.min()), -2e-7)
            self.assertGreaterEqual(
                float((arrays[f"k{k}__group2__head_mass"] - arrays[f"k{k}__group16__head_mass"]).mean()),
                -2e-7,
            )

    def test_pair_overlap_matches_known_intersections(self):
        pair_probs = np.array(
            [[0.6, 0.4, 0, 0], [0, 0.6, 0.4, 0], [0, 0, 0.6, 0.4], [0.4, 0, 0, 0.6]],
            dtype=np.float32,
        )
        probs = np.tile(np.repeat(pair_probs, 2, axis=0), (2, 1))[None]
        _, arrays = compute_specialization_metrics(probs, [4], ks=(2,))
        overlap = arrays["k2__pair_overlap"][0]
        self.assertEqual(overlap[0, 0], 1)
        self.assertEqual(overlap[0, 1], 0.5)
        self.assertEqual(overlap[0, 2], 0)
        self.assertEqual(overlap[0, 4], 1)
        np.testing.assert_array_equal(overlap, overlap.T)

    def test_short_prefix_reports_effective_k_and_excludes_padding(self):
        probs = np.zeros((2, 16, 4), dtype=np.float32)
        probs[0, :, :3] = [0.5, 0.3, 0.2]
        probs[1, :, 0] = 1
        shared = [[0, 1, 2, -1, -1], [0, -1, -1, -1, -1]]
        report, arrays = compute_specialization_metrics(
            probs, [3, 1], ks=(5,), shared_indices_by_k={5: shared}
        )
        self.assertEqual(report["by_k"]["5"]["effective_k"], [3, 1])
        self.assertEqual(report["by_k"]["5"]["short_prefix_queries"], 2)
        self.assertEqual(arrays["k5__group2__indices"].shape, (2, 8, 3))
        np.testing.assert_array_equal(arrays["k5__group2__indices"][1, :, 1:], -1)
        np.testing.assert_allclose(arrays["k5__pair_overlap"], 1)
        np.testing.assert_allclose(arrays["k5__shared_dsa__head_mass"], 1)

    def test_invalid_shared_indices_are_rejected(self):
        probs = np.tile(np.array([0.5, 0.3, 0.2, 0], np.float32), (1, 16, 1))
        for indices, reason in (
            ([[0, 0]], "unique"),
            ([[0, 3]], "causal"),
            ([[0, 4]], "causal"),
            ([[0, -2]], "padding"),
            ([[0, -1]], "exactly"),
            ([[0.0, 1.0]], "integers"),
        ):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, reason):
                compute_specialization_metrics(probs, [3], ks=(2,), shared_indices_by_k={2: indices})

    def test_invalid_probabilities_are_rejected(self):
        valid = np.tile(np.array([0.5, 0.3, 0.2, 0], np.float32), (1, 16, 1))
        future = valid.copy()
        future[:, :, 3] = 0.1
        negative = valid.copy()
        negative[:, :, 3] = -0.1
        for probs, reason in ((future, "causal"), (negative, "nonnegative"), (valid * 1.1, "normalized")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                compute_specialization_metrics(probs, [3], ks=(1,))

    def test_sparse_output_is_renormalized_and_relative_l2_is_measured(self):
        probs = np.tile(np.array([0.75, 0.25], np.float32), (1, 16, 1))
        values = np.zeros((2, 16, 1), dtype=np.float32)
        values[0] = 2
        values[1] = 10
        for v in (values, values[None]):
            report, arrays = compute_specialization_metrics(probs, [2], ks=(1, 2), values=v)
            np.testing.assert_allclose(arrays["dense_output"], 4)
            np.testing.assert_allclose(arrays["k1__group2__sparse_output"], 2)
            np.testing.assert_allclose(arrays["k1__group2__relative_l2"], 0.5)
            np.testing.assert_allclose(arrays["k2__group2__relative_l2"], 0)
            self.assertIn("not output error", report["relative_l2_definition"]["selection_objective"])

    def test_zero_selected_mass_has_explicit_output_convention(self):
        probs = np.tile(np.eye(2, dtype=np.float32), (8, 1))[None]
        values = np.ones((2, 16, 1), dtype=np.float32)
        report, arrays = compute_specialization_metrics(probs, [2], ks=(1,), values=values)
        self.assertEqual(report["by_k"]["1"]["policies"]["group2"]["zero_selected_mass_heads"], 8)
        np.testing.assert_array_equal(arrays["k1__group2__indices"], 0)
        np.testing.assert_array_equal(arrays["k1__group2__relative_l2"][0, 1::2], 1)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
