import unittest

import numpy as np

from coordinate_geometry.part4_statistics import (bootstrap, delta, describe, number,
                                                  outcome, paired_records, resampling_plan)


class StatisticsTests(unittest.TestCase):
    def test_delta_floor_and_missing(self):
        self.assertEqual(delta(1e-5, 1e-8), 3)
        self.assertEqual(delta(0, 1e-30), 0)
        self.assertIsNone(delta(None, 1))

    def test_outcomes(self):
        self.assertEqual([outcome(d, q) for d, q in ((1, 1), (1, 0), (0, 1), (0, 0))],
                         ["both", "diagonal_only", "rotated_only", "neither"])

    def test_reject_nonfinite_metrics(self):
        for value in ("nan", "inf", "-1"):
            with self.assertRaises(ValueError):
                number(value)
        self.assertIsNone(number(""))

    def test_same_draws_across_angles_and_algorithm(self):
        a = resampling_plan(["i1", "i0", "i2"], 200, 5, "planar|100")
        b = resampling_plan(["i2", "i1", "i0"], 200, 5, "planar|100")
        np.testing.assert_array_equal(a[1], b[1])

    def test_cluster_resampling_not_iid_rows(self):
        # Two clusters, five identical observations each. Resampling whole
        # clusters permits all-zero/all-ten medians; row bootstrap is not used.
        plan = resampling_plan(["a", "b"], 1000, 7, "p20|100")
        interval = bootstrap({"a": [0] * 5, "b": [10] * 5}, plan)
        self.assertEqual(interval, [0, 10])

    def test_missing_metrics_do_not_become_fake_attainment(self):
        plan = resampling_plan(["a", "b"], 200, 7, "x")
        self.assertIsNone(bootstrap({"a": [], "b": []}, plan))
        self.assertEqual(describe([])["n"], 0)
        self.assertEqual(bootstrap({"a": [2], "b": []}, plan), [2, 2])

    def sample_pair(self):
        base = dict(algorithm="adam", regime="transfer", study="planar", dimension="2", kappa="10.0",
                    init_id="i0", init_seed_key="key", pair_id="p0", cluster_id="i0", orientation_id="identity",
                    angle_degrees="", status="finite_cap", gradient_calls="10", first_attainment="2",
                    threshold_retained="True", r_10="1e-8")
        d = dict(base, run_id="d", role="reference", reference_run_id="")
        q = dict(base, run_id="q", role="rotated", reference_run_id="d", orientation_id="deg45", angle_degrees="45.0",
                 threshold_retained="False", r_10="1e-4")
        p = {k: q[k] for k in ("algorithm", "regime", "study", "dimension", "kappa", "pair_id", "cluster_id", "orientation_id")}
        p.update(rotated_run_id="q", reference_run_id="d", outcome="both", reference_retained="True", rotated_retained="False",
                 attainment_ratio="1", correspondence_expected="False", delta_10="4")
        return d, q, p

    def test_historical_vs_retained(self):
        d, q, p = self.sample_pair()
        result, _ = paired_records([d, q], [p], [10])
        self.assertEqual(result[0]["historical_outcome"], "both")
        self.assertEqual(result[0]["retained_outcome"], "diagonal_only")

    def test_missing_duplicate_or_invalid_pairs_rejected(self):
        d, q, p = self.sample_pair()
        for runs, pairs in (([q], [p]), ([d, q, q], [p]), ([d, q], [p, p]), ([d, q], [])):
            with self.assertRaises(ValueError):
                paired_records(runs, pairs, [10])

    def test_seed_and_delta_mismatches_rejected(self):
        d, q, p = self.sample_pair()
        with self.assertRaises(ValueError):
            paired_records([d, dict(q, init_seed_key="other")], [p], [10])
        with self.assertRaises(ValueError):
            paired_records([d, q], [dict(p, delta_10="5")], [10])

    def test_retained_run_requires_historical_attainment(self):
        d, q, p = self.sample_pair()
        # Keep the saved pair internally consistent so this is caught by the
        # run invariant, rather than merely by a stale derived outcome/ratio.
        d["first_attainment"] = ""
        p.update(outcome="rotated_only", attainment_ratio="")
        with self.assertRaisesRegex(ValueError, "retention requires historical attainment"):
            paired_records([d, q], [p], [10])

    def test_first_attainment_is_positive_integer_within_actual_calls(self):
        for first in ("0", "-1", "2.5", "11", "99", "nan", "inf", True):
            with self.subTest(first=first):
                d, q, p = self.sample_pair()
                q["first_attainment"] = first
                if isinstance(first, str) and first not in ("nan", "inf"):
                    p["attainment_ratio"] = str(float(first) / 2)
                with self.assertRaises(ValueError):
                    paired_records([d, q], [p], [10])

    def test_unpaired_reference_runs_receive_attainment_validation(self):
        d, q, p = self.sample_pair()
        for changes in (
            dict(first_attainment="", threshold_retained="True"),
            dict(first_attainment="2.5"),
            dict(first_attainment="11"),
            dict(first_attainment="2", gradient_calls="1"),
        ):
            with self.subTest(changes=changes):
                extra = dict(d, run_id="unpaired", **changes)
                with self.assertRaises(ValueError):
                    paired_records([d, q, extra], [p], [10])

    def test_failed_and_unexecuted_runs_preserve_missing_attainment(self):
        d, q, p = self.sample_pair()
        for status, calls in (("nonfinite", "4"), ("explosive", "3"),
                              ("not_run_tuning_failure", "0")):
            with self.subTest(status=status):
                extra = dict(d, run_id="unpaired", status=status, gradient_calls=calls,
                             first_attainment="", threshold_retained="False", r_10="")
                result, _ = paired_records([d, q, extra], [p], [10])
                self.assertEqual(result[0]["historical_outcome"], "both")
        # A run may attain the threshold and subsequently fail; do not convert
        # this legitimate historical success into final retention or reject it.
        extra = dict(d, run_id="unpaired", status="nonfinite", gradient_calls="4",
                     first_attainment="4", threshold_retained="False", r_10="")
        paired_records([d, q, extra], [p], [10])


if __name__ == "__main__":
    unittest.main()
