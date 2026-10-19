"""Regression checks for complete plans, truthful fields and report denominators."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from coordinate_geometry import part3_experiments as pipeline
from coordinate_geometry import verify_part3 as checker

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = json.loads((ROOT / "configs/part3_smoke.json").read_text())
    config["tuning"].update(max_steps=3, candidates=2, inits_per_orientation=1, transfer_orientations=1)
    config["evaluation"].update(max_steps=4, checkpoints=[2, 4], record_every=2)
    config["pilot"].update(orientations=1, inits_per_orientation=1, max_rounds=1)
    config["algorithms"] = {"gd": {"range": [.05, 1.998], "params": {}},
                             "adam": {"range": [.001, .3], "params": {}}}
    config["studies"] = [{"name": "planar", "dimension": 2, "condition_numbers": [10],
                          "evaluation": {"angles_degrees": [0, 45], "inits": 1}},
                         {"name": "spatial", "dimension": 3, "condition_numbers": [10],
                          "evaluation": {"rotations": 1, "inits_per_rotation": 1}}]
    return config


class Part3VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.baseline = cls.root / "baseline"
        cls.config = small_config()
        path = cls.root / "config.json"
        path.write_text(json.dumps(cls.config), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--config", str(path), "--output", str(cls.baseline)])

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.workspace_copy = tempfile.TemporaryDirectory(dir=self.root)
        self.output = Path(self.workspace_copy.name) / "output"
        shutil.copytree(self.baseline, self.output)

    def tearDown(self):
        self.workspace_copy.cleanup()

    def records(self, name="evaluation"):
        return checker.read_jsonl(self.output / "raw" / f"{name}.jsonl")

    def save_records(self, records, name="evaluation"):
        (self.output / "raw" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_independent_full_plan_and_sampled_fields_pass(self):
        report = checker.verify(self.output, sample=1000)
        self.assertTrue(report["passed"])
        self.assertEqual(report["complete_stage_records"]["evaluation"], 20)
        self.assertEqual(report["pairs_checked"], 10)

    def test_missing_stage_record_and_duplicate_are_rejected_before_sampling(self):
        original = self.records()
        for records in ([], original[:-1], original + [original[0]]):
            with self.subTest(count=len(records)):
                self.save_records(records)
                with self.assertRaisesRegex(ValueError, "coverage|duplicate"):
                    checker.verify(self.output, sample=1)

    def test_empty_output_is_not_a_verified_study(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            output = Path(directory)
            (output / "run_config.json").write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError, "Missing required artifact"):
                checker.verify(output)

    def test_expected_seed_and_candidate_sets_are_checked_for_every_record(self):
        original = self.records("tuning")
        for field, value in (("candidate_index", 999), ("learning_rate", 12.), ("root_seed", 123)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[0][field] = value
                self.save_records(changed, "tuning")
                with self.assertRaisesRegex(ValueError, "specification"):
                    checker.verify(self.output, sample=1)
        self.save_records(original, "tuning")
        changed = copy.deepcopy(original)
        changed[0]["init"]["key"][2] += 1
        self.save_records(changed, "tuning")
        with self.assertRaisesRegex(ValueError, "init"):
            checker.verify(self.output, sample=1)

    def test_sampled_rerun_checks_final_position_trace_and_result_fields(self):
        original = self.records()[0]
        for field, value in (("final_relative_loss", 123456.), ("gradient_calls", 0),
                             ("threshold_retained", not original["threshold_retained"]),
                             ("failure_step", 999), ("final_position", [100.] * original["dimension"])):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                self.assertTrue(checker.recheck(changed))
        changed = copy.deepcopy(original)
        changed["trace"][-1][2] += 1.
        self.assertIn("trace differs", checker.recheck(changed))

    def test_final_checkpoint_contradiction_is_not_hidden_by_sampling(self):
        records = self.records()
        records[-1]["final_relative_loss"] = 123456.
        self.save_records(records)
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            checker.verify(self.output, sample=1)

    def test_wrong_tables_and_budget_totals_are_rejected(self):
        path = self.output / "tables/evaluation_runs.csv"
        original = path.read_text()
        path.write_text(original.replace("finite_cap", "invented_status", 1))
        with self.assertRaisesRegex(ValueError, "evaluation_runs.csv"):
            checker.verify(self.output, sample=1)
        path.write_text(original)
        budget_path = self.output / "budget_summary.json"
        budget = json.loads(budget_path.read_text())
        budget["runs"]["evaluation"] -= 1
        budget_path.write_text(json.dumps(budget))
        with self.assertRaisesRegex(ValueError, "denominators"):
            checker.verify(self.output, sample=1)

    def test_candidate_score_summaries_are_recomputed_from_raw(self):
        path = self.output / "selected_rates.json"
        selections = json.loads(path.read_text())
        selections["selections"][0]["candidate_scores"][0] = 999.
        path.write_text(json.dumps(selections))
        with self.assertRaisesRegex(ValueError, "score/selection summary"):
            checker.verify(self.output, sample=1)

    def test_statistical_pairing_metadata_is_checked_in_both_exported_tables(self):
        # In particular, collapsing cluster labels must not silently change the
        # resampling unit used by Module 4, even when numeric values are intact.
        tables = {
            "evaluation_runs.csv": ("cluster_id", "pair_id", "reference_run_id", "orientation_id", "init_id", "init_seed_key"),
            "pairs.csv": ("cluster_id", "pair_id", "reference_run_id", "rotated_run_id", "orientation_id"),
        }
        for name, columns in tables.items():
            path = self.output / "tables" / name
            original = path.read_bytes()
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for column in columns:
                with self.subTest(table=name, column=column):
                    changed = copy.deepcopy(rows)
                    for row in changed:
                        row[column] = "invented_single_cluster_or_pair"
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(changed)
                    try:
                        with self.assertRaisesRegex(ValueError, name):
                            checker.verify(self.output, sample=1000)
                    finally:
                        path.write_bytes(original)

    def test_numerical_failures_are_reported_separately_without_becoming_passed(self):
        records = self.records()
        records[0]["final_position"][0] += 10.
        self.save_records(records)
        report = checker.verify(self.output, sample=1000)
        self.assertTrue(report["structural_checks_passed"])
        self.assertFalse(report["numerical_checks_passed"])
        self.assertFalse(report["passed"])
        self.assertTrue(any("final_position" in message for message in report["errors"]))

    def test_invalid_sample_replaces_an_old_passed_marker(self):
        marker = self.output / "verification.json"
        for sample in ("0", "-1", "1.2", "nan"):
            with self.subTest(sample=sample):
                marker.write_text('{"passed": true}')
                with contextlib.redirect_stdout(io.StringIO()):
                    code = checker.main(["--output", str(self.output), "--sample", sample])
                self.assertEqual(code, 1)
                self.assertFalse(json.loads(marker.read_text())["passed"])

    def test_read_only_verification_preserves_historical_identity_and_all_files(self):
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(["--output", str(self.output), "--sample", "1000", "--read-only"]), 0)
        after = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_legacy_source_identity_can_differ_from_current_checkout(self):
        # Recreate the old v1 artifact schema, rather than using current source
        # hashes as a prerequisite for checking an archived completed study.
        (self.output / "execution_origin.json").unlink()
        protocol_path = self.output / "frozen_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        for field in ("provenance_version", "configuration", "execution_origin_sha256"):
            protocol.pop(field, None)
        fields = ("root_seed", "tolerance", "explosive_threshold", "tuning", "pilot", "stability_cap_fraction", "algorithms")
        protocol["protocol_sha256"] = checker.digest({k: self.config[k] for k in fields})
        protocol_path.write_text(json.dumps(protocol))
        path = self.output / "selected_rates.json"
        selected = json.loads(path.read_text())
        selected["protocol_sha256"] = protocol["protocol_sha256"]
        path.write_text(json.dumps(selected))
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["protocol_sha256"] = protocol["protocol_sha256"]
        manifest["source_sha256"] = {"historical_implementation.py": "1" * 64}
        path.write_text(json.dumps(manifest))
        before = path.read_bytes()
        self.assertTrue(checker.verify(self.output, sample=1000)["passed"])
        self.assertEqual(path.read_bytes(), before)

    def test_unchanged_study_subset_keeps_valid_old_cache_without_reporting_it(self):
        config = copy.deepcopy(self.config)
        config["studies"] = config["studies"][:1]
        path = Path(self.workspace_copy.name) / "subset.json"
        path.write_text(json.dumps(config))
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--config", str(path), "--output", str(self.output), "--stage", "report"])
        report = checker.verify(self.output, sample=1000)
        self.assertEqual(report["complete_stage_records"]["evaluation"], 10)
        records = self.records()
        inactive = next(r for r in records if r["study"] == "spatial")
        inactive["init"]["scale"] = 123.
        self.save_records(records)
        with self.assertRaisesRegex(ValueError, "specification differs"):
            checker.verify(self.output, sample=1)

    def test_tuning_failure_placeholders_keep_the_evaluation_denominator(self):
        config = small_config()
        config["algorithms"] = {"adam": {"range": [1e6, 1e7], "params": {}}}
        config["explosive_threshold"] = 2.
        config["theory_references"] = False
        path = Path(self.workspace_copy.name) / "failed.json"
        output = Path(self.workspace_copy.name) / "failed"
        path.write_text(json.dumps(config))
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--config", str(path), "--output", str(output)])
        report = checker.verify(output, sample=1000)
        self.assertEqual(report["not_run_placeholders"], 8)
        self.assertEqual(report["complete_stage_records"]["evaluation"], 8)
        self.assertEqual(report["numerical_runs_rechecked"]["evaluation"], 0)
        placeholders = json.loads((output / "raw/evaluation_not_run.json").read_text())
        (output / "raw/evaluation_not_run.json").write_text(json.dumps(placeholders[:-1]))
        with self.assertRaisesRegex(ValueError, "coverage"):
            checker.verify(output, sample=1)


if __name__ == "__main__":
    unittest.main()
