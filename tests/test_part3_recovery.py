"""Protocol identity, safe resumption and report/configuration regression checks."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coordinate_geometry import part3_experiments as experiments
from coordinate_geometry.part3_design import evaluation_units

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = json.loads((ROOT / "configs/part3_smoke.json").read_text(encoding="utf-8"))
    config["algorithms"] = {"gd": {"range": [0.1, 1.9], "params": {}}}
    config["tuning"].update(max_steps=2, candidates=2, inits_per_orientation=1, transfer_orientations=1)
    config["pilot"].update(orientations=1, inits_per_orientation=1, max_rounds=1)
    config["evaluation"].update(max_steps=4, checkpoints=[2, 4], record_every=2)
    config["theory_references"] = False
    config["studies"] = [
        {"name": "planar", "dimension": 2, "condition_numbers": [10],
         "evaluation": {"angles_degrees": [0, 45], "inits": 1}},
        {"name": "p3", "dimension": 3, "condition_numbers": [10],
         "evaluation": {"rotations": 1, "inits_per_rotation": 1}},
    ]
    return config


class Part3RecoveryTests(unittest.TestCase):
    def run_stage(self, config, root, stage="all"):
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            return experiments.main(["--config", str(path), "--output", str(root / "out"),
                                     "--stage", stage, "--workers", "1"])

    def published(self, root):
        return {name: (root / "out" / name).read_bytes() for name in
                ("execution_origin.json", "run_config.json", "manifest.json", "budget_summary.json")}

    def specs(self):
        config = small_config()
        study = config["studies"][0]
        unit = evaluation_units(study, 10)[0]
        return [experiments.make_spec(
            config, stage="evaluation", regime="transfer", round_index=0, study=study,
            kappa=10, unit=unit, algorithm="gd", learning_rate=rate, params={},
            candidate_index=None, max_steps=2, record_every=2, checkpoints=[2],
            cell="all", run_id=f"case-{index}") for index, rate in enumerate((0.1, 0.2))]

    def test_changed_source_cannot_relabel_cached_runs_or_published_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            self.run_stage(config, root)
            before = self.published(root)
            raw = (root / "out/raw/evaluation.jsonl").read_bytes()
            sources = experiments.current_sources()
            changed = dict(sources, **{"runner.py": "0" * 64})
            with patch.object(experiments, "current_sources", return_value=changed):
                with self.assertRaisesRegex(SystemExit, "source changed"):
                    self.run_stage(config, root, "evaluate")
            self.assertEqual(self.published(root), before)
            self.assertEqual((root / "out/raw/evaluation.jsonl").read_bytes(), raw)

    def test_legacy_output_is_read_only_without_inventing_an_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            legacy = out / "run_config.json"
            legacy.write_text('{"legacy": true}', encoding="utf-8")
            before = legacy.read_bytes()
            with self.assertRaisesRegex(SystemExit, "legacy|unknown"):
                self.run_stage(small_config(), root, "pilot")
            self.assertEqual(legacy.read_bytes(), before)
            self.assertFalse((out / "execution_origin.json").exists())

    def test_interrupted_tail_is_removed_before_missing_runs_are_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            specs = self.specs()
            experiments.run_batch(specs[:1], path, 1, "first", lambda _: None)
            first_line = path.read_bytes()
            with path.open("ab") as handle:
                handle.write(b'{"run_id":"interrupted')
            experiments.run_batch(specs, path, 1, "resume", lambda _: None)
            self.assertTrue(path.read_bytes().startswith(first_line))
            self.assertEqual([r["run_id"] for r in experiments.read_jsonl(path)], ["case-0", "case-1"])

    def test_valid_unterminated_record_gets_a_separator_without_duplication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            specs = self.specs()
            experiments.run_batch(specs[:1], path, 1, "first", lambda _: None)
            path.write_bytes(path.read_bytes().rstrip(b"\n"))
            experiments.run_batch(specs, path, 1, "resume", lambda _: None)
            self.assertEqual(len(experiments.read_jsonl(path)), 2)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_internal_corruption_is_rejected_without_rewriting_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            specs = self.specs()
            experiments.run_batch(specs, path, 1, "first", lambda _: None)
            lines = path.read_bytes().splitlines(keepends=True)
            corrupted = lines[0] + b"not JSON\n" + lines[1]
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                experiments.run_batch(specs, path, 1, "resume", lambda _: None)
            self.assertEqual(path.read_bytes(), corrupted)

    def test_source_rejection_does_not_even_repair_an_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            specs = self.specs()
            experiments.run_batch(specs[:1], path, 1, "first", lambda _: None)
            corrupted = path.read_bytes() + b'{"interrupted":'
            path.write_bytes(corrupted)
            with patch.object(experiments, "source_fingerprint", return_value="different"):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    experiments.run_batch(specs, path, 1, "resume", lambda _: None)
            self.assertEqual(path.read_bytes(), corrupted)

    def test_unchanged_study_subset_keeps_seeds_and_excludes_old_cached_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            self.run_stage(config, root)
            origin = (root / "out/execution_origin.json").read_bytes()
            raw = (root / "out/raw/evaluation.jsonl").read_bytes()
            subset = copy.deepcopy(config)
            subset["studies"] = subset["studies"][1:]
            a = evaluation_units(config["studies"][1], 10)
            b = evaluation_units(subset["studies"][0], 10)
            self.assertEqual(a, b)
            self.run_stage(subset, root, "report")
            with (root / "out/tables/evaluation_runs.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({r["study"] for r in rows}, {"p3"})
            self.assertEqual((root / "out/execution_origin.json").read_bytes(), origin)
            self.assertEqual((root / "out/raw/evaluation.jsonl").read_bytes(), raw)
            self.assertEqual(json.loads((root / "out/run_config.json").read_text())["studies"], subset["studies"])

    def test_study_mutations_and_report_budget_changes_preserve_saved_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            self.run_stage(config, root)
            before = self.published(root)
            mutations = [
                lambda c: c["studies"][0].update(initial_scale=100),
                lambda c: c["studies"][1].update(dimension=4),
                lambda c: c["studies"][0].update(condition_numbers=[100]),
                lambda c: c["studies"][0]["evaluation"].update(inits=2),
                lambda c: c["evaluation"].update(max_steps=8, checkpoints=[2, 8]),
            ]
            for mutation in mutations:
                changed = copy.deepcopy(config)
                mutation(changed)
                with self.assertRaises(SystemExit):
                    self.run_stage(changed, root, "report")
                self.assertEqual(self.published(root), before)

    def test_report_rejects_a_missing_record_before_republishing_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            self.run_stage(config, root)
            before = self.published(root)
            path = root / "out/raw/evaluation.jsonl"
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(b"".join(lines[1:]))
            with self.assertRaisesRegex(RuntimeError, "incomplete saved plan"):
                self.run_stage(config, root, "report")
            self.assertEqual(self.published(root), before)

    def test_discrete_configuration_rejects_coercion_and_invalid_ranges(self):
        paths = [
            ("root_seed",), ("schema_version",), ("tuning", "max_steps"), ("tuning", "candidates"),
            ("tuning", "inits_per_orientation"), ("tuning", "transfer_orientations"),
            ("evaluation", "max_steps"), ("evaluation", "record_every"),
            ("pilot", "orientations"), ("pilot", "inits_per_orientation"), ("pilot", "max_rounds"),
            ("studies", 1, "dimension"), ("studies", 0, "evaluation", "inits"),
            ("studies", 1, "evaluation", "rotations"), ("studies", 1, "evaluation", "inits_per_rotation"),
            ("evaluation", "checkpoints", 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for setting in paths:
                for value in (True, False, 2.9, 2.0, "2", float("nan"), float("inf"), -1):
                    with self.subTest(setting=setting, value=value):
                        config = small_config()
                        target = config
                        for name in setting[:-1]:
                            target = target[name]
                        target[setting[-1]] = value
                        path.write_text(json.dumps(config), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            experiments.load_config(path)
            valid = small_config()
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(experiments.load_config(path), valid)

    def test_integer_valued_condition_numbers_keep_the_existing_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = small_config()
            config["studies"][0]["condition_numbers"] = [10.0]
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = experiments.load_config(path)
            original = evaluation_units(small_config()["studies"][0], 10)
            self.assertEqual(evaluation_units(loaded["studies"][0], 10.0), original)
            config["studies"][0]["condition_numbers"] = [1.5]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed keys stay exact"):
                experiments.load_config(path)


if __name__ == "__main__":
    unittest.main()
