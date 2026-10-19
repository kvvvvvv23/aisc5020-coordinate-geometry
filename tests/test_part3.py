"""Module 3 checks: generators, seed streams, selection rule, pilot rules and the pipeline."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry import RunConfig, matched_pair, run
from coordinate_geometry.part3_design import (Orientation, build_instance, canonical_initial, evaluation_orientations,
                                              evaluation_units, generator, retune_units, search_units, seed_key,
                                              spectrum, theory_parameters)
from coordinate_geometry.part3_experiments import (execute_spec, load_config, main, make_spec, paired_rows,
                                                   read_jsonl, run_batch, select_cells)
from coordinate_geometry.part3_tuning import (candidate_grid, candidate_score, check_range, revise_range,
                                              select_rate, stability_cap)

ROOT = Path(__file__).resolve().parents[1]
PLANAR = {"name": "planar", "dimension": 2, "condition_numbers": [1, 100],
          "evaluation": {"angles_degrees": [0, 15, 45, 90], "inits": 3}}
ROTATED = {"name": "p6", "dimension": 6, "condition_numbers": [100],
           "evaluation": {"rotations": 3, "inits_per_rotation": 2}}


def tiny_config() -> dict:
    config = json.loads((ROOT / "configs" / "part3_smoke.json").read_text(encoding="utf-8"))
    config["tuning"].update(max_steps=60, candidates=3, inits_per_orientation=2, transfer_orientations=2)
    config["evaluation"].update(max_steps=120, checkpoints=[40, 120], record_every=40)
    config["pilot"].update(orientations=1, inits_per_orientation=1, max_rounds=2)
    config["algorithms"] = {"gd": {"range": [0.05, 1.998], "params": {}},
                            "adam": {"range": [0.003, 0.3], "params": {}}}
    config["studies"] = [{"name": "planar", "dimension": 2, "condition_numbers": [100],
                          "evaluation": {"angles_degrees": [0, 45, 90], "inits": 2}},
                         {"name": "p3", "dimension": 3, "condition_numbers": [10],
                          "evaluation": {"rotations": 2, "inits_per_rotation": 1}}]
    return config


class GeneratorTests(unittest.TestCase):
    def test_spectrum_endpoints_and_spacing(self):
        assert_allclose(spectrum(2, 1000), [1e-3, 1.0])
        values = spectrum(20, 1e4)
        self.assertEqual(values[0], 1e-4)
        self.assertEqual(values[-1], 1.0)
        assert_allclose(np.diff(np.log(values)), np.log(1e4) / 19, rtol=1e-12)
        assert_allclose(spectrum(5, 1), np.ones(5))
        with self.assertRaises(ValueError):
            spectrum(3, 10.5)  # seed keys need integer condition numbers

    def test_loss_normalization_matches_shared_matched_pair(self):
        d = spectrum(7, 100)
        q = Orientation("haar00", "haar", key=(9, 9)).matrix(7, 11)
        key = seed_key("evaluation", 2, "x", 100, index=3)
        _, rotated, z_shared, w_shared = matched_pair(d, q, generator(11, key))
        z0 = canonical_initial(d, generator(11, key).normal(size=7))
        assert_allclose(z0, z_shared, rtol=0, atol=0)
        assert_allclose(rotated.loss(q @ z0), 1.0, atol=1e-13)
        assert_allclose(np.linalg.norm(q @ z0), np.linalg.norm(z0), rtol=1e-13)
        unit = canonical_initial(d, generator(11, key).normal(size=7), "unit_norm", 3.0)
        assert_allclose(np.linalg.norm(unit), 3.0, rtol=1e-14)

    def test_build_instance_is_deterministic_and_paired(self):
        unit = evaluation_units(ROTATED, 100)[1]
        spec = {"dimension": 6, "kappa": 100, "orientation": unit["orientation"], "init": unit["init"]}
        p1, w1, z1 = build_instance(spec, 5)
        p2, w2, _ = build_instance(copy.deepcopy(spec), 5)
        assert_allclose(w1, w2, rtol=0, atol=0)
        assert_allclose(p1.hessian, p2.hessian, rtol=0, atol=0)
        q = p1.rotation
        assert_allclose(q.T @ q, np.eye(6), atol=1e-13)
        assert_allclose(np.linalg.eigvalsh(p1.hessian), spectrum(6, 100), rtol=1e-11)
        assert_allclose(p1.gradient(w1), q @ (spectrum(6, 100) * z1), atol=1e-13)
        assert_allclose(p1.loss(w1), 1.0, rtol=1e-13)
        p3, w3, _ = build_instance(spec, 6)  # another root seed is another study
        self.assertFalse(np.allclose(p3.rotation, q))

    def test_planar_units_share_directions_and_reference_angle_zero(self):
        units = evaluation_units(PLANAR, 100)
        self.assertEqual(len(units), 4 * 3)
        keys = {}
        for u in units:
            keys.setdefault(u["init"]["init_id"], set()).add(tuple(u["init"]["key"]))
            self.assertTrue(u["reference_key"].endswith("|identity|" + u["init"]["init_id"]))
        self.assertTrue(all(len(v) == 1 for v in keys.values()))  # shared across angles
        roles = [u["role"] for u in units if u["orientation"]["orientation_id"] == "identity"]
        self.assertEqual(roles, ["reference"] * 3)
        clusters = {u["cluster_id"] for u in units}
        self.assertEqual(len(clusters), 3)  # resample directions jointly across angles

    def test_rotation_units_pair_each_rotation_with_its_own_reference(self):
        units = evaluation_units(ROTATED, 100)
        self.assertEqual(len(units), 3 * 2 * 2)
        pairs = {}
        for u in units:
            pairs.setdefault(u["pair_id"], []).append(u)
        self.assertEqual(len(pairs), 6)
        for members in pairs.values():
            self.assertEqual(sorted(m["role"] for m in members), ["reference", "rotated"])
            self.assertEqual(members[0]["init"], members[1]["init"])
        self.assertEqual(len({u["cluster_id"] for u in units}), 3)  # cluster = rotation

    def test_stage_streams_are_disjoint(self):
        keys = set()
        for study in (PLANAR, ROTATED):
            for kappa in study["condition_numbers"]:
                for stage, regime in (("pilot", "none"), ("tuning", "transfer")):
                    for u in search_units(study, kappa, 3, stage=stage, regime=regime, round_index=0,
                                          orientations=3, inits=2):
                        keys.add(("init",) + tuple(u["init"]["key"]))
                        if u["orientation"]["key"]:
                            keys.add(("rot",) + tuple(u["orientation"]["key"]))
                for o, code in evaluation_orientations(study, kappa):
                    for u in retune_units(study, kappa, o, code, 2):
                        keys.add(("init",) + tuple(u["init"]["key"]))
                for u in evaluation_units(study, kappa):
                    keys.add(("init",) + tuple(u["init"]["key"]))
                    if u["orientation"]["key"]:
                        keys.add(("rot",) + tuple(u["orientation"]["key"]))
        stages = {k[1] for k in keys}
        self.assertEqual(stages, {1, 2, 3})
        # Pilot round r uses fresh seeds.
        first = search_units(ROTATED, 100, 3, stage="pilot", regime="none", round_index=0, orientations=2, inits=1)
        second = search_units(ROTATED, 100, 3, stage="pilot", regime="none", round_index=1, orientations=2, inits=1)
        self.assertNotEqual(first[-1]["init"]["key"], second[-1]["init"]["key"])
        self.assertNotEqual(first[-1]["orientation"]["key"], second[-1]["orientation"]["key"])

    def test_searches_include_diagonal_basis(self):
        units = search_units(PLANAR, 100, 3, stage="tuning", regime="transfer", round_index=0,
                             orientations=4, inits=5)
        ids = sorted({u["orientation"]["orientation_id"] for u in units})
        self.assertEqual(len(ids), 4)
        self.assertIn("identity", ids)
        self.assertEqual(len(units), 20)
        for u in units:
            if u["orientation"]["kind"] == "angle":
                self.assertTrue(0 <= u["orientation"]["degrees"] < 90)


class TuningRuleTests(unittest.TestCase):
    def test_grid_and_stability_ranges(self):
        grid = candidate_grid(0.002, 1.998, 16)
        self.assertEqual(len(grid), 16)
        self.assertEqual((grid[0], grid[-1]), (0.002, 1.998))
        assert_allclose(np.diff(np.log(grid)), np.log(1.998 / 0.002) / 15, rtol=1e-12)
        self.assertEqual(stability_cap("gd", {}), 2.0)
        self.assertAlmostEqual(stability_cap("heavy_ball", {"beta": 0.9}), 3.8)
        self.assertIsNone(stability_cap("adam", {}))
        check_range("gd", {}, 0.002, 1.998, 0.999)
        with self.assertRaises(ValueError):
            check_range("gd", {}, 0.002, 2.0, 0.999)
        with self.assertRaises(ValueError):
            check_range("heavy_ball", {"beta": 0.9}, 0.01, 3.8, 0.999)

    def test_score_floor_median_and_failures(self):
        self.assertEqual(candidate_score([1e-3, 1e-5, 1e-20], ["finite_cap"] * 3), -5.0)
        self.assertEqual(candidate_score([1e-30, 1e-40], ["finite_cap"] * 2), -16.0)
        self.assertEqual(candidate_score([1e-8, 1e14], ["finite_cap", "explosive"]), math.inf)
        self.assertEqual(candidate_score([1e-8, None], ["finite_cap", "nonfinite"]), math.inf)
        # Missing the threshold within the budget is not a failure.
        self.assertEqual(candidate_score([0.5, 0.5], ["finite_cap"] * 2), math.log10(0.5))

    def test_selection_ties_failures_and_boundaries(self):
        rates = [0.1, 0.2, 0.4, 0.8]
        chosen = select_rate(rates, [-3.0, -16.0, -16.0, math.inf])
        self.assertEqual((chosen["selected_rate"], chosen["boundary"], chosen["candidates_failed"]), (0.2, None, 1))
        self.assertEqual(select_rate(rates, [-9.0, -2, -1, 0])["boundary"], "lower")
        self.assertEqual(select_rate(rates, [0, -1, -2, -9.0])["boundary"], "upper")
        plateau = select_rate(rates, [-16.0, -16.0, -10.0, -5.0])  # floor plateau starting at the edge
        self.assertEqual((plateau["selected_rate"], plateau["boundary"], plateau["edge_plateau"]), (0.1, None, "lower"))
        self.assertEqual(plateau["tied_candidates"], 2)
        strict = select_rate(rates, [-16.0, -15.0, -10.0, -5.0])
        self.assertEqual((strict["boundary"], strict["edge_plateau"]), ("lower", None))
        quick = select_rate(rates, [-16.0, -16.0, -16.0, -5.0], attainment=[900, 300, 300, math.inf],
                            tie_break="earliest_attainment")
        self.assertEqual((quick["selected_rate"], quick["boundary"]), (0.2, None))
        edge = select_rate(rates, [-16.0, -16.0, -10.0, -5.0], attainment=[100, 300, math.inf, math.inf],
                           tie_break="earliest_attainment")
        self.assertEqual(edge["boundary"], "lower")  # strictly faster than its tied neighbour
        failure = select_rate(rates, [math.inf] * 4)
        self.assertTrue(failure["tuning_failure"])
        self.assertIsNone(failure["selected_rate"])
        with self.assertRaises(ValueError):
            select_rate([0.2, 0.1], [0, 0])

    def test_pilot_revision_respects_stability_cap(self):
        capped = revise_range("gd", {}, 0.002, 1.998, {"upper"}, 10, 0.999)
        self.assertFalse(capped["changed"])
        self.assertEqual(capped["notes"], ["stability_capped"])
        partly = revise_range("gd", {}, 0.002, 0.5, {"upper", "lower"}, 10, 0.999)
        self.assertEqual((partly["low"], partly["high"]), (0.0002, 1.998))
        adam = revise_range("adam", {}, 1e-4, 1.0, {"upper"}, 10, 0.999)
        self.assertEqual((adam["low"], adam["high"], adam["changed"]), (1e-4, 10.0, True))
        self.assertFalse(revise_range("adam", {}, 1e-4, 1.0, set(), 10, 0.999)["changed"])

    def test_theory_parameters_match_proposal_and_module1(self):
        d = spectrum(20, 1e4)
        self.assertAlmostEqual(theory_parameters("gd", d)["learning_rate"], 2 / (1 + 1e-4))
        hb = theory_parameters("heavy_ball", d)
        self.assertAlmostEqual(hb["learning_rate"], 4 / (1 + 0.01) ** 2)
        self.assertAlmostEqual(hb["beta"], (0.99 / 1.01) ** 2)
        self.assertIsNone(theory_parameters("adam", d))
        rho = (1e4 - 1) / (1e4 + 1)
        self.assertEqual(math.ceil(math.log(1e-6) / (2 * math.log(rho))), 34539)  # proposal eq. (11)
        try:
            from coordinate_geometry.gd_momentum import optimal_gd_parameters, optimal_heavy_ball_parameters
        except ImportError:
            self.skipTest("module 1 spectral helpers are not merged on this branch")
        self.assertAlmostEqual(optimal_gd_parameters(d)[0], theory_parameters("gd", d)["learning_rate"])
        eta, beta, _ = optimal_heavy_ball_parameters(d)
        self.assertAlmostEqual(eta, hb["learning_rate"])
        self.assertAlmostEqual(beta, hb["beta"])


class PipelineTests(unittest.TestCase):
    def test_repository_configs_validate(self):
        main_config = load_config(ROOT / "configs" / "part3.json")
        self.assertEqual(main_config["tuning"]["candidates"], 16)
        self.assertEqual(main_config["evaluation"]["checkpoints"], [1000, 10000, 50000])
        load_config(ROOT / "configs" / "part3_smoke.json")

    def test_invalid_config_is_rejected(self):
        config = tiny_config()
        config["evaluation"]["checkpoints"] = [40, 100]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_record_matches_direct_shared_runner_call(self):
        config = tiny_config()
        unit = evaluation_units(PLANAR, 100)[5]
        spec = make_spec(config, stage="evaluation", regime="transfer", round_index=0, study=PLANAR, kappa=100,
                         unit=unit, algorithm="adam", learning_rate=0.03, params={"epsilon": 1e-8},
                         candidate_index=None, max_steps=120, record_every=40, checkpoints=[40, 120],
                         cell="all", run_id="x")
        record = execute_spec(spec)
        problem, w0, _ = build_instance(spec, config["root_seed"])
        direct = run(problem, w0, RunConfig("adam", 0.03, 120, record_every=40))
        self.assertEqual(record["status"], direct.status)
        assert_allclose(record["final_position"], direct.final_position, rtol=0, atol=0)
        self.assertEqual(record["checkpoints"]["120"], direct.final_loss / direct.initial_loss)
        self.assertEqual(record["gradient_calls"], 120)

    def test_end_to_end_pipeline_resume_and_protocol_guard(self):
        self._end_to_end(tiny_config())

    def test_end_to_end_with_attainment_tie_break(self):
        config = tiny_config()
        config["tuning"]["tie_break"] = "earliest_attainment"
        self._end_to_end(config)

    def _end_to_end(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path, out = tmp / "c.json", tmp / "out"
            path.write_text(json.dumps(config), encoding="utf-8")
            args = ["--config", str(path), "--output", str(out), "--workers", "1"]
            self.assertEqual(main(args + ["--stage", "all"]), 0)
            evaluation = read_jsonl(out / "raw" / "evaluation.jsonl")
            # gd: transfer, retune, theory; adam: transfer, retune.
            per_algorithm_units = 3 * 2 + 2 * 2 * 1
            self.assertEqual(len(evaluation), per_algorithm_units * 5)
            pairs = paired_rows(evaluation, [40, 120], 1e-16)
            self.assertEqual(len(pairs), (2 * 2 + 2) * 5)
            by_id = {r["run_id"]: r for r in evaluation}
            for pair in pairs:
                a, b = by_id[pair["reference_run_id"]], by_id[pair["rotated_run_id"]]
                self.assertEqual(a["init"], b["init"])
                self.assertEqual(a["orientation"]["orientation_id"], "identity")
                if pair["algorithm"] == "gd" and pair["regime"] != "retune":
                    self.assertAlmostEqual(pair["delta_120"], 0.0, places=6)  # P1 correspondence
                    self.assertTrue(pair["correspondence_expected"])
                    self.assertIsNone(pair["separation_step"])
                if pair["algorithm"] == "adam" and pair["angle_degrees"] == 45.0:
                    self.assertFalse(pair["correspondence_expected"])
                if pair["regime"] == "retune" and pair["reference_rate"] != pair["rotated_rate"]:
                    self.assertFalse(pair["correspondence_expected"])
                if pair["angle_degrees"] == 90.0 and pair["regime"] != "retune":
                    self.assertAlmostEqual(pair["delta_120"], 0.0, places=6)  # signed permutation
            # Selections are reproducible from the saved raw tuning runs.
            saved = json.loads((out / "selected_rates.json").read_text(encoding="utf-8"))["selections"]
            protocol = json.loads((out / "frozen_protocol.json").read_text(encoding="utf-8"))
            again = select_cells(read_jsonl(out / "raw" / "tuning.jsonl"),
                                 {a: v["candidates"] for a, v in protocol["algorithms"].items()},
                                 config, ("regime", "algorithm", "study", "kappa", "cell"))
            self.assertEqual([s["selected_rate"] for s in again], [s["selected_rate"] for s in saved])
            for name in ("evaluation_runs.csv", "pairs.csv", "selected_rates.csv", "tuning_scores.csv",
                         "status_summary.csv", "evaluation_traces.csv", "failures.jsonl"):
                self.assertTrue((out / "tables" / name).exists(), name)
            # Independent reconstruction and re-execution of every saved run.
            from coordinate_geometry.verify_part3 import main as verify
            self.assertEqual(verify(["--output", str(out), "--sample", "100000"]), 0)
            # Resume: rerunning executes nothing new and leaves the raw file unchanged.
            before = (out / "raw" / "evaluation.jsonl").read_bytes()
            self.assertEqual(main(args + ["--stage", "evaluate"]), 0)
            self.assertEqual((out / "raw" / "evaluation.jsonl").read_bytes(), before)
            # Changing a protocol field after freezing is refused.
            config["tuning"]["loss_floor"] = 1e-12
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(args + ["--stage", "tune"])

    def test_resume_refuses_a_different_specification(self):
        config = tiny_config()
        unit = evaluation_units(PLANAR, 100)[0]
        kwargs = dict(stage="evaluation", regime="transfer", round_index=0, study=PLANAR, kappa=100, unit=unit,
                      algorithm="gd", params={}, candidate_index=None, max_steps=10, record_every=10,
                      checkpoints=[10], cell="all", run_id="same")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            quiet = lambda message: None
            run_batch([make_spec(config, learning_rate=0.1, **kwargs)], path, 1, "t", quiet)
            with self.assertRaises(RuntimeError):
                run_batch([make_spec(config, learning_rate=0.2, **kwargs)], path, 1, "t", quiet)


if __name__ == "__main__":
    unittest.main()
