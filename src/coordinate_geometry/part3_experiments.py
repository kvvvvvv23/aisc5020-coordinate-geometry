"""Role 3 (@Sans): pilot, two tuning regimes, batch evaluation and failure logs.

Stages (run in order, each resumable):

  pilot     search candidate ranges on pilot seeds, expand a boundary optimum
            at most ``max_rounds - 1`` times, then freeze every candidate rate
  tune      transfer regime (one rate per algorithm/study/kappa from four tuning
            orientations) and retuning regime (one rate per evaluation orientation)
  evaluate  held-out evaluation runs for both regimes plus spectral-information
            theory references for GD and heavy ball
  report    tidy CSV tables, paired table, failure log, budget and manifest

  python -m coordinate_geometry.part3_experiments --config configs/part3.json \
      --output outputs/part3 --stage all --workers 4

Every run goes through the shared ``run`` contract; this module never
re-implements an optimizer. Each record keeps the seed keys needed to rebuild
its problem and initial point. Pilot, tuning and evaluation instances come
from disjoint seed streams (see ``part3_design``).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from .contracts import ALGORITHMS, RunConfig
from .part3_design import (build_instance, evaluation_orientations, evaluation_units,
                           kappa_code, retune_units, search_units, spectrum, theory_parameters,
                           validate_study)
from .part3_tuning import (TIE_BREAKS, candidate_grid, candidate_score, check_range, revise_range, run_failed,
                           select_rate, stability_cap)
from .runner import run

SCHEMA_VERSION = 1
STUDY_STATUSES = ("part3_smoke", "main_protocol")
PARAM_KEYS = ("beta", "beta1", "beta2", "epsilon", "basis_source")
PROTOCOL_FIELDS = ("root_seed", "tolerance", "explosive_threshold", "tuning", "pilot",
                   "stability_cap_fraction", "algorithms")
NOT_RUN = "not_run_tuning_failure"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol_hash(config: dict[str, Any]) -> str:
    """Hash of every setting that changes candidate rates or tuning outcomes."""
    return sha256_text(canonical_json({key: config[key] for key in PROTOCOL_FIELDS}))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("study_status") not in STUDY_STATUSES:
        raise ValueError(f"study_status must be one of {STUDY_STATUSES}")
    for key in (*PROTOCOL_FIELDS, "evaluation", "studies", "theory_references"):
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    if not isinstance(config["root_seed"], int) or config["root_seed"] < 0:
        raise ValueError("root_seed must be a non-negative integer")
    tuning, evaluation, pilot = config["tuning"], config["evaluation"], config["pilot"]
    for section, keys in ((tuning, ("max_steps", "candidates", "inits_per_orientation",
                                    "transfer_orientations", "loss_floor", "tie_atol")),
                          (evaluation, ("max_steps", "checkpoints", "record_every")),
                          (pilot, ("orientations", "inits_per_orientation", "max_rounds", "expansion_factor"))):
        missing = [k for k in keys if k not in section]
        if missing:
            raise ValueError(f"config section is missing {missing}")
    checkpoints = evaluation["checkpoints"]
    if sorted(set(checkpoints)) != checkpoints or checkpoints[-1] != evaluation["max_steps"]:
        raise ValueError("checkpoints must be ascending and end at the evaluation budget")
    if any(b % evaluation["record_every"] for b in checkpoints):
        raise ValueError("every checkpoint must be a multiple of record_every")
    if tuning.get("tie_break", "smaller_rate") not in TIE_BREAKS:
        raise ValueError(f"tuning.tie_break must be one of {TIE_BREAKS}")
    if tuning["transfer_orientations"] < 1 or pilot["orientations"] < 1:
        raise ValueError("searches need at least the diagonal orientation")
    if pilot["max_rounds"] < 1:
        raise ValueError("pilot needs at least one round")
    fraction = config["stability_cap_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("stability_cap_fraction must lie in (0, 1)")
    enabled = enabled_algorithms(config)
    if not enabled:
        raise ValueError("enable at least one algorithm")
    for name, spec in config["algorithms"].items():
        if name not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {name!r}")
        unknown = set(spec.get("params", {})) - set(PARAM_KEYS)
        if unknown:
            raise ValueError(f"{name}: unknown parameters {sorted(unknown)}")
        low, high = spec["range"]
        check_range(name, spec.get("params", {}), low, high, fraction)
        candidate_grid(low, high, tuning["candidates"])
    names = [s["name"] for s in config["studies"]]
    if len(set(names)) != len(names) or any("|" in n for n in names):
        raise ValueError("study names must be unique and must not contain '|'")
    for study in config["studies"]:
        validate_study(study)
    return config


def enabled_algorithms(config: dict[str, Any]) -> list[str]:
    return [name for name in ALGORITHMS
            if name in config["algorithms"] and config["algorithms"][name].get("enabled", True)]


def algorithm_params(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    return dict(config["algorithms"][algorithm].get("params", {}))


# ---------------------------------------------------------------------------
# Run specifications and execution
# ---------------------------------------------------------------------------

def make_spec(config: dict[str, Any], *, stage: str, regime: str, round_index: int,
              study: dict[str, Any], kappa: float, unit: dict[str, Any], algorithm: str,
              learning_rate: float, params: dict[str, Any], candidate_index: int | None,
              max_steps: int, record_every: int, checkpoints: list[int], cell: str,
              run_id: str, reference_run_id: str | None = None) -> dict[str, Any]:
    spec = {
        "run_id": run_id, "stage": stage, "regime": regime, "round": round_index,
        "study": study["name"], "dimension": int(study["dimension"]), "kappa": float(kappa),
        "control": float(kappa) == 1.0,
        "orientation": unit["orientation"], "init": unit["init"],
        "role": unit.get("role"), "pair_id": unit.get("pair_id"),
        "reference_run_id": reference_run_id, "cluster_id": unit.get("cluster_id"),
        "cell": cell, "algorithm": algorithm, "candidate_index": candidate_index,
        "learning_rate": float(learning_rate), "params": params,
        "max_steps": int(max_steps), "record_every": int(record_every),
        "checkpoints": list(checkpoints), "tolerance": float(config["tolerance"]),
        "explosive_threshold": float(config["explosive_threshold"]),
        "root_seed": int(config["root_seed"]),
    }
    spec["spec_sha256"] = sha256_text(canonical_json(spec))
    return spec


def run_config(spec: dict[str, Any]) -> RunConfig:
    return RunConfig(algorithm=spec["algorithm"], learning_rate=spec["learning_rate"],
                     max_steps=spec["max_steps"], tolerance=spec["tolerance"],
                     explosive_threshold=spec["explosive_threshold"],
                     record_every=spec["record_every"], **spec["params"])


def execute_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the instance from seeds, call the shared runner, summarize the result."""
    generation_started = time.perf_counter()
    problem, w0, z0 = build_instance(spec, spec["root_seed"])
    generation_seconds = time.perf_counter() - generation_started
    result = run(problem, w0, run_config(spec))
    by_step = {entry.step: entry.relative_loss for entry in result.trace}
    failed = run_failed(result.status)
    after_attainment = None
    if result.first_attainment is not None:
        recorded = [entry.relative_loss for entry in result.trace
                    if entry.step >= result.first_attainment and entry.relative_loss is not None]
        after_attainment = max(recorded) if recorded else None
    record = {key: value for key, value in spec.items() if key not in ("checkpoints",)}
    record.update({
        "status": result.status,
        "gradient_calls": result.gradient_calls,
        "first_attainment": result.first_attainment,
        "threshold_retained": result.threshold_retained,
        "initial_loss": result.initial_loss,
        "final_relative_loss": result.final_loss / result.initial_loss,
        "failure_step": result.trace[-1].step if failed else None,
        "checkpoints": {str(b): by_step.get(b) for b in spec["checkpoints"]},
        "max_recorded_relative_after_attainment": after_attainment,
        "initial_norm": float(np.linalg.norm(w0)),
        "canonical_initial_norm": float(np.linalg.norm(z0)),
        "initial_gradient_norm": float(np.linalg.norm(problem.gradient(w0))),
        "generation_seconds": generation_seconds,
        "setup_seconds": result.metadata["setup_seconds"],
        "evaluation_seconds": result.metadata["evaluation_seconds"],
        "basis_source": result.metadata["basis_source"],
    })
    if spec["stage"] == "evaluation" or failed:
        record["trace"] = [[e.step, e.relative_loss, e.gradient_norm_before_step] for e in result.trace]
        record["final_position"] = result.final_position.tolist()
    return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines) - 1:  # interrupted final write; rerun that spec
                continue
            raise
    return records


def run_batch(specs: list[dict[str, Any]], path: Path, workers: int, label: str,
              log: "Logger") -> list[dict[str, Any]]:
    """Execute missing specs, appending one JSON line per run; resume from ``path``."""
    ids = [s["run_id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{label}: duplicate run identifiers in plan")
    done = {}
    for record in read_jsonl(path):
        done[record["run_id"]] = record
    wanted = {s["run_id"]: s for s in specs}
    for run_id, record in done.items():
        if run_id in wanted and record.get("spec_sha256") != wanted[run_id]["spec_sha256"]:
            raise RuntimeError(f"{label}: {path} holds a different specification for {run_id}; "
                               "use a fresh output directory instead of mixing protocols")
    todo = [s for s in specs if s["run_id"] not in done]
    log(f"{label}: {len(specs)} planned, {len(specs) - len(todo)} already saved, {len(todo)} to run "
        f"({sum(s['max_steps'] for s in todo):,} gradient calls), workers={workers}")
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        report_every = max(1, len(todo) // 20)
        with path.open("a", encoding="utf-8") as handle:
            def save(index: int, record: dict[str, Any]) -> None:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                done[record["run_id"]] = record
                if (index + 1) % report_every == 0 or index + 1 == len(todo):
                    handle.flush()
                    elapsed = time.perf_counter() - started
                    eta = elapsed / (index + 1) * (len(todo) - index - 1)
                    log(f"{label}: {index + 1}/{len(todo)} runs, {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
            if workers <= 1:
                for index, spec in enumerate(todo):
                    save(index, execute_spec(spec))
            else:
                context = mp.get_context("spawn")
                chunk = max(1, min(16, len(todo) // (workers * 8)))
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    for index, record in enumerate(pool.map(execute_spec, todo, chunksize=chunk)):
                        save(index, record)
    return [done[run_id] for run_id in ids]


class Logger:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_cells(records: Iterable[dict[str, Any]], candidates: dict[str, list[float]],
                 config: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Group tuning records by cell and candidate, score, and select per cell."""
    grouped: dict[tuple, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[tuple(record[k] for k in keys)][record["candidate_index"]].append(record)
    floor, tie = config["tuning"]["loss_floor"], config["tuning"]["tie_atol"]
    selections = []
    for cell_key, by_candidate in sorted(grouped.items(), key=lambda item: str(item[0])):
        cell = dict(zip(keys, cell_key))
        rates = candidates[cell["algorithm"]]
        if sorted(by_candidate) != list(range(len(rates))):
            raise RuntimeError(f"incomplete candidate set for {cell}")
        sizes = {len(v) for v in by_candidate.values()}
        if len(sizes) != 1:
            raise RuntimeError(f"unequal tuning runs per candidate for {cell}")
        scores, failed_runs, medians_finite, attainment = [], [], [], []
        for index in range(len(rates)):
            runs = by_candidate[index]
            scores.append(candidate_score([r["final_relative_loss"] for r in runs],
                                          [r["status"] for r in runs], floor))
            failed_runs.append(sum(run_failed(r["status"]) for r in runs))
            finite = [math.log10(max(r["final_relative_loss"], floor)) for r in runs
                      if not run_failed(r["status"])]
            medians_finite.append(float(np.median(finite)) if finite else None)
            attainment.append(float(np.median([math.inf if r["first_attainment"] is None else r["first_attainment"]
                                               for r in runs])))
        selection = select_rate(rates, scores, tie, attainment, config["tuning"].get("tie_break", "smaller_rate"))
        selections.append({**cell, **selection, "runs_per_candidate": sizes.pop(),
                           "candidate_rates": rates,
                           "candidate_scores": [s if math.isfinite(s) else None for s in scores],
                           "candidate_failed_runs": failed_runs,
                           "candidate_median_of_nonfailed": medians_finite,
                           "candidate_median_attainment": [a if math.isfinite(a) else None for a in attainment]})
    return selections


# ---------------------------------------------------------------------------
# Stage: pilot
# ---------------------------------------------------------------------------

def search_specs(config: dict[str, Any], *, stage: str, regime: str, round_index: int,
                 algorithm: str, rates: list[float], study: dict[str, Any], kappa: float,
                 units: list[dict[str, Any]], cell: str) -> list[dict[str, Any]]:
    params = algorithm_params(config, algorithm)
    budget = config["tuning"]["max_steps"]
    specs = []
    prefix = f"pilot|r{round_index}" if stage == "pilot" else f"tuning|{regime}"
    for index, rate in enumerate(rates):
        for unit in units:
            run_id = (f"{prefix}|{algorithm}|{study['name']}|k{kappa_code(kappa)}|{cell}|"
                      f"{unit['orientation']['orientation_id']}|{unit['init']['init_id']}|c{index:02d}")
            specs.append(make_spec(config, stage=stage, regime=regime, round_index=round_index, study=study,
                                   kappa=kappa, unit=unit, algorithm=algorithm, learning_rate=rate,
                                   params=params, candidate_index=index, max_steps=budget,
                                   record_every=budget, checkpoints=[budget], cell=cell, run_id=run_id))
    return specs


def pilot_specs(config: dict[str, Any], algorithm: str, rates: list[float], round_index: int) -> list[dict[str, Any]]:
    pilot = config["pilot"]
    specs = []
    for study in config["studies"]:
        for kappa in study["condition_numbers"]:
            units = search_units(study, kappa, config["root_seed"], stage="pilot", regime="none",
                                 round_index=round_index, orientations=pilot["orientations"],
                                 inits=pilot["inits_per_orientation"])
            specs.extend(search_specs(config, stage="pilot", regime="none", round_index=round_index,
                                      algorithm=algorithm, rates=rates, study=study, kappa=kappa,
                                      units=units, cell="all"))
    return specs


def stage_pilot(config: dict[str, Any], out: Path, workers: int, log: Logger) -> dict[str, Any]:
    tuning, pilot = config["tuning"], config["pilot"]
    fraction = config["stability_cap_fraction"]
    state = {alg: {"range": list(config["algorithms"][alg]["range"]), "history": [], "resolved": False,
                   "notes": []} for alg in enabled_algorithms(config)}
    all_selections, timing = [], defaultdict(lambda: [0.0, 0])
    for round_index in range(pilot["max_rounds"]):
        active = [alg for alg, s in state.items() if not s["resolved"]]
        if not active:
            break
        for algorithm in active:
            low, high = state[algorithm]["range"]
            rates = candidate_grid(low, high, tuning["candidates"])
            specs = pilot_specs(config, algorithm, rates, round_index)
            records = run_batch(specs, out / "raw" / f"pilot_round{round_index}_{algorithm}.jsonl", workers,
                                f"pilot round {round_index} {algorithm}", log)
            for r in records:
                key = (algorithm, r["dimension"])
                timing[key][0] += r["evaluation_seconds"]
                timing[key][1] += r["gradient_calls"]
            selections = select_cells(records, {algorithm: rates}, config, ("algorithm", "study", "kappa"))
            for s in selections:
                s["round"] = round_index
            all_selections.extend(selections)
            hits = {s["boundary"] for s in selections if s["boundary"]}
            failures = [f"{s['study']}|k{kappa_code(s['kappa'])}" for s in selections if s["tuning_failure"]]
            revision = revise_range(algorithm, algorithm_params(config, algorithm), low, high, hits,
                                    pilot["expansion_factor"], fraction)
            entry = {"round": round_index, "range": [low, high], "boundary_hits": sorted(hits),
                     "tuning_failures": failures, "revision": revision,
                     "selected": {f"{s['study']}|k{kappa_code(s['kappa'])}": s["selected_rate"] for s in selections}}
            state[algorithm]["history"].append(entry)
            if not revision["changed"]:
                state[algorithm]["resolved"] = True
                state[algorithm]["notes"] = revision["notes"]
            elif round_index + 1 < pilot["max_rounds"]:
                state[algorithm]["range"] = [revision["low"], revision["high"]]
                log(f"pilot {algorithm}: boundary {sorted(hits)} -> range {state[algorithm]['range']}")
            else:
                state[algorithm]["notes"] = revision["notes"] + ["unresolved_boundary"]
                state[algorithm]["resolved"] = True
                log(f"pilot {algorithm}: boundary unresolved after {pilot['max_rounds']} rounds; "
                    "frozen at the last tested range and reported as a range limitation")
    frozen = {}
    for algorithm, s in state.items():
        low, high = s["range"]
        frozen[algorithm] = {
            "range": [low, high], "candidates": candidate_grid(low, high, tuning["candidates"]),
            "params": algorithm_params(config, algorithm),
            "stability_cap": stability_cap(algorithm, algorithm_params(config, algorithm)),
            "pilot_rounds": len(s["history"]), "notes": s["notes"], "history": s["history"],
        }
    cost = estimate_cost(config, timing, frozen)
    protocol = {"study_status": config["study_status"], "frozen": True,
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": protocol_hash(config), "algorithms": frozen,
                "pilot_seconds_per_gradient_call": {f"{a}|p{p}": v[0] / v[1] for (a, p), v in timing.items() if v[1]},
                "cost_estimate": cost}
    write_json(out / "frozen_protocol.json", protocol)
    write_json(out / "pilot_selections.json", all_selections)
    log(f"pilot frozen: {json.dumps({a: v['range'] for a, v in frozen.items()})}")
    log(f"estimated remaining single-worker time: {cost['total_single_worker_hours']:.2f} h")
    return protocol


# ---------------------------------------------------------------------------
# Stage: tune
# ---------------------------------------------------------------------------

def load_protocol(config: dict[str, Any], out: Path) -> dict[str, Any]:
    path = out / "frozen_protocol.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run --stage pilot first so ranges are frozen before tuning")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["protocol_sha256"] != protocol_hash(config):
        raise SystemExit("frozen_protocol.json was made with different tuning/algorithm settings; "
                         "rerun the pilot in a fresh output directory")
    missing = set(enabled_algorithms(config)) - set(protocol["algorithms"])
    if missing:
        raise SystemExit(f"frozen protocol lacks algorithms {sorted(missing)}")
    return protocol


def tuning_plan(config: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tuning = config["tuning"]
    specs = []
    for algorithm in enabled_algorithms(config):
        rates = protocol["algorithms"][algorithm]["candidates"]
        for study in config["studies"]:
            for kappa in study["condition_numbers"]:
                units = search_units(study, kappa, config["root_seed"], stage="tuning", regime="transfer",
                                     round_index=0, orientations=tuning["transfer_orientations"],
                                     inits=tuning["inits_per_orientation"])
                specs.extend(search_specs(config, stage="tuning", regime="transfer", round_index=0,
                                          algorithm=algorithm, rates=rates, study=study, kappa=kappa,
                                          units=units, cell="all"))
                for orientation, code in evaluation_orientations(study, kappa):
                    units = retune_units(study, kappa, orientation, code, tuning["inits_per_orientation"])
                    specs.extend(search_specs(config, stage="tuning", regime="retune", round_index=0,
                                              algorithm=algorithm, rates=rates, study=study, kappa=kappa,
                                              units=units, cell=orientation.orientation_id))
    return specs


def stage_tune(config: dict[str, Any], out: Path, workers: int, log: Logger) -> list[dict[str, Any]]:
    protocol = load_protocol(config, out)
    candidates = {a: v["candidates"] for a, v in protocol["algorithms"].items()}
    records = run_batch(tuning_plan(config, protocol), out / "raw" / "tuning.jsonl", workers, "tuning", log)
    selections = select_cells(records, candidates, config, ("regime", "algorithm", "study", "kappa", "cell"))
    fraction = config["stability_cap_fraction"]
    for s in selections:
        cap = stability_cap(s["algorithm"], algorithm_params(config, s["algorithm"]))
        at_cap = cap is not None and s["candidate_rates"][-1] >= cap * fraction * (1 - 1e-12)
        s["stability_capped"] = s["boundary"] == "upper" and at_cap
        s["range_limitation"] = s["boundary"] is not None and not s["stability_capped"]
    write_json(out / "selected_rates.json", {"protocol_sha256": protocol["protocol_sha256"],
                                             "selections": selections})
    n_fail = sum(s["tuning_failure"] for s in selections)
    n_boundary = sum(s["boundary"] is not None for s in selections)
    log(f"tuning: {len(selections)} selections, {n_fail} tuning failures, {n_boundary} on a grid boundary")
    return selections


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------

def evaluation_plan(config: dict[str, Any], selections: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Return (specs to run, placeholder records for cells whose tuning failed)."""
    evaluation = config["evaluation"]
    lookup = {(s["regime"], s["algorithm"], s["study"], float(s["kappa"]), s["cell"]): s for s in selections}
    specs, placeholders = [], []
    for study in config["studies"]:
        for kappa in study["condition_numbers"]:
            eigenvalues = spectrum(int(study["dimension"]), kappa)
            units = evaluation_units(study, kappa)
            for algorithm in enabled_algorithms(config):
                regimes = ["transfer", "retune"]
                if config["theory_references"] and theory_parameters(algorithm, eigenvalues) is not None:
                    regimes.append("theory")
                for regime in regimes:
                    for unit in units:
                        orientation_id = unit["orientation"]["orientation_id"]
                        run_id = (f"evaluation|{regime}|{algorithm}|{study['name']}|k{kappa_code(kappa)}|"
                                  f"{orientation_id}|{unit['init']['init_id']}")
                        reference = (f"evaluation|{regime}|{algorithm}|{unit['reference_key']}"
                                     if unit["role"] == "rotated" else None)
                        params = algorithm_params(config, algorithm)
                        if regime == "theory":
                            theory = theory_parameters(algorithm, eigenvalues)
                            rate = theory.pop("learning_rate")
                            params.update(theory)
                        else:
                            cell = "all" if regime == "transfer" else orientation_id
                            selection = lookup.get((regime, algorithm, study["name"], float(kappa), cell))
                            if selection is None:
                                raise SystemExit(f"no tuning selection for {(regime, algorithm, study['name'], kappa, cell)}")
                            if selection["tuning_failure"]:
                                placeholders.append({
                                    "run_id": run_id, "stage": "evaluation", "regime": regime,
                                    "study": study["name"], "dimension": int(study["dimension"]),
                                    "kappa": float(kappa), "control": float(kappa) == 1.0,
                                    "orientation": unit["orientation"], "init": unit["init"],
                                    "role": unit["role"], "pair_id": unit["pair_id"],
                                    "reference_run_id": reference, "cluster_id": unit["cluster_id"],
                                    "algorithm": algorithm, "learning_rate": None, "params": params,
                                    "status": NOT_RUN, "gradient_calls": 0, "first_attainment": None,
                                    "threshold_retained": False, "checkpoints": {}})
                                continue
                            rate = selection["selected_rate"]
                        specs.append(make_spec(config, stage="evaluation", regime=regime, round_index=0,
                                               study=study, kappa=kappa, unit=unit, algorithm=algorithm,
                                               learning_rate=rate, params=params, candidate_index=None,
                                               max_steps=evaluation["max_steps"],
                                               record_every=evaluation["record_every"],
                                               checkpoints=evaluation["checkpoints"],
                                               cell="all" if regime != "retune" else orientation_id,
                                               run_id=run_id, reference_run_id=reference))
    return specs, placeholders


def stage_evaluate(config: dict[str, Any], out: Path, workers: int, log: Logger) -> list[dict[str, Any]]:
    load_protocol(config, out)
    path = out / "selected_rates.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run --stage tune first")
    selections = json.loads(path.read_text(encoding="utf-8"))["selections"]
    specs, placeholders = evaluation_plan(config, selections)
    if placeholders:
        log(f"evaluation: {len(placeholders)} runs not executed because their tuning failed")
    write_json(out / "raw" / "evaluation_not_run.json", placeholders)
    return run_batch(specs, out / "raw" / "evaluation.jsonl", workers, "evaluation", log) + placeholders


# ---------------------------------------------------------------------------
# Stage: report (tidy tables)
# ---------------------------------------------------------------------------

def flat(record: dict[str, Any], checkpoints: list[int]) -> dict[str, Any]:
    orientation, init = record["orientation"], record["init"]
    row = {
        "run_id": record["run_id"], "stage": record["stage"], "regime": record["regime"],
        "round": record.get("round", 0), "study": record["study"], "dimension": record["dimension"],
        "kappa": record["kappa"], "control": record.get("control", False),
        "orientation_id": orientation["orientation_id"], "orientation_kind": orientation["kind"],
        "angle_degrees": orientation.get("degrees"),
        "rotation_seed_key": ".".join(map(str, orientation["key"])) if orientation.get("key") else None,
        "init_id": init["init_id"], "init_seed_key": ".".join(map(str, init["key"])),
        "initial_normalization": init["normalization"], "initial_scale": init["scale"],
        "role": record.get("role"), "pair_id": record.get("pair_id"),
        "reference_run_id": record.get("reference_run_id"), "cluster_id": record.get("cluster_id"),
        "cell": record.get("cell"), "algorithm": record["algorithm"],
        "candidate_index": record.get("candidate_index"), "learning_rate": record["learning_rate"],
        "beta": record["params"].get("beta"), "beta1": record["params"].get("beta1"),
        "beta2": record["params"].get("beta2"), "epsilon": record["params"].get("epsilon"),
        "status": record["status"], "gradient_calls": record["gradient_calls"],
        "first_attainment": record["first_attainment"],
        "threshold_retained": record["threshold_retained"],
        "final_relative_loss": record.get("final_relative_loss"),
        "failure_step": record.get("failure_step"),
        "max_recorded_relative_after_attainment": record.get("max_recorded_relative_after_attainment"),
        "initial_norm": record.get("initial_norm"),
        "initial_gradient_norm": record.get("initial_gradient_norm"),
        "generation_seconds": record.get("generation_seconds"),
        "setup_seconds": record.get("setup_seconds"),
        "evaluation_seconds": record.get("evaluation_seconds"),
    }
    for b in checkpoints:
        row[f"r_{b}"] = record.get("checkpoints", {}).get(str(b))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def log_gap(r: float | None, floor: float) -> float | None:
    return None if r is None else math.log10(max(r, floor))


SEPARATION_TOLERANCE = 1e-6  # |log10 r_rotated - log10 r_reference| at a recorded step


def signed_permutation(orientation: dict[str, Any]) -> bool:
    """Exact planar angles that are multiples of 90 degrees permute coordinates with signs."""
    return orientation["kind"] == "identity" or (
        orientation["kind"] == "angle" and float(orientation["degrees"]) % 90 == 0)


def correspondence_expected(rotated: dict[str, Any], reference: dict[str, Any]) -> bool:
    """True when the pair coincides in exact arithmetic (Results P1, P2, P4 and Sec. 2.2).

    GD, heavy ball and common-eigenbasis Adam are equivariant for every rotation;
    Adam and diagonal preconditioning only for signed permutations. Both runs must
    share every optimizer setting, which holds for transfer and theory pairs.
    """
    if rotated["learning_rate"] != reference["learning_rate"] or rotated["params"] != reference["params"]:
        return False
    if rotated["algorithm"] in ("gd", "heavy_ball", "eigenbasis_adam"):
        return True
    return signed_permutation(rotated["orientation"])


def separation_step(rotated: dict[str, Any], reference: dict[str, Any]) -> int | None:
    """First recorded step at which the two relative-loss traces differ in log10 by > tolerance."""
    tiny = float(np.finfo(np.float64).tiny)
    other = {step: rel for step, rel, _ in reference.get("trace", [])}
    for step, rel, _ in rotated.get("trace", []):
        if step not in other:
            continue
        a, b = rel, other[step]
        if a is None or b is None:
            if (a is None) != (b is None):
                return step
            continue
        if abs(math.log10(max(a, tiny)) - math.log10(max(b, tiny))) > SEPARATION_TOLERANCE:
            return step
    return None


def paired_rows(evaluation: list[dict[str, Any]], checkpoints: list[int], floor: float) -> list[dict[str, Any]]:
    by_id = {r["run_id"]: r for r in evaluation}
    rows = []
    for rotated in evaluation:
        if rotated.get("role") != "rotated":
            continue
        reference = by_id.get(rotated["reference_run_id"])
        if reference is None:
            raise RuntimeError(f"missing reference run {rotated['reference_run_id']}")
        d = reference["first_attainment"] is not None
        q = rotated["first_attainment"] is not None
        row = {
            "pair_id": rotated["pair_id"], "regime": rotated["regime"], "algorithm": rotated["algorithm"],
            "study": rotated["study"], "dimension": rotated["dimension"], "kappa": rotated["kappa"],
            "control": rotated.get("control", False),
            "orientation_id": rotated["orientation"]["orientation_id"],
            "angle_degrees": rotated["orientation"].get("degrees"), "cluster_id": rotated["cluster_id"],
            "reference_run_id": reference["run_id"], "rotated_run_id": rotated["run_id"],
            "reference_rate": reference["learning_rate"], "rotated_rate": rotated["learning_rate"],
            "reference_status": reference["status"], "rotated_status": rotated["status"],
            "reference_first_attainment": reference["first_attainment"],
            "rotated_first_attainment": rotated["first_attainment"],
            "outcome": "both" if d and q else "diagonal_only" if d else "rotated_only" if q else "neither",
            "attainment_ratio": (rotated["first_attainment"] / reference["first_attainment"]) if d and q else None,
            "reference_retained": reference["threshold_retained"], "rotated_retained": rotated["threshold_retained"],
            "reference_max_after_attainment": reference.get("max_recorded_relative_after_attainment"),
            "rotated_max_after_attainment": rotated.get("max_recorded_relative_after_attainment"),
            "correspondence_expected": correspondence_expected(rotated, reference),
            "separation_step": separation_step(rotated, reference),
        }
        for b in checkpoints:
            rq = log_gap(rotated.get("checkpoints", {}).get(str(b)), floor)
            ri = log_gap(reference.get("checkpoints", {}).get(str(b)), floor)
            row[f"delta_{b}"] = rq - ri if rq is not None and ri is not None else None
        rows.append(row)
    return rows


def pilot_files(out: Path) -> list[Path]:
    """Raw pilot files that the frozen protocol actually used (stale rounds are ignored)."""
    path = out / "frozen_protocol.json"
    if not path.exists():
        return sorted((out / "raw").glob("pilot_round*.jsonl"))
    protocol = json.loads(path.read_text(encoding="utf-8"))
    return [out / "raw" / f"pilot_round{r}_{a}.jsonl"
            for a, v in protocol["algorithms"].items() for r in range(v["pilot_rounds"])]


def stage_report(config: dict[str, Any], out: Path, log: Logger, plot: bool) -> None:
    checkpoints = config["evaluation"]["checkpoints"]
    floor = config["tuning"]["loss_floor"]
    tables = out / "tables"
    pilot = [r for p in pilot_files(out) for r in read_jsonl(p)]
    tuning = read_jsonl(out / "raw" / "tuning.jsonl")
    evaluation = read_jsonl(out / "raw" / "evaluation.jsonl")
    not_run_path = out / "raw" / "evaluation_not_run.json"
    evaluation += json.loads(not_run_path.read_text(encoding="utf-8")) if not_run_path.exists() else []
    budget = config["tuning"]["max_steps"]
    write_csv(tables / "pilot_runs.csv", [flat(r, [budget]) for r in pilot])
    write_csv(tables / "tuning_runs.csv", [flat(r, [budget]) for r in tuning])
    eval_rows = [flat(r, checkpoints) for r in evaluation]
    write_csv(tables / "evaluation_runs.csv", eval_rows)
    trace_rows = [{"run_id": r["run_id"], "step": s, "relative_loss": rel, "gradient_norm_before_step": g}
                  for r in evaluation for s, rel, g in r.get("trace", [])]
    write_csv(tables / "evaluation_traces.csv", trace_rows,
              ["run_id", "step", "relative_loss", "gradient_norm_before_step"])
    pairs = paired_rows(evaluation, checkpoints, floor)
    write_csv(tables / "pairs.csv", pairs)
    failures = [{"run_id": r["run_id"], "stage": r["stage"], "status": r["status"],
                 "failure_step": r.get("failure_step"), "learning_rate": r["learning_rate"],
                 "final_relative_loss_last_finite": r.get("final_relative_loss"),
                 "last_finite_position": r.get("final_position"), "trace": r.get("trace"),
                 "orientation": r["orientation"], "init": r["init"], "params": r["params"]}
                for r in pilot + tuning + evaluation if r["status"] != "finite_cap"]
    (tables / "failures.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n" for f in failures), encoding="utf-8")
    selection_path = out / "selected_rates.json"
    if selection_path.exists():
        selections = json.loads(selection_path.read_text(encoding="utf-8"))["selections"]
        write_csv(tables / "selected_rates.csv", [{k: v for k, v in s.items() if not k.startswith("candidate_")}
                                                  for s in selections])
        write_csv(tables / "tuning_scores.csv", [
            {"regime": s["regime"], "algorithm": s["algorithm"], "study": s["study"], "kappa": s["kappa"],
             "cell": s["cell"], "candidate_index": i, "learning_rate": rate, "score": score,
             "failed_runs": failed, "median_of_nonfailed": median, "selected": i == s["selected_index"]}
            for s in selections for i, (rate, score, failed, median) in enumerate(zip(
                s["candidate_rates"], s["candidate_scores"], s["candidate_failed_runs"],
                s["candidate_median_of_nonfailed"]))])
    status_counts: dict[tuple, int] = defaultdict(int)
    for r in eval_rows:
        status_counts[(r["regime"], r["algorithm"], r["study"], r["kappa"], r["role"], r["status"],
                       r["first_attainment"] is not None)] += 1
    write_csv(tables / "status_summary.csv", [
        {"regime": k[0], "algorithm": k[1], "study": k[2], "kappa": k[3], "role": k[4], "status": k[5],
         "attained_threshold": k[6], "runs": n} for k, n in sorted(status_counts.items(), key=str)],
        ["regime", "algorithm", "study", "kappa", "role", "status", "attained_threshold", "runs"])
    calls: dict[str, int] = defaultdict(int)
    for r in pilot:
        calls["pilot"] += r["gradient_calls"]
    for r in tuning:
        calls[f"tuning_{r['regime']}"] += r["gradient_calls"]
    for r in evaluation:
        calls[f"evaluation_{r['regime']}"] += r["gradient_calls"]
    write_json(out / "budget_summary.json", {
        "gradient_calls_by_stage": dict(calls),
        "tuning_horizon": budget, "evaluation_horizon": config["evaluation"]["max_steps"],
        "note": ("Rates are selected at the tuning horizon and evaluated at the longer evaluation "
                 "horizon; a rate that is best at the tuning horizon need not be best at the longer one."),
        "runs": {"pilot": len(pilot), "tuning": len(tuning), "evaluation": len(evaluation)},
        "failures": {"total": len(failures),
                     "by_stage_status": {f"{s}|{t}": sum(1 for f in failures if f["stage"] == s and f["status"] == t)
                                         for s in ("pilot", "tuning", "evaluation")
                                         for t in ("explosive", "nonfinite", NOT_RUN)}},
    })
    if plot:
        make_plots(out)
    log(f"report: {len(eval_rows)} evaluation rows, {len(pairs)} pairs, {len(failures)} non-finite_cap records")


def make_plots(out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install optional plots dependencies: python -m pip install -e '.[plots]'") from exc
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sources = [("pilot", out / "pilot_selections.json", lambda s: True)]
    selected = out / "selected_rates.json"
    if selected.exists():
        sources.append(("transfer", selected, lambda s: s["regime"] == "transfer"))
    for name, path, keep in sources:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        selections = [s for s in (value["selections"] if isinstance(value, dict) else value) if keep(s)]
        if name == "pilot":  # the final round of each algorithm
            last = {}
            for s in selections:
                last[s["algorithm"]] = max(last.get(s["algorithm"], 0), s["round"])
            selections = [s for s in selections if s["round"] == last[s["algorithm"]]]
        algorithms = sorted({s["algorithm"] for s in selections}, key=ALGORITHMS.index)
        if not algorithms:
            continue
        fig, axes = plt.subplots(1, len(algorithms), figsize=(4.2 * len(algorithms), 3.6), squeeze=False)
        for axis, algorithm in zip(axes[0], algorithms):
            for s in (s for s in selections if s["algorithm"] == algorithm):
                scores = [np.nan if v is None else v for v in s["candidate_scores"]]
                line, = axis.semilogx(s["candidate_rates"], scores, marker=".",
                                      label=f"{s['study']} κ={s['kappa']:g}")
                if s["selected_index"] is not None:
                    axis.plot(s["selected_rate"], s["selected_score"], "o", mfc="none", color=line.get_color())
            axis.set(title=algorithm, xlabel="learning rate",
                     ylabel="median log10 r_B (failed = gap)")
            axis.grid(True, alpha=0.3)
        axes[0][-1].legend(fontsize=6)
        fig.suptitle(f"{name} scores at the tuning horizon (circles: selected; gaps: a failed run)")
        fig.tight_layout()
        fig.savefig(figures / f"{name}_scores.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Cost estimate and manifest
# ---------------------------------------------------------------------------

def estimate_cost(config: dict[str, Any], timing: dict, frozen: dict[str, Any]) -> dict[str, Any]:
    protocol = {"algorithms": {a: {"candidates": v["candidates"]} for a, v in frozen.items()}}
    tune_calls: dict[tuple, int] = defaultdict(int)
    for spec in tuning_plan(config, protocol):
        tune_calls[(spec["algorithm"], spec["dimension"])] += spec["max_steps"]
    eval_calls: dict[tuple, int] = defaultdict(int)
    dummy = []
    for spec in tuning_plan(config, protocol):
        if spec["candidate_index"] == 0 and spec["init"]["init_id"].endswith("i00"):
            dummy.append({"regime": spec["regime"], "algorithm": spec["algorithm"], "study": spec["study"],
                          "kappa": spec["kappa"], "cell": spec["cell"], "tuning_failure": False,
                          "selected_rate": 1e-3})
    specs, _ = evaluation_plan(config, dummy)
    for spec in specs:
        eval_calls[(spec["algorithm"], spec["dimension"])] += spec["max_steps"]
    seconds = 0.0
    fallback = max((v[0] / v[1] for v in timing.values() if v[1]), default=3e-5)
    for key in set(tune_calls) | set(eval_calls):
        per_call = timing[key][0] / timing[key][1] if timing.get(key, (0, 0))[1] else fallback
        seconds += per_call * (tune_calls.get(key, 0) + eval_calls.get(key, 0))
    return {"tuning_gradient_calls": sum(tune_calls.values()),
            "evaluation_gradient_calls": sum(eval_calls.values()),
            "tuning_runs": len(tuning_plan(config, protocol)), "evaluation_runs": len(specs),
            "total_single_worker_hours": seconds / 3600,
            "basis": "pilot per-call evaluation-loop time by algorithm and dimension; excludes process start-up"}


def git_state(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": head, "dirty_src_or_configs": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_src_or_configs": None}


def write_manifest(out: Path, config_path: Path, config: dict[str, Any], args: argparse.Namespace,
                   stage_seconds: dict[str, float]) -> None:
    package = Path(__file__).resolve().parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}
    try:
        blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas")
    except (TypeError, AttributeError):
        blas = None
    previous = {}
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
    history = previous.get("invocations", [])
    history.append({"utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
                    "workers": args.workers, "stage_wall_seconds": stage_seconds})
    write_json(path, {
        "study_status": config["study_status"],
        "config_path": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_sha256": protocol_hash(config), "git": git_state(package.parents[1]),
        "source_sha256": sources, "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "blas": blas, "thread_env": {k: os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "timing_note": ("Per-run seconds come from the shared runner loop under the stated worker count; "
                        "with several workers they share the CPU and are not cost measurements."),
        "invocations": history,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/part3.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part3"))
    parser.add_argument("--stage", choices=("pilot", "tune", "evaluate", "report", "all"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="also draw score curves (needs '.[plots]')")
    args = parser.parse_args(argv)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(name, "1")  # inherited by spawned workers
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run_config.json", config)
    log = Logger(args.output / "run.log")
    log(f"part3 {args.stage}: config={args.config} status={config['study_status']} workers={args.workers}")
    stages = ("pilot", "tune", "evaluate", "report") if args.stage == "all" else (args.stage,)
    seconds = {}
    for stage in stages:
        started = time.perf_counter()
        if stage == "pilot":
            stage_pilot(config, args.output, args.workers, log)
        elif stage == "tune":
            stage_tune(config, args.output, args.workers, log)
        elif stage == "evaluate":
            stage_evaluate(config, args.output, args.workers, log)
        else:
            stage_report(config, args.output, log, args.plot)
        seconds[stage] = time.perf_counter() - started
    write_manifest(args.output, args.config, config, args, seconds)
    log(f"part3 {args.stage} finished: {json.dumps({k: round(v, 1) for k, v in seconds.items()})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
