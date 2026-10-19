"""Independent check of saved module 3 outputs.

Uses its own reconstruction of spectra, rotations and initial points from the
seed keys stored in each record (it does not import ``part3_design``), then

1. re-executes a random sample of pilot, tuning and evaluation runs through the
   shared runner and compares status, attainment and checkpoint losses exactly;
2. re-derives every selected rate from the raw tuning records with a separate
   implementation of the median/failure/tie rule;
3. checks accounting: complete candidate sets, disjoint seed stages, paired runs
   sharing their initial key, failure records retaining a finite last iterate.

  python -m coordinate_geometry.verify_part3 --output outputs/part3 --sample 60
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

from .contracts import RunConfig
from .problems import QuadraticProblem
from .runner import run

STAGE = {"pilot": 1, "tuning": 2, "evaluation": 3}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rebuild(record: dict) -> tuple[QuadraticProblem, np.ndarray]:
    p, kappa = int(record["dimension"]), float(record["kappa"])
    if p == 1:
        d = np.ones(1)
    else:
        d = np.geomspace(1 / kappa, 1.0, p)  # same primitive and formula, so values match bitwise
        d[0], d[-1] = 1 / kappa, 1.0
    o = record["orientation"]
    if o["kind"] == "identity":
        q = np.eye(p)
    elif o["kind"] == "angle":
        t = np.deg2rad(o["degrees"])
        q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    else:
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(record["root_seed"], spawn_key=o["key"])))
        a, r = np.linalg.qr(g.standard_normal((p, p)))
        q = a * np.sign(np.where(np.diag(r) == 0, 1.0, np.diag(r)))
    init = record["init"]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(record["root_seed"], spawn_key=init["key"])))
    u = g.standard_normal(p)
    if init["normalization"] == "loss":
        z = np.sqrt(2 * init["scale"] / float(u @ (d * u))) * u
    else:
        z = init["scale"] * u / float(np.linalg.norm(u))
    return QuadraticProblem(d, q), q @ z


def recheck(record: dict) -> list[str]:
    problem, w0 = rebuild(record)
    params = record["params"]
    config = RunConfig(record["algorithm"], record["learning_rate"], record["max_steps"],
                       tolerance=record["tolerance"], explosive_threshold=record["explosive_threshold"],
                       record_every=record["record_every"], **params)
    result = run(problem, w0, config)
    problems = []
    if result.status != record["status"]:
        problems.append(f"status {result.status} != {record['status']}")
    if result.first_attainment != record["first_attainment"]:
        problems.append(f"attainment {result.first_attainment} != {record['first_attainment']}")
    if abs(result.initial_loss - record["initial_loss"]) > 1e-13 * max(1.0, record["initial_loss"]):
        problems.append("initial loss differs")
    trace = {e.step: e.relative_loss for e in result.trace}
    for step, saved in record["checkpoints"].items():
        mine = trace.get(int(step))
        if (mine is None) != (saved is None) or (mine is not None and mine != saved):
            problems.append(f"checkpoint {step}: {mine} != {saved}")
    return problems


def reselect(records: list[dict], floor: float, tie: float, tie_break: str) -> dict[tuple, float | None]:
    cells: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        cells[(r["regime"], r["algorithm"], r["study"], r["kappa"], r["cell"], r.get("round", 0))][
            r["candidate_index"]].append(r)
    chosen = {}
    for key, by_index in cells.items():
        scored = []
        for index in sorted(by_index):
            runs = by_index[index]
            if any(r["status"] in ("explosive", "nonfinite") for r in runs):
                continue
            score = statistics.median(math.log10(max(r["final_relative_loss"], floor)) for r in runs)
            when = statistics.median(math.inf if r["first_attainment"] is None else r["first_attainment"] for r in runs)
            scored.append((score, when if tie_break == "earliest_attainment" else 0, runs[0]["learning_rate"]))
        if not scored:
            chosen[key] = None
            continue
        best = min(s for s, _, _ in scored)
        chosen[key] = min((w, rate) for s, w, rate in scored if s <= best + tie)[1]
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("outputs/part3"))
    parser.add_argument("--sample", type=int, default=60, help="runs re-executed per stage")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args(argv)
    config = json.loads((args.output / "run_config.json").read_text(encoding="utf-8"))
    raw = args.output / "raw"
    protocol_path = args.output / "frozen_protocol.json"
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))["algorithms"]
        pilot_paths = [raw / f"pilot_round{r}_{a}.jsonl" for a, v in frozen.items() for r in range(v["pilot_rounds"])]
    else:
        pilot_paths = sorted(raw.glob("pilot_round*.jsonl"))
    pilot = [r for p in pilot_paths for r in read_jsonl(p)]
    tuning = read_jsonl(raw / "tuning.jsonl")
    evaluation = read_jsonl(raw / "evaluation.jsonl")
    errors: list[str] = []
    rng = random.Random(args.seed)

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    for name, records in (("pilot", pilot), ("tuning", tuning), ("evaluation", evaluation)):
        for r in records:
            check(r["init"]["key"][0] == STAGE[name], f"{r['run_id']}: init key not in the {name} stream")
            if r["orientation"].get("key"):
                # Retuning selects a rate for an evaluation orientation, so it reuses that
                # rotation (evaluation stream) with fresh tuning initial points.
                expected = STAGE["evaluation"] if r.get("regime") == "retune" else STAGE[name]
                check(r["orientation"]["key"][0] == expected, f"{r['run_id']}: rotation key not in the expected stream")
            if r["status"] in ("explosive", "nonfinite"):
                check(r.get("final_position") is not None and all(map(math.isfinite, r["final_position"])),
                      f"{r['run_id']}: failure without a finite last iterate")
        failed_checks = 0
        sample = rng.sample(records, min(args.sample, len(records)))
        for r in sample:
            issues = recheck(r)
            failed_checks += bool(issues)
            errors.extend(f"{r['run_id']}: {i}" for i in issues)
        print(f"{name}: {len(records)} records, re-executed {len(sample)}, mismatches {failed_checks}")

    floor, tie = config["tuning"]["loss_floor"], config["tuning"]["tie_atol"]
    tie_break = config["tuning"].get("tie_break", "smaller_rate")
    selected_path = args.output / "selected_rates.json"
    if selected_path.exists():
        saved = json.loads(selected_path.read_text(encoding="utf-8"))["selections"]
        mine = reselect(tuning, floor, tie, tie_break)
        for s in saved:
            key = (s["regime"], s["algorithm"], s["study"], s["kappa"], s["cell"], 0)
            check(mine.get(key, "missing") == s["selected_rate"],
                  f"selection {key}: recomputed {mine.get(key, 'missing')} vs saved {s['selected_rate']}")
            check(s["runs_per_candidate"] * len(s["candidate_rates"]) ==
                  sum(1 for r in tuning if (r["regime"], r["algorithm"], r["study"], r["kappa"], r["cell"]) == key[:5]),
                  f"selection {key}: tuning run count mismatch")
        print(f"selections: {len(saved)} recomputed independently")
    pilot_path = args.output / "pilot_selections.json"
    if pilot_path.exists():
        mine = reselect([{**r, "cell": "all"} for r in pilot], floor, tie, tie_break)
        for s in json.loads(pilot_path.read_text(encoding="utf-8")):
            key = ("none", s["algorithm"], s["study"], s["kappa"], "all", s["round"])
            check(mine.get(key, "missing") == s["selected_rate"], f"pilot selection {key} differs")

    init_stage = defaultdict(set)
    for name, records in (("pilot", pilot), ("tuning", tuning), ("evaluation", evaluation)):
        for r in records:
            init_stage[tuple(r["init"]["key"])].add(name)
    check(all(len(v) == 1 for v in init_stage.values()), "an initial key is shared between stages")
    by_id = {r["run_id"]: r for r in evaluation}
    pairs = 0
    for r in evaluation:
        if r.get("role") == "rotated":
            ref = by_id.get(r["reference_run_id"])
            check(ref is not None, f"{r['run_id']}: reference run missing")
            if ref is not None:
                pairs += 1
                check(ref["init"]["key"] == r["init"]["key"], f"{r['run_id']}: reference uses another z0")
                check(ref["orientation"]["kind"] == "identity", f"{r['run_id']}: reference is not diagonal")
                if r["regime"] in ("transfer", "theory"):
                    check(ref["learning_rate"] == r["learning_rate"], f"{r['run_id']}: transfer rates differ")
    print(f"pairs checked: {pairs}")
    report = {"errors": errors, "passed": not errors, "sample_per_stage": args.sample, "seed": args.seed}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        print("\n".join(errors[:40]))
        print(f"FAILED: {len(errors)} problems")
        return 1
    print("verify_part3: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
