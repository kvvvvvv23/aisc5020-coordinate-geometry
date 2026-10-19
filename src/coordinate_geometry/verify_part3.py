"""Complete artifact checks plus independent plan reconstruction and sampled reruns.

No production Part 3 planner or design helper is imported. Historical source
identities need not equal today's checkout. --read-only preserves all artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

from .contracts import ALGORITHMS, RunConfig
from .problems import QuadraticProblem
from .runner import run

STAGE = {"pilot": 1, "tuning": 2, "evaluation": 3}
NOT_RUN = "not_run_tuning_failure"
FAILED = {"explosive", "nonfinite"}
RTOL, ATOL = 1e-8, 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if type(a) is int and type(b) is int:
        return a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    require(path.is_file(), f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def keyed(records, field, label):
    result = {}
    for record in records:
        key = record[field]
        require(key not in result, f"{label}: duplicate {field} {key}")
        result[key] = record
    return result


def csv_rows(path):
    require(path.is_file(), f"Missing required table: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def text_matches(text, value):
    if value is None:
        return text == ""
    if isinstance(value, bool):
        return text == str(value)
    if isinstance(value, (float, int)):
        try:
            return same(float(text), value)
        except ValueError:
            return False
    return text == str(value)


def table_compare(path, expected, keys):
    actual = csv_rows(path)
    index = {}
    for row in actual:
        key = tuple(row[k] for k in keys)
        require(key not in index, f"{path.name}: duplicate row {key}")
        index[key] = row
    require(len(actual) == len(expected), f"{path.name}: wrong row count")
    for row in expected:
        key = tuple("" if row[k] is None else str(row[k]) for k in keys)
        require(key in index, f"{path.name}: missing row {key}")
        require(all(k in index[key] and text_matches(index[key][k], value) for k, value in row.items()), f"{path.name}: fields differ for {key}")


def check_tables(output, active, selections, config):
    tables = output / "tables"
    for stage in STAGE:
        expected = []
        for r in active[stage]:
            fields = {k: r.get(k) for k in ("run_id", "stage", "regime", "study", "dimension", "kappa", "algorithm", "learning_rate", "status", "gradient_calls", "first_attainment", "threshold_retained", "final_relative_loss", "failure_step", "role", "pair_id", "reference_run_id", "cluster_id")}
            o, init = r["orientation"], r["init"]
            fields.update({"orientation_id": o["orientation_id"], "orientation_kind": o["kind"],
                           "angle_degrees": o.get("degrees"),
                           "rotation_seed_key": ".".join(map(str, o["key"])) if o.get("key") else None,
                           "init_id": init["init_id"], "init_seed_key": ".".join(map(str, init["key"])),
                           "initial_normalization": init["normalization"], "initial_scale": init["scale"]})
            fields.update({f"r_{b}": r.get("checkpoints", {}).get(str(b)) for b in (config["evaluation"]["checkpoints"] if stage == "evaluation" else [config["tuning"]["max_steps"]])})
            expected.append(fields)
        table_compare(tables / f"{stage}_runs.csv", expected, ["run_id"])
    evaluation = active["evaluation"]
    refs = {r["run_id"]: r for r in evaluation}
    pairs = []
    floor = config["tuning"]["loss_floor"]
    for r in evaluation:
        if r["role"] != "rotated":
            continue
        ref = refs[r["reference_run_id"]]
        d, q = ref["first_attainment"] is not None, r["first_attainment"] is not None
        row = {"pair_id": r["pair_id"], "regime": r["regime"], "algorithm": r["algorithm"],
               "study": r["study"], "dimension": r["dimension"], "kappa": r["kappa"], "control": r.get("control", False),
               "cluster_id": r["cluster_id"], "orientation_id": r["orientation"]["orientation_id"],
               "angle_degrees": r["orientation"].get("degrees"),
               "reference_run_id": ref["run_id"], "rotated_run_id": r["run_id"], "reference_status": ref["status"], "rotated_status": r["status"],
               "reference_rate": ref["learning_rate"], "rotated_rate": r["learning_rate"],
               "outcome": "both" if d and q else "diagonal_only" if d else "rotated_only" if q else "neither",
               "attainment_ratio": r["first_attainment"] / ref["first_attainment"] if d and q else None,
               "reference_retained": ref["threshold_retained"], "rotated_retained": r["threshold_retained"]}
        signed = r["orientation"]["kind"] == "identity" or (r["orientation"]["kind"] == "angle" and r["orientation"]["degrees"] % 90 == 0)
        row["correspondence_expected"] = (r["learning_rate"] == ref["learning_rate"] and r["params"] == ref["params"] and
                                         (r["algorithm"] in ("gd", "heavy_ball", "eigenbasis_adam") or signed))
        reference_trace = {s: loss for s, loss, _ in ref.get("trace", [])}
        row["separation_step"] = None
        for step, loss, _ in r.get("trace", []):
            if step not in reference_trace:
                continue
            other = reference_trace[step]
            if (loss is None) != (other is None) or (loss is not None and other is not None and
                    abs(math.log10(max(loss, np.finfo(float).tiny)) - math.log10(max(other, np.finfo(float).tiny))) > 1e-6):
                row["separation_step"] = step
                break
        row.update({"reference_first_attainment": ref["first_attainment"], "rotated_first_attainment": r["first_attainment"],
                    "reference_max_after_attainment": ref.get("max_recorded_relative_after_attainment"),
                    "rotated_max_after_attainment": r.get("max_recorded_relative_after_attainment")})
        for b in config["evaluation"]["checkpoints"]:
            x, y = r.get("checkpoints", {}).get(str(b)), ref.get("checkpoints", {}).get(str(b))
            row[f"delta_{b}"] = math.log10(max(x, floor)) - math.log10(max(y, floor)) if x is not None and y is not None else None
        pairs.append(row)
    table_compare(tables / "pairs.csv", pairs, ["regime", "algorithm", "pair_id"])
    traces = [{"run_id": r["run_id"], "step": s, "relative_loss": rel, "gradient_norm_before_step": g} for r in evaluation for s, rel, g in r.get("trace", [])]
    table_compare(tables / "evaluation_traces.csv", traces, ["run_id", "step"])
    counts = Counter((r["regime"], r["algorithm"], r["study"], r["kappa"], r["role"], r["status"], r["first_attainment"] is not None) for r in evaluation)
    status_rows = [dict(zip(("regime", "algorithm", "study", "kappa", "role", "status", "attained_threshold", "runs"), (*key, count))) for key, count in counts.items()]
    table_compare(tables / "status_summary.csv", status_rows, ["regime", "algorithm", "study", "kappa", "role", "status", "attained_threshold"])
    all_records = active["pilot"] + active["tuning"] + evaluation
    failures = {r["run_id"]: r for r in all_records if r["status"] != "finite_cap"}
    require((tables / "failures.jsonl").is_file(), "Missing failure log")
    saved_failures = keyed(read_jsonl(tables / "failures.jsonl"), "run_id", "failure log")
    require(saved_failures.keys() == failures.keys(), "Failure log does not preserve the full failure denominator")
    for identity, f in saved_failures.items():
        r = failures[identity]
        require(all(same(f.get(k), r.get(k)) for k in ("stage", "status", "failure_step", "learning_rate", "trace", "orientation", "init", "params")), f"{identity}: failure log differs")
        require(same(f.get("last_finite_position"), r.get("final_position")) and same(f.get("final_relative_loss_last_finite"), r.get("final_relative_loss")), f"{identity}: failure last iterate differs")
    budget = read_json(output / "budget_summary.json")
    calls = defaultdict(int)
    for r in all_records:
        calls[r["stage"] if r["stage"] == "pilot" else f"{r['stage']}_{r['regime']}"] += r["gradient_calls"]
    require(budget["gradient_calls_by_stage"] == dict(calls), "Budget gradient-call totals differ")
    require(budget["runs"] == {stage: len(active[stage]) for stage in STAGE}, "Budget run denominators differ")
    require(budget["tuning_horizon"] == config["tuning"]["max_steps"] and budget["evaluation_horizon"] == config["evaluation"]["max_steps"], "Budget horizons differ")
    require(budget["failures"]["total"] == len(failures), "Budget failure count differs")
    for stage in STAGE:
        for status in (*sorted(FAILED), NOT_RUN):
            require(budget["failures"]["by_stage_status"][f"{stage}|{status}"] == sum(r["stage"] == stage and r["status"] == status for r in failures.values()), "Failure status summary differs")
    selection_fields = ("regime", "algorithm", "study", "kappa", "cell", "selected_rate", "selected_index", "tuning_failure", "runs_per_candidate")
    table_compare(tables / "selected_rates.csv", [{k: s[k] for k in selection_fields} for s in selections], ["regime", "algorithm", "study", "kappa", "cell"])
    scores = [{"regime": s["regime"], "algorithm": s["algorithm"], "study": s["study"], "kappa": s["kappa"],
               "cell": s["cell"], "candidate_index": i, "learning_rate": rate,
               "score": s["candidate_scores"][i], "failed_runs": s["candidate_failed_runs"][i],
               "median_of_nonfailed": s["candidate_median_of_nonfailed"][i], "selected": i == s["selected_index"]}
              for s in selections for i, rate in enumerate(s["candidate_rates"])]
    table_compare(tables / "tuning_scores.csv", scores, ["regime", "algorithm", "study", "kappa", "cell", "candidate_index"])
    return len(pairs)


def rebuild(record):
    p, kappa = record["dimension"], record["kappa"]
    d = np.ones(1) if p == 1 else np.geomspace(1 / kappa, 1., p)
    if p > 1:
        d[0], d[-1] = 1 / kappa, 1.
    o = record["orientation"]
    if o["kind"] == "identity":
        q = np.eye(p)
    elif o["kind"] == "angle":
        t = np.deg2rad(o["degrees"])
        q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    else:
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(record["root_seed"], spawn_key=o["key"])))
        a, r = np.linalg.qr(g.standard_normal((p, p)))
        q = a * np.sign(np.where(np.diag(r) == 0, 1., np.diag(r)))
    init = record["init"]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(record["root_seed"], spawn_key=init["key"])))
    u = g.standard_normal(p)
    z = np.sqrt(2 * init["scale"] / float(u @ (d * u))) * u if init["normalization"] == "loss" else init["scale"] * u / float(np.linalg.norm(u))
    return QuadraticProblem(d, q), q @ z


def recheck(record):
    problem, w0 = rebuild(record)
    result = run(problem, w0, RunConfig(record["algorithm"], record["learning_rate"], record["max_steps"],
                 tolerance=record["tolerance"], explosive_threshold=record["explosive_threshold"],
                 record_every=record["record_every"], **record["params"]))
    values = {"status": result.status, "first_attainment": result.first_attainment,
              "gradient_calls": result.gradient_calls, "threshold_retained": result.threshold_retained,
              "initial_loss": result.initial_loss, "final_relative_loss": result.final_loss / result.initial_loss,
              "failure_step": result.trace[-1].step if result.status in FAILED else None}
    errors = [f"{key} differs" for key, value in values.items() if not same(record.get(key), value)]
    trace = {e.step: e.relative_loss for e in result.trace}
    for step, saved in record["checkpoints"].items():
        if not same(saved, trace.get(int(step))):
            errors.append(f"checkpoint {step} differs")
    if "final_position" in record and not same(record["final_position"], result.final_position.tolist()):
        errors.append("final_position differs")
    actual_trace = [[e.step, e.relative_loss, e.gradient_norm_before_step] for e in result.trace]
    if "trace" in record and not same(record["trace"], actual_trace):
        errors.append("trace differs")
    after = [e.relative_loss for e in result.trace if result.first_attainment is not None
             and e.step >= result.first_attainment and e.relative_loss is not None]
    if not same(record.get("max_recorded_relative_after_attainment"), max(after) if after else None):
        errors.append("post-attainment maximum differs")
    return errors


def selection_facts(records, rates, config):
    by_candidate = defaultdict(list)
    for r in records:
        by_candidate[r["candidate_index"]].append(r)
    require(set(by_candidate) == set(range(len(rates))), "Incomplete candidate set")
    require(len({len(v) for v in by_candidate.values()}) == 1, "Unequal candidate repetitions")
    floor, tie = config["tuning"]["loss_floor"], config["tuning"]["tie_atol"]
    earliest = config["tuning"].get("tie_break", "smaller_rate") == "earliest_attainment"
    scores, counts, medians, attainment = [], [], [], []
    for i in range(len(rates)):
        runs = by_candidate[i]
        failed = sum(r["status"] in FAILED for r in runs)
        values = [math.log10(max(r["final_relative_loss"], floor)) for r in runs if r["status"] not in FAILED]
        scores.append(math.inf if failed else statistics.median(values))
        counts.append(failed)
        medians.append(statistics.median(values) if values else None)
        attainment.append(statistics.median(math.inf if r["first_attainment"] is None else r["first_attainment"] for r in runs))
    valid = [i for i, score in enumerate(scores) if math.isfinite(score)]
    tied = [i for i in valid if scores[i] <= min(scores[j] for j in valid) + tie] if valid else []
    index = min(tied, key=lambda i: (attainment[i] if earliest else 0., i)) if tied else None
    edge = "lower" if index == 0 else "upper" if index == len(rates) - 1 else None
    neighbour = 1 if edge == "lower" else len(rates) - 2 if edge == "upper" else None
    strict = edge is not None and (neighbour not in tied or (earliest and attainment[neighbour] > attainment[index]))
    return {"tuning_failure": index is None, "selected_index": index,
            "selected_rate": rates[index] if index is not None else None,
            "selected_score": scores[index] if index is not None else None,
            "boundary": edge if strict else None, "edge_plateau": edge if not strict else None,
            "candidates_failed": len(rates) - len(valid), "tied_candidates": len(tied),
            "runs_per_candidate": len(records) // len(rates), "candidate_rates": rates,
            "candidate_scores": [s if math.isfinite(s) else None for s in scores],
            "candidate_failed_runs": counts, "candidate_median_of_nonfailed": medians,
            "candidate_median_attainment": [a if math.isfinite(a) else None for a in attainment]}


def reselect(records, floor, tie, tie_break):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[(r["regime"], r["algorithm"], r["study"], r["kappa"], r["cell"], r.get("round", 0))][r["candidate_index"]].append(r)
    chosen = {}
    for key, candidates in grouped.items():
        scored = []
        for runs in candidates.values():
            if not any(r["status"] in FAILED for r in runs):
                score = statistics.median(math.log10(max(r["final_relative_loss"], floor)) for r in runs)
                when = statistics.median(math.inf if r["first_attainment"] is None else r["first_attainment"] for r in runs)
                scored.append((score, when if tie_break == "earliest_attainment" else 0., runs[0]["learning_rate"]))
        if not scored:
            chosen[key] = None
        else:
            best = min(s for s, _, _ in scored)
            chosen[key] = min((w, rate) for s, w, rate in scored if s <= best + tie)[1]
    return chosen


def validate_record(record, spec, config):
    identity = spec["run_id"]
    placeholder = spec["learning_rate"] is None
    for field, expected in spec.items():
        if placeholder and field in ("round", "cell", "candidate_index", "max_steps", "record_every", "tolerance", "explosive_threshold", "root_seed"):
            continue
        require(same(record.get(field), expected), f"{identity}: specification differs: {field}")
    if placeholder:
        for key, value in {"status": NOT_RUN, "gradient_calls": 0, "first_attainment": None,
                           "threshold_retained": False, "checkpoints": {}}.items():
            require(same(record.get(key), value), f"{identity}: invalid tuning-failure placeholder {key}")
        require(not record.get("trace") and not record.get("final_position"), f"{identity}: placeholder contains an executed trajectory")
        return
    status, calls = record["status"], record["gradient_calls"]
    require(status in FAILED | {"finite_cap"}, f"{identity}: invalid status")
    require(type(calls) is int and 1 <= calls <= spec["max_steps"], f"{identity}: invalid gradient calls")
    require(status != "finite_cap" or calls == spec["max_steps"], f"{identity}: finite run stopped before budget")
    gap, loss = record["final_relative_loss"], record["initial_loss"]
    require(math.isfinite(loss) and loss > 0 and math.isfinite(gap) and gap >= 0, f"{identity}: invalid loss")
    attainment = record["first_attainment"]
    require(attainment is None or (type(attainment) is int and 1 <= attainment <= calls), f"{identity}: invalid attainment")
    require(type(record["threshold_retained"]) is bool and record["threshold_retained"] ==
            (status == "finite_cap" and gap <= spec["tolerance"]), f"{identity}: inconsistent threshold retention")
    require(record["failure_step"] == (calls if status in FAILED else None), f"{identity}: inconsistent failure step")
    checkpoints = config["evaluation"]["checkpoints"] if spec["stage"] == "evaluation" else [spec["max_steps"]]
    require(set(record["checkpoints"]) == set(map(str, checkpoints)), f"{identity}: missing/extra checkpoints")
    if status == "finite_cap":
        require(same(record["checkpoints"][str(spec["max_steps"])], gap), f"{identity}: final checkpoint contradicts final loss")
    if spec["stage"] == "evaluation" or status in FAILED:
        position = np.asarray(record["final_position"], dtype=float)
        require(position.shape == (spec["dimension"],) and np.all(np.isfinite(position)), f"{identity}: missing finite last position")
        trace = record["trace"]
        require(trace and trace[0][0] == 0 and trace[0][1] == 1. and trace[0][2] is None, f"{identity}: invalid initial trace")
        steps = [e[0] for e in trace]
        require(all(type(s) is int for s in steps) and steps == sorted(set(steps)) and steps[-1] == calls, f"{identity}: invalid trace steps")
        traced = {s: rel for s, rel, _ in trace}
        require(all(same(value, traced.get(int(step))) for step, value in record["checkpoints"].items()), f"{identity}: checkpoint contradicts saved trace")
        require(status != "nonfinite" or trace[-1][1] is None, f"{identity}: nonfinite trace marker missing")
        require(status != "explosive" or (trace[-1][1] is not None and trace[-1][1] > spec["explosive_threshold"]), f"{identity}: explosive threshold was not crossed")


def key_for(study, kappa, stage, purpose, regime="none", round_index=0, orientation=0, index=0):
    code = int.from_bytes(hashlib.sha256(study.get("seed_study", study["name"]).encode()).digest()[:4], "big")
    return [STAGE[stage], purpose, code, int(kappa), {"none": 0, "transfer": 1, "retune": 2}[regime], round_index, orientation, index]


def rng_for(config, key):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(config["root_seed"], spawn_key=key)))


def orientation(identity="identity", kind="identity", degrees=None, key=None):
    return {"orientation_id": identity, "kind": kind, "degrees": degrees, "key": key}


def initial(study, identity, key):
    return {"init_id": identity, "key": key, "normalization": study.get("initial_normalization", "loss"), "scale": float(study.get("initial_scale", 1.0))}


def orientations(config, study, kappa, stage=None, regime="none", round_index=0, count=None):
    planar = "angles_degrees" in study["evaluation"]
    if stage is None:
        if planar:
            return [orientation("identity" if a == 0 else f"deg{a:g}", "identity" if a == 0 else "angle", None if a == 0 else float(a)) for a in study["evaluation"]["angles_degrees"]]
        return [orientation()] + [orientation(f"haar{r:02d}", "haar", key=key_for(study, kappa, "evaluation", 1, index=r)) for r in range(study["evaluation"]["rotations"])]
    result = [orientation()]
    for j in range(1, count):
        key = key_for(study, kappa, stage, 1, regime, round_index, index=j)
        prefix = "pilot-rot" if stage == "pilot" else "tune-rot"
        if planar:
            angle = float(rng_for(config, key).uniform(0., 90.))
            result.append(orientation(f"{prefix}{j:02d}-deg{angle:.4f}", "angle", angle, key))
        else:
            result.append(orientation(f"{prefix}{j:02d}", "haar", key=key))
    return result


def search_units(config, study, kappa, stage, regime, round_index, count, inits, retune=None):
    if retune is not None:
        o, code = retune
        scope = 0 if "angles_degrees" in study["evaluation"] else code
        return [{"orientation": o, "init": initial(study, f"retune-s{scope}-i{i:02d}", key_for(study, kappa, "tuning", 2, "retune", orientation=scope, index=i))} for i in range(inits)]
    return [{"orientation": o, "init": initial(study, f"{stage[:4]}-o{j}-i{i:02d}", key_for(study, kappa, stage, 2, regime, round_index, j, i))}
            for j, o in enumerate(orientations(config, study, kappa, stage, regime, round_index, count)) for i in range(inits)]


def evaluation_units(config, study, kappa):
    tag = f"{study['name']}|k{int(kappa)}"
    os = orientations(config, study, kappa)
    result = []
    def add(o, identity, init, role, pair, cluster):
        result.append({"orientation": o, "init": init, "role": role, "pair_id": pair, "reference_key": f"{tag}|identity|{identity}", "cluster_id": cluster})
    if "angles_degrees" in study["evaluation"]:
        for o in os:
            for i in range(study["evaluation"]["inits"]):
                name = f"eval-i{i:02d}"
                init = initial(study, name, key_for(study, kappa, "evaluation", 2, index=i))
                add(o, name, init, "reference" if o["kind"] == "identity" else "rotated", f"{tag}|{o['orientation_id']}|{name}", f"{tag}|{name}")
    else:
        for code, o in enumerate(os[1:], 1):
            for i in range(study["evaluation"]["inits_per_rotation"]):
                name = f"eval-r{code-1:02d}-i{i:02d}"
                init = initial(study, name, key_for(study, kappa, "evaluation", 2, orientation=code, index=i))
                pair = f"{tag}|{o['orientation_id']}|{name}"
                for basis, role in ((os[0], "reference"), (o, "rotated")):
                    add(basis, name, init, role, pair, f"{tag}|{o['orientation_id']}")
    return result


def grid(low, high, count):
    rates = np.geomspace(low, high, count)
    rates[0], rates[-1] = low, high
    return list(map(float, rates))


def algorithms(config):
    return [a for a in ALGORITHMS if a in config["algorithms"] and config["algorithms"][a].get("enabled", True)]


def expected_spec(config, study, kappa, unit, algorithm, rate, stage, regime, round_index, cell, candidate=None):
    evaluation = stage == "evaluation"
    prefix = f"pilot|r{round_index}" if stage == "pilot" else f"tuning|{regime}"
    o, init = unit["orientation"], unit["init"]
    identity = (f"evaluation|{regime}|{algorithm}|{study['name']}|k{int(kappa)}|{o['orientation_id']}|{init['init_id']}" if evaluation else
                f"{prefix}|{algorithm}|{study['name']}|k{int(kappa)}|{cell}|{o['orientation_id']}|{init['init_id']}|c{candidate:02d}")
    return {"run_id": identity, "stage": stage, "regime": regime, "round": round_index,
            "study": study["name"], "dimension": study["dimension"], "kappa": float(kappa), "control": kappa == 1,
            "orientation": o, "init": init, "role": unit.get("role"), "pair_id": unit.get("pair_id"),
            "reference_run_id": f"evaluation|{regime}|{algorithm}|{unit['reference_key']}" if unit.get("role") == "rotated" else None,
            "cluster_id": unit.get("cluster_id"), "cell": cell, "algorithm": algorithm, "candidate_index": candidate,
            "learning_rate": rate, "params": dict(config["algorithms"][algorithm].get("params", {})),
            "max_steps": config["evaluation" if evaluation else "tuning"]["max_steps"],
            "record_every": config["evaluation"]["record_every"] if evaluation else config["tuning"]["max_steps"],
            "tolerance": config["tolerance"], "explosive_threshold": config["explosive_threshold"], "root_seed": config["root_seed"]}


def plans(config, frozen, selections, include_evaluation=True):
    result = {name: {} for name in STAGE}
    selected = {(s["regime"], s["algorithm"], s["study"], s["kappa"], s["cell"]): s for s in selections}
    for a in algorithms(config):
        f = frozen[a]
        require(len(f["history"]) == f["pilot_rounds"] and 1 <= f["pilot_rounds"] <= config["pilot"]["max_rounds"], "Invalid pilot round coverage")
        require(same(f["candidates"], grid(*f["range"], config["tuning"]["candidates"])), f"{a}: frozen grid differs")
        for study in config["studies"]:
            for kappa in study["condition_numbers"]:
                for stage in ("pilot", "tuning"):
                    for r in (range(f["pilot_rounds"]) if stage == "pilot" else [0]):
                        rates = grid(*f["history"][r]["range"], config["tuning"]["candidates"]) if stage == "pilot" else f["candidates"]
                        search = config["pilot" if stage == "pilot" else "tuning"]
                        regime = "none" if stage == "pilot" else "transfer"
                        count = search["orientations" if stage == "pilot" else "transfer_orientations"]
                        units = search_units(config, study, kappa, stage, regime, r, count, search["inits_per_orientation"])
                        cells = [(regime, "all", units)]
                        if stage == "tuning":
                            cells += [("retune", o["orientation_id"], search_units(config, study, kappa, stage, "retune", 0, 0, search["inits_per_orientation"], (o, code)))
                                      for code, o in enumerate(orientations(config, study, kappa))]
                        for regime, cell, units in cells:
                            for i, rate in enumerate(rates):
                                for unit in units:
                                    spec = expected_spec(config, study, kappa, unit, a, rate, stage, regime, r, cell, i)
                                    result[stage][spec["run_id"]] = spec
                for regime in (["transfer", "retune"] + (["theory"] if config["theory_references"] and a in ("gd", "heavy_ball") else [])) if include_evaluation else []:
                    for unit in evaluation_units(config, study, kappa):
                        cell = unit["orientation"]["orientation_id"] if regime == "retune" else "all"
                        if regime == "theory":
                            mu = 1. if study["dimension"] == 1 else 1. / kappa
                            rate = 2 / (1 + mu) if a == "gd" else 4 / (1 + math.sqrt(mu))**2
                        else:
                            key = (regime, a, study["name"], float(kappa), cell)
                            require(key in selected, f"Missing selection cell {key}")
                            rate = selected[key]["selected_rate"]
                        spec = expected_spec(config, study, kappa, unit, a, rate, "evaluation", regime, 0, cell)
                        if regime == "theory" and a == "heavy_ball":
                            spec["params"]["beta"] = ((1 - math.sqrt(mu)) / (1 + math.sqrt(mu)))**2
                        result["evaluation"][spec["run_id"]] = spec
    return result


def verify(output, sample=60, seed=20261018):
    require(type(sample) is int and sample > 0, "sample must be a positive integer")
    config = read_json(output / "run_config.json")
    protocol = read_json(output / "frozen_protocol.json")
    selected_document = read_json(output / "selected_rates.json")
    selections = selected_document["selections"]
    require(selected_document["protocol_sha256"] == protocol["protocol_sha256"], "Selected rates belong to another protocol")
    frozen = protocol["algorithms"]
    require(set(frozen) == set(algorithms(config)), "Frozen algorithm coverage differs")
    protocol_fields = ("root_seed", "tolerance", "explosive_threshold", "tuning", "pilot", "stability_cap_fraction", "algorithms")
    protocol_config = protocol.get("configuration", config)
    hash_fields = (*protocol_fields, "studies", "evaluation", "theory_references") if protocol.get("provenance_version") == 2 else protocol_fields
    require(protocol["protocol_sha256"] == digest({k: protocol_config[k] for k in hash_fields}), "Frozen protocol identity differs")
    origin_path = output / "execution_origin.json"
    require(protocol.get("provenance_version") != 2 or origin_path.exists(), "Missing version-2 execution origin")
    origin = read_json(origin_path) if origin_path.exists() else None
    if origin:
        require(origin["origin_sha256"] == digest({k: v for k, v in origin.items() if k != "origin_sha256"}), "Execution origin digest differs")
        require(origin["source_fingerprint"] == digest(origin["source_sha256"]), "Execution source fingerprint differs")
        origin_config = origin["config"]
        require(origin["config_sha256"] == digest(origin_config), "Execution origin configuration digest differs")
        allowed = {s["name"]: s for s in origin_config["studies"]}
        require(all(s["name"] in allowed and s == allowed[s["name"]] for s in config["studies"]), "Active studies differ from execution origin")
        ignored = {"studies"}
        require({k: v for k, v in config.items() if k not in ignored} == {k: v for k, v in origin_config.items() if k not in ignored}, "Active execution settings differ from origin")
    else:
        origin_config = config
    manifest = read_json(output / "manifest.json")
    require(manifest["protocol_sha256"] == protocol["protocol_sha256"], "Manifest protocol identity differs")
    require(isinstance(manifest["source_sha256"], dict) and manifest["source_sha256"], "Missing historical source identity")
    if origin:
        require(manifest["execution_origin_sha256"] == origin["origin_sha256"] and protocol["execution_origin_sha256"] == origin["origin_sha256"], "Execution origin references differ")
        require(manifest["source_sha256"] == origin["source_sha256"], "Manifest overwrote historical source identity")
        require(manifest["config_sha256"] == digest(config), "Manifest active configuration differs")
    expected = plans(config, frozen, selections)
    active_names = {s["name"] for s in config["studies"]}
    raw = output / "raw"
    stage_records = {"pilot": [r for a, f in frozen.items() for i in range(f["pilot_rounds"]) for r in read_jsonl(raw / f"pilot_round{i}_{a}.jsonl")],
                     "tuning": read_jsonl(raw / "tuning.jsonl"), "evaluation": read_jsonl(raw / "evaluation.jsonl")}
    placeholders = read_json(raw / "evaluation_not_run.json")
    stage_records["evaluation"] += placeholders
    origin_search = expected if {s["name"] for s in origin_config["studies"]} == active_names else plans(origin_config, frozen, selections, include_evaluation=False)
    all_chosen = reselect(stage_records["tuning"], config["tuning"]["loss_floor"], config["tuning"]["tie_atol"], config["tuning"].get("tie_break", "smaller_rate"))
    origin_studies = {s["name"]: s for s in origin_config["studies"]}
    active = {}
    for stage, records in stage_records.items():
        keyed(records, "run_id", stage)
        active[stage] = [r for r in records if r["study"] in active_names]
        actual = {r["run_id"] for r in active[stage]}
        require(actual == set(expected[stage]), f"{stage}: planned run coverage differs (missing {len(set(expected[stage])-actual)}, extra {len(actual-set(expected[stage]))})")
        for r in active[stage]:
            validate_record(r, expected[stage][r["run_id"]], config)
        for r in records:
            if origin and r["status"] != NOT_RUN:
                require(r.get("execution_source_sha256") == origin["source_fingerprint"], f"{r['run_id']}: execution source identity differs")
            require(r["study"] in {s["name"] for s in origin_config["studies"]}, "Unknown inactive cached study")
            if r["study"] not in active_names:
                if stage != "evaluation":
                    require(r["run_id"] in origin_search[stage], f"{r['run_id']}: unknown inactive cached run")
                    validate_record(r, origin_search[stage][r["run_id"]], origin_config)
                else:
                    study = origin_studies[r["study"]]
                    require(r["kappa"] in study["condition_numbers"] and r["algorithm"] in algorithms(origin_config), "Invalid inactive evaluation cell")
                    units = [u for u in evaluation_units(origin_config, study, r["kappa"]) if u["orientation"]["orientation_id"] == r["orientation"]["orientation_id"] and u["init"]["init_id"] == r["init"]["init_id"]]
                    require(len(units) == 1, "Invalid inactive evaluation unit")
                    cell = r["orientation"]["orientation_id"] if r["regime"] == "retune" else "all"
                    key = (r["regime"], r["algorithm"], r["study"], r["kappa"], cell, 0)
                    require(r["regime"] in ("transfer", "retune", "theory"), "Unknown inactive regime")
                    if r["regime"] == "theory":
                        require(origin_config["theory_references"] and r["algorithm"] in ("gd", "heavy_ball"), "Unexpected inactive theory run")
                        mu = 1. if study["dimension"] == 1 else 1. / r["kappa"]
                        rate = 2 / (1 + mu) if r["algorithm"] == "gd" else 4 / (1 + math.sqrt(mu))**2
                    else:
                        require(key in all_chosen, "Inactive evaluation has no tuning cell")
                        rate = all_chosen[key]
                        wanted_ids = {identity for identity, item in origin_search["tuning"].items() if (item["regime"], item["algorithm"], item["study"], item["kappa"], item["cell"]) == key[:5]}
                        require(wanted_ids <= {x["run_id"] for x in stage_records["tuning"]}, "Incomplete inactive tuning cell")
                    spec = expected_spec(origin_config, study, r["kappa"], units[0], r["algorithm"], rate, "evaluation", r["regime"], 0, cell)
                    if r["regime"] == "theory" and r["algorithm"] == "heavy_ball":
                        spec["params"]["beta"] = ((1 - math.sqrt(mu)) / (1 + math.sqrt(mu)))**2
                    validate_record(r, spec, origin_config)
    tuning_groups = defaultdict(list)
    for r in active["tuning"]:
        tuning_groups[(r["regime"], r["algorithm"], r["study"], r["kappa"], r["cell"], 0)].append(r)
    active_selections = [s for s in selections if s["study"] in active_names]
    selection_keys = [(s["regime"], s["algorithm"], s["study"], s["kappa"], s["cell"], 0) for s in active_selections]
    require(len(set(selection_keys)) == len(selection_keys) and set(selection_keys) == set(tuning_groups), "Selection cell coverage differs")
    floor, tie = config["tuning"]["loss_floor"], config["tuning"]["tie_atol"]
    chosen = reselect(active["tuning"], floor, tie, config["tuning"].get("tie_break", "smaller_rate"))
    for s, key in zip(active_selections, selection_keys):
        require(same(s["selected_rate"], chosen[key]) and s["tuning_failure"] == (chosen[key] is None), f"{key}: selected rate differs")
        rates = frozen[s["algorithm"]]["candidates"]
        require(same(s["candidate_rates"], rates) and s["runs_per_candidate"] * len(rates) == len(tuning_groups[key]), f"{key}: candidate coverage differs")
        facts = selection_facts(tuning_groups[key], rates, config)
        require(all(same(s.get(k), v) for k, v in facts.items()), f"{key}: candidate score/selection summary differs")
    pilot_selected = [s for s in read_json(output / "pilot_selections.json") if s["study"] in active_names]
    pilot_chosen = reselect([{**r, "cell": "all"} for r in active["pilot"]], floor, tie, config["tuning"].get("tie_break", "smaller_rate"))
    keys = [("none", s["algorithm"], s["study"], s["kappa"], "all", s["round"]) for s in pilot_selected]
    require(len(set(keys)) == len(keys) and set(keys) == set(pilot_chosen), "Pilot selection coverage differs")
    require(all(same(s["selected_rate"], pilot_chosen[k]) for s, k in zip(pilot_selected, keys)), "Pilot selection differs")
    for s, key in zip(pilot_selected, keys):
        records = [r for r in active["pilot"] if (r["algorithm"], r["study"], r["kappa"], r["round"]) == (s["algorithm"], s["study"], s["kappa"], s["round"])]
        rates = grid(*frozen[s["algorithm"]]["history"][s["round"]]["range"], config["tuning"]["candidates"])
        require(all(same(s.get(k), v) for k, v in selection_facts(records, rates, config).items()), f"{key}: pilot score summary differs")
    # Tables are verified before expensive numerical sampling. Missing data cannot
    # be hidden by the sample size or by a missing optional input file.
    pairs = check_tables(output, active, active_selections, config)
    numerical = {}
    numerical_errors = []
    rng = random.Random(seed)
    for stage, records in active.items():
        executed = [r for r in records if r["status"] != NOT_RUN]
        sampled = rng.sample(executed, min(sample, len(executed)))
        for r in sampled:
            try:
                errors = recheck(r)
            except Exception as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
            numerical_errors.extend(f"{r['run_id']}: {error}" for error in errors)
        numerical[stage] = len(sampled)
    return {"passed": not numerical_errors, "errors": numerical_errors, "sample_per_stage": sample, "seed": seed,
            "structural_checks_passed": True, "numerical_checks_passed": not numerical_errors,
            "complete_stage_records": {k: len(v) for k, v in active.items()}, "numerical_runs_rechecked": numerical,
            "not_run_placeholders": sum(r["status"] == NOT_RUN for r in active["evaluation"]), "pairs_checked": pairs,
            "historical_source_identity_preserved": True, "numerical_tolerance": {"rtol": RTOL, "atol": ATOL}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/part3"))
    parser.add_argument("--sample", default=60, nargs="?", const="", help="positive number of runs re-executed per stage")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--read-only", action="store_true", help="print verification without changing any artifact")
    args = parser.parse_args(argv)
    marker = args.output / "verification.json"
    if not args.read_only and marker.exists():
        marker.unlink()
    try:
        try:
            sample = int(args.sample)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample must be a positive integer") from exc
        report = verify(args.output, sample, args.seed)
    except Exception as exc:
        report = {"passed": False, "structural_checks_passed": False, "numerical_checks_passed": None,
                  "errors": [f"{type(exc).__name__}: {exc}"], "sample_per_stage": args.sample, "seed": args.seed}
    if not args.read_only and args.output.is_dir():
        marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
