"""Part 4: read-only source audit, paired statistics and reproducible bootstrap.

No tuning or optimizer reruns are performed here. Saved historical observations
remain unchanged. Use --input /path/to/outputs/part3 --output outputs/part4/stats.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

GROUP = ("regime", "algorithm", "study", "kappa", "angle")
OUTCOMES = ("both", "diagonal_only", "rotated_only", "neither")
STATUSES = {"finite_cap", "explosive", "nonfinite", "not_run_tuning_failure"}


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    if not records:
        raise ValueError("refusing to write an empty result without a schema")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def number(text):
    if text is None or text == "":
        return None
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid nonnegative finite metric: {text!r}")
    return value


def boolean(text):
    if text not in ("True", "False"):
        raise ValueError(f"invalid boolean {text!r}")
    return text == "True"


def outcome(d, q):
    return "both" if d and q else "diagonal_only" if d else "rotated_only" if q else "neither"


def delta(q, d, floor=1e-16):
    return None if q is None or d is None else math.log10(max(q, floor)) - math.log10(max(d, floor))


def paired_records(evaluation, saved_pairs, checkpoints, tolerance=1e-6):
    index = {r["run_id"]: r for r in evaluation}
    if len(index) != len(evaluation):
        raise ValueError("duplicate run_id")
    # Check unique runs before pairing, including references with no paired row.
    attainments = {}
    for r in evaluation:
        if r["status"] not in STATUSES:
            raise ValueError("unknown status")
        first = number(r["first_attainment"])
        if first is not None:
            if isinstance(r["first_attainment"], bool) or first <= 0 or not first.is_integer():
                raise ValueError("first attainment must be a positive integer")
            calls = number(r["gradient_calls"])
            if (isinstance(r["gradient_calls"], bool) or calls is None
                    or not calls.is_integer() or first > calls):
                raise ValueError("first attainment exceeds valid integer gradient calls")
        attainments[r["run_id"]] = first
        retained = boolean(r["threshold_retained"])
        if retained and first is None:
            raise ValueError("final retention requires historical attainment")
        final = number(r[f"r_{checkpoints[-1]}"])
        expected_kept = r["status"] == "finite_cap" and final is not None and final <= tolerance
        if retained != expected_kept:
            raise ValueError("retention inconsistent with final checkpoint")
    seen, result, maximum = set(), [], 0.0
    for p in saved_pairs:
        rid = p["rotated_run_id"]
        if rid in seen:
            raise ValueError("duplicate rotated pair")
        seen.add(rid)
        if rid not in index or p["reference_run_id"] not in index:
            raise ValueError("missing pair reference")
        q, d = index[rid], index[p["reference_run_id"]]
        if q["reference_run_id"] != d["run_id"] or q["role"] != "rotated" or d["role"] != "reference":
            raise ValueError("inconsistent pair link or role")
        for key in ("algorithm", "regime", "study", "dimension", "kappa", "init_id", "init_seed_key"):
            if q[key] != d[key]:
                raise ValueError(f"pair mismatch: {key}")
        for key in ("algorithm", "regime", "study", "dimension", "kappa", "pair_id", "cluster_id", "orientation_id"):
            if p[key] != q[key]:
                raise ValueError(f"saved pair metadata differs: {key}")
        td, tq = attainments[d["run_id"]], attainments[q["run_id"]]
        historic = outcome(td is not None, tq is not None)
        if p["outcome"] != historic:
            raise ValueError("historical success outcome differs")
        reference_kept, rotated_kept = boolean(d["threshold_retained"]), boolean(q["threshold_retained"])
        if boolean(p["reference_retained"]) != reference_kept or boolean(p["rotated_retained"]) != rotated_kept:
            raise ValueError("saved retention fields differ")
        ratio = tq / td if td is not None and tq is not None else None
        supplied_ratio = number(p["attainment_ratio"])
        if (ratio is None) != (supplied_ratio is None) or (ratio is not None and not math.isclose(ratio, supplied_ratio, rel_tol=1e-12)):
            raise ValueError("attainment ratio differs")
        row = {key: q[key] for key in ("regime", "algorithm", "study", "dimension", "kappa", "orientation_id", "init_id", "cluster_id")}
        row.update(angle=q["angle_degrees"] if q["study"] == "planar" else "all_rotations",
                   pair_id=q["pair_id"], rotated_run_id=rid, reference_run_id=d["run_id"],
                   historical_outcome=historic, retained_outcome=outcome(reference_kept, rotated_kept),
                   attainment_ratio=ratio, reference_status=d["status"], rotated_status=q["status"],
                   correspondence_expected=boolean(p["correspondence_expected"]))
        for budget in checkpoints:
            a, b = number(q[f"r_{budget}"]), number(d[f"r_{budget}"])
            effect = delta(a, b)
            saved = None if p[f"delta_{budget}"] == "" else float(p[f"delta_{budget}"])
            if (effect is None) != (saved is None):
                raise ValueError("missing saved delta differs")
            if effect is not None:
                maximum = max(maximum, abs(effect - saved))
                if not math.isfinite(saved) or not math.isclose(effect, saved, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("saved delta differs")
            row[f"delta_{budget}"] = effect
        result.append(row)
    expected = {r["run_id"] for r in evaluation if r["role"] == "rotated"}
    if seen != expected:
        raise ValueError("paired coverage differs from evaluation")
    return result, maximum


def resampling_plan(units, repetitions, seed, namespace):
    """Same namespace/units => same draws across angles, algorithms and metrics."""
    if type(repetitions) is not int or repetitions < 100:
        raise ValueError("use at least 100 bootstrap repetitions")
    ordered = sorted(set(units))
    if not ordered:
        raise ValueError("no sampling units")
    digest = hashlib.sha256(f"{seed}|{namespace}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return ordered, rng.integers(0, len(ordered), size=(repetitions, len(ordered)))


def bootstrap(values_by_unit, plan, estimator="median"):
    """Cluster-only resampling: each selected unit contributes every paired row.

    Missing metrics remain missing within their sampled unit; no artificial
    replacement values are assigned. Return None if no finite observations exist.
    """
    units, draws = plan
    arrays = [np.asarray(values_by_unit.get(u, []), dtype=float) for u in units]
    if any(np.any(~np.isfinite(a)) for a in arrays):
        raise ValueError("nonfinite bootstrap values")
    if estimator not in ("median", "mean"):
        raise ValueError("unknown estimator")
    reduce = np.median if estimator == "median" else np.mean
    samples = []
    # Vectorized common balanced case; do not flatten clusters into iid rows.
    sizes = {len(a) for a in arrays}
    if len(sizes) == 1 and next(iter(sizes)) > 0:
        sampled = np.asarray(arrays)[draws].reshape(len(draws), -1)
        estimates = np.median(sampled, axis=1) if estimator == "median" else np.mean(sampled, axis=1)
        return [float(x) for x in np.quantile(estimates, [.025, .975])]
    for draw in draws:
        sample = np.concatenate([arrays[i] for i in draw])
        if sample.size:
            samples.append(float(reduce(sample)))
    return None if not samples else [float(x) for x in np.quantile(samples, [.025, .975])]


def describe(values):
    values = list(values)
    if not values:
        return dict(n=0, median=None, q25=None, q75=None)
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    return dict(n=len(values), median=float(median), q25=float(q25), q75=float(q75))


def aggregate(evaluation, paired, checkpoints, repetitions, seed):
    groups, by_run = defaultdict(list), defaultdict(list)
    plans, angle_units = {}, {}
    for p in paired:
        groups[tuple(p[k] for k in GROUP)].append(p)
        namespace = (p["study"], p["kappa"])
        angle_units.setdefault(namespace, set()).add(p["init_id"] if p["study"] == "planar" else p["cluster_id"])
    for namespace, units in angle_units.items():
        plans[namespace] = resampling_plan(units, repetitions, seed, "|".join(namespace))
    summaries, success = [], []
    for keys, items in sorted(groups.items()):
        meta = dict(zip(GROUP, keys))
        study, kappa = meta["study"], meta["kappa"]
        plan = plans[(study, kappa)]
        unit_key = "init_id" if study == "planar" else "cluster_id"
        # Require shared units across angle curves; no silent unmatched bootstrap.
        if set(p[unit_key] for p in items) != set(plan[0]):
            raise ValueError("incomplete sampling-unit coverage across comparison cells")
        for definition, field in (("ever_attained", "historical_outcome"), ("final_retained", "retained_outcome")):
            counts = Counter(p[field] for p in items)
            row = dict(meta, definition=definition, n_pairs=len(items), **{name: counts[name] for name in OUTCOMES})
            for side in ("diagonal", "rotated"):
                successes = [int(p[field] in ("both", side + "_only")) for p in items]
                units = defaultdict(list)
                for p, value in zip(items, successes):
                    units[p[unit_key]].append(value)
                interval = bootstrap(units, plan, "mean")
                row.update({f"{side}_rate": float(np.mean(successes)),
                            f"{side}_ci_low": interval[0], f"{side}_ci_high": interval[1]})
            success.append(row)
        for metric in [f"delta_{b}" for b in checkpoints] + ["attainment_ratio"]:
            units = defaultdict(list)
            for p in items:
                if p[metric] is not None:
                    units[p[unit_key]].append(p[metric])
            description = describe(v for a in units.values() for v in a)
            interval = bootstrap(units, plan)
            summaries.append(dict(meta, metric=metric, **description, n_pairs_total=len(items),
                                  n_missing=len(items)-description["n"], n_units=len(plan[0]),
                                  ci_low=interval[0] if interval else None, ci_high=interval[1] if interval else None))
    for r in evaluation:
        # Unique actual runs, never multiply the denominator by reusing a reference.
        angle = r["angle_degrees"] if r["study"] == "planar" and r["role"] == "rotated" else ("reference" if r["role"] == "reference" else "all_rotations")
        by_run[(r["regime"], r["algorithm"], r["study"], r["kappa"], angle, r["role"])].append(r)
    run_summary = []
    for keys, items in sorted(by_run.items()):
        meta = dict(zip((*GROUP, "role"), keys))
        status = Counter(r["status"] for r in items)
        for budget in checkpoints:
            values = [number(r[f"r_{budget}"]) for r in items]
            run_summary.append(dict(meta, checkpoint=budget, n_runs=len(items),
                                    ever_attained=sum(r["first_attainment"] != "" for r in items),
                                    final_retained=sum(boolean(r["threshold_retained"]) for r in items),
                                    **{s: status[s] for s in sorted(STATUSES)},
                                    **describe(math.log10(max(v, 1e-16)) for v in values if v is not None)))
    return summaries, success, run_summary


def analyze(source, output, repetitions=2000, seed=502004):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("analysis output must be outside the historical input directory")
    config = json.loads((source / "run_config.json").read_text())
    if config["study_status"] != "main_protocol":
        raise ValueError("formal report requires main_protocol, not smoke")
    evaluation, supplied = rows(source / "tables/evaluation_runs.csv"), rows(source / "tables/pairs.csv")
    checkpoints = config["evaluation"]["checkpoints"]
    paired, error = paired_records(evaluation, supplied, checkpoints, config["tolerance"])
    summaries, success, run_summary = aggregate(evaluation, paired, checkpoints, repetitions, seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("paired_observations.csv", paired), ("paired_effects.csv", summaries),
                          ("paired_success.csv", success), ("run_summary.csv", run_summary)):
        write_csv(output / name, records)
    failure_counts = Counter()
    with (source / "tables/failures.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            failure_counts[(r["stage"], r["status"])] += 1
    failures = [dict(stage=k[0], status=k[1], count=v) for k, v in sorted(failure_counts.items())]
    write_csv(output / "failures_by_stage.csv", failures or [dict(stage="none", status="none", count=0)])
    coverage = []
    for algorithm in sorted({r["algorithm"] for r in evaluation}):
        items = [r for r in evaluation if r["algorithm"] == algorithm]
        coverage.append(dict(algorithm=algorithm, n=len(items),
                             ever_attained=sum(r["first_attainment"] != "" for r in items),
                             retained=sum(boolean(r["threshold_retained"]) for r in items)))
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob("*")) if p.is_file()}
    report = dict(input_manifest=json.loads((source / "manifest.json").read_text()),
                  study_status=config["study_status"], evaluation_runs=len(evaluation), pairs=len(paired),
                  max_delta_recalculation_error=error, source_files_sha256=hashes,
                  bootstrap_repetitions=repetitions, bootstrap_seed=seed, numpy=np.__version__, python=platform.python_version(),
                  bootstrap="planar: joint init-ID draws across angles; p20: rotation-cluster draws with all paired initializations",
                  intervals="pointwise descriptive percentile 95%; not simultaneous; 10 independent units per cell",
                  historical_observations_not_rerun=True, historical_verification_is_not_current_reproducibility=True,
                  cost_benchmark_completed=False, coverage=coverage,
                  failure_counts=failures, preserved_source=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/part4/stats"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=502004)
    args = parser.parse_args()
    report = analyze(args.input, args.output, args.bootstrap, args.seed)
    print(json.dumps({k: report[k] for k in ("evaluation_runs", "pairs", "max_delta_recalculation_error", "bootstrap_repetitions")}, indent=2))


if __name__ == "__main__":
    main()
