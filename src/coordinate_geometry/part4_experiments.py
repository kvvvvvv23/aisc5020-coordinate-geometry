"""Part 4 standalone P5 validation; NOT the Part 3 held-out experiment.

Run with python -m coordinate_geometry.part4_experiments --output outputs/part4.
The angle sweep uses analytic spectral information and is labeled accordingly.
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path

import numpy as np

from .contracts import RunConfig
from .preconditioning import effective_hessian, p5_condition_number
from .problems import QuadraticProblem, matched_pair, rotation_2d
from .runner import run


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(output: Path, steps=2000, *, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    curves, traces, runs = [], [], []
    max_error = 0.0
    for kappa in (10, 100, 1000, 10000):
        d = np.array([1 / kappa, 1.0])
        for angle in range(91):
            problem = QuadraticProblem(d, rotation_2d(np.deg2rad(angle)))
            eigenvalues = np.linalg.eigvalsh(effective_hessian(problem.hessian))
            predicted = p5_condition_number(problem.hessian)
            measured = float(eigenvalues[-1] / eigenvalues[0])
            error = abs(predicted - measured) / measured
            max_error = max(error, max_error)
            curves.append(dict(kappa=kappa, angle_degrees=angle,
                               predicted=predicted, measured=measured, relative_error=error))
        for angle in (0, 15, 30, 45, 60, 75, 90):
            # Reinitialize the same stream across angles: identical canonical direction.
            _, problem, _, initial = matched_pair(d, rotation_2d(np.deg2rad(angle)),
                                                  np.random.default_rng(20261019))
            for algorithm, rate in (("diagonal_gd", 1.0), ("gd", 2 / (1 + 1 / kappa))):
                result = run(problem, initial, RunConfig(algorithm, rate, steps, record_every=10))
                runs.append(dict(kappa=kappa, angle_degrees=angle, algorithm=algorithm,
                                 learning_rate=rate, gradient_calls=result.gradient_calls,
                                 status=result.status, first_attainment=result.first_attainment,
                                 threshold_retained=result.threshold_retained,
                                 final_relative_loss=result.final_loss / result.initial_loss,
                                 experiment_origin="part4_analytic_reference_not_part3"))
                traces.extend(dict(kappa=kappa, angle_degrees=angle, algorithm=algorithm,
                                   step=t.step, relative_loss=t.relative_loss) for t in result.trace)
    assert max_error < 1e-8, max_error
    write_csv(output / "condition_numbers.csv", curves)
    write_csv(output / "standalone_runs.csv", runs)
    write_csv(output / "standalone_traces.csv", traces)
    summary = dict(max_relative_condition_error=max_error, condition_checks=len(curves),
                   standalone_runs=len(runs), steps=steps, seed=20261019,
                   formal_part3_analysis_completed=False,
                   notes="Standalone theoretical reference, one direction per spectrum; no tuned ranking.")
    (output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    if plots:
        plot(output, curves, traces)
    return summary


def plot(output, curves, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, path):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=180)
        data = buffer.getvalue()
        if len(data) < 100: raise ValueError("empty figure")
        temporary = path.with_suffix(".png.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    fig, ax = plt.subplots(figsize=(7, 4))
    for kappa in (10, 100, 1000, 10000):
        rows = [r for r in curves if r["kappa"] == kappa]
        line, = ax.plot([r["angle_degrees"] for r in rows], [r["predicted"] for r in rows],
                        label=f"kappa={kappa}")
        ax.scatter([r["angle_degrees"] for r in rows[::5]],
                   [r["measured"] for r in rows[::5]], s=12, color=line.get_color())
    ax.set(xlabel="Rotation angle (degrees)", ylabel="Effective condition number", yscale="log",
           title="P5: theory (lines), eigensolver (dots)")
    ax.legend(); fig.tight_layout(); save(fig, output / "effective_condition.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, kappa in zip(axes.flat, (10, 100, 1000, 10000)):
        for angle in (0, 15, 30, 45):
            rows = [r for r in traces if r["kappa"] == kappa and r["angle_degrees"] == angle
                    and r["algorithm"] == "diagonal_gd"]
            ax.semilogy([r["step"] for r in rows],
                        [max(r["relative_loss"], 1e-16) for r in rows], label=f"{angle} deg")
        ax.axhline(1e-6, color="grey", linestyle="--", linewidth=.7)
        ax.set(title=f"kappa={kappa}", xlabel="Gradient calls", ylabel="Relative loss (floor 1e-16)")
    axes[0, 0].legend(); fig.suptitle("Standalone diagonal GD, eta=1; not a tuned comparison")
    fig.tight_layout(); save(fig, output / "preconditioned_trajectories.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/part4"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no-plots", action="store_true", help="Run numerical checks without matplotlib")
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.steps, plots=not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
