"""Generate Part 4 figures from saved CSV only; no optimizer execution."""
import argparse
import io
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .part4_statistics import rows

ALGORITHMS = ("gd", "heavy_ball", "adam", "eigenbasis_adam", "diagonal_gd")
LABELS = ("GD", "Heavy ball", "Adam", "Eigenbasis Adam", "Diagonal GD")
COLORS = ("#4477AA", "#228833", "#CC6677", "#AA3377", "#EE9933")


def save(fig, path, dpi):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    data = buffer.getvalue()
    if len(data) < 100: raise ValueError("empty figure")
    temporary = path.with_suffix(".png.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def generate(source, output, budgets=(1000, 10000, 50000)):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    effects, success, run_summary = (rows(source / name) for name in
                                    ("paired_effects.csv", "paired_success.csv", "run_summary.csv"))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for budget in budgets:
        for regime in ("transfer", "retune"):
            fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
            images = []
            for ax, algorithm, label in zip(axes, ALGORITHMS, LABELS):
                matrix = []
                for kappa in (1, 10, 100, 1000, 10000):
                    line = []
                    for angle in ("reference", "15.0", "30.0", "45.0", "60.0", "75.0", "90.0"):
                        found = [r for r in run_summary if r["study"] == "planar" and r["regime"] == regime
                                 and r["algorithm"] == algorithm and float(r["kappa"]) == kappa
                                 and r["angle"] == angle and int(r["checkpoint"]) == budget]
                        if len(found) != 1: raise ValueError("gap-map cell missing or duplicated")
                        line.append(float(found[0]["median"]))
                    matrix.append(line)
                images.append(ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=-16, vmax=0))
                ax.set(title=label, xticks=range(7), xticklabels=["0", "15", "30", "45", "60", "75", "90"],
                       yticks=range(5), yticklabels=["1", "10", "100", "1000", "10000"], xlabel="Angle (deg)")
            axes[0].set_ylabel("Condition number")
            fig.suptitle(f"2D {regime}, B={budget}: median log10 relative loss (floor 1e-16)")
            fig.subplots_adjust(top=.82, bottom=.18, right=.92, wspace=.2)
            fig.colorbar(images[0], cax=fig.add_axes([.94, .2, .012, .55]))
            save(fig, output / f"planar_gap_{regime}_B{budget}.png", dpi=160); plt.close(fig)
        # Every kappa (including isotropic control), both regimes, all algorithms.
        fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
        for i, regime in enumerate(("transfer", "retune")):
            for j, kappa in enumerate((1, 10, 100, 1000, 10000)):
                ax = axes[i, j]
                for algorithm, label, color in zip(ALGORITHMS, LABELS, COLORS):
                    selected = sorted([r for r in effects if r["study"] == "planar" and r["regime"] == regime
                                       and r["algorithm"] == algorithm and float(r["kappa"]) == kappa
                                       and r["metric"] == f"delta_{budget}"], key=lambda r: float(r["angle"]))
                    x = [float(r["angle"]) for r in selected]
                    ax.plot(x, [float(r["median"]) for r in selected], "o-", markersize=2.5, linewidth=1, color=color, label=label)
                    ax.fill_between(x, [float(r["ci_low"]) for r in selected],
                                    [float(r["ci_high"]) for r in selected], color=color, alpha=.10)
                ax.axhline(0, color="grey", lw=.5)
                ax.set(title=f"{regime}; kappa={kappa}", xlabel="Angle (degrees)", ylabel="Median paired log10 gap")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        fig.suptitle(f"2D, B={budget}: joint initial-direction bootstrap, pointwise 95% bands")
        fig.tight_layout(rect=(0, .06, 1, .94)); save(fig, output / f"planar_effect_B{budget}.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
        for ax, regime in zip(axes, ("transfer", "retune")):
            for a, (algorithm, label, color) in enumerate(zip(ALGORITHMS, LABELS, COLORS)):
                selected = sorted([r for r in effects if r["study"] == "p20" and r["regime"] == regime
                                   and r["algorithm"] == algorithm and r["metric"] == f"delta_{budget}"],
                                  key=lambda r: float(r["kappa"]))
                ys = np.arange(len(selected)) + (a - 2) * .12
                for y, r in zip(ys, selected):
                    ax.plot([float(r["ci_low"]), float(r["ci_high"])], [y, y], color=color, lw=1.5)
                ax.scatter([float(r["median"]) for r in selected], ys, s=15, color=color, label=label)
            ax.axvline(0, color="grey", lw=.7)
            ax.set(yticks=range(5), yticklabels=["1 (control)", "10", "100", "1000", "10000"],
                   title=regime, xlabel="Median paired log10 gap", ylabel="Condition number")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"p=20, B={budget}: rotation-cluster percentile 95% intervals")
        fig.tight_layout(); save(fig, output / f"p20_effect_B{budget}.png", dpi=170); plt.close(fig)
    # Historical attainment and final retention (unique runs; no duplicated refs).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, regime in zip(axes, ("transfer", "retune")):
        selected = [r for r in run_summary if r["regime"] == regime and r["checkpoint"] == "50000"]
        ever, retained = [], []
        for algorithm in ALGORITHMS:
            a = [r for r in selected if r["algorithm"] == algorithm]
            n = sum(int(r["n_runs"]) for r in a)
            ever.append(100 * sum(int(r["ever_attained"]) for r in a) / n)
            retained.append(100 * sum(int(r["final_retained"]) for r in a) / n)
        x = np.arange(5)
        ax.bar(x-.18, ever, .36, label="Ever attained", color="#AABBDD")
        ax.bar(x+.18, retained, .36, label="Final retained", color="#4477AA")
        ax.set(xticks=x, xticklabels=["GD", "HB", "Adam", "Basis\nAdam", "Diag\nGD"],
               ylim=(0, 108), title=regime, ylabel="Percent of unique evaluation runs")
        for index, value in enumerate(retained): ax.text(index+.18, value+1, f"{value:.1f}", ha="center", fontsize=8)
    axes[0].legend(); fig.suptitle("Threshold 1e-6: historical attainment is not final retention")
    fig.tight_layout(); save(fig, output / "success_retention.png", dpi=180); plt.close(fig)
    # Full four-way retained table across all pairs (counts, not independent n).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cats = ("both", "diagonal_only", "rotated_only", "neither")
    for ax, regime in zip(axes, ("transfer", "retune")):
        image = np.array([[sum(int(r[c]) for r in success if r["definition"] == "final_retained"
                                and r["regime"] == regime and r["algorithm"] == algorithm) for c in cats] for algorithm in ALGORITHMS])
        ax.imshow(image, cmap="Blues", aspect="auto")
        for i in range(5):
            for j in range(4): ax.text(j, i, str(image[i, j]), ha="center", va="center", color="white" if image[i,j]>image.max()/2 else "black")
        ax.set(xticks=range(4), xticklabels=["Both", "Diagonal\nonly", "Rotated\nonly", "Neither"],
               yticks=range(5), yticklabels=LABELS, title=regime)
    fig.suptitle("Final-retained paired outcomes; repeated references are not independent samples")
    fig.tight_layout(); save(fig, output / "paired_success_table.png", dpi=180); plt.close(fig)
    failures = rows(source / "failures_by_stage.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([r["stage"]+" / "+r["status"] for r in failures], [int(r["count"]) for r in failures], color="#CC6677")
    ax.set(ylabel="Recorded failed runs", title="Failures in pilot/tuning; zero explosive/nonfinite evaluation runs")
    fig.tight_layout(); save(fig, output / "failure_counts.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/part4/stats"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part4/figures"))
    args = parser.parse_args(); generate(args.input, args.output)


if __name__ == "__main__": main()
