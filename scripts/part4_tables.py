"""Generate the report's LaTeX table from a Part 4 statistics directory."""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"gd": "GD", "heavy_ball": "Heavy ball", "adam": "Adam",
          "eigenbasis_adam": "Eigenbasis Adam", "diagonal_gd": "Diagonal GD"}


def generate(source, output):
    """Write the final-budget counts table and return its output path."""
    with (Path(source) / "run_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        runs = list(csv.DictReader(handle))
    lines = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{llrrr}\toprule",
             r"Regime & Method & Unique runs & Ever attained & Final retained \\ \midrule"]
    for regime in ("transfer", "retune", "theory"):
        for algorithm, label in LABELS.items():
            selected = [row for row in runs if row["regime"] == regime
                        and row["algorithm"] == algorithm and row["checkpoint"] == "50000"]
            if selected:
                counts = [sum(int(row[key]) for row in selected)
                          for key in ("n_runs", "ever_attained", "final_retained")]
                lines.append(f"{regime} & {label} & {counts[0]} & {counts[1]} & {counts[2]} " + r"\\")
    lines.extend([r"\bottomrule\end{tabular}",
                  r"\caption{Unique evaluation runs; references counted once. Theory is separate.}",
                  r"\end{table}"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/part4/stats")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/part4_generated_tables.tex")
    args = parser.parse_args(argv)
    generate(args.input, args.output)


if __name__ == "__main__":
    main()
