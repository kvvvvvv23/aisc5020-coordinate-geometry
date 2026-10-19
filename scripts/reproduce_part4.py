"""Reproduce only Module 4, not the expensive Part 3 tuning/evaluation.

Run from project root: python scripts/reproduce_part4.py --report
Use --config to select another Part 4 configuration. Paths inside that configuration
are relative to the project root. Requires numpy, matplotlib; --report additionally
requires pdflatex on PATH.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def project_path(value, root):
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def run(*command, cwd, env, log=None):
    if log is not None:
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def reproduce(config_path, *, report=False):
    root = ROOT
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    source = project_path(config["input"], root)
    statistics = project_path(config["statistics_output"], root)
    output = root / "outputs/part4"
    figures = output / "figures"
    scratch = root / "work/part4"
    temporary = scratch / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "src"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
               MPLCONFIGDIR=str(scratch / "matplotlib"))

    def execute(*command, cwd=root, log=None):
        run(*command, cwd=cwd, env=env, log=log)

    execute(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q")
    execute(sys.executable, "-m", "coordinate_geometry.part4_experiments",
            "--output", str(output))
    execute(sys.executable, "-m", "coordinate_geometry.part4_statistics",
            "--input", str(source), "--output", str(statistics),
            "--bootstrap", str(config["bootstrap_repetitions"]),
            "--seed", str(config["bootstrap_seed"]))
    execute(sys.executable, "-m", "coordinate_geometry.part4_figures",
            "--input", str(statistics), "--output", str(figures))
    execute(sys.executable, str(root / "scripts/part4_tables.py"),
            "--input", str(statistics),
            "--output", str(root / "docs/part4_generated_tables.tex"))
    if report:
        compiler = shutil.which("pdflatex")
        if compiler is None:
            raise SystemExit("Statistics are complete. Install a LaTeX distribution to rebuild the included PDF.")
        build = (scratch / "report-build").resolve()
        build.mkdir(parents=True, exist_ok=True)
        for pass_number in (1, 2):
            execute(compiler, "-interaction=nonstopmode", "-halt-on-error",
                    f"-output-directory={build}", "part4_report.tex",
                    cwd=root / "docs", log=build / f"report_build_{pass_number}.log")
        pdf_output = output / "pdf"
        pdf_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / "part4_report.pdf", pdf_output / "part4_report.pdf")
    print("Part 4 recomputed. This does NOT assert a fully passing historical optimizer replay.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/part4.json")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    reproduce(args.config, report=args.report)


if __name__ == "__main__":
    main()
