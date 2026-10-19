"""Part 4 entrypoint paths, table CLI, and isolated report build regressions."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_summary(directory, count, retained):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("regime", "algorithm", "checkpoint",
                                                    "n_runs", "ever_attained", "final_retained"))
        writer.writeheader()
        writer.writerow(dict(regime="transfer", algorithm="adam", checkpoint=50000,
                             n_runs=count, ever_attained=count, final_retained=retained))


class Part4PipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "work/test-part4-pipeline"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        (self.project / "scripts").mkdir()
        (self.project / "docs").mkdir()
        shutil.copy2(ROOT / "scripts/part4_tables.py", self.project / "scripts/part4_tables.py")
        self.config = self.project / "alternate.json"
        self.config.write_text(json.dumps(dict(input="saved-part3", statistics_output="work/alternate-stats",
                                               bootstrap_repetitions=17, bootstrap_seed=42)), encoding="utf-8")
        self.statistics = self.project / "work/alternate-stats"
        write_summary(self.project / "outputs/part4/stats", 999, 998)
        self.pipeline = load_script("reproduce_part4.py")
        self.pipeline.ROOT = self.project

    def stage_runner(self, commands):
        actual_run = subprocess.run

        def execute(command, **kwargs):
            commands.append((command, kwargs))
            if "coordinate_geometry.part4_statistics" in command:
                destination = Path(command[command.index("--output") + 1])
                write_summary(destination, 7, 5)
            elif "coordinate_geometry.part4_figures" in command:
                source = Path(command[command.index("--input") + 1])
                # Read the selected input, so a stale/default input gives a different result.
                with (source / "run_summary.csv").open(encoding="utf-8") as handle:
                    selected_count = next(csv.DictReader(handle))["n_runs"]
                destination = Path(command[command.index("--output") + 1])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "selected_count.txt").write_text(selected_count, encoding="utf-8")
            elif Path(command[1]).name == "part4_tables.py":
                # Exercise the real table CLI in the isolated project, not a mocked writer.
                return actual_run(command, **kwargs)
            elif command[0] == "fixture-pdflatex":
                build = Path(next(value.split("=", 1)[1] for value in command
                                  if value.startswith("-output-directory=")))
                (build / "part4_report.pdf").write_bytes(b"fixture-pdf")
                (build / "part4_report.aux").write_text("fixture-aux", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        return execute

    def test_scripts_are_import_safe(self):
        # Imports must not parse the caller's flags, read statistics, or start stages.
        with patch.object(sys, "argv", ["caller", "--unrelated-flag"]), \
                patch.object(subprocess, "run", side_effect=AssertionError("executed on import")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            tables = load_script("part4_tables.py")
            pipeline = load_script("reproduce_part4.py")
        self.assertTrue(callable(tables.generate))
        self.assertTrue(callable(pipeline.main))
        self.assertEqual(output.getvalue(), "")

    def test_alternate_config_flows_to_figures_and_real_table_cli(self):
        commands = []
        with patch.object(subprocess, "run", side_effect=self.stage_runner(commands)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.pipeline.main(["--config", str(self.config)])
        output = self.project / "outputs/part4"
        self.assertEqual((output / "figures/selected_count.txt").read_text(), "7")
        table = (self.project / "docs/part4_generated_tables.tex").read_text(encoding="utf-8")
        self.assertIn("transfer & Adam & 7 & 7 & 5", table)
        self.assertNotIn("999", table)
        self.assertIn("999", (output / "stats/run_summary.csv").read_text())
        statistics_command = next(command for command, _ in commands
                                  if "coordinate_geometry.part4_statistics" in command)
        self.assertEqual(Path(statistics_command[statistics_command.index("--input") + 1]),
                         self.project / "saved-part3")
        self.assertEqual(statistics_command[statistics_command.index("--bootstrap") + 1], "17")
        self.assertEqual(statistics_command[statistics_command.index("--seed") + 1], "42")
        figure_command = next(command for command, _ in commands
                              if "coordinate_geometry.part4_figures" in command)
        self.assertEqual(Path(figure_command[figure_command.index("--input") + 1]), self.statistics)

    def test_report_build_uses_work_and_copies_deliverable(self):
        commands = []
        (self.project / "docs/part4_report.tex").write_text("fixture-source", encoding="utf-8")
        with patch.object(subprocess, "run", side_effect=self.stage_runner(commands)), \
                patch.object(shutil, "which", return_value="fixture-pdflatex"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.pipeline.main(["--config", str(self.config), "--report"])
        compile_calls = [(command, kwargs) for command, kwargs in commands if command[0] == "fixture-pdflatex"]
        self.assertEqual(len(compile_calls), 2)
        for command, kwargs in compile_calls:
            build = Path(next(value.split("=", 1)[1] for value in command
                              if value.startswith("-output-directory=")))
            self.assertTrue(build.is_absolute())
            self.assertEqual(build, self.project / "work/part4/report-build")
            self.assertEqual(kwargs["cwd"], self.project / "docs")
            self.assertEqual(command[-1], "part4_report.tex")
            self.assertEqual(Path(kwargs["stdout"].name).parent, build)
        self.assertEqual((self.project / "outputs/part4/pdf/part4_report.pdf").read_bytes(), b"fixture-pdf")
        self.assertEqual(sorted(path.name for path in (self.project / "docs").iterdir()),
                         ["part4_generated_tables.tex", "part4_report.tex"])

    def test_table_cli_accepts_explicit_input_and_output(self):
        write_summary(self.statistics, 11, 9)
        destination = self.project / "nested/table.tex"
        environment = dict(os.environ, TEMP=str(self.project), TMP=str(self.project), TMPDIR=str(self.project))
        subprocess.run([sys.executable, str(ROOT / "scripts/part4_tables.py"),
                        "--input", str(self.statistics), "--output", str(destination)],
                       cwd=self.project, env=environment, check=True, capture_output=True, text=True)
        self.assertIn("transfer & Adam & 11 & 11 & 9", destination.read_text(encoding="utf-8"))
        self.assertFalse((self.project / "docs/part4_generated_tables.tex").exists())


if __name__ == "__main__":
    unittest.main()
