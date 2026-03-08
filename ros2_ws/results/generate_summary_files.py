#!/usr/bin/env python3
"""Generate summary CSV files from per-method raw result CSVs.

The script expects a results directory shaped like:

    results/
      GrGSL/*.csv
      PMFS/*.csv
      ...

Raw files ending in ``_variance.csv`` are ignored. By default, summaries are
written into the same results directory.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable


METHOD_DISPLAY_NAMES = {
    "GrGSL": "GrGSL",
    "ParticleFilter": "Particle Filter",
    "PMFS": "PMFS",
    "Spiral": "Spiral",
    "SurgeCast": "Surge-Cast",
    "SurgeSpiral": "Surge-Spiral",
}

SUMMARY_METHOD_ORDER = [
    "PMFS",
    "GrGSL",
    "SurgeCast",
    "Spiral",
    "SurgeSpiral",
    "ParticleFilter",
]

FILE_METHOD_ORDER = [
    "GrGSL",
    "PMFS",
    "ParticleFilter",
    "Spiral",
    "SurgeCast",
    "SurgeSpiral",
]


@dataclass(frozen=True)
class FileSummary:
    method_dir: str
    method: str
    path: Path
    scenario_file: str
    scenario: str
    simulation: str
    speed: str
    runs: int
    mean_error: float
    std_error: float
    mean_time: float
    std_time: float
    errors: tuple[float, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class MethodSummary:
    method_dir: str
    method: str
    scenarios: int
    runs: int
    mean_error: float
    std_error: float
    mean_time: float
    std_time: float


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def display_name(method_dir: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method_dir, method_dir)


def sort_method_dirs(method_dirs: Iterable[Path], method_order: list[str]) -> list[Path]:
    order = {name: index for index, name in enumerate(method_order)}
    return sorted(method_dirs, key=lambda path: (order.get(path.name, 10_000), path.name.lower()))


def parse_scenario(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "", ""

    scenario = parts[0]
    speed = parts[-1]
    simulation = "_".join(parts[1:])
    return scenario, simulation, speed


def read_raw_file(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    errors: list[float] = []
    times: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"error", "search_time"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                errors.append(float(row["error"]))
                times.append(float(row["search_time"]))
            except ValueError as exc:
                raise ValueError(f"{path}:{row_number} contains a non-numeric summary value") from exc

    return tuple(errors), tuple(times)


def collect_file_summaries(results_dir: Path) -> list[FileSummary]:
    method_dirs = [
        path
        for path in results_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith((".", "_"))
        and any(path.glob("*.csv"))
    ]

    summaries: list[FileSummary] = []
    for method_dir in sort_method_dirs(method_dirs, FILE_METHOD_ORDER):
        raw_files = sorted(
            path
            for path in method_dir.glob("*.csv")
            if not path.name.endswith("_variance.csv")
        )
        for raw_file in raw_files:
            errors, times = read_raw_file(raw_file)
            scenario, simulation, speed = parse_scenario(raw_file.stem)
            summaries.append(
                FileSummary(
                    method_dir=method_dir.name,
                    method=display_name(method_dir.name),
                    path=raw_file,
                    scenario_file=raw_file.stem,
                    scenario=scenario,
                    simulation=simulation,
                    speed=speed,
                    runs=len(errors),
                    mean_error=mean(errors),
                    std_error=sample_std(errors),
                    mean_time=mean(times),
                    std_time=sample_std(times),
                    errors=errors,
                    times=times,
                )
            )

    return summaries


def collect_method_summaries(file_summaries: list[FileSummary]) -> list[MethodSummary]:
    grouped: dict[str, list[FileSummary]] = {}
    for file_summary in file_summaries:
        grouped.setdefault(file_summary.method_dir, []).append(file_summary)

    summaries: list[MethodSummary] = []
    for method_dir in SUMMARY_METHOD_ORDER + sorted(set(grouped).difference(SUMMARY_METHOD_ORDER)):
        method_files = grouped.get(method_dir)
        if not method_files:
            continue

        errors = [value for file_summary in method_files for value in file_summary.errors]
        times = [value for file_summary in method_files for value in file_summary.times]
        summaries.append(
            MethodSummary(
                method_dir=method_dir,
                method=display_name(method_dir),
                scenarios=len(method_files),
                runs=len(errors),
                mean_error=mean(errors),
                std_error=sample_std(errors),
                mean_time=mean(times),
                std_time=sample_std(times),
            )
        )

    return summaries


def source_file_for(results_dir: Path, raw_file: Path) -> str:
    relative = raw_file.relative_to(results_dir)
    return str(PureWindowsPath("results", relative))


def write_summary_csv(output_dir: Path, method_summaries: list[MethodSummary]) -> None:
    output_path = output_dir / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "method",
                "scenarios",
                "runs",
                "target_error_mean",
                "actual_error_mean",
                "target_error_std",
                "actual_error_std",
                "target_time_mean",
                "actual_time_mean",
                "target_time_std",
                "actual_time_std",
            ]
        )
        for summary in method_summaries:
            writer.writerow(
                [
                    summary.method,
                    summary.scenarios,
                    summary.runs,
                    f"{round(summary.mean_error, 2):.6f}",
                    f"{summary.mean_error:.6f}",
                    f"{round(summary.std_error, 2):.6f}",
                    f"{summary.std_error:.6f}",
                    f"{round(summary.mean_time, 2):.6f}",
                    f"{summary.mean_time:.6f}",
                    f"{round(summary.std_time, 2):.6f}",
                    f"{summary.std_time:.6f}",
                ]
            )


def write_parsed_summary_results(output_dir: Path, method_summaries: list[MethodSummary]) -> None:
    output_path = output_dir / "parsed_summary_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Method", "Metric", "mean", "std", "runs"])
        for summary in method_summaries:
            writer.writerow([summary.method, "Error", summary.mean_error, summary.std_error, summary.runs])
            writer.writerow([summary.method, "Time", summary.mean_time, summary.std_time, summary.runs])


def write_parsed_file_summary_results(
    results_dir: Path,
    output_dir: Path,
    file_summaries: list[FileSummary],
) -> None:
    output_path = output_dir / "parsed_file_summary_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "method",
                "scenario_file",
                "source_file",
                "runs",
                "mean_error",
                "std_error",
                "mean_time",
                "std_time",
            ]
        )
        for summary in file_summaries:
            writer.writerow(
                [
                    summary.method,
                    summary.scenario_file,
                    source_file_for(results_dir, summary.path),
                    summary.runs,
                    summary.mean_error,
                    summary.std_error,
                    summary.mean_time,
                    summary.std_time,
                ]
            )


def write_summary_by_scenario(output_dir: Path, file_summaries: list[FileSummary]) -> None:
    output_path = output_dir / "summary_by_scenario.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "method",
                "scenario",
                "simulation",
                "speed",
                "runs",
                "mean_error",
                "std_error",
                "mean_search_time",
                "std_search_time",
            ]
        )
        for summary in file_summaries:
            writer.writerow(
                [
                    summary.method,
                    summary.scenario,
                    summary.simulation,
                    summary.speed,
                    summary.runs,
                    f"{summary.mean_error:.6f}",
                    f"{summary.std_error:.6f}",
                    f"{summary.mean_time:.6f}",
                    f"{summary.std_time:.6f}",
                ]
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate result summary CSV files from results/<method> raw CSVs."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing method subdirectories. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated summary CSVs. Defaults to results_dir.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_summaries = collect_file_summaries(results_dir)
    if not file_summaries:
        raise SystemExit(f"No raw CSV files found under method folders in {results_dir}")

    method_summaries = collect_method_summaries(file_summaries)
    write_summary_csv(output_dir, method_summaries)
    write_parsed_summary_results(output_dir, method_summaries)
    write_parsed_file_summary_results(results_dir, output_dir, file_summaries)
    write_summary_by_scenario(output_dir, file_summaries)

    print(f"Wrote summaries for {len(method_summaries)} methods and {len(file_summaries)} scenario files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
