#!/usr/bin/env python3
"""Plot paired before/after TMol benchmark results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

CASE_KEYS = ["device", "workflow", "pdb", "batch", "threads"]


def _json_paths(inputs: Iterable[Path]) -> list[Path]:
    paths = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*.json")))
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise FileNotFoundError(input_path)
    return paths


def _load_variant(inputs: Iterable[Path], variant: str) -> pd.DataFrame:
    records = []
    for path in _json_paths(inputs):
        payload = json.loads(path.read_text())
        if payload.get("engine") != "tmol":
            continue
        records.append(
            {
                "variant": variant,
                "device": payload.get("device", "cpu"),
                "workflow": payload["workflow"],
                "pdb": Path(payload["pdb"]).stem,
                "batch": int(payload["batch"]),
                "threads": int(payload["threads"]),
                "milliseconds_per_pose": 1000
                * float(payload["timing"]["median_seconds_per_pose"]),
            }
        )
    if not records:
        raise ValueError(f"no TMol records found for {variant}")
    return pd.DataFrame.from_records(records)


def _median_of_runs(series: pd.Series) -> float:
    return statistics.median(float(value) for value in series)


def _comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    baseline = (
        baseline.groupby(CASE_KEYS, as_index=False)
        .milliseconds_per_pose.agg(_median_of_runs)
        .rename(columns={"milliseconds_per_pose": "baseline_milliseconds_per_pose"})
    )
    candidate = (
        candidate.groupby(CASE_KEYS, as_index=False)
        .milliseconds_per_pose.agg(_median_of_runs)
        .rename(columns={"milliseconds_per_pose": "candidate_milliseconds_per_pose"})
    )
    comparison = baseline.merge(candidate, on=CASE_KEYS, validate="one_to_one")
    comparison["speedup"] = (
        comparison.baseline_milliseconds_per_pose
        / comparison.candidate_milliseconds_per_pose
    )
    comparison["percent_change"] = 100 * (comparison.speedup - 1)
    comparison["baseline_label"] = baseline_label
    comparison["candidate_label"] = candidate_label
    comparison["case"] = comparison.apply(
        lambda row: f"{row.workflow}; {row.pdb}; B{row.batch}; T{row.threads}",
        axis=1,
    )
    return comparison.sort_values(CASE_KEYS)


def _plot(comparison: pd.DataFrame, candidate_label: str, output_dir: Path) -> None:
    devices = list(dict.fromkeys(comparison.device))
    figure, axes = plt.subplots(
        1,
        len(devices),
        figsize=(8 * len(devices), max(4.5, 0.42 * len(comparison))),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, device in zip(axes.flat, devices):
        selected = comparison[comparison.device == device].copy()
        colors = [
            "#2ca02c" if value >= 0 else "#d62728" for value in selected.percent_change
        ]
        axis.barh(selected.case, selected.percent_change, color=colors)
        axis.axvline(0, color="0.25", linewidth=1)
        axis.set_title(device.upper())
        axis.set_xlabel(f"{candidate_label} throughput change (%)")
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"optimization_speedup.{extension}", dpi=240)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_variant(args.baseline, args.baseline_label)
    candidate = _load_variant(args.candidate, args.candidate_label)
    comparison = _comparison(
        baseline, candidate, args.baseline_label, args.candidate_label
    )
    if args.device is not None:
        comparison = comparison[comparison.device == args.device]
        if comparison.empty:
            raise ValueError(f"no paired {args.device} records found")
    comparison.to_csv(args.output_dir / "optimization_summary.csv", index=False)
    _plot(comparison, args.candidate_label, args.output_dir)


if __name__ == "__main__":
    main()
