#!/usr/bin/env python3
"""Plot core scaling from rosetta_comparison.py JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

ENGINE_LABELS = {
    "tmol": "TMol",
    "pyrosetta": "PyRosetta",
    "rosetta": "Rosetta application",
}
MARKERS = {"tmol": "o", "pyrosetta": "s", "rosetta": "^"}


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


def _load_records(inputs: Iterable[Path]) -> pd.DataFrame:
    records = []
    for path in _json_paths(inputs):
        try:
            payload = json.loads(path.read_text())
            timing = payload["timing"]
            metadata = payload["engine_metadata"]
            record = {
                "path": str(path),
                "engine": payload["engine"],
                "workflow": payload["workflow"],
                "device": payload.get("device", "cpu"),
                "pdb": Path(payload["pdb"]).stem,
                "batch": int(payload["batch"]),
                "threads": int(payload["threads"]),
                "residues": metadata.get("n_residues"),
                "milliseconds_per_pose": 1000
                * float(timing["median_seconds_per_pose"]),
                "poses_per_second": float(timing["poses_per_second"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        records.append(record)
    if not records:
        raise ValueError("no rosetta_comparison.py result records found")
    frame = pd.DataFrame.from_records(records)
    frame["case"] = frame.apply(
        lambda row: f"{row.workflow}; {row.device}; {row.pdb}; B{row.batch}", axis=1
    )
    return frame.sort_values(
        ["workflow", "device", "pdb", "batch", "engine", "threads"]
    )


def _subplot_grid(workflows: list[str]):
    ncols = min(2, len(workflows))
    nrows = (len(workflows) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 4.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for axis in flat_axes[len(workflows) :]:
        axis.set_visible(False)
    return figure, flat_axes


def _series_label(engine: str, group: pd.DataFrame, multiple_cases: bool) -> str:
    label = ENGINE_LABELS.get(engine, engine)
    if multiple_cases:
        label += (
            f" ({group.iloc[0].device}, {group.iloc[0].pdb}, B{group.iloc[0].batch})"
        )
    return label


def _plot_scaling(
    frame: pd.DataFrame,
    value: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
    ideal: bool = False,
) -> None:
    workflows = list(dict.fromkeys(frame.workflow))
    figure, axes = _subplot_grid(workflows)
    for axis, workflow in zip(axes, workflows):
        workflow_frame = frame[frame.workflow == workflow]
        multiple_cases = workflow_frame.case.nunique() > 1
        for (engine, _), group in workflow_frame.groupby(
            ["engine", "case"], sort=False
        ):
            group = group.sort_values("threads")
            axis.plot(
                group.threads,
                group[value],
                marker=MARKERS.get(engine, "o"),
                linewidth=2,
                label=_series_label(engine, group, multiple_cases),
            )
        if ideal and not workflow_frame.empty:
            thread_counts = sorted(workflow_frame.threads.unique())
            axis.plot(
                thread_counts,
                thread_counts,
                linestyle="--",
                linewidth=1,
                color="0.5",
                label="ideal",
            )
        axis.set_title(workflow)
        axis.set_xlabel("Allocated CPU cores")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(workflow_frame.threads.unique()))
        axis.get_xaxis().set_major_formatter("{x:g}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"{filename}.{extension}", dpi=240)
    plt.close(figure)


def _derive_scaling(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["engine", "workflow", "device", "pdb", "batch"]
    baseline = (
        frame.sort_values("threads")
        .groupby(keys, as_index=False, sort=False)
        .first()[keys + ["threads", "milliseconds_per_pose"]]
        .rename(
            columns={
                "threads": "baseline_threads",
                "milliseconds_per_pose": "baseline_milliseconds_per_pose",
            }
        )
    )
    scaling = frame.merge(baseline, on=keys)
    scaling["speedup"] = (
        scaling.baseline_milliseconds_per_pose / scaling.milliseconds_per_pose
    )
    scaling["efficiency"] = scaling.speedup / (
        scaling.threads / scaling.baseline_threads
    )
    return scaling.sort_values(keys + ["threads"])


def _plot_tmol_advantage(frame: pd.DataFrame, output_dir: Path) -> None:
    competitors = frame[frame.engine != "tmol"]
    tmol = frame[frame.engine == "tmol"]
    keys = ["workflow", "device", "pdb", "batch", "threads"]
    merged = competitors.merge(tmol, on=keys, suffixes=("_competitor", "_tmol"))
    if merged.empty:
        return
    merged["tmol_speedup"] = (
        merged.milliseconds_per_pose_competitor / merged.milliseconds_per_pose_tmol
    )
    workflows = list(dict.fromkeys(merged.workflow))
    figure, axes = _subplot_grid(workflows)
    for axis, workflow in zip(axes, workflows):
        workflow_frame = merged[merged.workflow == workflow]
        multiple_cases = (
            workflow_frame[["device", "pdb", "batch"]].drop_duplicates().shape[0] > 1
        )
        for (engine, device, pdb, batch), group in workflow_frame.groupby(
            ["engine_competitor", "device", "pdb", "batch"], sort=False
        ):
            label = ENGINE_LABELS.get(engine, engine)
            if multiple_cases:
                label += f" ({device}, {pdb}, B{batch})"
            axis.plot(
                group.threads,
                group.tmol_speedup,
                marker=MARKERS.get(engine, "o"),
                linewidth=2,
                label=label,
            )
        axis.axhline(1, color="0.35", linestyle="--", linewidth=1)
        axis.set_title(workflow)
        axis.set_xlabel("Allocated CPU cores")
        axis.set_ylabel("TMol speedup over competitor")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(workflow_frame.threads.unique()))
        axis.get_xaxis().set_major_formatter("{x:g}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"tmol_advantage.{extension}", dpi=240)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_records(args.inputs)
    scaling = _derive_scaling(frame)
    frame.to_csv(args.output_dir / "scaling_summary.csv", index=False)
    scaling.to_csv(args.output_dir / "scaling_relative.csv", index=False)
    _plot_scaling(
        frame,
        "milliseconds_per_pose",
        "Median latency per pose (ms)",
        "latency_scaling",
        args.output_dir,
    )
    _plot_scaling(
        scaling,
        "speedup",
        "Speedup over smallest core count",
        "core_speedup",
        args.output_dir,
        ideal=True,
    )
    _plot_scaling(
        scaling,
        "efficiency",
        "Parallel efficiency",
        "parallel_efficiency",
        args.output_dir,
    )
    _plot_tmol_advantage(frame, args.output_dir)


if __name__ == "__main__":
    main()
