#!/usr/bin/env python3
"""Plot independent-process throughput scaling for TMol and PyRosetta."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESS_PATTERN = re.compile(r"-t(?P<processes>[0-9]+)-r[0-9]+[.]json$")
ENGINE_LABELS = {"tmol": "TMol", "pyrosetta": "PyRosetta"}
MARKERS = {"tmol": "o", "pyrosetta": "s"}


def _load_process_results(path: Path, engine: str) -> pd.DataFrame:
    records = []
    for result_path in sorted(path.rglob("*.json")):
        match = PROCESS_PATTERN.search(result_path.name)
        if match is None:
            continue
        payload = json.loads(result_path.read_text())
        if payload.get("engine") != engine:
            continue
        records.append(
            {
                "engine": engine,
                "workflow": payload["workflow"],
                "pdb": Path(payload["pdb"]).stem,
                "processes": int(match.group("processes")),
                "worker_poses_per_second": float(payload["timing"]["poses_per_second"]),
            }
        )
    if not records:
        raise ValueError(f"no {engine} process-scaling results found in {path}")
    return pd.DataFrame.from_records(records)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["engine", "workflow", "pdb", "processes"]
    aggregate = (
        frame.groupby(keys, as_index=False)
        .agg(
            poses_per_second=("worker_poses_per_second", "sum"),
            workers_measured=("worker_poses_per_second", "size"),
        )
        .sort_values(keys)
    )
    incomplete = aggregate[aggregate.workers_measured != aggregate.processes]
    if not incomplete.empty:
        cases = incomplete[keys + ["workers_measured"]].to_dict("records")
        raise ValueError(f"incomplete process groups: {cases}")
    baselines = (
        aggregate.sort_values("processes")
        .groupby(["engine", "workflow", "pdb"], as_index=False)
        .first()[["engine", "workflow", "pdb", "poses_per_second"]]
        .rename(columns={"poses_per_second": "baseline_poses_per_second"})
    )
    aggregate = aggregate.merge(baselines, on=["engine", "workflow", "pdb"])
    aggregate["speedup"] = (
        aggregate.poses_per_second / aggregate.baseline_poses_per_second
    )
    aggregate["parallel_efficiency"] = aggregate.speedup / aggregate.processes
    return aggregate


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


def _plot(
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
        for engine, group in workflow_frame.groupby("engine", sort=False):
            group = group.sort_values("processes")
            axis.plot(
                group.processes,
                group[value],
                marker=MARKERS[engine],
                linewidth=2,
                label=ENGINE_LABELS[engine],
            )
        if ideal:
            process_counts = sorted(workflow_frame.processes.unique())
            axis.plot(
                process_counts,
                process_counts,
                color="0.5",
                linestyle="--",
                linewidth=1,
                label="ideal",
            )
        axis.set_title(workflow)
        axis.set_xlabel("One-core worker processes")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(workflow_frame.processes.unique()))
        axis.get_xaxis().set_major_formatter("{x:g}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"{filename}.{extension}", dpi=240)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmol", type=Path, required=True)
    parser.add_argument("--pyrosetta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(
        [
            _load_process_results(args.tmol, "tmol"),
            _load_process_results(args.pyrosetta, "pyrosetta"),
        ],
        ignore_index=True,
    )
    aggregate = _aggregate(raw)
    aggregate.to_csv(args.output_dir / "process_scaling_summary.csv", index=False)
    _plot(
        aggregate,
        "poses_per_second",
        "Aggregate throughput (poses/s)",
        "process_throughput",
        args.output_dir,
    )
    _plot(
        aggregate,
        "speedup",
        "Speedup over one process",
        "process_speedup",
        args.output_dir,
        ideal=True,
    )
    _plot(
        aggregate,
        "parallel_efficiency",
        "Parallel efficiency",
        "process_efficiency",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
