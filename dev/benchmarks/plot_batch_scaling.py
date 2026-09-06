#!/usr/bin/env python3
"""Plot latency, throughput, and cold-call cost across benchmark batch sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


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


def _mode(payload: dict) -> str:
    if payload["engine"] != "tmol":
        return "pose loop"
    metadata = payload.get("engine_metadata", {})
    if not metadata.get("reuse_topology"):
        return "one-shot"
    return "reuse + CUDA graph" if metadata.get("cuda_graph") else "reuse"


def _load(inputs: Iterable[Path]) -> pd.DataFrame:
    records = []
    for path in _json_paths(inputs):
        try:
            payload = json.loads(path.read_text())
            timing = payload["timing"]
            first_call = payload.get("first_call_seconds")
            records.append(
                {
                    "path": str(path),
                    "engine": payload["engine"],
                    "device": payload.get("device", "cpu"),
                    "workflow": payload["workflow"],
                    "pdb": Path(payload["pdb"]).stem,
                    "batch": int(payload["batch"]),
                    "threads": int(payload["threads"]),
                    "mode": _mode(payload),
                    "total_milliseconds": 1000 * float(timing["median_seconds"]),
                    "milliseconds_per_pose": 1000
                    * float(timing["median_seconds_per_pose"]),
                    "poses_per_second": float(timing["poses_per_second"]),
                    "first_call_milliseconds": (
                        1000 * float(first_call) if first_call is not None else None
                    ),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not records:
        raise ValueError("no benchmark result records found")
    frame = pd.DataFrame.from_records(records)
    frame["series"] = frame.apply(
        lambda row: (
            f"{row.engine} {row.device}; {row['mode']}; T{row.threads}; {row.pdb}"
        ),
        axis=1,
    )
    frame["cold_to_steady_ratio"] = (
        frame.first_call_milliseconds / frame.total_milliseconds
    )
    return frame.sort_values(["workflow", "series", "batch"])


def _plot_metric(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    workflows = list(dict.fromkeys(frame.workflow))
    figure, axes = plt.subplots(
        1,
        len(workflows),
        figsize=(7.2 * len(workflows), 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, workflow in zip(axes.flat, workflows):
        selected = frame[frame.workflow == workflow]
        for series, group in selected.groupby("series", sort=False):
            # Profiled cases may duplicate an ordinary timing record. Collapse
            # repeated runs instead of drawing multiple values at one x point.
            group = (
                group.groupby("batch", as_index=False)[column]
                .median()
                .sort_values("batch")
            )
            axis.plot(group.batch, group[column], marker="o", linewidth=2, label=series)
        axis.set_title(workflow)
        axis.set_xlabel("Pose batch size")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(selected.batch.unique()))
        axis.get_xaxis().set_major_formatter("{x:g}")
        if (selected[column].dropna() > 0).all():
            axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"{filename}.{extension}", dpi=240)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load(args.inputs)
    frame.to_csv(args.output_dir / "batch_scaling_summary.csv", index=False)
    _plot_metric(
        frame,
        "poses_per_second",
        "Throughput (poses/s)",
        "batch_throughput",
        args.output_dir,
    )
    _plot_metric(
        frame,
        "milliseconds_per_pose",
        "Median latency per pose (ms)",
        "batch_latency_per_pose",
        args.output_dir,
    )
    _plot_metric(
        frame,
        "total_milliseconds",
        "Median total call latency (ms)",
        "batch_total_latency",
        args.output_dir,
    )
    _plot_metric(
        frame,
        "cold_to_steady_ratio",
        "First-call / steady-state latency",
        "batch_cold_overhead",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
