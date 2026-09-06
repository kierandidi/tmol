#!/usr/bin/env python3
"""Summarize host/operator and CUDA launch event counts from Chrome traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_CATEGORIES = {
    "cpu_op": "CPU operator events",
    "cuda_runtime": "CUDA runtime calls",
    "kernel": "CUDA kernel events",
}


def _profile(value: str) -> tuple[str, Path]:
    try:
        label, path_string = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("profiles must use LABEL=PATH") from error
    if not label:
        raise argparse.ArgumentTypeError("profile label cannot be empty")
    path = Path(path_string)
    if path.is_dir():
        path = path / "trace.json"
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"trace not found: {path}")
    return label, path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        type=_profile,
        metavar="LABEL=PATH",
        help="Chrome trace file or profile directory; repeat for comparisons",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = []
    profile_labels = []
    for label, path in args.profile:
        profile_labels.append(label)
        payload = json.loads(path.read_text())
        counts = Counter(event.get("cat") for event in payload["traceEvents"])
        for category, display_name in _CATEGORIES.items():
            records.append(
                {
                    "profile": label,
                    "category": display_name,
                    "event_count": counts[category],
                    "trace": str(path),
                }
            )

    frame = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "profile_event_counts.csv", index=False)

    pivot = frame.pivot(index="category", columns="profile", values="event_count")
    pivot = pivot.reindex(index=_CATEGORIES.values(), columns=profile_labels)
    axis = pivot.plot.bar(figsize=(9.0, 5.2), width=0.78)
    axis.set_xlabel("")
    axis.set_ylabel("Events in one profiled minimization")
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Execution mode")
    for container in axis.containers:
        axis.bar_label(container, fmt="{:,.0f}", padding=3, fontsize="small")
    figure = axis.get_figure()
    figure.tight_layout()
    for extension in ("png", "svg", "pdf"):
        figure.savefig(args.output_dir / f"profile_event_counts.{extension}", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    main()
