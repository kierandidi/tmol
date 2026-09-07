#!/usr/bin/env python3
"""Summarize stable Nsight Compute metrics and optimization rules."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = (
    ("GPU Speed Of Light Throughput", "Duration"),
    ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
    ("GPU Speed Of Light Throughput", "Memory Throughput"),
    ("Launch Statistics", "Registers Per Thread"),
    ("Launch Statistics", "Waves Per SM"),
    ("Occupancy", "Theoretical Occupancy"),
    ("Occupancy", "Achieved Occupancy"),
    ("Memory Workload Analysis", "L1/TEX Hit Rate"),
    ("Memory Workload Analysis", "L2 Hit Rate"),
    ("Source Counters", "Branch Efficiency"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("details_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.details_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    lines = [
        "# Nsight Compute kernel summary",
        "",
        "Metrics and rule estimates describe the one profiled kernel launch; "
        "they are not whole-workflow speedup predictions.",
        "",
        "| Section | Metric | Value |",
        "|---|---|---:|",
    ]
    by_key = {(row["Section Name"], row["Metric Name"]): row for row in rows}
    for section, metric in METRICS:
        row = by_key.get((section, metric))
        if row is None:
            continue
        value = f"{row['Metric Value']} {row['Metric Unit']}".strip()
        lines.append(f"| {section} | {metric} | {value} |")

    rules = [row for row in rows if row.get("Rule Description")]
    if rules:
        lines += [
            "",
            "## Profiler rules",
            "",
            "| Scope | Rule | Estimated opportunity | Description |",
            "|---|---|---:|---|",
        ]
        for row in rules:
            estimate = row.get("Estimated Speedup", "")
            estimate_type = row.get("Estimated Speedup Type", "")
            opportunity = " ".join(filter(None, (estimate, estimate_type))) or "—"
            description = row["Rule Description"].replace("|", "\\|")
            lines.append(
                f"| {row['Rule Type']} | {row['Rule Name']} | "
                f"{opportunity} | {description} |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
