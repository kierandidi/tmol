#!/usr/bin/env python3
"""Summarize the CPU score-term worker-count sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


NAME = re.compile(
    r"score-grad-protein(?P<size>\d+)-t(?P<threads>\d+)"
    r"-s(?P<strategy>[^-]+)-w(?P<workers>\d+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input.glob("*.json")):
        match = NAME.fullmatch(path.stem)
        if match is None:
            continue
        payload = json.loads(path.read_text())
        timing = payload["timing"]
        rows.append(
            {
                "nominal_residues": int(match["size"]),
                "threads": int(match["threads"]),
                "strategy": match["strategy"],
                "requested_term_workers": int(match["workers"]),
                "median_ms": 1000 * timing["median_seconds"],
                "poses_per_second": timing["poses_per_second"],
                "source_file": str(path),
            }
        )

    rows.sort(
        key=lambda row: (
            row["nominal_residues"],
            row["threads"],
            row["strategy"],
            row["requested_term_workers"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
