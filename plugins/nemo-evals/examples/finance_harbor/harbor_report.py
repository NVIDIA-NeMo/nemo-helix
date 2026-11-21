#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the finance cookbook's report from a Harbor taskset run's downloaded results.

Reads every ``row_scores.jsonl`` a shard's verifier wrote under the run's download directory and pools the rows,
so the numbers are exact dataset means rather than means of shard means. The output has the keys
``../finance_gym/finance_report.py`` writes for the Gym route: overall, ``by_capability`` and ``by_task``.

    nemo evals runs download "$RUN_ID" --workspace default
    python3 harbor_report.py "$RUN_ID" --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: Report name for each component, as the cookbook's ``metrics()`` names them.
NAMES = {
    "task_correct": "task_accuracy",
    "answer_f1": "answer_f1",
    "trace_closed": "closed_trace_rate",
    "program_syntax_valid": "python_syntax_rate",
    "hit_max_new_tokens": "hit_max_new_tokens",
    "exact": "exact_match",
    "relaxed": "relaxed_match",
}


def read_rows(root: Path) -> list[dict[str, Any]]:
    """Every row of every shard under ``root``; a row id seen twice keeps its last attempt's line."""
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("row_scores.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["id"]] = row
    return list(rows.values())


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scored = [row for row in rows if row["status"] == "scored"]
    values: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        for name, value in row["scores"].items():
            values[name].append(value)
    summary: dict[str, Any] = {
        "expected_samples": len(rows),
        "scored_samples": len(scored),
        "coverage": len(scored) / len(rows) if rows else 0.0,
    }
    for name, report_name in NAMES.items():
        if values.get(name):
            summary[report_name] = sum(values[name]) / len(values[name])
    return summary


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_capability[row["capability"]].append(row)
        by_task[row["task"]].append(row)
    return {
        **summarize(rows),
        "by_capability": {name: summarize(group) for name, group in sorted(by_capability.items())},
        "by_task": {name: summarize(group) for name, group in sorted(by_task.items())},
        "failures": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows} - {"scored"})
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("download", type=Path, help="The directory `nemo evals runs download` wrote")
    parser.add_argument("--output", type=Path, help="Write the report here instead of stdout")
    args = parser.parse_args(argv)
    rows = read_rows(args.download)
    if not rows:
        sys.exit(f"{args.download}: no row_scores.jsonl found")
    text = json.dumps(report(rows), indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
