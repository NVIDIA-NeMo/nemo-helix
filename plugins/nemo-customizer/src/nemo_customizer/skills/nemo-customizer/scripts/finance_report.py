#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine finance eval jobs' row scores into the Nemotron finance cookbook's report.

Score each capability's test set in its own eval job with the ``finance-qa`` metric, download each
job's row scores, and pass them here labeled by capability:

    nemo jobs results download row-scores --job <ocr-job> --output-file ocr.jsonl
    nemo jobs results download row-scores --job <reasoning-job> --output-file reasoning.jsonl
    nemo jobs results download row-scores --job <code-job> --output-file code.jsonl
    python3 finance_report.py visual_ocr=ocr.jsonl reasoning=reasoning.jsonl \\
      code_generation=code.jsonl --output report.json

The report has the shape of ``metrics()`` in the cookbook's ``evaluation.py`` (NVIDIA-NeMo/Nemotron,
``usage-cookbook/Nemotron-3.5-Super-VL/lora-finance``): the overall summary, then
``by_capability`` and ``by_task``. Each summary gives the means of ``exact``, ``relaxed``,
``task_correct``, ``answer_f1`` and ``trace_closed``, and ``python_syntax_rate`` over the rows that
have ``program_syntax_valid``. ``coverage`` is the rows scored over all rows in the files. A row whose
inference or scoring failed is not scored, as in the cookbook. ``hit_max_new_tokens`` is a count;
a row without that score, because the job set no ``max_new_tokens``, counts as not hitting it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC = "finance-qa"
SCORES = ("exact", "relaxed", "task_correct", "answer_f1", "trace_closed")


def _flag(value: Any) -> bool | None:
    """A boolean score from a row-scores file, or None when it is missing or NaN."""
    if value is None or value == "NaN" or (isinstance(value, float) and math.isnan(value)):
        return None
    return bool(value)


def read_rows(path: Path, capability: str, metric: str = METRIC) -> tuple[list[dict[str, Any]], int]:
    """Return the scored rows of one row-scores file, and the number of rows in it."""
    rows: list[dict[str, Any]] = []
    total = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        record = json.loads(line)
        if (record.get("metric_errors") or {}).get(metric):
            continue
        outputs = {output["name"]: output["value"] for output in (record.get("metrics") or {}).get(metric) or []}
        if not outputs:
            continue
        if not isinstance(outputs.get("task"), str):
            raise ValueError(f"{path}:{number}: the {metric} scores have no task label")
        row: dict[str, Any] = {"task": outputs["task"], "capability": capability}
        for name in SCORES:
            if name == "answer_f1":
                row[name] = float(outputs[name])
            elif (flag := _flag(outputs.get(name))) is None:
                raise ValueError(f"{path}:{number}: the {metric} scores have no {name}")
            else:
                row[name] = flag
        row["program_syntax_valid"] = _flag(outputs.get("program_syntax_valid"))
        row["hit_max_new_tokens"] = bool(_flag(outputs.get("hit_max_new_tokens")))
        rows.append(row)
    return rows, total


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The cookbook's summary of a group of scored rows."""
    total = len(items)
    syntax = [row["program_syntax_valid"] for row in items if row["program_syntax_valid"] is not None]

    def mean(name: str) -> float:
        return sum(row[name] for row in items) / total if total else 0.0

    return {
        "samples": total,
        "exact_accuracy": mean("exact"),
        "relaxed_accuracy": mean("relaxed"),
        "closed_trace_rate": mean("trace_closed"),
        "python_syntax_rate": sum(syntax) / len(syntax) if syntax else None,
        "task_accuracy": mean("task_correct"),
        "answer_f1": mean("answer_f1"),
    }


def report(rows: list[dict[str, Any]], expected: int, tolerance: float) -> dict[str, Any]:
    """The cookbook's ``metrics()`` report: overall, then by capability and by task."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
        by_capability[row["capability"]].append(row)
    result = summarize(rows)
    result.update(
        {
            "expected_samples": expected,
            "evaluated_samples": len(rows),
            "coverage": len(rows) / expected if expected else 0.0,
            "numeric_relative_tolerance": tolerance,
            "hit_max_new_tokens": sum(row["hit_max_new_tokens"] for row in rows),
            "by_capability": {name: summarize(items) for name, items in sorted(by_capability.items())},
            "by_task": {name: summarize(items) for name, items in sorted(by_task.items())},
        }
    )
    return result


def _source(value: str) -> tuple[str, Path]:
    capability, separator, path = value.partition("=")
    if not separator or not capability or not path:
        raise argparse.ArgumentTypeError(f"expected CAPABILITY=PATH, got {value!r}")
    return capability, Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sources",
        nargs="+",
        type=_source,
        metavar="CAPABILITY=PATH",
        help="A capability name and its job's row-scores JSONL, for example visual_ocr=ocr.jsonl",
    )
    parser.add_argument("--metric", default=METRIC, help=f"The metric key in the row scores (default {METRIC})")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="The metric's relative tolerance, recorded in the report (default 0.05)",
    )
    parser.add_argument("--output", type=Path, help="Write the report here instead of to stdout")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    expected = 0
    for capability, path in args.sources:
        scored, total = read_rows(path, capability, args.metric)
        rows.extend(scored)
        expected += total
    text = json.dumps(report(rows, expected, args.tolerance), indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
