#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine the finance Gym jobs' results into the Nemotron finance cookbook's report.

Run one ``agent-evaluate`` job per capability, each against that capability's model or adapter, download each
job's results bundle, and pass the bundles here, as ``.tar.gz`` files or extracted directories:

    nemo jobs results download agent-eval-results --job <ocr-job> --output-file ocr.tar.gz
    nemo jobs results download agent-eval-results --job <reasoning-job> --output-file reasoning.tar.gz
    nemo jobs results download agent-eval-results --job <code-job> --output-file code.tar.gz
    python3 finance_report.py ocr.tar.gz reasoning.tar.gz code.tar.gz --output report.json

The report has the shape of ``metrics()`` in the cookbook's ``evaluation.py`` (NVIDIA-NeMo/Nemotron,
``usage-cookbook/Nemotron-3.5-Super-VL/lora-finance``): the overall summary, then ``by_capability`` and
``by_task``. Each summary gives the means of ``exact``, ``relaxed``, ``task_correct``, ``answer_f1`` and
``trace_closed``, and ``python_syntax_rate`` over the rows that have ``program_syntax_valid``.

- Each row's task and capability come from the Gym row, as ``convert_rows.py`` writes them.
- A failed trial is not scored, as the cookbook skips a failed request, but it counts toward
  ``expected_samples``: ``coverage`` is the rows scored over all rows.
- ``hit_max_new_tokens`` is a count of rows.
- ``numeric_relative_tolerance`` is ``--tolerance``, since the bundles don't record it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

#: The Gym runner's metric type in a bundle's scores.
METRIC_TYPE = "gym_reward"
SCORES = ("exact", "relaxed", "task_correct", "answer_f1", "trace_closed", "hit_max_new_tokens")
_BUNDLE_FILES = ("tasks.jsonl", "scores.jsonl")


def _jsonl(text: str, where: str) -> list[dict[str, Any]]:
    records = []
    # Splitting on "\n" only: str.splitlines() also splits on U+2028, which JSON can hold unescaped.
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}:{number}: not a JSON record: {error}") from error
    return records


def _shallowest(names: list[str], file_name: str) -> str | None:
    matches = [name for name in names if PurePosixPath(name).name == file_name]
    if not matches:
        return None
    return min(matches, key=lambda name: len(PurePosixPath(name).parts))


def read_bundle(path: Path) -> dict[str, list[dict[str, Any]]]:
    """A results bundle's ``tasks.jsonl`` and ``scores.jsonl``, from a ``.tar.gz`` or a directory."""
    files: dict[str, list[dict[str, Any]]] = {}
    if path.is_dir():
        names = [str(found.relative_to(path)) for found in path.rglob("*.jsonl")]
        for file_name in _BUNDLE_FILES:
            if (name := _shallowest(names, file_name)) is not None:
                files[file_name] = _jsonl((path / name).read_text(encoding="utf-8"), str(path / name))
    else:
        with tarfile.open(path) as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            for file_name in _BUNDLE_FILES:
                if (name := _shallowest(list(members), file_name)) is not None:
                    handle = archive.extractfile(members[name])
                    assert handle is not None  # members are regular files
                    text = io.TextIOWrapper(handle, encoding="utf-8").read()
                    files[file_name] = _jsonl(text, f"{path}!{name}")
    missing = [name for name in _BUNDLE_FILES if name not in files]
    if missing:
        raise ValueError(f"{path}: not an agent-eval results bundle; no {' or '.join(missing)}")
    return files


def bundle_rows(path: Path) -> tuple[list[dict[str, Any]], int, set[str]]:
    """The scored rows of one bundle, the number of rows expected, and the bundle's run IDs."""
    files = read_bundle(path)
    extras = {task["id"]: (task.get("metadata") or {}).get("gym_row_extras") or {} for task in files["tasks.jsonl"]}
    scores = [record for record in files["scores.jsonl"] if record.get("metric_type") == METRIC_TYPE]
    if not scores:
        raise ValueError(f"{path}: no {METRIC_TYPE} scores; is this a Gym job's results bundle?")
    rows: list[dict[str, Any]] = []
    tried: set[str] = set()
    for record in scores:
        task_id = record["task_id"]
        tried.add(task_id)
        row_extras = extras.get(task_id)
        if row_extras is None or "task" not in row_extras or "capability" not in row_extras:
            raise ValueError(f"{path}: task {task_id} has no task or capability; convert rows with convert_rows.py")
        outputs = {output["name"]: output["value"] for output in record.get("outputs") or []}
        if record.get("status") != "completed" or not outputs:
            continue
        missing = [name for name in SCORES if outputs.get(name) is None]
        if missing:
            raise ValueError(f"{path}: trial {record['trial_id']} has no {', '.join(missing)}; is this finance_qa?")
        row: dict[str, Any] = {name: float(outputs[name]) for name in SCORES}
        syntax = outputs.get("program_syntax_valid")
        row["program_syntax_valid"] = None if syntax is None else float(syntax)
        row.update(task=row_extras["task"], capability=row_extras["capability"])
        rows.append(row)
    # A task with no trial at all still counts: it's a row the job was asked to score.
    expected = len(scores) + len(set(extras) - tried)
    return rows, expected, {record["run_id"] for record in scores}


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
            "hit_max_new_tokens": int(sum(row["hit_max_new_tokens"] for row in rows)),
            "by_capability": {name: summarize(items) for name, items in sorted(by_capability.items())},
            "by_task": {name: summarize(items) for name, items in sorted(by_task.items())},
        }
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundles", nargs="+", type=Path, help="Each job's agent-eval-results bundle")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="The environment's numeric tolerance, recorded in the report (default 0.05, the environment's default)",
    )
    parser.add_argument("--output", type=Path, help="Write the report here instead of to stdout")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    expected = 0
    seen_runs: set[str] = set()
    try:
        for path in args.bundles:
            scored, total, runs = bundle_rows(path.expanduser())
            if repeated := runs & seen_runs:
                raise ValueError(f"{path}: run {sorted(repeated)[0]} is in an earlier bundle too")
            seen_runs |= runs
            rows.extend(scored)
            expected += total
    except ValueError as error:
        sys.exit(str(error))
    text = json.dumps(report(rows, expected, args.tolerance), indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
