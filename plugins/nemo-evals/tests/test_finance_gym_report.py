# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finance Gym example's report turns Gym jobs' results bundles into the cookbook's ``metrics()`` shape."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/finance_gym"


def _load() -> ModuleType:
    # No bytecode: a `__pycache__` beside the example would be uploaded with it.
    writes_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location("finance_gym_report", EXAMPLE / "finance_report.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = writes_bytecode


finance_report = _load()

CORRECT = {
    "exact": 1.0,
    "relaxed": 1.0,
    "task_correct": 1.0,
    "answer_f1": 1.0,
    "trace_closed": 1.0,
    "hit_max_new_tokens": 0.0,
}
WRONG = {
    "exact": 0.0,
    "relaxed": 0.0,
    "task_correct": 0.0,
    "answer_f1": 0.5,
    "trace_closed": 0.0,
    "hit_max_new_tokens": 1.0,
}


def _bundle(root: Path, run_id: str, rows: list[tuple[str, str, dict[str, float] | None]]) -> Path:
    """A results bundle as a Gym job writes it: one task per row, and one score per trial.

    Each row is ``(task, capability, scores)``; ``None`` scores make a failed trial.
    """
    bundle = root / "agent-eval-results"
    bundle.mkdir(parents=True)
    tasks, scores = [], []
    for index, (task, capability, outputs) in enumerate(rows):
        task_id = f"gym-{run_id}-{index}"
        extras = {"expected_answer": "1", "task": task, "capability": capability, "row_id": f"{capability}/{index}"}
        tasks.append({"id": task_id, "inputs": {"gym_row": {}}, "metadata": {"gym_row_extras": extras}})
        record: dict[str, Any] = {
            "id": f"{run_id}:{task_id}:{task_id}:0:gym_reward",
            "metric_type": "gym_reward",
            "run_id": run_id,
            "task_id": task_id,
            "trial_id": f"{task_id}:0",
        }
        if outputs is None:
            record.update(status="failed", outputs=[], diagnostics=[{"severity": "error"}])
        else:
            values = {"reward": outputs["task_correct"], **outputs}
            record.update(status="completed", outputs=[{"name": k, "value": v} for k, v in values.items()])
        scores.append(record)
    for name, records in (("tasks.jsonl", tasks), ("scores.jsonl", scores)):
        (bundle / name).write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return root


def _tar(directory: Path, path: Path) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        archive.add(directory / "agent-eval-results", arcname="agent-eval-results")
    return path


def test_three_capability_jobs_make_the_cookbooks_report(tmp_path: Path) -> None:
    ocr = _bundle(
        tmp_path / "ocr",
        "run-ocr",
        [("TAT-DQA/span", "visual_ocr", CORRECT), ("TAT-DQA/multi-span", "visual_ocr", WRONG)],
    )
    reasoning = _bundle(
        tmp_path / "reasoning",
        "run-reasoning",
        [("TAT-DQA/arithmetic", "reasoning", CORRECT), ("TAT-DQA/count", "reasoning", None)],
    )
    code = _bundle(
        tmp_path / "code",
        "run-code",
        [
            ("CodeFinQA", "code_generation", {**CORRECT, "program_syntax_valid": 1.0}),
            ("CodeTAT-QA", "code_generation", {**WRONG, "program_syntax_valid": 0.0}),
        ],
    )
    output = tmp_path / "report.json"

    finance_report.main(
        [str(ocr), str(_tar(reasoning, tmp_path / "reasoning.tar.gz")), str(code), "--output", str(output)]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert {
        key: result[key] for key in ("expected_samples", "evaluated_samples", "coverage", "hit_max_new_tokens")
    } == {
        "expected_samples": 6,
        "evaluated_samples": 5,
        "coverage": pytest.approx(5 / 6),
        "hit_max_new_tokens": 2,
    }
    assert result["task_accuracy"] == pytest.approx(3 / 5)
    assert result["answer_f1"] == pytest.approx((1 + 0.5 + 1 + 1 + 0.5) / 5)
    # Only the code rows have a syntax score.
    assert result["python_syntax_rate"] == pytest.approx(0.5)
    assert result["by_capability"]["visual_ocr"]["python_syntax_rate"] is None
    assert result["numeric_relative_tolerance"] == 0.05
    assert sorted(result["by_capability"]) == ["code_generation", "reasoning", "visual_ocr"]
    assert result["by_capability"]["reasoning"]["samples"] == 1
    assert sorted(result["by_task"]) == [
        "CodeFinQA",
        "CodeTAT-QA",
        "TAT-DQA/arithmetic",
        "TAT-DQA/multi-span",
        "TAT-DQA/span",
    ]
    assert result["by_task"]["TAT-DQA/multi-span"] == {
        "samples": 1,
        "exact_accuracy": 0.0,
        "relaxed_accuracy": 0.0,
        "closed_trace_rate": 0.0,
        "python_syntax_rate": None,
        "task_accuracy": 0.0,
        "answer_f1": 0.5,
    }


def test_a_task_with_no_trial_still_counts_against_coverage(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "ocr", "run-ocr", [("TAT-DQA/span", "visual_ocr", CORRECT)])
    tasks = root / "agent-eval-results/tasks.jsonl"
    extras = {"expected_answer": "1", "task": "TAT-DQA/span", "capability": "visual_ocr", "row_id": "visual_ocr/9"}
    with tasks.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "gym-never-ran", "metadata": {"gym_row_extras": extras}}) + "\n")

    rows, expected, _ = finance_report.bundle_rows(root)

    assert (len(rows), expected) == (1, 2)


def test_a_job_with_no_trial_counts_all_its_rows_against_coverage(tmp_path: Path) -> None:
    ocr = _bundle(tmp_path / "ocr", "run-ocr", [("TAT-DQA/span", "visual_ocr", CORRECT)])
    reasoning = _bundle(
        tmp_path / "reasoning",
        "run-reasoning",
        [("TAT-DQA/arithmetic", "reasoning", CORRECT), ("TAT-DQA/count", "reasoning", CORRECT)],
    )
    (reasoning / "agent-eval-results/scores.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "report.json"

    finance_report.main([str(ocr), str(reasoning), "--output", str(output)])

    result = json.loads(output.read_text(encoding="utf-8"))
    assert (result["expected_samples"], result["evaluated_samples"]) == (3, 1)
    assert result["coverage"] == pytest.approx(1 / 3)
    assert sorted(result["by_capability"]) == ["visual_ocr"]


def test_the_same_job_twice_is_refused(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "ocr", "run-ocr", [("TAT-DQA/span", "visual_ocr", CORRECT)])

    with pytest.raises(SystemExit, match="run-ocr"):
        finance_report.main([str(root), str(root)])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("not_gym", "no gym_reward scores"),
        ("no_capability", "no task or capability"),
        ("no_component", "has no answer_f1"),
    ],
)
def test_a_bundle_the_report_cannot_read_is_refused(tmp_path: Path, change: str, message: str) -> None:
    root = _bundle(tmp_path / "ocr", "run-ocr", [("TAT-DQA/span", "visual_ocr", CORRECT)])
    bundle = root / "agent-eval-results"
    if change == "not_gym":
        scores = bundle / "scores.jsonl"
        scores.write_text(scores.read_text().replace('"gym_reward"', '"llm-judge"'), encoding="utf-8")
    elif change == "no_capability":
        tasks = bundle / "tasks.jsonl"
        tasks.write_text(tasks.read_text().replace('"capability": "visual_ocr", ', ""), encoding="utf-8")
    else:
        scores = bundle / "scores.jsonl"
        scores.write_text(scores.read_text().replace('"answer_f1"', '"other"'), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        finance_report.main([str(root)])


def test_a_directory_without_a_bundle_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not an agent-eval results bundle"):
        finance_report.main([str(tmp_path)])
