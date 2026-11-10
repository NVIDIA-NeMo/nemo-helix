# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "nemo_customizer"
    / "skills"
    / "nemo-customizer"
    / "scripts"
    / "finance_report.py"
)
CAPABILITY_TASKS = {
    "visual_ocr": ("TAT-DQA/span", "TAT-DQA/multi-span"),
    "reasoning": ("TAT-DQA/arithmetic", "TAT-DQA/count"),
    "code_generation": ("CodeFinQA", "CodeTAT-QA"),
}


def _load_script():
    spec = importlib.util.spec_from_file_location("finance_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["finance_report"] = module
    spec.loader.exec_module(module)
    return module


finance_report = _load_script()


def cookbook_metrics(rows: list[dict], expected: int, tolerance: float) -> dict:
    """``metrics()`` from the cookbook's evaluation.py, the oracle for the report.

    Copied from NVIDIA-NeMo/Nemotron (Apache-2.0) at 18c74c09a,
    usage-cookbook/Nemotron-3.5-Super-VL/lora-finance/nemo-automodel/scripts/evaluation.py lines 73-107,
    with only line wrapping changed.
    """

    def summarize(items: list[dict]) -> dict:
        total = len(items)
        return {
            "samples": total,
            "exact_accuracy": sum(row["exact"] for row in items) / total if total else 0.0,
            "relaxed_accuracy": sum(row["relaxed"] for row in items) / total if total else 0.0,
            "closed_trace_rate": sum(row["trace_closed"] for row in items) / total if total else 0.0,
            "python_syntax_rate": (
                sum(row["program_syntax_valid"] for row in items if row["program_syntax_valid"] is not None)
                / sum(row["program_syntax_valid"] is not None for row in items)
                if any(row["program_syntax_valid"] is not None for row in items)
                else None
            ),
            "task_accuracy": sum(row["task_correct"] for row in items) / total if total else 0.0,
            "answer_f1": sum(row["answer_f1"] for row in items) / total if total else 0.0,
        }

    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped_capabilities: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
        grouped_capabilities[row["capability"]].append(row)
    result = summarize(rows)
    result.update(
        {
            "expected_samples": expected,
            "evaluated_samples": len(rows),
            "coverage": len(rows) / expected if expected else 0.0,
            "numeric_relative_tolerance": tolerance,
            "hit_max_new_tokens": sum(row["hit_max_new_tokens"] for row in rows),
            "by_capability": {name: summarize(items) for name, items in sorted(grouped_capabilities.items())},
            "by_task": {name: summarize(items) for name, items in sorted(grouped.items())},
        }
    )
    return result


def _cookbook_row(rng: random.Random, capability: str) -> dict[str, Any]:
    """One scored row as the cookbook records it."""
    task = rng.choice(CAPABILITY_TASKS[capability])
    exact = rng.random() < 0.4
    relaxed = exact or rng.random() < 0.3
    task_correct = relaxed if capability != "visual_ocr" else (exact or rng.random() < 0.2)
    return {
        "task": task,
        "capability": capability,
        "exact": exact,
        "relaxed": relaxed,
        "task_correct": task_correct,
        "answer_f1": rng.choice([0.0, 0.4, 2 / 3, 1.0]) if capability == "visual_ocr" else float(task_correct),
        "trace_closed": rng.random() < 0.9,
        "program_syntax_valid": (rng.random() < 0.8) if capability == "code_generation" else None,
        "hit_max_new_tokens": rng.random() < 0.1,
    }


def _row_score(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """The same row as a line of a job's row-scores JSONL from the finance-qa metric."""
    outputs = [{"name": name, "value": row[name]} for name in finance_report.SCORES]
    if row["program_syntax_valid"] is not None:
        outputs.append({"name": "program_syntax_valid", "value": row["program_syntax_valid"]})
    outputs.append({"name": "hit_max_new_tokens", "value": row["hit_max_new_tokens"]})
    outputs.append({"name": "task", "value": row["task"]})
    return {
        "row_index": row_index,
        "item": {"messages": []},
        "sample": {"output_text": "..."},
        "metrics": {"finance-qa": outputs},
        "requests": [],
    }


def _failed_row(row_index: int) -> dict[str, Any]:
    """A row whose inference failed with ignore_request_failure: NaN outputs and a metric error."""
    outputs = [{"name": name, "value": "NaN"} for name in (*finance_report.SCORES, "hit_max_new_tokens")]
    return {
        "row_index": row_index,
        "item": {"messages": []},
        "sample": {},
        "metrics": {"finance-qa": outputs},
        "requests": [],
        "metric_errors": {"finance-qa": "Inference request failed: timed out"},
    }


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_matches_the_cookbook_report(tmp_path: Path) -> None:
    rng = random.Random(1112)
    cookbook_rows: list[dict[str, Any]] = []
    argv: list[str] = []
    expected = 0
    for capability, count in (("visual_ocr", 120), ("reasoning", 90), ("code_generation", 150)):
        rows = [_cookbook_row(rng, capability) for _ in range(count)]
        records = [_row_score(row, index) for index, row in enumerate(rows)]
        records += [_failed_row(count + index) for index in range(rng.randint(1, 5))]
        expected += len(records)
        cookbook_rows += rows
        argv.append(f"{capability}={_write(tmp_path / f'{capability}.jsonl', records)}")

    output = tmp_path / "report.json"
    finance_report.main([*argv, "--output", str(output)])

    report = json.loads(output.read_text())
    assert report == cookbook_metrics(cookbook_rows, expected, 0.05)
    assert report["evaluated_samples"] == 360
    assert 0 < report["coverage"] < 1
    assert set(report["by_capability"]) == set(CAPABILITY_TASKS)
    assert len(report["by_task"]) == 6
    assert report["by_capability"]["visual_ocr"]["python_syntax_rate"] is None
    assert report["by_capability"]["code_generation"]["python_syntax_rate"] is not None


def test_rows_without_hit_max_new_tokens_count_as_not_hitting(tmp_path: Path) -> None:
    record = _row_score(_cookbook_row(random.Random(0), "reasoning"), 0)
    record["metrics"]["finance-qa"] = [o for o in record["metrics"]["finance-qa"] if o["name"] != "hit_max_new_tokens"]
    rows, total = finance_report.read_rows(_write(tmp_path / "rows.jsonl", [record]), "reasoning")
    assert total == 1
    assert rows[0]["hit_max_new_tokens"] is False


def test_rows_of_other_metrics_are_not_scored(tmp_path: Path) -> None:
    record = {"row_index": 0, "item": {}, "sample": {}, "metrics": {"string-check": []}, "requests": []}
    rows, total = finance_report.read_rows(_write(tmp_path / "rows.jsonl", [record, {}]), "reasoning")
    assert (rows, total) == ([], 2)


def test_a_row_without_a_task_label_is_an_error(tmp_path: Path) -> None:
    record = _row_score(_cookbook_row(random.Random(0), "reasoning"), 0)
    record["metrics"]["finance-qa"] = [o for o in record["metrics"]["finance-qa"] if o["name"] != "task"]
    with pytest.raises(ValueError, match="rows.jsonl:1: the finance-qa scores have no task label"):
        finance_report.read_rows(_write(tmp_path / "rows.jsonl", [record]), "reasoning")


def test_metric_key_can_be_renamed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record = _row_score(_cookbook_row(random.Random(0), "reasoning"), 0)
    record["metrics"] = {"finance": record["metrics"]["finance-qa"]}
    path = _write(tmp_path / "rows.jsonl", [record])
    finance_report.main([f"reasoning={path}", "--metric", "finance", "--tolerance", "0.01"])
    report = json.loads(capsys.readouterr().out)
    assert (report["evaluated_samples"], report["numeric_relative_tolerance"]) == (1, 0.01)


@pytest.mark.parametrize("source", ["ocr.jsonl", "=ocr.jsonl", "visual_ocr="])
def test_sources_need_a_capability_and_a_path(source: str) -> None:
    with pytest.raises(SystemExit):
        finance_report.main([source])
