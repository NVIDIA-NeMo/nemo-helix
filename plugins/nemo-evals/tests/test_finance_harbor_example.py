# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finance Harbor example: shard tasks, the in-sandbox metric, and the pooled report."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from harbor.models.task.config import TaskConfig
from harbor.models.task.verifier_mode import task_has_any_separate_verifier
from nhx_evals_sdk.agent_eval.runtimes.harbor.dataset_runner import parse_shard

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "finance_harbor"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def make_tasks() -> ModuleType:
    return _load("finance_harbor_make_tasks", EXAMPLE / "make_tasks.py")


@pytest.fixture(scope="module")
def score() -> ModuleType:
    # The metric imports `scoring` and `cookbook` as top-level modules, as it does inside its image.
    scorer = str(EXAMPLE.parent / "finance_gym" / "environment" / "resources_servers" / "finance_qa")
    sys.path.insert(0, scorer)
    try:
        return _load("finance_harbor_score", EXAMPLE / "metric" / "score.py")
    finally:
        sys.path.remove(scorer)


@pytest.fixture(scope="module")
def harbor_report() -> ModuleType:
    return _load("finance_harbor_report", EXAMPLE / "harbor_report.py")


def _gym_row(index: int, capability: str, task: str, *, image: str | None = None) -> dict[str, Any]:
    content: Any = f"Task: {task}\nQuestion {index}"
    if image is not None:
        content = [
            {"type": "input_image", "image_url": image, "detail": "auto"},
            {"type": "input_text", "text": content},
        ]
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_output_tokens": 256,
        },
        "expected_answer": str(index),
        "task": task,
        "capability": capability,
        "row_id": f"{capability}/{index}",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_each_shard_is_a_task_whose_verifier_shares_the_sandbox(make_tasks: ModuleType, tmp_path: Path) -> None:
    reasoning = _write_rows(
        tmp_path / "reasoning.jsonl", [_gym_row(i, "reasoning", "TAT-DQA/arithmetic") for i in range(3)]
    )
    ocr = _write_rows(
        tmp_path / "ocr.jsonl",
        [_gym_row(0, "visual_ocr", "TAT-DQA/span", image="fileset://default/finance-images#images/a.png")],
    )

    written = make_tasks.make_tasks(
        [reasoning, ocr], tmp_path / "out", shard_size=2, prefix="finance", agent_timeout_sec=60
    )

    assert [task.name for task in written] == [
        "finance-reasoning-0000",
        "finance-reasoning-0001",
        "finance-visual-ocr-0000",
    ]
    for task in written:
        config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
        assert not task_has_any_separate_verifier(config)
        assert config.environment.network_mode.value == "no-network"
        # One build for the dataset: every task's image context holds the same bytes.
        assert {f.name: f.read_bytes() for f in (task / "environment").iterdir()} == {
            f.name: f.read_bytes() for f in (written[0] / "environment").iterdir()
        }
    first = parse_shard((written[0] / "instruction.md").read_text())
    assert [row.id for row in first.rows] == ["reasoning/0", "reasoning/1"]
    assert first.rows[0].params == {"temperature": 0, "max_tokens": 256}
    image_row = parse_shard((written[2] / "instruction.md").read_text()).rows[0]
    assert image_row.messages[0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "fileset://default/finance-images#images/a.png", "detail": "auto"},
    }
    references = (written[1] / "tests" / "references.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in references] == ["reasoning/2"]


def test_a_shard_too_large_for_one_instruction_is_refused(make_tasks: ModuleType, tmp_path: Path) -> None:
    big = _gym_row(0, "reasoning", "TAT-DQA/arithmetic")
    big["responses_create_params"]["input"][0]["content"] += "x" * (1024 * 1024)
    source = _write_rows(tmp_path / "big.jsonl", [big])
    with pytest.raises(ValueError, match="lower --shard-size"):
        make_tasks.make_tasks([source], tmp_path / "out", shard_size=1, prefix="f", agent_timeout_sec=60)


def _completion(content: str, *, reasoning: str | None = None, tokens: int = 10) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message}], "usage": {"completion_tokens": tokens}}


def test_the_metric_scores_answered_rows_and_reports_the_rest(score: ModuleType) -> None:
    references = [
        {"id": f"r/{i}", "expected_answer": "42", "task": "TAT-DQA/arithmetic", "capability": "reasoning"}
        for i in range(4)
    ]
    responses = [
        {"id": "r/0", "status": "ok", "response": _completion("<think>sum</think>\n42")},
        {"id": "r/1", "status": "ok", "response": _completion("<think>guess</think>\n7")},
        {"id": "r/2", "status": "timeout", "response": None, "error": "slow"},
    ]

    rewards, rows, summary = score.score_shard(responses, references)

    assert [row["status"] for row in rows] == ["scored", "scored", "timeout", "missing"]
    assert rewards["reward"] == rewards["task_correct"] == 0.5
    assert summary["rows"] == 4 and summary["scored"] == 2
    assert summary["sums"]["task_correct"] == 1.0 and summary["counts"]["task_correct"] == 2


def test_the_metric_reads_a_completion_as_gym_inference_provider_does(score: ModuleType) -> None:
    """Without a reasoning parser, Gym keeps the message content and drops ``reasoning_content``."""
    response = score.gym_response(_completion("42", reasoning="long trace", tokens=99))
    assert response == {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}],
        "usage": {"output_tokens": 99},
    }


def test_the_report_pools_rows_across_shards(harbor_report: ModuleType, tmp_path: Path) -> None:
    def shard(name: str, rows: list[dict[str, Any]]) -> None:
        directory = tmp_path / name / "verifier"
        directory.mkdir(parents=True)
        (directory / "row_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    scored = {"task_correct": 1.0, "answer_f1": 1.0}
    shard("a", [{"id": "x/0", "task": "T1", "capability": "x", "status": "scored", "scores": scored}])
    shard(
        "b",
        [
            {
                "id": "x/1",
                "task": "T1",
                "capability": "x",
                "status": "scored",
                "scores": {**scored, "task_correct": 0.0},
            },
            {
                "id": "x/2",
                "task": "T1",
                "capability": "x",
                "status": "scored",
                "scores": {**scored, "task_correct": 0.0},
            },
            {"id": "x/3", "task": "T1", "capability": "x", "status": "error", "scores": None},
        ],
    )

    report = harbor_report.report(harbor_report.read_rows(tmp_path))

    # Pooled over rows, not shards: a mean of shard means would give 0.5.
    assert report["task_accuracy"] == pytest.approx(1 / 3)
    assert (report["expected_samples"], report["scored_samples"], report["coverage"]) == (4, 3, 0.75)
    assert report["failures"] == {"error": 1}
    assert report["by_task"]["T1"]["task_accuracy"] == pytest.approx(1 / 3)
