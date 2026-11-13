# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finance Gym example's scorer gives the cookbook's scores for a Gym rollout.

``cookbook.py`` is the cookbook's scoring code, unchanged. What can go wrong is the wrapping: rebuilding the chat
message the cookbook reads from a Responses API response, the per-task token limit, and the reward components.
So each case is scored twice, through ``scoring.score`` on a Gym response and through ``cookbook.cookbook_row`` on
the chat message that response stands for, and the two must agree.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SERVER = Path(__file__).resolve().parents[1] / "examples/finance_gym/environment/resources_servers/finance_qa"


def _load(name: str) -> ModuleType:
    # The scorer imports its sibling `cookbook` the way Gym runs it, from its own directory. No bytecode: a
    # `__pycache__` beside the example would be uploaded with the environment.
    sys.path.insert(0, str(SERVER))
    writes_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(f"finance_gym_{name}", SERVER / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVER))
        sys.dont_write_bytecode = writes_bytecode


cookbook = _load("cookbook")
scoring = _load("scoring")

TASKS = {
    "TAT-DQA/span": "tat_dqa",
    "TAT-DQA/multi-span": "tat_dqa",
    "TAT-DQA/arithmetic": "tat_dqa",
    "TAT-DQA/count": "tat_dqa",
    "CodeFinQA": "program_synthesis",
    "CodeTAT-QA": "program_synthesis",
}


def _cookbook_limit(task: str) -> int:
    """The cookbook's per-task completion limit (evaluation.py lines 428-431, with its default 768)."""
    return min(768, 256 if task.endswith(("arithmetic", "count")) else 512)


def _response(texts: list[str], *, reasoning: str | None = None, field: str = "summary", tokens: int | None) -> dict:
    """A Responses API response as Gym records it: an optional reasoning item, then the output text."""
    output: list[dict[str, Any]] = []
    if reasoning is not None:
        output.append({"type": "reasoning", "id": "rs_1", field: [{"type": f"{field}_text", "text": reasoning}]})
    output.append(
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []} for text in texts],
        }
    )
    usage = None if tokens is None else {"input_tokens": 600, "output_tokens": tokens, "total_tokens": 600 + tokens}
    return {"id": "resp_1", "object": "response", "output": output, "usage": usage}


def _cookbook_scores(task: str, reference: str, message: dict[str, Any], tokens: int | None) -> dict[str, float]:
    row = cookbook.cookbook_row(
        task=task,
        task_family=TASKS[task],
        answer=reference,
        response={"choices": [{"message": message}], "usage": {"completion_tokens": tokens}},
        task_limit=_cookbook_limit(task),
        tolerance=0.05,
    )
    scores = {name: float(value) for name, value in row.items() if value is not None}
    return scores


def _number(rng: random.Random) -> float:
    return round(rng.uniform(-5000, 5000), rng.choice([0, 1, 2, 3]))


def _format_number(rng: random.Random, value: float, scale: str) -> str:
    text = rng.choice([f"{value:,}", f"{value}", f"{value:.2f}", f"{value:,.1f}"])
    if rng.random() < 0.2:
        text = f"${text}" if not text.startswith("-") else f"-${text[1:]}"
    if scale:
        text = f"{text}{rng.choice(['', ' '])}{scale}"
    return rng.choice([text, f"{text}.", f" {text} ", text.upper()])


def _answer_pair(rng: random.Random, task: str) -> tuple[str, str]:
    """A reference and a prediction that is often right, often close, and sometimes unrelated."""
    words = ["revenue", "Net Sales", "cost of goods", "2019", "2018", "EBITDA", "the total", "Q4"]
    if task.endswith("span"):
        count = rng.randint(2, 3) if task.endswith("multi-span") else 1
        spans = rng.sample(words, count)
        reference = rng.choice([" | ", "##", "; "]).join(spans)
        predicted = rng.sample(spans, len(spans)) if rng.random() < 0.6 else rng.sample(words, count)
        if rng.random() < 0.3:
            predicted = [span.upper() for span in predicted]
        return reference, rng.choice([" | ", "##", ";", ", "]).join(predicted)
    value = _number(rng)
    scale = rng.choice(["", "", "%", "percent", "thousand", "million", "billion"])
    predicted = value * rng.choice([1, 1, 1.03, 0.96, 1.051, 10])
    predicted_scale = scale if rng.random() < 0.8 else rng.choice(["", "%", "million"])
    return _format_number(rng, value, scale), _format_number(rng, round(predicted, 4), predicted_scale)


def _case(rng: random.Random) -> tuple[str, str, dict[str, Any], dict[str, Any], int | None]:
    """A task, its reference, a Gym response, the chat message it stands for, and its token count."""
    task = rng.choice(list(TASKS))
    reference, prediction = _answer_pair(rng, task)
    reasoning = rng.choice(["revenue = 120\nanswer = revenue * 2", "answer = (120", "Read the table.", ""])
    tokens = rng.choice([None, 100, 255, 256, 300, 511, 512])
    form = rng.randrange(6)
    if form in (0, 1):
        # A reasoning parser split the trace into its own output item.
        field = "summary" if form == 0 else "content"
        response = _response([prediction], reasoning=reasoning, field=field, tokens=tokens)
        return task, reference, response, {"content": prediction, "reasoning_content": reasoning}, tokens
    text = {
        2: f"{reasoning}\n</think>\n{prediction}",
        3: f"<think>\n{reasoning}\n</think>\n\n{prediction}",
        4: f"<think>\n{reasoning} and then",
        5: prediction,
    }[form]
    # Form 5 also splits the text across two parts, which the scorer must join.
    parts = [text[: len(text) // 2], text[len(text) // 2 :]] if form == 5 else [text]
    return task, reference, _response(parts, tokens=tokens), {"content": text}, tokens


def test_the_scorer_agrees_with_the_cookbook_on_generated_rows() -> None:
    rng = random.Random(1112)
    for _ in range(600):
        task, reference, response, message, tokens = _case(rng)

        scores = scoring.score(task=task, expected_answer=reference, response=response)

        assert scores == pytest.approx(_cookbook_scores(task, reference, message, tokens)), (task, response)


def test_generated_rows_reach_every_outcome() -> None:
    """The comparison above isn't vacuous: every score takes each of its values somewhere."""
    rng = random.Random(1112)
    seen: dict[str, set[float]] = {}
    partial_f1 = False
    for _ in range(600):
        task, reference, response, _, _ = _case(rng)
        for name, value in scoring.score(task=task, expected_answer=reference, response=response).items():
            if name == "answer_f1" and not math.isclose(value, round(value)):
                partial_f1 = True
            else:
                seen.setdefault(name, set()).add(value)
    assert set(seen) == {*scoring.COMPONENTS, "program_syntax_valid"}
    assert all(values >= {0.0, 1.0} for values in seen.values()), seen
    assert partial_f1


@pytest.mark.parametrize("task", sorted(TASKS))
def test_each_task_uses_the_cookbooks_token_limit(task: str) -> None:
    limit = _cookbook_limit(task)
    assert scoring.token_limit(task) == limit

    def hit(tokens: int) -> float:
        response = _response(["1"], tokens=tokens)
        return scoring.score(task=task, expected_answer="1", response=response)["hit_max_new_tokens"]

    assert (hit(limit - 1), hit(limit)) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("task", "has_syntax"), [(task, family == "program_synthesis") for task, family in TASKS.items()]
)
def test_only_code_tasks_get_a_syntax_score(task: str, has_syntax: bool) -> None:
    response = _response(["<think>\nanswer = 1 + 1\n</think>\n2"], tokens=20)

    scores = scoring.score(task=task, expected_answer="2", response=response)

    assert ("program_syntax_valid" in scores) is has_syntax
    assert set(scores) - {"program_syntax_valid"} == set(scoring.COMPONENTS)


def test_a_response_with_no_output_scores_zero() -> None:
    scores = scoring.score(task="TAT-DQA/span", expected_answer="2019", response={"output": [], "usage": None})

    assert scores == dict.fromkeys(scoring.COMPONENTS, 0.0)


def test_the_tolerance_reaches_the_numeric_match() -> None:
    response = _response(["<think>\nx\n</think>\n104"], tokens=10)

    assert scoring.score(task="TAT-DQA/arithmetic", expected_answer="100", response=response)["relaxed"] == 1.0
    loose = scoring.score(task="TAT-DQA/arithmetic", expected_answer="100", response=response, tolerance=0.01)
    assert loose["relaxed"] == 0.0


def test_by_task_averages_task_accuracy_and_answer_f1_per_task() -> None:
    rollouts = [
        {"task": "TAT-DQA/span", "task_correct": 1.0, "answer_f1": 1.0},
        {"task": "TAT-DQA/span", "task_correct": 0.0, "answer_f1": 0.5},
        {"task": "CodeFinQA", "task_correct": 1.0, "answer_f1": 1.0},
    ]

    assert scoring.by_task(rollouts) == {
        "by_task/CodeFinQA/answer_f1": 1.0,
        "by_task/CodeFinQA/task_accuracy": 1.0,
        "by_task/TAT-DQA/span/answer_f1": 0.75,
        "by_task/TAT-DQA/span/task_accuracy": 0.5,
    }
