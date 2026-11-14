# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finance cookbook's per-row scores for one Gym rollout.

Kept apart from ``app.py`` so it runs without NeMo Gym: ``app.py`` passes the rollout's Responses API response
as a dict. The scoring itself is ``cookbook.cookbook_row``, unchanged; this module only rebuilds its inputs.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import cookbook  # ty: ignore[unresolved-import]

#: Tasks the cookbook scores as program synthesis, which adds the Python syntax check.
PROGRAM_SYNTHESIS_TASKS = frozenset({"CodeFinQA", "CodeTAT-QA"})
#: Tasks the cookbook gives 256 new tokens. Every other task gets 512.
SHORT_ANSWER_TASKS = frozenset({"TAT-DQA/arithmetic", "TAT-DQA/count"})
#: The scores every row gets. ``program_syntax_valid`` is added for code tasks only.
COMPONENTS = ("exact", "relaxed", "task_correct", "answer_f1", "trace_closed", "hit_max_new_tokens")


def token_limit(task: str) -> int:
    """The cookbook's new-token limit for ``task``."""
    return 256 if task in SHORT_ANSWER_TASKS else 512


def chat_message(response: Mapping[str, Any]) -> dict[str, str]:
    """The Responses API ``response`` as the chat message the cookbook reads.

    Needs ``uses_reasoning_parser: true`` on Gym's ``inference_provider``, which otherwise drops
    ``reasoning_content``. With it, Gym wraps that reasoning in ``<think>`` in front of the content, then moves
    every ``<think>`` block into a reasoning item. Putting the blocks back in front of the content rebuilds that
    text, which the cookbook splits as it would the provider's message.

    Gym doesn't keep where the blocks sat, so a block written after the answer moves in front of it. That can change
    the cookbook's answer only when the model returns no reasoning of its own.
    """
    blocks = ""
    text = ""
    for item in response.get("output") or []:
        if item.get("type") == "reasoning":
            for part in list(item.get("summary") or []) + list(item.get("content") or []):
                # An empty block still goes back: its tags move the cookbook's split.
                if part.get("text") is not None:
                    blocks += f"<think>{part['text']}</think>"
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("text"):
                    text += part["text"]
    return {"content": blocks + text}


def score(*, task: str, expected_answer: str, response: Mapping[str, Any], tolerance: float = 0.05) -> dict[str, float]:
    """The cookbook's scores for one row, as reward components."""
    usage = response.get("usage") or {}
    scores = cookbook.cookbook_row(
        task=task,
        task_family="program_synthesis" if task in PROGRAM_SYNTHESIS_TASKS else "tat_dqa",
        answer=expected_answer,
        response={
            "choices": [{"message": chat_message(response)}],
            "usage": {"completion_tokens": usage.get("output_tokens") or 0},
        },
        task_limit=token_limit(task),
        tolerance=tolerance,
    )
    components = {name: float(scores[name]) for name in COMPONENTS}
    if scores["program_syntax_valid"] is not None:
        components["program_syntax_valid"] = float(scores["program_syntax_valid"])
    return components


def by_task(rollouts: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Task accuracy and answer F1 per task, as ``by_task/<task>/<name>``.

    Gym reports these as the server's own metrics when it runs the environment itself. On the platform, use
    ``finance_report.py`` instead.
    """
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rollout in rollouts:
        for key, name in (("task_correct", "task_accuracy"), ("answer_f1", "answer_f1")):
            if rollout.get(key) is not None:
                values[rollout["task"]][name].append(float(rollout[key]))
    return {
        f"by_task/{task}/{name}": sum(samples) / len(samples)
        for task, names in sorted(values.items())
        for name, samples in names.items()
    }
