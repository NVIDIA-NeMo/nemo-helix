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

    The model server must run with ``uses_reasoning_parser: true``; without it, Gym's ``inference_provider``
    drops the provider's ``reasoning_content``. With it, Gym wraps that reasoning in ``<think>`` tags in front of
    the content, then moves every ``<think>`` block into one reasoning item, a summary part each, in order.

    So the first part is the model's own reasoning, which becomes ``reasoning_content``: the cookbook puts it
    back inside ``<think>`` before splitting the answer off at the first ``</think>``. Any later part was a
    ``<think>`` block the model wrote in its content, as the cookbook's prompt asks. It goes back in front of the
    content, where the cookbook's split still finds it.
    """
    message_text = ""
    reasoning_texts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("text"):
                    message_text += part["text"]
        elif item.get("type") == "reasoning":
            parts = list(item.get("summary") or []) + list(item.get("content") or [])
            for part in parts:
                # Keep empty texts too: an empty <think></think> block the model wrote still goes back.
                if part.get("text") is not None:
                    reasoning_texts.append(part["text"])

    if not reasoning_texts:
        return {"content": message_text, "reasoning_content": ""}

    own_reasoning = reasoning_texts[0]
    written_blocks = ""
    for text in reasoning_texts[1:]:
        written_blocks += f"<think>{text}</think>"
    return {"content": written_blocks + message_text, "reasoning_content": own_reasoning}


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
