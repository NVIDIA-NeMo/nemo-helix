# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finance question answering metric runtime implementation.

The scoring functions port ``evaluation.py`` from the Nemotron finance LoRA cookbook (NVIDIA-NeMo/Nemotron,
``usage-cookbook/Nemotron-3.5-Super-VL/lora-finance/nemo-automodel/scripts/evaluation.py`` at ``18c74c09a``),
so scores computed here match the cookbook's per-row scores.
"""

import ast
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from typing import Any

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import (
    build_template_context,
    render_default_output_text_candidate_or_raise,
    render_template_or_raise,
    template_metric_repr,
)
from nhx_evals_sdk.values.metrics import FinanceQA

__all__ = ["FinanceQAMetric"]

_NUMBER_RE = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*(%|percent|thousand|million|billion)?")
_SPAN_SEPARATOR_RE = re.compile(r"\s*(?:\||##|;)\s*")
_THINK_END = "</think>"
TAT_DQA_TASKS = ("TAT-DQA/span", "TAT-DQA/multi-span", "TAT-DQA/arithmetic", "TAT-DQA/count")


def normalize(value: object) -> str:
    """NFKC, casefold, collapse whitespace, and strip whitespace and ``.,;:!?`` from both ends."""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?")


def number_and_scale(value: object) -> tuple[float | None, str]:
    """Parse a number with an optional scale (``percent``, ``thousand``, ``million``, ``billion``)."""
    text = normalize(value).replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    match = _NUMBER_RE.fullmatch(text)
    if not match:
        return None, ""
    scale = match.group(2) or ""
    if scale == "%":
        scale = "percent"
    return float(match.group(1)), scale


def relaxed(prediction: str, reference: str, tolerance: float) -> bool:
    """Numeric match within a relative tolerance when both parse as numbers, else a normalized exact match.

    When the reference has a scale, the prediction must have the same one.
    """
    (left, left_scale), (right, right_scale) = number_and_scale(prediction), number_and_scale(reference)
    if left is not None and right is not None:
        if right_scale and left_scale != right_scale:
            return False
        return abs(left - right) <= max(1e-6, abs(right) * tolerance)
    return normalize(prediction) == normalize(reference)


def answer_tokens(value: object) -> list[str]:
    """The ``[a-z0-9]+`` tokens of the normalized value."""
    return re.findall(r"[a-z0-9]+", normalize(value))


def token_f1(prediction: str, reference: str) -> float:
    """Token F1 over ``answer_tokens``. Two empty answers score 1."""
    predicted, expected = Counter(answer_tokens(prediction)), Counter(answer_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted and not expected:
        return 1.0
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def unordered_spans(value: object) -> list[str]:
    """The sorted, normalized, non-empty spans of a value split on ``|``, ``##`` or ``;``."""
    return sorted(normalize(part) for part in _SPAN_SEPARATOR_RE.split(str(value)) if normalize(part))


def task_score(task: str, prediction: str, reference: str, tolerance: float) -> tuple[bool, float]:
    """Return ``(task_correct, answer_f1)`` for the row's task.

    - Multi-span: the sorted span lists must be equal, and F1 is token F1.
    - Span: a normalized exact match, and F1 is token F1.
    - Any other task: ``relaxed``, and F1 is 1 or 0 to match it.
    """
    if task.endswith("multi-span"):
        matched = unordered_spans(prediction) == unordered_spans(reference)
        return matched, token_f1(prediction, reference)
    if task.endswith("span"):
        score = token_f1(prediction, reference)
        return normalize(prediction) == normalize(reference), score
    matched = relaxed(prediction, reference, tolerance)
    return matched, float(matched)


def split_raw_completion(raw: str) -> tuple[str, str]:
    """Return ``(answer, reasoning)``, split on the first ``</think>``.

    A reasoning parser may drop the opening ``<think>`` and leave ``</think>`` in the content, so the split
    uses the closing tag alone.
    """
    text = str(raw or "").strip()
    if _THINK_END not in text:
        return text, ""
    reasoning, answer = text.split(_THINK_END, 1)
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>", 1)[1]
    return answer.strip(), reasoning.strip()


def completion_text(message: Mapping[str, Any]) -> str:
    """Rebuild the raw completion from a chat message: ``<think>\\n{reasoning}\\n</think>\\n{content}``.

    The reasoning is ``reasoning_content``, or ``reasoning`` when that is empty. Without reasoning, the
    completion is the content alone. The run config's ``reasoning.end_token`` handling leaves the trace's own
    ``<think>`` and ``</think>`` in ``reasoning_content``; they are dropped so the trace is wrapped once.
    """
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(reasoning, str) and reasoning.strip().endswith(_THINK_END):
        reasoning = reasoning.strip()[: -len(_THINK_END)].removeprefix("<think>")
    raw = f"<think>\n{reasoning}\n{_THINK_END}\n{content}" if reasoning else str(content)
    return raw.strip()


def program_syntax_valid(reasoning: str) -> bool:
    """Whether the reasoning trace parses as Python. It is parsed, never run.

    A trace too deeply nested for the parser counts as invalid rather than failing the row.
    """
    try:
        ast.parse(reasoning)
    except (SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _first_message(response: object) -> Mapping[str, Any] | None:
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, Mapping) else None


def _completion_tokens(response: object) -> int | None:
    """``usage.completion_tokens`` as the cookbook reads it (a missing count is 0), or None without usage."""
    if not isinstance(response, Mapping) or not isinstance(response.get("usage"), Mapping):
        return None
    tokens = response["usage"].get("completion_tokens")
    if tokens is None:
        return 0
    if isinstance(tokens, bool) or not isinstance(tokens, int | float):
        return None
    return int(tokens)


def _as_text(value: object, template_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    raise TypeError(f"The {template_name} must be a string or a number.")


class FinanceQAMetric(FinanceQA):
    """The per-row scores of the Nemotron finance LoRA cookbook.

    The prediction is the text after the first ``</think>`` of the raw completion, and the reasoning is the text
    before it. Without a ``candidate`` template, the raw completion is rebuilt from the model response's
    ``reasoning_content`` (or ``reasoning``) and ``content``, or taken from the output text when there is no
    response.

    Scores per row:

    - ``exact``: the normalized prediction equals the normalized reference.
    - ``relaxed``: numbers within ``tolerance`` relative error with matching scale, else ``exact``.
    - ``task_correct``: sorted spans for multi-span tasks, ``exact`` for span tasks, ``relaxed`` otherwise.
    - ``answer_f1``: token F1 for span and multi-span tasks, ``relaxed`` as 1 or 0 otherwise.
    - ``trace_closed``: the reasoning is non-empty or the raw completion contains ``</think>``. With a
      reasoning parser, any response with reasoning counts, even one cut off mid-trace, as in the cookbook.
    - ``program_syntax_valid``: for ``program_synthesis_tasks`` only, the reasoning parses as Python.
    - ``hit_max_new_tokens``: declared only when ``max_new_tokens`` is set, and emitted when the response
      reports usage: whether ``usage.completion_tokens`` reached it. Its aggregate sum is the count.
    - ``task``: the rendered task, as a label for grouping rows.

    The task must be one of the TAT-DQA tasks (``TAT-DQA/span``, ``TAT-DQA/multi-span``,
    ``TAT-DQA/arithmetic``, ``TAT-DQA/count``) or one of ``program_synthesis_tasks``; any other value is a
    row error, so a task template that renders the wrong text fails loudly instead of changing the scoring.
    """

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [
            MetricOutputSpec.boolean("exact"),
            MetricOutputSpec.boolean("relaxed"),
            MetricOutputSpec.boolean("task_correct"),
            MetricOutputSpec.continuous_score("answer_f1"),
            MetricOutputSpec.boolean("trace_closed"),
            MetricOutputSpec.boolean("program_syntax_valid", required=False),
            *(
                [MetricOutputSpec.boolean("hit_max_new_tokens", required=False)]
                if self.max_new_tokens is not None
                else []
            ),
            MetricOutputSpec.label("task"),
        ]

    def _render(self, name: str, template: str, input: MetricInput, context: dict[str, Any], metric_repr: str) -> str:
        value = render_template_or_raise(
            template_name=name,
            template=template,
            context=context,
            item=input.row.data,
            sample=input.candidate,
            metric_repr=metric_repr,
        )
        return _as_text(value, name)

    def _raw_completion(self, input: MetricInput, context: dict[str, Any], metric_repr: str) -> str:
        if self.candidate:
            return self._render("candidate", self.candidate, input, context, metric_repr)
        message = _first_message(input.candidate.response)
        if message is not None:
            return completion_text(message)
        output_text = render_default_output_text_candidate_or_raise(
            sample=input.candidate, metric_name=self.__class__.__name__
        )
        return _as_text(output_text, "candidate")

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        context = build_template_context(input.row.data, input.candidate)
        metric_repr = template_metric_repr(self)
        reference = self._render("reference", self.reference, input, context, metric_repr)
        task = self._render("task", self.task, input, context, metric_repr)
        if task not in TAT_DQA_TASKS and task not in self.program_synthesis_tasks:
            raise ValueError(
                f"{metric_repr} rendered the task {task!r}, which is not a TAT-DQA task "
                f"({', '.join(TAT_DQA_TASKS)}) or one of program_synthesis_tasks. Check the `task` template."
            )
        raw = self._raw_completion(input, context, metric_repr)

        prediction, reasoning = split_raw_completion(raw)
        task_correct, answer_f1 = task_score(task, prediction, reference, self.tolerance)
        outputs = [
            MetricOutput(name="exact", value=normalize(prediction) == normalize(reference)),
            MetricOutput(name="relaxed", value=relaxed(prediction, reference, self.tolerance)),
            MetricOutput(name="task_correct", value=task_correct),
            MetricOutput(name="answer_f1", value=answer_f1),
            MetricOutput(name="trace_closed", value=bool(reasoning) or _THINK_END in raw),
        ]
        if task in self.program_synthesis_tasks:
            outputs.append(MetricOutput(name="program_syntax_valid", value=program_syntax_valid(reasoning)))
        completion_tokens = _completion_tokens(input.candidate.response)
        if self.max_new_tokens is not None and completion_tokens is not None:
            outputs.append(MetricOutput(name="hit_max_new_tokens", value=completion_tokens >= self.max_new_tokens))
        outputs.append(MetricOutput(name="task", value=task))
        return MetricResult(outputs=outputs)
