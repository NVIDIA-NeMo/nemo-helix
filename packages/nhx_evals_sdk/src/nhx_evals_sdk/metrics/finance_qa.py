# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finance question answering metric runtime implementation.

The metric scores as ``evaluation.py`` in the Nemotron finance LoRA cookbook does (NVIDIA-NeMo/Nemotron,
``usage-cookbook/Nemotron-3.5-Super-VL/lora-finance/nemo-automodel/scripts/evaluation.py`` at ``18c74c09a``). It
is a preset of ``answer-match``: ``FinanceQA.to_answer_match()`` holds the cookbook's choices as configuration,
and this module adds the one score that isn't about the answer, ``hit_max_new_tokens``.
"""

from collections.abc import Mapping

from nhx_evals_sdk.metrics.answer_match import SelectUnmatchedError, score_answer
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import template_metric_repr
from nhx_evals_sdk.values.metrics import FINANCE_QA_TAT_DQA_TASKS, FinanceQA

__all__ = ["FinanceQAMetric"]

TAT_DQA_TASKS = FINANCE_QA_TAT_DQA_TASKS
_ANSWER_OUTPUTS = ("exact", "relaxed", "task_correct", "answer_f1", "trace_closed", "program_syntax_valid")


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

    Every score but ``hit_max_new_tokens`` comes from ``to_answer_match()``, the cookbook as an ``answer-match``
    configuration.
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

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        metric_repr = template_metric_repr(self)
        try:
            scored = score_answer(
                self.to_answer_match(), input, metric_repr=metric_repr, metric_name=self.__class__.__name__
            )
        except SelectUnmatchedError as error:
            raise ValueError(
                f"{metric_repr} rendered the task {error.value!r}, which is not a TAT-DQA task "
                f"({', '.join(TAT_DQA_TASKS)}) or one of program_synthesis_tasks. Check the `task` template."
            ) from None
        by_name = {output.name: output for output in scored}
        outputs = [by_name[name] for name in _ANSWER_OUTPUTS if name in by_name]
        completion_tokens = _completion_tokens(input.candidate.response)
        if self.max_new_tokens is not None and completion_tokens is not None:
            outputs.append(MetricOutput(name="hit_max_new_tokens", value=completion_tokens >= self.max_new_tokens))
        outputs.append(by_name["task"])
        return MetricResult(outputs=outputs)
