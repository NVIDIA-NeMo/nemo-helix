# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-criterion component scoring for the Legal Agent Benchmark (LAB).

LAB is a *rubric* benchmark: each task carries several pass/fail criteria, and its
Harbor verifier judges every one, writing the outcome to
``<trial>/verifier/scores.json``. The SDK's built-in
:class:`~nhx_evals_sdk.agent_eval.runtimes.harbor.runtime.HarborRewardMetric`
scores only the single scalar reward LAB emits (``full_task`` by default: ``1.0``
iff every criterion passes).

This metric reads that same ``scores.json`` and turns the rubric into first-class
**component** scores, so one run reports both the official reward *and* the
criterion breakdown. It is a plain SDK metric — a small object with ``type`` /
``output_spec`` / ``compute_scores`` and no base class — attached alongside
``HarborRewardMetric`` (see ``run_legal_agent_bench.py``).

Schema note: the keys read below (``n_criteria``, ``n_passed``, ``all_pass``,
``judge_error_count``, and ``criteria_results[].verdict``) match the ``scores.json``
LAB's Harbor verifier writes.
If you run a LAB build whose verifier writes a different schema, adjust the key
names here; unreadable or missing scores degrade to zeros rather than raising, so
one crashed trial never fails the whole run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult

logger = logging.getLogger(__name__)

_PASS_VERDICT = "pass"


class LabCriteriaMetric:
    """Score LAB's rubric criteria as components, read from the trial's verifier output.

    Outputs (all derived from ``<harbor_trial_dir>/verifier/scores.json``):

    * ``criteria_pass_rate`` — fraction of criteria the judge passed (``0.0``–``1.0``).
    * ``all_criteria_pass``  — ``True`` iff every criterion passed (LAB's ``full_task``).
    * ``n_passed`` / ``n_criteria`` — the raw counts behind the rate.
    * ``judge_error_count`` — judge/infrastructure failures. Treat ``> 0`` as an
      infrastructure problem, not a model failure: a rubric graded with a broken
      judge is not a real ``0``.
    """

    def __init__(self, *, metric_type: str = "lab_criteria", scores_relpath: str = "verifier/scores.json") -> None:
        self._type = metric_type
        self._scores_relpath = scores_relpath

    @property
    def type(self) -> str:
        return self._type

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("criteria_pass_rate"),
            MetricOutputSpec.boolean("all_criteria_pass"),
            MetricOutputSpec.discrete_score("n_passed"),
            MetricOutputSpec.discrete_score("n_criteria"),
            MetricOutputSpec.discrete_score("judge_error_count"),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        scores = self._load_scores(input)

        n_criteria = _as_int(scores.get("n_criteria"))
        n_passed = _as_int(scores.get("n_passed"))
        # Fall back to counting verdicts if the aggregate counts are absent.
        results = scores.get("criteria_results")
        if n_criteria == 0 and isinstance(results, list):
            n_criteria = len(results)
            n_passed = sum(1 for entry in results if _verdict_is_pass(entry))

        pass_rate = (n_passed / n_criteria) if n_criteria else 0.0
        all_pass = bool(scores.get("all_pass")) if "all_pass" in scores else (n_criteria > 0 and n_passed == n_criteria)
        judge_errors = _as_int(scores.get("judge_error_count"))

        return MetricResult(
            outputs=[
                MetricOutput(name="criteria_pass_rate", value=pass_rate),
                MetricOutput(name="all_criteria_pass", value=all_pass),
                MetricOutput(name="n_passed", value=n_passed),
                MetricOutput(name="n_criteria", value=n_criteria),
                MetricOutput(name="judge_error_count", value=judge_errors),
            ]
        )

    def _load_scores(self, input: MetricInput) -> dict[str, Any]:
        """Read LAB's ``scores.json`` from the Harbor trial directory the runner stamped on the trial."""
        trial_dir = input.candidate.metadata.get("harbor_trial_dir")
        if not isinstance(trial_dir, str):
            logger.warning("LabCriteriaMetric: trial has no 'harbor_trial_dir' metadata; scoring zeros")
            return {}
        scores_path = Path(trial_dir) / self._scores_relpath
        try:
            data = json.loads(scores_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("LabCriteriaMetric: could not read %s (%s); scoring zeros", scores_path, exc)
            return {}
        return data if isinstance(data, dict) else {}


def _verdict_is_pass(entry: Any) -> bool:
    return isinstance(entry, dict) and str(entry.get("verdict", "")).strip().lower() == _PASS_VERDICT


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
