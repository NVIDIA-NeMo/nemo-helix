# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example-local pipeline: AgentEvaluator run + deterministic gate.

Thin convenience wrapper used by this example only (not SDK API): it runs the
SDK's :class:`~nemo_evaluator_sdk.agent_eval.evaluator.AgentEvaluator` and applies
the example-local gate. Two seams keep it backend-agnostic:

* **verify-enable is inverted to data**: callers pass ``extra_metrics`` to append
  (e.g. a verifier-reward metric). The pipeline never introspects a runtime's
  config to decide what to score.
* **environment prep is an injected hook**: ``prepare_task`` (e.g. "build the task
  image") runs per task before execution, so Docker/build specifics live in the
  caller, not here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTarget, AgentEvalTrial
from nemo_evaluator_sdk.metrics.protocol import Metric

from .gating import GateThresholds, evaluate_gate, load_baseline_summary, write_gate_report


@dataclass(frozen=True)
class PipelineConfig:
    """Run-level knobs shared by the online and offline paths."""

    parallelism: int = 1
    write_dashboard: bool = True
    write_gate: bool = True
    gate_thresholds: GateThresholds | None = None
    baseline_summary_path: Path | None = None


class AgentEvalPipeline:
    """Run tasks through ``AgentEvaluator`` (online or offline) and apply the gate."""

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        extra_metrics: Sequence[Metric] = (),
    ) -> None:
        self.config = config or PipelineConfig()
        self._extra_metrics = list(extra_metrics)

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        *,
        target: AgentEvalTarget,
        labels: dict[str, str] | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
        prepare_task: Callable[[AgentEvalTask], object] | None = None,
    ) -> AgentEvalResult:
        """Online path: optionally prep each task, run the target, score, gate."""
        prepared = [self._with_extra_metrics(task) for task in tasks]
        if prepare_task is not None:
            for task in prepared:
                prepare_task(task)

        result = await AgentEvaluator().run(
            tasks=prepared,
            target=target,
            config=self._run_config(output_dir=output_dir, run_id=run_id, labels=labels),
        )
        self._persist_and_gate(result, output_dir)
        return result

    async def score_trials(
        self,
        tasks: Sequence[AgentEvalTask],
        *,
        trials: Sequence[AgentEvalTrial],
        labels: dict[str, str] | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> AgentEvalResult:
        """Offline path: score already-captured trials (no agent execution)."""
        prepared = [self._with_extra_metrics(task) for task in tasks]
        result = await AgentEvaluator().run(
            tasks=prepared,
            trials=list(trials),
            config=self._run_config(output_dir=output_dir, run_id=run_id, labels=labels),
        )
        self._persist_and_gate(result, output_dir)
        return result

    def _run_config(
        self,
        *,
        output_dir: Path | None,
        run_id: str | None,
        labels: dict[str, str] | None,
    ) -> AgentEvalRunConfig:
        return AgentEvalRunConfig(
            work_dir=output_dir,
            run_id=run_id,
            parallelism=self.config.parallelism,
            labels=dict(labels or {}),
        )

    def _with_extra_metrics(self, task: AgentEvalTask) -> AgentEvalTask:
        """Append injected metrics, honoring task-authored metrics and avoiding duplicate types."""
        if not self._extra_metrics:
            return task
        metrics: list[Metric] = list(task.metrics)
        existing_types = {metric.type for metric in metrics}
        appended = [metric for metric in self._extra_metrics if metric.type not in existing_types]
        if not appended:
            return task
        return task.model_copy(update={"metrics": metrics + appended})

    def _persist_and_gate(self, result: AgentEvalResult, output_dir: Path | None) -> None:
        """Store the run (when a directory was given) and write the gate report beside it."""
        if output_dir is None:
            return
        location = result.persist(write_dashboard=self.config.write_dashboard)
        if not self.config.write_gate:
            return
        baseline = (
            load_baseline_summary(self.config.baseline_summary_path)
            if self.config.baseline_summary_path is not None
            else None
        )
        report = evaluate_gate(result, thresholds=self.config.gate_thresholds, baseline_summary=baseline)
        write_gate_report(report, location.output_dir)


__all__ = [
    "AgentEvalPipeline",
    "GateThresholds",
    "PipelineConfig",
]
