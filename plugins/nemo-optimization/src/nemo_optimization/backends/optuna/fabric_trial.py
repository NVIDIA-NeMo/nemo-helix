# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentEvaluator + FabricAgentRuntime trial evaluator for Optuna studies."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.metrics import ToolCallCountMetric
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.metrics.tunable_rag_evaluator import TunableRagEvaluatorMetric
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.evidence import EVIDENCE_TRACE
from nemo_evaluator_sdk.values.models import Model
from pydantic import ValidationError

from nemo_optimization.backends.optuna.atif_metadata import build_atif_trial_tags
from nemo_optimization.backends.optuna.config_overlay import apply_suggestions
from nemo_optimization.backends.optuna.study_driver import StudyDriverError

logger = logging.getLogger(__name__)


class FabricTrialEvaluator:
    """Run one Optuna trial repetition through Fabric and reduce evaluator scores."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        metric_names: Sequence[str],
        output_dir: Path,
        experiment_id: str,
    ) -> None:
        self._payload = copy.deepcopy(dict(payload))
        self._metric_names = tuple(metric_names)
        self._output_dir = output_dir
        self._experiment_id = experiment_id
        self._eval_config = _eval_config(payload)
        fabric_eval = self._eval_config.get("fabric") if isinstance(self._eval_config.get("fabric"), Mapping) else {}
        if self._eval_config.get("run_hook") is not None:
            raise StudyDriverError(
                "eval.run_hook is no longer supported: declare MCP servers statically under mcp.servers and "
                "score any per-run audit with an evaluator."
            )
        self._fabric_base_dir = _optional_path(
            fabric_eval.get("base_dir") if isinstance(fabric_eval, Mapping) else None
        )
        self._timeout_s = int(fabric_eval.get("timeout_s", 600) if isinstance(fabric_eval, Mapping) else 600)
        self._capture_trajectory = bool(
            fabric_eval.get("capture_trajectory", True) if isinstance(fabric_eval, Mapping) else True
        )
        general = self._eval_config.get("general")
        general = general if isinstance(general, Mapping) else {}
        self._parallelism = int(general.get("max_concurrency", 4))
        self._trace_map: list[dict[str, Any]] = []
        # Validate dataset/metrics once at construction so config errors fail before the study loop.
        build_agent_eval_tasks(self._payload)

    def evaluate(
        self,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> dict[str, float]:
        del trial_overlay  # reserved for profile overlays; runtime uses path-resolved payload
        # ``suggestions`` are Fabric dotted paths (from study_driver.suggestions_by_path).
        trial_payload = apply_suggestions(self._payload, suggestions)
        # Rebuild tasks from the path-resolved payload so search-space paths under
        # eval.evaluators (and dataset settings) affect this trial's scoring.
        tasks = build_agent_eval_tasks(trial_payload)
        runtime = FabricAgentRuntime(
            config=_runtime_agent_config(trial_payload),
            base_dir=self._fabric_base_dir,
            work_root=self._trial_work_root(trial_number, rep),
            timeout_s=self._timeout_s,
            capture_trajectory=self._capture_trajectory,
            trajectory_extra=build_atif_trial_tags(
                experiment_id=self._experiment_id,
                trial_number=trial_number,
                rep=rep,
            ),
        )
        result = AgentEvaluator().run_sync(
            tasks=tasks,
            target=runtime,
            config=AgentEvalRunConfig(
                work_dir=self._trial_output_dir(trial_number, rep),
                parallelism=self._parallelism,
                # Keep scoring the rest of the dataset when one metric raises;
                # reduce_agent_eval_scores skips FAILED task scores.
                fail_fast=False,
            ),
        )
        # ``run`` no longer stores the bundle; keep writing one per trial as before. No dashboard —
        # each trial bundle is intermediate evidence for the study, not something anyone opens.
        result.persist(write_dashboard=False)
        self._record_traces(result, trial_number=trial_number, rep=rep)
        self._write_trace_map()
        return reduce_agent_eval_scores(result.scores, self._metric_names)

    def _trial_work_root(self, trial_number: int, rep: int) -> Path:
        return self._output_dir / "evidence" / f"trial-{trial_number:03d}" / f"rep-{rep:03d}"

    def _trial_output_dir(self, trial_number: int, rep: int) -> Path:
        return self._output_dir / "agent_eval" / f"trial-{trial_number:03d}" / f"rep-{rep:03d}"

    def _record_traces(self, result: AgentEvalResult, *, trial_number: int, rep: int) -> None:
        for trial in result.trials:
            trace = trial.evidence.descriptors.get(EVIDENCE_TRACE) if trial.evidence is not None else None
            if trace is None:
                continue
            self._trace_map.append(
                {
                    "experiment_id": self._experiment_id,
                    "trial_number": trial_number,
                    "rep": rep,
                    "row_id": trial.task_id,
                    "task_id": trial.task_id,
                    "trial_id": trial.id,
                    "trace_ref": trace.ref,
                    "trace_format": trace.format,
                }
            )

    def _write_trace_map(self) -> None:
        if not self._trace_map:
            return
        path = self._output_dir / "trial_trace_map.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._trace_map, indent=2) + "\n", encoding="utf-8")


def build_agent_eval_tasks(payload: Mapping[str, Any]) -> list[AgentEvalTask]:
    eval_config = _eval_config(payload)
    rows = _load_dataset_rows(eval_config)
    metrics = _build_metrics(payload, eval_config)
    tasks: list[AgentEvalTask] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", index))
        instruction = _row_instruction(row, row_id)
        answer = row.get("answer") or row.get("expected_answer") or row.get("reference") or row.get("label") or ""
        tasks.append(
            AgentEvalTask(
                id=row_id,
                intent=instruction,
                inputs={"instruction": instruction},
                reference={"answer": str(answer)},
                metrics=copy.deepcopy(metrics),
                metadata={"optimizer_dataset_index": index},
            )
        )
    return tasks


def _row_instruction(row: Mapping[str, Any], row_id: str) -> str:
    """Build the agent prompt for one dataset row.

    Prefer explicit instruction fields. Otherwise match the phishing-agent eval
    convention of ``subject\\n\\nbody`` when both are present.
    """
    for key in ("instruction", "question", "prompt"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)

    subject = row.get("subject")
    body = row.get("body") if row.get("body") is not None else row.get("input")
    if subject is not None and str(subject).strip() and body is not None and str(body).strip():
        return f"{subject}\n\n{body}"
    if body is not None and str(body).strip():
        return str(body)

    raise StudyDriverError(f"Dataset row {row_id!r} has no instruction/question/body/input.")


def reduce_agent_eval_scores(scores: Sequence[AgentEvalTaskScore], metric_names: Sequence[str]) -> dict[str, float]:
    """Reduce per-task metric scores into one float per study objective.

    Failed / incomplete task scores are skipped so a single bad dataset row does
    not fail the whole Optuna trial. The study still fails if a metric has no
    completed samples left to average.
    """
    reduced: dict[str, float] = {}
    for metric_name in metric_names:
        values: list[float] = []
        skipped = 0
        for score in scores:
            if score.status != AgentEvalScoreStatus.COMPLETED:
                skipped += 1
                logger.warning(
                    "Skipping non-completed agent-eval score for Optuna reduction (metric=%s task=%s status=%s): %s",
                    score.metric_type,
                    score.task_id,
                    score.status,
                    score.diagnostics,
                )
                continue
            for output in score.outputs:
                if output.name == metric_name:
                    values.append(float(output.value))
        if not values:
            detail = f" ({skipped} non-completed score(s) skipped)" if skipped else ""
            raise StudyDriverError(f"Agent evaluation did not produce metric output {metric_name!r}{detail}.")
        if skipped:
            logger.info(
                "Averaged metric %r over %d completed sample(s) (%d skipped)",
                metric_name,
                len(values),
                skipped,
            )
        reduced[metric_name] = sum(values) / len(values)
    return reduced


def _build_metrics(payload: Mapping[str, Any], eval_config: Mapping[str, Any]) -> list[Metric]:
    evaluators = eval_config.get("evaluators")
    if not isinstance(evaluators, Mapping) or not evaluators:
        raise StudyDriverError("eval.evaluators must declare at least one evaluator.")
    metrics: list[Metric] = []
    for evaluator in evaluators.values():
        if not isinstance(evaluator, Mapping):
            continue
        evaluator_type = evaluator.get("_type") or evaluator.get("type")
        if evaluator_type in {"tunable_rag_evaluator", "tunable-rag-evaluator"}:
            metrics.append(_build_tunable_rag_metric(payload, evaluator))
        elif evaluator_type in {"tool_call_count", "tool-call-count"}:
            metrics.append(_build_tool_call_count_metric(evaluator))
        else:
            raise StudyDriverError(f"Unsupported evaluator type for optimize trial path: {evaluator_type!r}")
    if not metrics:
        raise StudyDriverError("No supported eval.evaluators were found.")
    return metrics


def _build_tool_call_count_metric(evaluator: Mapping[str, Any]) -> ToolCallCountMetric:
    tool_name = evaluator.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise StudyDriverError("tool_call_count evaluator requires a non-empty tool_name.")
    expected_calls = evaluator.get("expected_calls", 1)
    # A bool is an int in Python and a float would silently truncate, so both are rejected.
    if isinstance(expected_calls, bool) or not isinstance(expected_calls, int):
        raise StudyDriverError(
            f"tool_call_count evaluator expected_calls must be a non-negative integer, got {expected_calls!r}."
        )
    try:
        return ToolCallCountMetric(tool_name=tool_name.strip(), expected_calls=expected_calls)
    except ValidationError as exc:
        raise StudyDriverError(
            f"tool_call_count evaluator expected_calls must be a non-negative integer: {exc}"
        ) from exc


def _build_tunable_rag_metric(payload: Mapping[str, Any], evaluator: Mapping[str, Any]) -> TunableRagEvaluatorMetric:
    llm_name = str(evaluator.get("llm_name") or evaluator.get("judge_model") or "default")
    model = _model_from_fabric(payload, llm_name)
    return TunableRagEvaluatorMetric(
        model=model,
        judge_llm_prompt=str(evaluator.get("judge_llm_prompt") or ""),
        default_scoring=bool(evaluator.get("default_scoring", True)),
        default_score_weights=dict(evaluator.get("default_score_weights") or {}),
    )


def _model_from_fabric(payload: Mapping[str, Any], model_name: str) -> Model:
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise StudyDriverError("Fabric payload must declare models for tunable_rag_evaluator.")
    raw = models.get(model_name)
    if not isinstance(raw, Mapping):
        raise StudyDriverError(f"Judge model {model_name!r} not found under payload.models.")

    provider = str(raw.get("provider") or "openai").lower()
    if provider in {"openai", "nvidia"}:
        model_format = ModelFormat.OPEN_AI
    elif provider in {"nim", "nvidia_nim"}:
        model_format = ModelFormat.NVIDIA_NIM
    elif provider == "llama_stack":
        model_format = ModelFormat.LLAMA_STACK
    else:
        raise StudyDriverError(
            f"Judge model {model_name!r} has unsupported provider {provider!r}. "
            "Expected one of: openai, nvidia, nim, llama_stack."
        )
    model_id = str(raw.get("model") or raw.get("model_name") or model_name)
    url = str(raw.get("url") or raw.get("base_url") or "")
    if not url:
        raise StudyDriverError(f"Judge model {model_name!r} must declare 'url' or 'base_url'.")
    secret_ref = raw.get("api_key_secret") or raw.get("api_key_env")
    if secret_ref is not None and not isinstance(secret_ref, str):
        raise StudyDriverError(f"Judge model {model_name!r} api_key_secret/api_key_env must be a string.")
    return Model(
        url=url,
        name=model_id,
        format=model_format,
        api_key_secret=SecretRef(root=secret_ref) if secret_ref else None,
    )


def _load_dataset_rows(eval_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = eval_config.get("general", {}).get("dataset") if isinstance(eval_config.get("general"), Mapping) else None
    path = _dataset_path(dataset)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise StudyDriverError(f"Dataset must be a JSON list of rows: {path}")
    rows = [row for row in payload if isinstance(row, dict)]
    if len(rows) != len(payload):
        raise StudyDriverError(f"Dataset contains non-object rows: {path}")
    if not rows:
        raise StudyDriverError(f"Dataset is empty: {path}")
    return rows


def _dataset_path(dataset: Any) -> Path:
    if isinstance(dataset, str):
        return Path(dataset).expanduser()
    if isinstance(dataset, Mapping):
        file_path = dataset.get("file_path") or dataset.get("path")
        if isinstance(file_path, str):
            return Path(file_path).expanduser()
    raise StudyDriverError("eval.general.dataset must be a path string or mapping with file_path.")


def _eval_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    eval_config = payload.get("eval")
    if not isinstance(eval_config, Mapping):
        raise StudyDriverError("Fabric optimize payload must include an eval mapping for real trial execution.")
    return eval_config


def _runtime_agent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_config = copy.deepcopy(dict(config))
    runtime_config.pop("eval", None)
    runtime_config.pop("optimizer", None)
    return runtime_config


def _optional_path(value: Any) -> Path | None:
    return Path(value).expanduser() if isinstance(value, str) and value else None
