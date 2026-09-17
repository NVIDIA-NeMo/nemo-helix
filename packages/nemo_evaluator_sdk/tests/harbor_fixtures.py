# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Harbor 0.20 result builders for evaluator SDK tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from nemo_evaluator_sdk.metrics.protocol import (
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from pydantic import BaseModel

_ERROR_RESULT = Path(__file__).parent / "agent_eval" / "fixtures" / "harbor_error_result.json"
_FIXTURE_AGENT_RESULT: Mapping[str, float | int | None] = MappingProxyType(
    {
        "n_input_tokens": 100,
        "n_output_tokens": 10,
        "n_cache_tokens": 5,
        "cost_usd": 0.25,
    }
)


class ErrorAwareQualityMetric(BaseModel):
    """Test-only metric proving that error exclusions belong to each metric."""

    type: Literal["error_aware_quality"] = "error_aware_quality"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("quality", required=False)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trial = input.row.data["trial"]
        error = trial["error"]
        if isinstance(error, dict) and error.get("type") == "RuntimeError":
            return MetricResult(
                outputs=[],
                diagnostics=[
                    MetricDiagnostic(
                        message="quality excluded for RuntimeError",
                        details={"output": "quality", "reason": "excluded_error_type"},
                    )
                ],
            )
        return MetricResult(outputs=[MetricOutput(name="quality", value=1.0)])


def harbor_trial_result(
    trial_dir: Path,
    *,
    task_name: str,
    rewards: Mapping[str, float | int] | None,
    exception: str | Mapping[str, Any] | None = None,
    config: TrialConfig | None = None,
    task_path: Path | None = None,
    source: str | None = None,
    agent_result: Mapping[str, float | int | None] | None = _FIXTURE_AGENT_RESULT,
) -> TrialResult:
    """Build one coherent Harbor-valid result from the captured 0.20 fixture."""
    payload = TrialResult.model_validate_json(_ERROR_RESULT.read_bytes()).model_dump(mode="json")
    trial_name = trial_dir.name
    if task_path is not None:
        resolved_task_path = task_path
    elif config is not None and config.task.path is not None:
        resolved_task_path = config.task.path
    else:
        resolved_task_path = trial_dir.parent / "tasks" / task_name

    if source is not None:
        resolved_source = source
    elif config is not None and config.task.source is not None:
        resolved_source = config.task.source
    else:
        resolved_source = "nemo-evaluator-sdk-tests"

    payload.update(
        {
            "task_name": task_name,
            "trial_name": trial_name,
            "trial_uri": trial_dir.resolve().as_uri(),
            "task_id": {"path": str(resolved_task_path)},
            "source": resolved_source,
            "verifier_result": None if rewards is None else {"rewards": dict(rewards)},
            "agent_result": None if agent_result is None else dict(agent_result),
        }
    )
    result_config = config.model_dump(mode="json") if config is not None else payload["config"]
    assert isinstance(result_config, dict)
    result_config.update({"trial_name": trial_name, "trials_dir": str(trial_dir.parent)})
    task_config = result_config["task"]
    assert isinstance(task_config, dict)
    task_config.update({"path": str(resolved_task_path), "source": resolved_source})
    payload["config"] = result_config

    if exception is None:
        payload["exception_info"] = None
    else:
        template = payload["exception_info"]
        assert isinstance(template, dict)
        updates = {"exception_type": exception} if isinstance(exception, str) else dict(exception)
        payload["exception_info"] = {**template, **updates}

    return TrialResult.model_validate(payload)


def write_harbor_trial_result(
    trial_dir: Path,
    *,
    task_name: str,
    rewards: Mapping[str, float | int] | None,
    exception: str | Mapping[str, Any] | None = None,
    config: TrialConfig | None = None,
    task_path: Path | None = None,
    source: str | None = None,
    agent_result: Mapping[str, float | int | None] | None = _FIXTURE_AGENT_RESULT,
) -> TrialResult:
    """Write a coherent trial ``config.json`` and Harbor-valid ``result.json``."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "agent").mkdir(exist_ok=True)
    (trial_dir / "verifier").mkdir(exist_ok=True)
    (trial_dir / "agent" / "trajectory.json").write_text("{}", encoding="utf-8")

    result = harbor_trial_result(
        trial_dir,
        task_name=task_name,
        rewards=rewards,
        exception=exception,
        config=config,
        task_path=task_path,
        source=source,
        agent_result=agent_result,
    )
    (trial_dir / "config.json").write_text(result.config.model_dump_json(), encoding="utf-8")
    (trial_dir / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    return result
