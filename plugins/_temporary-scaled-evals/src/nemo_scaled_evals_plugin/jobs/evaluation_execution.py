# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform Job for one scaled-evals evaluation execution."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.jobs.api_factory import (
    HelixJobSpec,
    HelixJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
    StepLifecycle,
)
from nemo_scaled_evals_plugin.jobs.specs import EvaluationExecutionSpec
from nemo_scaled_evals_plugin.jobs.task_image_build import resolve_evaluation_secret_environment, resolve_executor
from pydantic import BaseModel


def _dispatcher_cls() -> type[Any]:
    from scaled_evals.dispatch.worker import Dispatcher

    return Dispatcher


class EvaluationExecutionJob(NemoJob):
    """Run one immutable scaled-evals evaluation execution."""

    name: ClassVar[str] = "evaluation-execution"
    description: ClassVar[str] = "Execute one scaled-evals evaluation attempt."
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = EvaluationExecutionSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HelixJobSpec:
        """Compile an evaluation execution into one Platform Jobs step."""
        del workspace, entity_client, job_name, async_sdk
        canonical = EvaluationExecutionSpec.model_validate(spec)
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="evaluation-execution",
                    executor=resolve_executor(
                        options,
                        profile=profile or "default",
                        module="nemo_scaled_evals_plugin.tasks.evaluation_execution",
                        resources=ResourcesSpec(
                            requests=ResourcesRequestsSpec(cpu="50m", memory="256Mi"),
                            limits=ResourcesLimitsSpec(cpu="1", memory="1Gi"),
                        ),
                    ),
                    environment=resolve_evaluation_secret_environment(canonical.runtime),
                    config=canonical.model_dump(mode="json"),
                    # Without this the platform cannot reap a hung dispatcher:
                    # the step stays active forever and the reconciler keeps
                    # releasing its claim.
                    lifecycle=StepLifecycle(staleness_timeout_seconds=canonical.deadline_seconds),
                )
            ]
        )

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute the specified execution, then publish its evidence and archive."""
        spec = EvaluationExecutionSpec.model_validate(config)
        dispatcher = _dispatcher_cls()()
        dispatcher.run(spec.evaluation_id, expected_execution_number=spec.execution_number)
        # A no-op when the execution was retried rather than terminalized.
        dispatcher.finalize(spec.evaluation_id)
        return {
            "status": "completed",
            "evaluation_id": spec.evaluation_id,
            "execution_number": spec.execution_number,
        }
