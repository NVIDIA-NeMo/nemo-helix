# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform Job for one scaled-evals evaluation execution."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.jobs.api_factory import (
    PlatformJobSpec,
    PlatformJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_scaled_evals_plugin.jobs.specs import EvaluationExecutionSpec
from nemo_scaled_evals_plugin.jobs.task_image_build import resolve_executor, resolve_secret_environment
from pydantic import BaseModel
from scaled_evals.dispatch.worker import Dispatcher


class EvaluationExecutionJob(NemoJob):
    """Run one immutable scaled-evals evaluation execution."""

    name: ClassVar[str] = "evaluation-execution"
    description: ClassVar[str] = "Execute one scaled-evals evaluation attempt."
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
    ) -> PlatformJobSpec:
        """Compile an evaluation execution into one Platform Jobs step."""
        del workspace, entity_client, job_name, async_sdk
        canonical = EvaluationExecutionSpec.model_validate(spec)
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
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
                    environment=resolve_secret_environment(),
                    config=canonical.model_dump(mode="json"),
                )
            ]
        )

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute the existing dispatcher for the specified execution."""
        spec = EvaluationExecutionSpec.model_validate(config)
        Dispatcher().run(
            spec.evaluation_id,
            maintain_claim=False,
            expected_execution_number=spec.execution_number,
        )
        return {
            "status": "completed",
            "evaluation_id": spec.evaluation_id,
            "execution_number": spec.execution_number,
        }
