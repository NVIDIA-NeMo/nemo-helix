# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundle Job: merge a run's task bundles into one bundle (M2 story 0's slice of M5 story 7).

It writes files only, as Job results: ``agent-eval-results`` (the Docker layout) and ``summary``.
The run controller saves the run's result record from them; workers never write records.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from nemo_evals.jobs.agent_evaluate import DEFAULT_RESULT_NAME, SUMMARY_FILE_NAME, SUMMARY_RESULT_NAME
from nemo_evals.taskset_runs.bundle_writer import write_taskset_bundle
from nemo_evals.taskset_runs.specs import TasksetBundleSpec
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import ContainerSpec, CPUExecutionProviderSpec, HelixJobSpec, HelixJobStep
from pydantic import BaseModel

BUNDLE_JOB_SOURCE = "evals.taskset-bundle"
BUNDLE_STEP_NAME = "taskset-bundle"
BUNDLE_COMMAND = ["nemo_evals.tasks.taskset_bundle"]


class TasksetBundleJob(NemoJob):
    """Merge one run's task bundles. Not registered under ``nemo.jobs``: no route, no CLI."""

    name: ClassVar[str] = "taskset-bundle"
    description: ClassVar[str] = "Merge a taskset run's task bundles into one result bundle."
    spec_schema: ClassVar[type[BaseModel] | None] = TasksetBundleSpec
    input_spec_schema: ClassVar[type[BaseModel] | None] = TasksetBundleSpec

    @classmethod
    async def compile(cls, *, spec: TasksetBundleSpec, profile: str, image: str, **_: Any) -> HelixJobSpec:  # ty: ignore[invalid-method-override]
        """One step on the run's pinned dispatcher image; it reads and writes Files only."""
        step = HelixJobStep(
            name=BUNDLE_STEP_NAME,
            executor=CPUExecutionProviderSpec(
                profile=profile,
                provider="cpu",
                container=ContainerSpec(image=image, entrypoint=["python", "-m"], command=BUNDLE_COMMAND),
            ),
            config=spec.model_dump(mode="json"),
        )
        return HelixJobSpec(steps=[step])

    def run(self, config: dict, *, ctx: JobContext, async_client: AsyncNemoClient) -> dict:
        """Write the merged bundle and its summary as this Job's results."""
        spec = TasksetBundleSpec.model_validate(config)
        with tempfile.TemporaryDirectory(prefix="taskset-bundle-out-") as scratch:
            out_dir = Path(scratch) / DEFAULT_RESULT_NAME
            result = asyncio.run(write_taskset_bundle(spec, out_dir, files=AsyncFilesClient.from_client(async_client)))
            artifact = ctx.results.save(DEFAULT_RESULT_NAME, out_dir)
            ctx.results.save(SUMMARY_RESULT_NAME, out_dir / SUMMARY_FILE_NAME)
        return {
            "status": "completed",
            "artifact": artifact.model_dump(),
            "tasks": result.summary.task_count,
            "trials": result.summary.trial_count,
            "scores": result.summary.score_count,
        }
