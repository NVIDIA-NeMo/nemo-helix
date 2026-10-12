# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform Job for one scaled-evals benchmark-run archive build."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec, HelixJobStep
from nemo_scaled_evals_plugin.jobs.specs import BenchmarkArchiveBuildSpec
from nemo_scaled_evals_plugin.jobs.task_image_build import resolve_executor, resolve_secret_environment
from pydantic import BaseModel
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from scaled_evals.dispatch.worker import Dispatcher


class BenchmarkArchiveBuildJob(NemoJob):
    """Claim and build one benchmark-run archive."""

    name: ClassVar[str] = "benchmark-archive-build"
    description: ClassVar[str] = "Build a scaled-evals benchmark-run archive."
    spec_schema: ClassVar[type[BaseModel]] = BenchmarkArchiveBuildSpec

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
        """Compile a benchmark archive build into one CPU container step."""
        del workspace, entity_client, job_name, async_sdk
        canonical = BenchmarkArchiveBuildSpec.model_validate(spec)
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="benchmark-archive-build",
                    executor=resolve_executor(
                        options,
                        profile=profile or "default",
                        module="nemo_scaled_evals_plugin.tasks.benchmark_archive_build",
                    ),
                    environment=resolve_secret_environment(),
                    config=canonical.model_dump(mode="json"),
                )
            ]
        )

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Claim the archive here, so the lease starts only once the pod runs."""
        spec = BenchmarkArchiveBuildSpec.model_validate(config)
        dispatcher = Dispatcher()
        with dispatcher.connect() as conn:
            claimed = BenchmarkArchiveRepository(conn).claim(
                claim_timeout=dispatcher.claim_timeout,
                run_id=spec.benchmark_run_id,
            )
        # None means another attempt holds a live lease; "failed" means the
        # claim exhausted its attempts. Neither leaves work for this Job.
        built = claimed is not None and claimed["status"] == "building"
        if built:
            dispatcher.build_benchmark_archive(claimed)
        return {"status": "completed", "benchmark_run_id": spec.benchmark_run_id, "built": built}
