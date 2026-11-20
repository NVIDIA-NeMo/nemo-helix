# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``user-sim.export``: writes a simulate job's best conversations as chat training data to a fileset."""

import asyncio
import json
from typing import ClassVar

import pandas as pd
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec
from nemo_user_sim_plugin.engine.export import apply_review_feedback, training_records
from nemo_user_sim_plugin.intake import review_feedback
from nemo_user_sim_plugin.jobs._common import (
    EVALUATIONS,
    PROFILES,
    TRAINING_DATA,
    TRAJECTORIES,
    fetch_result,
    read_result_frame,
    task_step,
)
from nemo_user_sim_plugin.jobs.spec import ExportSpec
from pydantic import BaseModel
from usersim.selection import select_trajectories


class ExportJob(NemoJob):
    name: ClassVar[str] = "export"
    description: ClassVar[str] = "Export a simulate job's passing conversations as chat training data."
    spec_schema: ClassVar[type[BaseModel]] = ExportSpec
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    @classmethod
    async def compile(
        cls, *, workspace, spec, entity_client, job_name, async_sdk, profile=None, options=None
    ) -> HelixJobSpec:
        spec = ExportSpec.model_validate(spec)
        return HelixJobSpec(steps=[task_step("export", "nemo_user_sim_plugin.tasks.export", spec.model_dump())])

    def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict:
        spec = ExportSpec.model_validate(config)
        work = ctx.storage.persistent
        curated, candidates, skipped = [], 0, []
        for job in spec.jobs:
            try:
                conversations = read_result_frame(
                    fetch_result(client, ctx.workspace, job, "trajectories", work / f"{job}-{TRAJECTORIES}")
                )
                evaluations = read_result_frame(
                    fetch_result(client, ctx.workspace, job, "evaluations", work / f"{job}-{EVALUATIONS}")
                )
            except NotFoundError:  # every conversation failed, or the run was never scored
                skipped.append(job)
                continue
            candidates += len(conversations)
            curated.append(select_trajectories(conversations, evaluations, profile=PROFILES[spec.quality]).curated)
        if not curated:
            raise ValueError(f"None of {', '.join(spec.jobs)} has scored conversations to export.")
        selected = pd.concat(curated, ignore_index=True)
        feedback = asyncio.run(
            review_feedback(
                AsyncIntakeClient.from_client(client.to_async()), ctx.workspace, set(selected.trajectory_id)
            )
        )
        kept = apply_review_feedback(selected, feedback, require_review=spec.require_review)

        output = work / TRAINING_DATA
        output.write_text("".join(json.dumps(record) + "\n" for record in training_records(kept)))
        files = FilesClient.from_client(client)
        fileset = CreateFilesetRequest(
            name=spec.fileset,
            purpose=FilesetPurpose.DATASET,
            description=f"Training data from UserSim runs {', '.join(spec.jobs)}",
        )
        files.create_fileset(workspace=ctx.workspace, body=fileset, exist_ok=True)
        files.upload_file(workspace=ctx.workspace, name=spec.fileset, path=TRAINING_DATA, content=output.read_bytes())
        ctx.results.save("training-data", output)
        return {
            "status": "completed",
            "candidates": candidates,
            "curated": len(selected),
            "exported": len(kept),
            "skipped_runs": skipped,
        }
