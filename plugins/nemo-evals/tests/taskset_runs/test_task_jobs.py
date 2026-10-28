# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task Job's create request matches the Jobs API contract."""

from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec, PreparedHarborEnvironment, PreparedImage
from nemo_evals.taskset_runs.task_jobs import task_job_request
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

IMAGE = PreparedImage(image_ref="registry.invalid/agent@sha256:" + "a" * 64, preparation_key="k")
ENVIRONMENT = PreparedHarborEnvironment(task_revision_digest="r", archive_digest="d", agent=IMAGE)
NAMES = ["run-0123456789abcdef-task-0", "run-0123456789abcdef-task-1"]


async def test_request_matches_the_jobs_api_contract(harbor_plan, harbor_run):
    request = await task_job_request(harbor_plan, harbor_run, 0, ENVIRONMENT)
    golden = CreateHelixJobRequest.model_validate_json(request.model_dump_json())
    assert golden == request
    assert request.source == TASK_JOB_SOURCE and request.name == NAMES[0]
    assert HarborTaskJobSpec.model_validate(request.spec).task_index == 0
    assert request.platform_spec.steps[-1].config == request.spec
