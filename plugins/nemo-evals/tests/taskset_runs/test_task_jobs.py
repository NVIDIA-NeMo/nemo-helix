# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task Job's create request matches the Jobs API contract."""

from nemo_agents_plugin.config import AgentsConfig, DeploymentsRunnerConfig
from nemo_evals.jobs.gateway_models import GATEWAY_URL_OVERRIDE_ENVVAR
from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec, PreparedHarborEnvironment, PreparedImage
from nemo_evals.taskset_runs.specs import TasksetRunPlan
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


async def test_a_task_job_reaches_the_gateway_the_run_was_submitted_with(harbor_plan, harbor_run, monkeypatch):
    """Recovery recompiles a create of unknown outcome and adopts the Job only if it matches.

    Reading the platform's current override instead would turn a redeploy that changed it into a
    job_name_conflict failure of the task.
    """
    plan = harbor_plan.model_copy(update={"gateway_url_override": "https://gw.ns.svc:8443"})
    later = AgentsConfig(deployments=DeploymentsRunnerConfig(gateway_url_override="https://gw2.ns.svc:8443"))
    monkeypatch.setattr(AgentsConfig, "get", classmethod(lambda cls: later))

    request = await task_job_request(plan, harbor_run, 0, ENVIRONMENT)

    environment = {v.name: v.value for v in request.platform_spec.steps[-1].environment or []}
    assert environment[GATEWAY_URL_OVERRIDE_ENVVAR] == "https://gw.ns.svc:8443"


def test_a_plan_stored_without_a_gateway_url_override_has_none(harbor_plan):
    stored = harbor_plan.model_dump(mode="json", exclude={"gateway_url_override"})

    assert TasksetRunPlan.model_validate(stored).gateway_url_override is None
