# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting a Gym runner as an agent-evaluation job, against a live platform.

Covers the seam the unit tests deliberately stub: that a live ``GymAgentTaskRunner`` survives
description as a target spec, transport to the service, and persistence — so what comes back out is
the evaluation that was configured going in.

**Scope is submission, not execution.** Running a Gym eval additionally needs the ``gym`` CLI on the
job's PATH and tasks carrying discovery-compatible ``gym_row`` inputs and ``gym_row_extras`` metadata.
The dataset path is optional provenance. This fixture supplies valid row content, and asserting on them here would make this test fail for reasons that
have nothing to do with what it covers. The job is therefore submitted and its stored spec inspected,
not run to completion.

Run directly::

    uv run pytest plugins/nemo-evaluator/tests/integration/test_submit_gym_agent_eval.py -v
"""

from __future__ import annotations

import sys
import uuid

import cloudpickle
import httpx
import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetadataItem,
    MetricInline,
    TaskInput,
    TaskInputs,
    TaskRef,
    TasksetInput,
    TasksetRef,
)
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import GymPlacement
from nemo_evaluator.sdk.job_resources import AgentEvaluatorJobResource
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values import SecretRef
from nemo_helix_plugin.client.errors import UnprocessableEntityError
from nemo_helix_plugin.sdk import NeMoHelix

WORKSPACE = "default"

#: Runs in CI, unlike the ``RUN_AGENT_EVAL_INTEGRATION`` siblings in this directory.
pytestmark = pytest.mark.integration


# Pickle metrics defined in this module BY VALUE, so the bundle embeds the class itself: the service
# resolves the taskset in its own process, which cannot import this test module.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class _AlwaysOneMetric:
    """A trivial metric so the stored task is valid. Never scored — this test never runs the job."""

    @property
    def type(self) -> str:
        return "always-one"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # pragma: no cover - never run
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


def _inline_metric() -> MetricInline:
    """Bundle the metric into the inline wire form a stored task requires."""
    bundle = bundle_metric(_AlwaysOneMetric(), CloudpickleMetricBundlePackager())
    return MetricInline.model_validate(bundle.model_dump(mode="json"))


def _stored_taskset(client: NeMoHelix) -> str:
    """A one-task taskset with valid Gym row content for submission validation."""
    task_name = _unique("gym-submit-task")
    client.evaluator.tasks.create(
        task_name,
        task=TaskInput(
            spec=EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Placeholder task; this test asserts on submission, not execution.",
                inputs=TaskInputs.model_validate({"gym_row": {"input": "Reply DONE"}}),
                metrics=[_inline_metric()],
            ),
            metadata=[MetadataItem(key="gym_row_extras", value={})],
        ),
    )
    taskset_name = _unique("gym-submit-suite")
    client.evaluator.tasksets.create(
        taskset_name,
        taskset=TasksetInput(tasks=[TaskRef(f"{WORKSPACE}/{task_name}")]),
    )
    return taskset_name


def test_a_live_gym_runner_submits_and_round_trips_through_the_service(subprocess_platform: str) -> None:
    client = NeMoHelix(base_url=subprocess_platform, workspace=WORKSPACE, max_retries=2)
    taskset_name = _stored_taskset(client)

    # Non-default values throughout: a field dropped anywhere along runner -> target -> wire ->
    # storage would come back as its default, which is exactly the silent divergence this path
    # exists to prevent.
    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
            bind_resources_server=False,
            num_repeats=3,
            concurrency=7,
            hydra_params={"simple_agent": {"responses_api_agents": {"x": 1}}},
            env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"},
            reward_key="score",
        )
    )

    job = client.evaluator.submit(tasks=TasksetRef(f"{WORKSPACE}/{taskset_name}"), target=runner)

    assert isinstance(job, AgentEvaluatorJobResource), (
        "a taskset submission must return the agent job resource, not the row-evaluation one"
    )
    assert job.name, "the service returned no job name"

    # The resource's own status route resolves for an agent job. Worth asserting rather than
    # assuming: `job_route_base_url` builds the status path from `/evaluate/jobs` while agent jobs
    # live under `/agent-evaluate/jobs`, and a review round questioned whether that 404s. It does
    # not — the status lookup ignores the collection prefix — but nothing else covers it, since the
    # execution path polls through `nhx.testing` rather than this resource.
    status = job.get_job_status()
    assert status.status, f"the agent job's status route returned no status: {status!r}"

    # Read back over the wire rather than trusting the submit response, so what is asserted is what
    # the service *stored*. Fetched with a plain GET because ``evaluator.get_job_resource`` is the
    # row-evaluation reader — it validates the job's spec as an ``EvaluateSpec``, which an agent
    # job's spec is not. An agent-job reader is worth having; it is not part of submission.
    fetched = httpx.get(
        f"{subprocess_platform}/apis/evaluator/v2/workspaces/{WORKSPACE}/agent-evaluate/jobs/{job.name}",
        timeout=30,
    )
    assert fetched.status_code == 200, fetched.text
    target = fetched.json()["spec"]["target"]

    # The target survived as a Gym target carrying the runner's own settings, rather than defaults
    # or a different kind.
    assert target["kind"] == "gym"
    assert target["resources_server"] == "mcqa"
    assert target["num_repeats"] == 3
    assert target["concurrency"] == 7
    assert target["reward_key"] == "score"
    assert target["bind_resources_server"] is False
    assert target["hydra_params"] == {"simple_agent": {"responses_api_agents": {"x": 1}}}
    assert target["env_vars"] == {"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"}


def test_a_secret_reference_and_agent_ref_name_survive_submission(subprocess_platform: str) -> None:
    """A secret reference and an agent instance, from the runner and the placement, on one target.

    The secret is created for real, so the reference names something the service can resolve rather
    than a string that happens to parse.
    """
    client = NeMoHelix(base_url=subprocess_platform, workspace=WORKSPACE, max_retries=2)
    secret_name = _unique("gym-model-key")
    client.secrets.create(name=secret_name, value="sk-not-a-real-key")
    taskset_name = _stored_taskset(client)

    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="mcqa",
            env_secrets={"EXTERNAL_MODEL_API_KEY": SecretRef(f"{WORKSPACE}/{secret_name}")},
        )
    )

    job = client.evaluator.submit(
        tasks=TasksetRef(f"{WORKSPACE}/{taskset_name}"),
        target=runner,
        placement=GymPlacement(agent_ref_name="mcqa_simple_agent"),
    )

    fetched = httpx.get(
        f"{subprocess_platform}/apis/evaluator/v2/workspaces/{WORKSPACE}/agent-evaluate/jobs/{job.name}",
        timeout=30,
    )
    assert fetched.status_code == 200, fetched.text
    target = fetched.json()["spec"]["target"]

    assert target["env_secrets"] == {"EXTERNAL_MODEL_API_KEY": f"{WORKSPACE}/{secret_name}"}
    assert target["agent_ref_name"] == "mcqa_simple_agent"


def test_an_environment_fileset_reaches_the_compiler_from_a_runner(subprocess_platform: str) -> None:
    """A FileSet environment named on the placement is resolved by the service, not dropped.

    Executing one needs a Kubernetes or Volcano profile with a shared PVC, which this subprocess
    deployment does not have — so the job is refused at compile time. That refusal *is* the
    assertion: reaching a FileSet-only branch of the compiler proves ``environment`` travelled from
    the placement through ``runner_to_target`` onto the spec. A dropped field would compile cleanly
    as an ordinary colocated Gym run, which is the silent wrong answer this guards.
    """
    client = NeMoHelix(base_url=subprocess_platform, workspace=WORKSPACE, max_retries=2)
    fileset_name = _unique("gym-env")
    client.files.filesets.create(name=fileset_name, purpose="environment")
    taskset_name = _stored_taskset(client)

    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="custom_greeting",
        )
    )
    placement = GymPlacement(environment=FilesetRef(root=f"{WORKSPACE}/{fileset_name}"))

    with pytest.raises(UnprocessableEntityError) as excinfo:
        client.evaluator.submit(tasks=TasksetRef(f"{WORKSPACE}/{taskset_name}"), target=runner, placement=placement)

    message = str(excinfo.value)
    assert fileset_name in message or "FileSet" in message, (
        f"the refusal must come from the FileSet path rather than a generic spec error: {message}"
    )
