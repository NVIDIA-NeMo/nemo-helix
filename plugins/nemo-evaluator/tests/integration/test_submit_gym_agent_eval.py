# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting a Gym runner as an agent-evaluation job, against a live platform.

Covers the seam the unit tests deliberately stub: that a live ``GymAgentTaskRunner`` survives
description as a target spec, transport to the service, and persistence — so what comes back out is
the evaluation that was configured going in.

**Scope is submission, not execution.** Running a Gym eval additionally needs the ``gym`` CLI on the
job's PATH and tasks carrying ``gym_dataset_path`` metadata from ``discover_gym_tasks``; neither is
a property of the submit path, and asserting on them here would make this test fail for reasons that
have nothing to do with what it covers. The job is therefore submitted and its stored spec inspected,
not run to completion.

Run directly::

    RUN_AGENT_EVAL_INTEGRATION=1 uv run pytest \\
        plugins/nemo-evaluator/tests/integration/test_submit_gym_agent_eval.py -v
"""

from __future__ import annotations

import os
import sys
import uuid

import cloudpickle
import httpx
import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetricInline,
    TaskInput,
    TaskInputs,
    TaskRef,
    TasksetInput,
    TasksetRef,
)
from nemo_evaluator.sdk.job_resources import AgentEvaluatorJobResource
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_platform_plugin.sdk import NeMoPlatform

WORKSPACE = "default"

#: Opt-in: shares the evaluator-plugin integration opt-in (spins a real ``nemo services`` platform).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"),
        reason="opt-in; set RUN_AGENT_EVAL_INTEGRATION=1 to run (spins real nemo services platforms)",
    ),
]


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


def _stored_taskset(client: NeMoPlatform) -> str:
    """A one-task taskset to reference. Its content is irrelevant — only the reference travels."""
    task_name = _unique("gym-submit-task")
    client.evaluator.tasks.create(
        task_name,
        task=TaskInput(
            spec=EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Placeholder task; this test asserts on submission, not execution.",
                inputs=TaskInputs(instruction="unused"),
                metrics=[_inline_metric()],
            )
        ),
    )
    taskset_name = _unique("gym-submit-suite")
    client.evaluator.tasksets.create(
        taskset_name,
        taskset=TasksetInput(tasks=[TaskRef(f"{WORKSPACE}/{task_name}")]),
    )
    return taskset_name


def test_a_live_gym_runner_submits_and_round_trips_through_the_service(subprocess_platform: str) -> None:
    client = NeMoPlatform(base_url=subprocess_platform, workspace=WORKSPACE, max_retries=2)
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
    # execution path polls through `nmp.testing` rather than this resource.
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
