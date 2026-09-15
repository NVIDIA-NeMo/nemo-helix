# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the agent-evaluation job (AALGO-297).

These exercise the job against *real* execution seams, across the dimensions that
matter for this work:

* target type — Model and Agent endpoint targets pointed at an IGW mock provider
  (canned response, so no real model or key);
* metric form — an inline metric bundle, plus a stored ``MetricRef`` resolved against
  the live entity store;
* execution mode — in-process ``run_local`` and service-side ``submit`` on both the
  subprocess and docker backends, against the session ``subprocess_platform`` /
  ``docker_platform`` fixtures in ``conftest.py``. (Docker submit is xfail today — the
  cpu-tasks image predates this work; tracked in AALGO-301.)

Marked ``integration`` (auto-applied to ``/integration/`` paths). Model/Agent tests
need only the running platform's IGW.

Run directly::

    uv run pytest plugins/nemo-evaluator/tests/integration/test_agent_evaluate_job.py -v
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

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
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalTaskInput,
    AgentTarget,
    HarborRunnerTarget,
    ModelTarget,
)
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec, EvaluateJob
from nemo_evaluator.metric_refs import MetricRef
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_evaluator_sdk.enums import AgentFormat, ModelFormat
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values import GenericAgent, Model, RunConfigOnline, RunConfigOnlineModel
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.scheduler import NemoJobScheduler
from nemo_platform_plugin.sdk import NeMoPlatform
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.testing import add_mock_provider
from nmp.testing.e2e import wait_for_platform_job

#: Opt-in: these tests spin real ``nemo services`` platforms (subprocess/docker/auth), so they're
#: kept out of the standard CI integration job. Run them locally (or on demand) with
#: ``RUN_AGENT_EVAL_INTEGRATION=1``.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"),
        reason="opt-in; set RUN_AGENT_EVAL_INTEGRATION=1 to run (spins real nemo services platforms)",
    ),
]

WORKSPACE = "default"

#: Headers a job's task client carries (mirrors ``get_task_nemo_client``): an internal service principal the
#: default PDP policy grants full permissions. Authenticates test-side calls against the
#: auth-enabled platform without standing up OIDC.
SERVICE_PRINCIPAL_HEADERS = {"X-NMP-Principal-Id": "service:evaluator", "X-NMP-Internal": "true"}


# Pickle metrics defined in this test module BY VALUE so the cloudpickle bundle embeds the class
# itself — the submit-backend task runs in a subprocess that can't import this test module, and a
# by-reference pickle would fail to hydrate there.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _OutputContainsMetric:
    """Custom metric: scores 1.0 iff the trial's output contains the expected token (case-insensitive).

    Validates what the agent actually produced — unlike a built-in clean-exit signal — and exercises
    the inline user-defined-metric path end to end (bundled, shipped in the spec, hydrated at run time).
    """

    def __init__(self, expected: str) -> None:
        self.expected = expected

    @property
    def type(self) -> str:
        return "output-contains"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean("contains")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # noqa: A002
        text = input.candidate.output_text or ""
        return MetricResult(outputs=[MetricOutput(name="contains", value=self.expected.lower() in text.lower())])


def _output_contains_metric(expected: str) -> MetricInline:
    """An inline, user-defined metric that validates the agent's output text."""
    bundle = bundle_metric(_OutputContainsMetric(expected), CloudpickleMetricBundlePackager())
    return MetricInline.model_validate(bundle.model_dump(mode="json"))


class _OutputScoreMetric:
    """Custom *numeric* metric: 1.0 iff the trial's output contains the expected token, else 0.0.

    The continuous-score counterpart to :class:`_OutputContainsMetric`. A numeric output aggregates
    into a real ``count``/``mean`` on the run result (a boolean output lands in ``nan_count`` instead),
    so a caller can assert on how many samples scored and their mean.
    """

    def __init__(self, expected: str) -> None:
        self.expected = expected

    @property
    def type(self) -> str:
        return "output-score"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("match")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # noqa: A002
        text = input.candidate.output_text or ""
        return MetricResult(
            outputs=[MetricOutput(name="match", value=1.0 if self.expected.lower() in text.lower() else 0.0)]
        )


def _bundle_dir(run_result: dict) -> Path:
    """The persisted run bundle directory (trials/scores/summary) from a run_local result."""
    return Path(run_result["artifact"]["artifact_url"].removeprefix("file://"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _chat_completion(content: str) -> dict:
    """An OpenAI ``chat.completion`` response body whose assistant message is ``content``."""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _igw_chat_url(base_url: str, model_entity: str) -> str:
    """OpenAI-compatible chat URL for a model entity routed through the platform's IGW."""
    return f"{base_url}/apis/inference-gateway/v2/workspaces/{WORKSPACE}/model/{model_entity}/-/v1/chat/completions"


def _unique(prefix: str) -> str:
    """A unique entity name, so a rerun (e.g. pytest-rerunfailures) doesn't 409 on an existing one."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.timeout(300)
def test_run_local_model_target_scores_a_real_trial(subprocess_platform: str) -> None:
    # dim 1 (Model endpoint target): generate a trial against an IGW mock provider that returns
    # "DONE" (no real model/key), then score the trial output with the inline metric.
    sdk = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(sdk, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    model_name = _unique("model-judge")
    add_mock_provider(sdk, workspace=WORKSPACE, name=model_name, mock_response_body=_chat_completion("DONE"))

    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="ask",
                intent="Obtain a one-word reply from the model.",
                inputs=TaskInputs(instruction="Reply with the single word DONE and nothing else."),
                metrics=[_output_contains_metric("DONE")],
            )
        ],
        target=ModelTarget(
            model=Model(
                url=_igw_chat_url(subprocess_platform, model_name), name=model_name, format=ModelFormat.OPEN_AI
            ),
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(),
        ),
    )

    result = NemoJobScheduler().run_local(AgentEvalJob, input_spec.model_dump(mode="json"))

    assert result["status"] == "completed"
    bundle = _bundle_dir(result)
    trials = _read_jsonl(bundle / "trials.jsonl")
    assert trials[0]["output"]["output_text"] == "DONE"
    scores = _read_jsonl(bundle / "scores.jsonl")
    assert scores[0]["outputs"][0]["value"] in (True, 1.0)


@pytest.mark.timeout(300)
def test_run_local_agent_target_scores_a_real_trial(subprocess_platform: str) -> None:
    # dim 1 (Agent endpoint target): a generic-HTTP agent posts to an IGW mock provider returning
    # "DONE"; response_path extracts the assistant content, then the inline metric scores it.
    sdk = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(sdk, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    agent_name = _unique("agent-judge")
    add_mock_provider(sdk, workspace=WORKSPACE, name=agent_name, mock_response_body=_chat_completion("DONE"))

    agent = GenericAgent(
        url=_igw_chat_url(subprocess_platform, agent_name),
        name=agent_name,
        format=AgentFormat.GENERIC,
        body={"model": agent_name, "messages": [{"role": "user", "content": "Reply with DONE."}]},
        response_path="$.choices[0].message.content",
    )
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="ask",
                intent="Obtain a one-word reply from the agent.",
                inputs=TaskInputs(instruction="Reply with the single word DONE and nothing else."),
                metrics=[_output_contains_metric("DONE")],
            )
        ],
        target=AgentTarget(agent=agent, params=RunConfigOnline()),
    )

    result = NemoJobScheduler().run_local(AgentEvalJob, input_spec.model_dump(mode="json"))

    assert result["status"] == "completed"
    bundle = _bundle_dir(result)
    trials = _read_jsonl(bundle / "trials.jsonl")
    assert trials[0]["output"]["output_text"] == "DONE"
    scores = _read_jsonl(bundle / "scores.jsonl")
    assert scores[0]["outputs"][0]["value"] in (True, 1.0)


# --- submit: service-side execution -----------------------------------------


def _offline_trials_input_spec() -> dict:
    """Submitter-facing spec: one precomputed trial scored offline by one inline metric.

    No target — so no online generation and no IGW. Runs entirely inside the task container,
    isolating the docker-backend-wiring + entrypoint condition (rather than also depending on a model
    endpoint the image cannot reach)."""
    return AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="say-done",
                intent="Agent follows a trivial instruction and exits cleanly.",
                inputs=TaskInputs(instruction="Reply with the single word DONE and nothing else."),
                metrics=[_output_contains_metric("DONE")],
            )
        ],
        trials=[
            AgentEvalTrial(
                id="t-1",
                task_id="say-done",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="DONE"),
            )
        ],
    ).model_dump(mode="json")


def _offline_row_eval_spec() -> dict:
    """An offline row (``EvaluateJob``) spec built with a JSON-native (inline-bundled) built-in metric.

    ExactMatch is in ``MetricsUnion``, so ``InlineMetricBundlePackager`` serializes it as declarative
    JSON — no cloudpickle bytes to survive on the create request body. No target — the dataset's
    ``model_output`` is scored directly. Seeds a *row* job alongside an agent-eval job so the mixed
    collection list endpoints can be exercised.
    """
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
        InlineMetricBundlePackager(),
    )
    return EvaluateInputSpec.model_validate(
        {
            "metrics": [bundle.model_dump(mode="json")],
            "dataset": [{"expected": "blue", "model_output": "blue"}],
        }
    ).model_dump(mode="json")


def _offline_agent_eval_spec_no_metrics() -> dict:
    """An offline agent-eval spec: one task (no metrics) scored against one precomputed trial.

    Metrics are optional on an agent task, so omitting them keeps the spec fully JSON-native (no
    cloudpickle) — enough to *create* a persisted AgentEvalJob record, which is all the list-endpoint
    regression needs.
    """
    return AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="say-done",
                intent="Agent follows a trivial instruction and exits cleanly.",
                inputs=TaskInputs(instruction="Reply with the single word DONE and nothing else."),
            )
        ],
        trials=[
            AgentEvalTrial(
                id="t-1",
                task_id="say-done",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="DONE"),
            )
        ],
    ).model_dump(mode="json")


@pytest.mark.timeout(300)
def test_mixed_job_types_list_endpoints_do_not_cross_render(subprocess_platform: str) -> None:
    """Regression (reported by Studio): a workspace holding BOTH a row ``EvaluateJob`` and an
    ``AgentEvalJob`` must not 500 either collection's list endpoint.

    Both job types are owned by the evaluator plugin. Before the fix they shared one ``source`` tag,
    so ``GET .../agent-evaluate/jobs`` returned every evaluator job and then rendered each against
    ``AgentEvalSpec`` — a row spec failed validation and the list 500'd (and vice versa). The fix
    gives agent-evaluate jobs a distinct ``source`` (``service.AGENT_EVAL_JOB_SOURCE``), so each list
    is scoped to its own collection. The list bug fires on *persisted* records, so this only creates
    one job of each type (no run/wait) and asserts the list endpoints; JSON-native specs keep create
    off the cloudpickle path.

    Runs in a fresh workspace so the assertions see only these two jobs — the shared subprocess DB may
    hold legacy agent-eval records written under the old shared source (the documented pre-fix caveat),
    which would otherwise still cross-render into the row list.
    """
    workspace = _unique("mixed-list")
    client = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=workspace)
    ).data()

    row_resp = NemoJobScheduler().submit_remote(
        EvaluateJob, _offline_row_eval_spec(), base_url=subprocess_platform, workspace=workspace, profile="default"
    )
    row_name = row_resp.get("name") or row_resp.get("id")
    agent_resp = NemoJobScheduler().submit_remote(
        AgentEvalJob,
        _offline_agent_eval_spec_no_metrics(),
        base_url=subprocess_platform,
        workspace=workspace,
        profile="default",
    )
    agent_name = agent_resp.get("name") or agent_resp.get("id")
    assert row_name and agent_name, f"submit responses carried no name/id: {row_resp}, {agent_resp}"

    base = f"{subprocess_platform}/apis/evaluator/v2/workspaces/{workspace}"
    # The core regression: the agent-evaluate list must 200 (not 500) in a mixed workspace, and it
    # must contain only the agent job — the row job is filtered out by the distinct source.
    agent_list = httpx.get(f"{base}/agent-evaluate/jobs", params={"page_size": 100}, timeout=30)
    assert agent_list.status_code == 200, agent_list.text
    agent_names = {item["name"] for item in agent_list.json()["data"]}
    assert agent_name in agent_names
    assert row_name not in agent_names

    # And symmetrically: the row list must 200 and exclude the agent job.
    row_list = httpx.get(f"{base}/evaluate/jobs", params={"page_size": 100}, timeout=30)
    assert row_list.status_code == 200, row_list.text
    row_names = {item["name"] for item in row_list.json()["data"]}
    assert row_name in row_names
    assert agent_name not in row_names


def _harbor_eval_input_spec() -> dict:
    """Minimal Harbor target submission; compilation must reject it before task execution."""
    return AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="harbor-task",
                intent="Exercise Harbor backend compatibility validation.",
                inputs=TaskInputs(instruction="Reply with DONE."),
            )
        ],
        target=HarborRunnerTarget(agent_name="oracle"),
    ).model_dump(mode="json")


@pytest.mark.timeout(420)
def test_submit_over_taskset_ref_resolves_and_scores(subprocess_platform: str) -> None:
    # dim 2 (stored taskset ref) x dim 3 (submit): store a metric + two tasks + a taskset, then submit
    # an agent eval whose `tasks` is a TasksetRef (no inline tasks). Server-side to_spec must load the
    # taskset, expand BOTH member tasks, and resolve each task's stored MetricRef — all against the
    # live entity store — before the job runs. A Model target -> IGW mock provider keeps it hermetic.
    client = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()

    model_name = _unique("taskset-model")
    add_mock_provider(client, workspace=WORKSPACE, name=model_name, mock_response_body=_chat_completion("DONE"))

    # Store the metric that both tasks will reference. Numeric, so the run aggregate carries a real
    # count/mean (a boolean output would land in nan_count and obscure whether scoring succeeded). The
    # cloudpickle packager is explicit: storing a custom metric to the service requires opting in.
    metric_name = _unique("done-score")
    client.evaluator.metrics.create(
        metric_name,
        metric=_OutputScoreMetric("DONE"),
        metric_bundle_packager=CloudpickleMetricBundlePackager(),
        workspace=WORKSPACE,
    )

    # Store two tasks that reference the metric, then group them in a taskset.
    task_names = [_unique("ask-a"), _unique("ask-b")]
    for name in task_names:
        client.evaluator.tasks.create(
            name,
            task=TaskInput(
                spec=EvaluatorTaskDefinition(
                    kind="evaluator",
                    intent="Obtain a one-word reply from the model.",
                    inputs=TaskInputs(instruction="Reply with the single word DONE and nothing else."),
                    metrics=[MetricRef(f"{WORKSPACE}/{metric_name}")],
                )
            ),
        )
    taskset_name = _unique("done-suite")
    client.evaluator.tasksets.create(
        taskset_name,
        taskset=TasksetInput(tasks=[TaskRef(f"{WORKSPACE}/{name}") for name in task_names]),
    )

    # The point of the test: reference the stored taskset instead of inlining the tasks.
    spec = AgentEvalInputSpec(
        tasks=TasksetRef(f"{WORKSPACE}/{taskset_name}"),
        target=ModelTarget(
            model=Model(
                url=_igw_chat_url(subprocess_platform, model_name), name=model_name, format=ModelFormat.OPEN_AI
            ),
            prompt_template={"messages": [{"role": "user", "content": "{{item.instruction}}"}]},
            params=RunConfigOnlineModel(),
        ),
    ).model_dump(mode="json")

    response = NemoJobScheduler().submit_remote(
        AgentEvalJob, spec, base_url=subprocess_platform, workspace=WORKSPACE, profile="default"
    )
    job_name = response.get("name") or response.get("id")
    assert job_name, f"submit response carried no job name/id: {response}"

    job = wait_for_platform_job(client, job_name, WORKSPACE, timeout=360)
    assert job.status == "completed", f"job {job_name} ended {job.status!r}: {getattr(job, 'status_details', None)}"

    # The taskset expanded to BOTH members and both were scored: the numeric metric aggregates to
    # count == number of members (one sample per task, one trial each), with no NaNs, and mean == 1.0
    # because the mock model returns "DONE" for every task (so every task's output contains "DONE").
    result = client.evaluator.agent_eval_results.retrieve(job_name, workspace=WORKSPACE)
    assert (result.target_kind, result.target_name) == ("model", model_name)
    assert result.scores.scores, "run produced no aggregated scores"
    aggregate = result.scores.scores[0]
    assert aggregate.nan_count == 0, f"metric failed to score some samples: nan_count={aggregate.nan_count}"
    assert aggregate.count == len(task_names), f"expected one scored sample per member, got count={aggregate.count}"
    assert aggregate.mean == 1.0, f"every member's output should score 1.0, got mean={aggregate.mean}"


@pytest.mark.timeout(420)
def test_submit_model_target_under_auth_forwards_identity_to_igw(auth_subprocess_platform: str) -> None:
    # dim 1 (Model target) x dim 3 (submit) under auth.enabled: the submitted task's get_task_nemo_client
    # identity (X-NMP-Principal-Id: service:evaluator) must be forwarded to the evaluator's IGW
    # inference client (AgentEvalJob._build_evaluator) — otherwise the IGW returns 401 and the job
    # fails. A clean completion proves the forwarded service-principal headers authenticate online
    # inference under auth, with no bearer. (Probed directly too: service headers -> 200, none -> 401.)
    sdk = NeMoPlatform(base_url=auth_subprocess_platform, default_headers=SERVICE_PRINCIPAL_HEADERS, max_retries=2)
    client_from_platform(sdk, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    model_name = _unique("auth-model")
    add_mock_provider(sdk, workspace=WORKSPACE, name=model_name, mock_response_body=_chat_completion("DONE"))

    spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="ask",
                intent="Obtain a one-word reply from the model.",
                inputs=TaskInputs(instruction="Reply with the single word DONE and nothing else."),
                metrics=[_output_contains_metric("DONE")],
            )
        ],
        target=ModelTarget(
            model=Model(
                url=_igw_chat_url(auth_subprocess_platform, model_name), name=model_name, format=ModelFormat.OPEN_AI
            ),
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(),
        ),
    ).model_dump(mode="json")

    response = NemoJobScheduler().submit_remote(
        AgentEvalJob,
        spec,
        base_url=auth_subprocess_platform,
        workspace=WORKSPACE,
        profile="default",
        headers=SERVICE_PRINCIPAL_HEADERS,
    )
    job_name = response.get("name") or response.get("id")
    assert job_name, f"submit response carried no job name/id: {response}"

    job = wait_for_platform_job(sdk, job_name, WORKSPACE, timeout=360)
    assert job.status == "completed", f"job {job_name} ended {job.status!r}: {getattr(job, 'status_details', None)}"

    # Persistence under auth: the result-entity write goes through the job's async task SDK
    # (get_async_task_nemo_client) as service:evaluator on-behalf-of the creator. A retrievable record here
    # proves that delegated identity actually authorized the entity write end-to-end (not just the
    # IGW inference call) — the key validation of the async task-SDK identity parity.
    result = sdk.evaluator.agent_eval_results.retrieve(job_name, workspace=WORKSPACE)
    assert result.job_id == job_name
    assert (result.target_kind, result.target_name) == ("model", model_name)
    assert result.bundle_ref


@pytest.mark.timeout(300)
def test_submit_harbor_target_to_docker_backend_fails_fast(docker_platform: str) -> None:
    workspace = _unique("harbor-docker-guard")
    client = NeMoPlatform(base_url=docker_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=workspace)
    ).data()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        NemoJobScheduler().submit_remote(
            AgentEvalJob,
            _harbor_eval_input_spec(),
            base_url=docker_platform,
            workspace=workspace,
            profile="default",
        )

    response = exc_info.value.response
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "profile 'default'" in detail
    assert "backend 'docker'" in detail
    assert "Harbor targets currently require local execution or the subprocess backend" in detail

    jobs = httpx.get(
        f"{docker_platform}/apis/evaluator/v2/workspaces/{workspace}/agent-evaluate/jobs",
        params={"page_size": 100},
        timeout=30,
    )
    assert jobs.status_code == 200, jobs.text
    assert jobs.json()["data"] == []


@pytest.mark.timeout(600)
@pytest.mark.xfail(
    reason="agent-eval can't run under the docker backend until the cpu-tasks image is rebuilt with "
    "this work: the published image predates the nemo_evaluator.tasks.agent_evaluate entrypoint "
    "(container exits with ModuleNotFoundError). This submits an offline trials spec (no online "
    "generation, no online target, no IGW), so the stale image is the only remaining failure cause — the "
    "xfail flips the moment the image ships the entrypoint. Tracked in AALGO-301.",
    strict=False,
)
def test_submit_to_docker_backend_runs_agent_eval(docker_platform: str) -> None:
    # dim 4 (docker backend): the platform routes cpu/default to the docker backend (no subprocess
    # executor is registered, so the step isn't rerouted), so the agent-eval task runs in the
    # cpu-tasks container. Verified to genuinely reach the docker backend (it creates a container
    # from the cpu-tasks image); it fails today because that image predates this work — hence xfail.
    # An offline trials spec keeps the task self-contained in-container, so this isolates the
    # backend-wiring + entrypoint condition rather than also depending on a live model endpoint.
    client = NeMoPlatform(base_url=docker_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()

    response = NemoJobScheduler().submit_remote(
        AgentEvalJob,
        _offline_trials_input_spec(),
        base_url=docker_platform,
        workspace=WORKSPACE,
        profile="default",
    )
    job_name = response.get("name") or response.get("id")
    assert job_name, f"submit response carried no job name/id: {response}"

    job = wait_for_platform_job(client, job_name, WORKSPACE, timeout=480)
    assert job.status == "completed"
