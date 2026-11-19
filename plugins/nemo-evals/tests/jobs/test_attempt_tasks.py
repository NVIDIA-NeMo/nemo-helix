# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-evaluate step configs leave their tasks on the job attempt, and workers read them back."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_evals.jobs import agent_compiler
from nemo_evals.jobs import attempt_tasks as attempt_tasks_module
from nemo_evals.jobs.agent_compiler import compile_agent_eval_job
from nemo_evals.jobs.agent_evaluate import AgentEvalJob, AsyncAgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec
from nemo_evals.jobs.attempt_tasks import (
    TASK_SERVICE_NAME,
    TASKS_DIGEST_KEY,
    TASKS_FROM_ATTEMPT,
    TASKS_FROM_KEY,
    AttemptTasksError,
    frozen_tasks_digest,
    load_attempt_tasks,
    step_config_without_tasks,
)
from nemo_evals.jobs.harbor_task_dispatch import HarborTaskAgentEvalJob, HarborTaskDispatchJob
from nemo_evals.jobs.harbor_task_spec import (
    HarborTaskJobSpec,
    PreparedHarborEnvironment,
    PreparedImage,
    build_task_job_input,
)
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.types import HelixJobAttemptResponse, HelixJobSpec, HelixJobStatus

BASE = "http://platform.test"
JOB = "eval-job"
ATTEMPT = "attempt-1"


@pytest.fixture
def task_spec(harbor_plan, harbor_run) -> HarborTaskJobSpec:
    agent = PreparedImage(image_ref="registry.invalid/agent@sha256:" + "a" * 64, preparation_key="ka")
    environment = PreparedHarborEnvironment(task_revision_digest="r", archive_digest="d", agent=agent)
    return build_task_job_input(harbor_plan, harbor_run, 1, environment)


def _ctx(*, job_id: str | None = JOB, attempt_id: str | None = ATTEMPT) -> JobContext:
    ctx = MagicMock(spec=JobContext)
    ctx.workspace = "default"
    ctx.job_id = job_id
    ctx.attempt_id = attempt_id
    return ctx


def _compiled(spec: AgentEvalSpec) -> HelixJobSpec:
    return HelixJobSpec.model_validate(compile_agent_eval_job(spec, use_subprocess=True).model_dump(mode="json"))


def _attempt_body(stored_spec: dict, platform_spec: HelixJobSpec) -> dict:
    now = datetime(2026, 11, 9, tzinfo=UTC)
    return HelixJobAttemptResponse(
        id=ATTEMPT,
        name=ATTEMPT,
        job=JOB,
        workspace="default",
        seq=0,
        spec=stored_spec,
        platform_spec=platform_spec,
        status=HelixJobStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    ).model_dump(mode="json")


class _Platform:
    """Serves one attempt and records each request path."""

    def __init__(self, body: dict | None, *, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        return httpx.Response(self.status_code, json=self.body or {"detail": "nope"}, request=request)

    def sync_client(self) -> NemoClient:
        return NemoClient(
            base_url=BASE,
            workspace="default",
            http_client=httpx.Client(transport=httpx.MockTransport(self)),
            owns_http_client=True,
        )


class _TaskClients:
    """Stands in for the platform's client provider and records each task client it builds."""

    def __init__(self, platform: _Platform) -> None:
        self.platform = platform
        self.built: list[tuple[str, str | None, NemoClient]] = []

    def __call__(self, service_name: str, *, workspace: str | None = None) -> NemoClient:
        client = self.platform.sync_client()
        self.built.append((service_name, workspace, client))
        return client


# --- Compile -----------------------------------------------------------------------------------------------


def test_the_step_config_carries_a_digest_of_the_tasks_instead_of_the_tasks(harbor_spec: AgentEvalSpec) -> None:
    dumped = harbor_spec.model_dump(mode="json")
    (step,) = _compiled(harbor_spec).steps
    assert "tasks" not in step.config and "trials" not in step.config
    assert step.config[TASKS_FROM_KEY] == TASKS_FROM_ATTEMPT
    assert step.config[TASKS_DIGEST_KEY] == frozen_tasks_digest(dumped)
    assert {key: value for key, value in step.config.items() if key not in {TASKS_FROM_KEY, TASKS_DIGEST_KEY}} == {
        key: value for key, value in dumped.items() if key not in {"tasks", "trials"}
    }


def test_the_step_config_does_not_grow_with_the_task_count(
    harbor_spec: AgentEvalSpec, harbor_resolved_task: Callable[..., Any]
) -> None:
    many = harbor_spec.model_copy(update={"tasks": [harbor_resolved_task(f"task-{i}") for i in range(300)]})
    (few_step,) = _compiled(harbor_spec).steps
    (many_step,) = _compiled(AgentEvalSpec.model_validate(many.model_dump(mode="json"))).steps
    assert len(json.dumps(many_step.config)) == len(json.dumps(few_step.config))


def test_a_step_config_kubernetes_cannot_hold_is_refused_at_compile(
    harbor_spec: AgentEvalSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_compiler, "MAX_STEP_CONFIG_BYTES", 100)
    with pytest.raises(HelixJobCompilationError, match="'agent-evaluate' step's config is .* bytes"):
        compile_agent_eval_job(harbor_spec, use_subprocess=True)
    with pytest.raises(HelixJobCompilationError):
        compile_agent_eval_job(harbor_spec, use_subprocess=False)


def test_offline_trials_leave_the_step_config_too() -> None:
    spec = {"target": None, "tasks": [{"id": "t"}], "trials": [{"id": "trial", "task_id": "t"}], "labels": {}}
    config = step_config_without_tasks(spec)
    assert set(config) == {"target", "labels", TASKS_FROM_KEY, TASKS_DIGEST_KEY}
    assert config[TASKS_DIGEST_KEY] != frozen_tasks_digest({**spec, "trials": None})


# --- Digest ------------------------------------------------------------------------------------------------


def test_the_digest_ignores_key_order_and_other_spec_fields() -> None:
    spec = {"tasks": [{"id": "t", "spec": {"b": 1, "a": "é"}}], "trials": None, "target": {"kind": "harbor"}}
    stored = json.loads(json.dumps({"kind": "agent-evaluate", **dict(reversed(list(spec.items())))}))
    stored["tasks"][0]["spec"] = {"a": "é", "b": 1}
    assert frozen_tasks_digest(stored) == frozen_tasks_digest(spec)
    assert frozen_tasks_digest(spec).startswith("sha256-v1:")


@pytest.mark.parametrize(
    "change",
    [
        lambda spec: spec["tasks"][0]["spec"]["metrics"][0].update(payload="b3RoZXI="),  # another base64 bundle
        lambda spec: spec["tasks"][0]["reference"].update(answer=1.0),  # int and float differ
        lambda spec: spec["tasks"].reverse(),
        lambda spec: spec.update(trials=[]),
    ],
)
def test_the_digest_changes_with_any_task_or_trial_change(change: Callable[[dict], None]) -> None:
    spec = {
        "tasks": [
            {"id": "a", "spec": {"metrics": [{"payload": "Y2xvdWRwaWNrbGU="}]}, "reference": {"answer": 1}},
            {"id": "b", "spec": {"metrics": []}, "reference": {}},
        ],
        "trials": None,
    }
    before = frozen_tasks_digest(spec)
    change(spec)
    assert frozen_tasks_digest(spec) != before


# --- Worker ------------------------------------------------------------------------------------------------


def _stored_and_compiled(spec: AgentEvalSpec) -> tuple[dict, HelixJobSpec, dict]:
    """The job record's spec as the Jobs API stores it, the compiled job, and its step config."""
    stored = {**json.loads(json.dumps(spec.model_dump(mode="json"))), "kind": "agent-evaluate"}
    compiled = _compiled(spec)
    return stored, compiled, compiled.steps[-1].config


def test_a_marked_config_gets_its_tasks_back_from_its_own_attempt(harbor_spec: AgentEvalSpec) -> None:
    stored, compiled, config = _stored_and_compiled(harbor_spec)
    platform = _Platform(_attempt_body(stored, compiled))
    restored = load_attempt_tasks(config, ctx=_ctx(), client=platform.sync_client())
    assert platform.paths == [f"/apis/jobs/v2/workspaces/default/jobs/{JOB}/attempts/{ATTEMPT}"]
    assert AgentEvalSpec.model_validate(restored) == harbor_spec


@pytest.mark.parametrize("job_class", [AgentEvalJob, AsyncAgentEvalJob])
def test_both_workers_run_the_attempts_tasks(
    harbor_spec: AgentEvalSpec, monkeypatch: pytest.MonkeyPatch, job_class: type[AgentEvalJob | AsyncAgentEvalJob]
) -> None:
    stored, compiled, config = _stored_and_compiled(harbor_spec)
    platform = _Platform(_attempt_body(stored, compiled))
    seen: list[AgentEvalSpec] = []

    def run(self: object, config: dict, **_: object) -> dict:
        # The real run validates the config against the spec, which forbids unknown keys.
        seen.append(AgentEvalSpec.model_validate(config))
        return {"status": "completed"}

    if job_class is AgentEvalJob:
        monkeypatch.setattr(AgentEvalJob, "_run_sync", run)
        AgentEvalJob().run(config, ctx=_ctx(), client=platform.sync_client())
    else:
        monkeypatch.setattr(AsyncAgentEvalJob, "_run_with_client", run)
        monkeypatch.setattr(attempt_tasks_module, "get_task_nemo_client", _TaskClients(platform))
        AsyncAgentEvalJob().run(config, ctx=_ctx(), async_client=MagicMock(spec=AsyncNemoClient))
    assert seen == [harbor_spec]
    assert len(platform.paths) == 1


def test_the_async_worker_reads_through_a_provider_built_task_client(
    harbor_spec: AgentEvalSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read uses a task client the platform's provider builds (its CA, transport and routing), then closes it.

    The async worker's own client is never used for it: a loop-local copy of it would lose those settings.
    """
    stored, compiled, config = _stored_and_compiled(harbor_spec)
    task_clients = _TaskClients(_Platform(_attempt_body(stored, compiled)))
    monkeypatch.setattr(attempt_tasks_module, "get_task_nemo_client", task_clients)
    async_client = MagicMock(spec=AsyncNemoClient)
    monkeypatch.setattr(AsyncAgentEvalJob, "_run_with_client", lambda self, config, **_: {"status": "completed"})

    AsyncAgentEvalJob().run(config, ctx=_ctx(), async_client=async_client)

    ((service_name, workspace, client),) = task_clients.built
    assert (service_name, workspace) == (TASK_SERVICE_NAME, "default")
    assert client.http_client.is_closed
    assert async_client.mock_calls == []


def test_a_config_without_the_marker_reads_nothing(harbor_spec: AgentEvalSpec) -> None:
    config = harbor_spec.model_dump(mode="json")
    assert load_attempt_tasks(config, ctx=_ctx(job_id=None, attempt_id=None), client=None) is config


def test_tasks_that_dont_match_the_digest_are_refused(harbor_spec: AgentEvalSpec) -> None:
    stored, compiled, config = _stored_and_compiled(harbor_spec)
    stored["tasks"] = stored["tasks"][:1]
    platform = _Platform(_attempt_body(stored, compiled))
    with pytest.raises(AttemptTasksError, match="don't match its step config"):
        load_attempt_tasks(config, ctx=_ctx(), client=platform.sync_client())


def test_a_failed_read_fails_the_run(harbor_spec: AgentEvalSpec) -> None:
    _, _, config = _stored_and_compiled(harbor_spec)
    platform = _Platform(None, status_code=403)
    with pytest.raises(AttemptTasksError, match="could not read the tasks"):
        load_attempt_tasks(config, ctx=_ctx(), client=platform.sync_client())


@pytest.mark.parametrize(
    ("override", "ctx", "message"),
    [
        ({TASKS_FROM_KEY: "fileset"}, _ctx(), "unknown task source 'fileset'"),
        ({TASKS_DIGEST_KEY: "md5:abc"}, _ctx(), "missing or not a sha256-v1: digest"),
        ({}, _ctx(attempt_id=None), "this run has no job attempt"),
    ],
)
def test_a_config_the_worker_cannot_trust_is_refused(
    harbor_spec: AgentEvalSpec, override: dict, ctx: JobContext, message: str
) -> None:
    _, _, config = _stored_and_compiled(harbor_spec)
    with pytest.raises(AttemptTasksError, match=message):
        load_attempt_tasks({**config, **override}, ctx=ctx, client=_Platform(None).sync_client())


# --- Taskset-run task Jobs ---------------------------------------------------------------------------------


async def test_taskset_run_task_jobs_keep_their_one_task_in_the_step_config(task_spec: HarborTaskJobSpec) -> None:
    (step,) = (await HarborTaskDispatchJob.compile(spec=task_spec, profile="k8s-cpu")).steps
    assert TASKS_FROM_KEY not in step.config and TASKS_FROM_KEY not in step.config["spec"]
    assert HarborTaskJobSpec.model_validate(step.config).spec.tasks == task_spec.spec.tasks


def test_taskset_run_task_jobs_never_read_their_attempt(
    task_spec: HarborTaskJobSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object, **__: object) -> dict:
        raise AssertionError("a taskset-run task Job read its attempt")

    monkeypatch.setattr("nemo_evals.jobs.agent_evaluate.load_attempt_tasks", refuse)
    for check in ("check_execution_deadline", "check_sandbox_server", "check_tasks_reach_the_gateway"):
        monkeypatch.setattr(f"nemo_evals.jobs.harbor_task_dispatch.{check}", lambda *_: None)
    seen: list[dict] = []
    monkeypatch.setattr(
        AsyncAgentEvalJob, "_run_with_client", lambda self, config, **_: seen.append(config) or {"status": "completed"}
    )
    HarborTaskAgentEvalJob().run(task_spec.model_dump(mode="json"), ctx=_ctx(), async_client=MagicMock())
    assert [AgentEvalSpec.model_validate(config) for config in seen] == [task_spec.spec]
