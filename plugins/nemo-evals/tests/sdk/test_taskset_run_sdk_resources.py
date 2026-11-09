# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Evaluator.taskset_runs``, run through the real typed client against ``httpx.MockTransport``.

Every test runs against the sync and the async resource.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from nemo_evals.api.fields import TasksetRef
from nemo_evals.api.taskset_run_schemas import (
    HarborTaskJobsExecutionInput,
    TasksetRunInput,
    TasksetRunStatus,
    request_digest,
)
from nemo_evals.jobs.agent_spec import (
    AgentEvalInputSpec,
    HarborBuiltinAgentSource,
    HarborRunnerTarget,
    RegisteredAgentSource,
)
from nemo_evals.runs_cli import run_request
from nemo_evals.sdk import taskset_run_resources
from nemo_evals.sdk.taskset_run_resources import (
    AsyncEvaluatorTasksetRunsResource,
    EvaluatorTasksetRunsResource,
    TasksetRunResults,
    TasksetRunSubmitError,
)
from nemo_helix_plugin.client.errors import (
    ConflictError,
    InternalServerError,
    NemoResponseValidationError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.client.retry import retry_delay
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY, RetryPolicy
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient, EvaluatorClient

BASE = "http://platform.test"
RUNS = "/apis/evals/v2/workspaces/default/agent-evaluate/runs"
RUN_ID = "run-0123456789abcdef"
DIGEST = "e" * 64
TASKSET_PATH = "/apis/evals/v2/workspaces/default/tasksets/suite/revisions/" + DIGEST
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
FAST_PLATFORM_POLICY = replace(PLATFORM_DEFAULT_RETRY_POLICY, backoff_base=0.0)

Handler = Callable[[httpx.Request], httpx.Response]


def run_status(**overrides: Any) -> dict[str, Any]:
    status = TasksetRunStatus(
        run_id=RUN_ID,
        workspace="default",
        taskset_ref=f"default/suite#{DIGEST}",
        phase="running",
        result_state="pending",
        accepted_at=NOW,
        timeout_seconds=3600,
        execution_deadline_at=NOW,
        finalization_deadline_at=NOW,
        idempotency_key="k",
    ).model_dump(mode="json", exclude_none=True)
    return status | overrides


def taskset(*names: str) -> dict[str, Any]:
    return {
        "id": "ts",
        "name": "suite",
        "workspace": "default",
        "tasks": [f"default/{name}#{'1' * 64}" for name in names],
        "revision": 1,
        "tags": {},
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }


def row(index: int, phase: str = "registered", **extra: Any) -> dict[str, Any]:
    return {"task": f"task-{index}", "task_index": index, "phase": phase} | extra


def page(items: list[dict[str, Any]], *, number: int = 1, pages: int = 1, sort: str | None = None) -> dict[str, Any]:
    pagination = {
        "page": number,
        "page_size": 1000,
        "current_page_size": len(items),
        "total_pages": pages,
        "total_results": len(items),
    }
    return {"data": items, "pagination": pagination} | ({"sort": sort} if sort else {})


def result_record(name: str) -> dict[str, Any]:
    return {
        "id": f"r-{name}",
        "name": name,
        "workspace": "default",
        "job_id": name,
        "scores": {"scores": [{"name": "reward", "count": 1, "nan_count": 0, "mean": 1.0}]},
        "bundle_ref": f"default/{name}#bundle",
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }


def harbor_request(target: HarborRunnerTarget | None = None, *, key: str | None = "key-1") -> TasksetRunInput:
    return TasksetRunInput(
        spec=AgentEvalInputSpec(
            tasks=TasksetRef(f"default/suite#{DIGEST}"),
            target=target or HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle")),
        ),
        execution=HarborTaskJobsExecutionInput(),
        idempotency_key=key,
    )


class Runs:
    """One resource, sync or async, behind the sync interface the tests use."""

    def __init__(self, resource: EvaluatorTasksetRunsResource | AsyncEvaluatorTasksetRunsResource) -> None:
        self.resource = resource

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self.resource, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            if inspect.isasyncgen(result):
                return asyncio.run(_collect(result))
            if inspect.isgenerator(result):
                return list(result)
            if inspect.iscoroutine(result):
                return asyncio.run(result)
            return result

        return call


async def _collect(generator: Any) -> list[Any]:
    return [item async for item in generator]


@pytest.fixture(params=["sync", "async"])
def make_runs(request: pytest.FixtureRequest) -> Callable[..., Runs]:
    def make(handler: Handler, *, retry: RetryPolicy | None = None) -> Runs:
        transport = httpx.MockTransport(handler)
        if request.param == "sync":
            client = EvaluatorClient(base_url=BASE, http_client=httpx.Client(transport=transport), retry=retry)
            return Runs(EvaluatorTasksetRunsResource(client))
        async_client = AsyncEvaluatorClient(
            base_url=BASE, http_client=httpx.AsyncClient(transport=transport), retry=retry
        )
        return Runs(AsyncEvaluatorTasksetRunsResource(async_client))

    return make


REAL_SUBMIT_RETRY = taskset_run_resources.SUBMIT_RETRY  # captured before no_backoff zeroes its backoff


def test_submit_retries_for_about_30_seconds():
    """2, 4, 8, 16 s: long enough to ride out an API pod restart after minutes of image builds."""
    failure = httpx.ConnectError("down")
    delays = [retry_delay(REAL_SUBMIT_RETRY, attempt, failure) for attempt in range(REAL_SUBMIT_RETRY.max_retries)]
    assert delays == [2, 4, 8, 16]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        taskset_run_resources, "SUBMIT_RETRY", replace(taskset_run_resources.SUBMIT_RETRY, backoff_base=0.0)
    )


class Sequence:
    """Answers each request with the next response, or raises the next transport error."""

    def __init__(self, *answers: httpx.Response | Callable[[httpx.Request], httpx.Response]) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, httpx.Response):
            return answer
        return answer(request)


def fail(error: type[httpx.TransportError]) -> Callable[[httpx.Request], httpx.Response]:
    def raise_(request: httpx.Request) -> httpx.Response:
        raise error("boom", request=request)

    return raise_


def ok_submit() -> httpx.Response:
    return httpx.Response(201, json=run_status(phase="accepted", warnings=[]))


def error_response(status: int, *, retry: str | None = None, **headers: str) -> httpx.Response:
    if retry is not None:
        headers["x-should-retry"] = retry
    return httpx.Response(status, json={"detail": f"status {status}"}, headers=headers)


# --- submit -------------------------------------------------------------------------------------------------


def test_submit_resends_a_retryable_503_with_the_same_body(make_runs):
    handler = Sequence(error_response(503, retry="true"), ok_submit())
    status = make_runs(handler).submit(harbor_request())
    assert status.run_id == RUN_ID
    assert len(handler.requests) == 2
    assert handler.requests[0].content == handler.requests[1].content
    assert handler.requests[0].url.path == RUNS


def test_submit_generates_one_key_and_sends_it_on_every_attempt(make_runs):
    handler = Sequence(error_response(503), ok_submit())
    make_runs(handler).submit(harbor_request(key=None))
    keys = [json.loads(r.content)["idempotency_key"] for r in handler.requests]
    assert len(keys) == 2 and keys[0] == keys[1] and keys[0]


def test_submit_sends_an_explicit_key_unchanged(make_runs):
    handler = Sequence(ok_submit())
    make_runs(handler).submit(harbor_request(key="mine"))
    assert json.loads(handler.requests[0].content)["idempotency_key"] == "mine"


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_submit_never_resends_a_body_that_may_have_arrived(make_runs, error):
    handler = Sequence(fail(error))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request(key="k1"))
    assert len(handler.requests) == 1
    assert raised.value.idempotency_key == "k1"
    assert raised.value.status_code is None and raised.value.request_id is None
    assert "idempotency_key=k1" in str(raised.value)


@pytest.mark.parametrize("status", [500, 502, 504])
def test_submit_does_not_resend_a_server_error_without_the_header(make_runs, status):
    handler = Sequence(error_response(status))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request())
    assert len(handler.requests) == 1
    assert raised.value.status_code == status


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_submit_resends_when_nothing_was_sent(make_runs, error):
    handler = Sequence(fail(error), ok_submit())
    assert make_runs(handler).submit(harbor_request()).run_id == RUN_ID
    assert len(handler.requests) == 2


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_submit_gives_up_after_five_unsent_attempts(make_runs, error):
    handler = Sequence(fail(error))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request(key=None))
    assert len(handler.requests) == 5
    assert raised.value.status_code is None
    assert raised.value.idempotency_key == json.loads(handler.requests[0].content)["idempotency_key"]


@pytest.mark.parametrize("status", [503, 429])
def test_submit_gives_up_after_five_retryable_answers(make_runs, status):
    handler = Sequence(error_response(status))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request())
    assert len(handler.requests) == 5
    assert raised.value.status_code == status


@pytest.mark.parametrize(
    "status,retry,cause",
    [
        (409, "false", ConflictError),
        (403, "false", PermissionDeniedError),
        (422, None, UnprocessableEntityError),
    ],
)
def test_submit_wraps_a_client_error_without_resending(make_runs, status, retry, cause):
    handler = Sequence(error_response(status, retry=retry))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request())
    assert len(handler.requests) == 1
    assert raised.value.status_code == status
    assert isinstance(raised.value.__cause__, cause)


def test_submit_keeps_a_generated_key_on_a_403_after_acceptance(make_runs):
    handler = Sequence(error_response(403, retry="false"))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request(key=None))
    assert raised.value.idempotency_key == json.loads(handler.requests[0].content)["idempotency_key"]


@pytest.mark.parametrize("policy", [None, PLATFORM_DEFAULT_RETRY_POLICY])
def test_submit_ignores_the_client_policy(make_runs, policy):
    timeout = Sequence(fail(httpx.ReadTimeout))
    with pytest.raises(TasksetRunSubmitError):
        make_runs(timeout, retry=policy).submit(harbor_request())
    assert len(timeout.requests) == 1
    outage = Sequence(error_response(503), ok_submit())
    make_runs(outage, retry=policy).submit(harbor_request())
    assert len(outage.requests) == 2


def test_submit_waits_300s_for_an_answer_and_10s_to_connect(make_runs):
    handler = Sequence(ok_submit())
    make_runs(handler).submit(harbor_request())
    timeout = handler.requests[0].extensions["timeout"]
    assert timeout["read"] == 300.0 and timeout["connect"] == 10.0


def test_submit_error_names_the_request_id(make_runs):
    handler = Sequence(error_response(500, **{"x-request-id": "req-1"}))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request())
    assert raised.value.request_id == "req-1"
    assert str(raised.value).endswith("; request_id=req-1")
    plain = Sequence(error_response(500))
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(plain).submit(harbor_request())
    assert raised.value.request_id is None and "request_id" not in str(raised.value)


@pytest.mark.parametrize(
    "response",
    [
        lambda _: httpx.Response(200, content=b"not json", headers={"x-request-id": "req-2"}),
        lambda _: httpx.Response(201, json={"phase": "accepted"}, headers={"x-request-id": "req-2"}),
    ],
    ids=["non-json", "no-run-id"],
)
def test_submit_does_not_resend_an_unparseable_success(make_runs, response):
    handler = Sequence(response)
    with pytest.raises(TasksetRunSubmitError) as raised:
        make_runs(handler).submit(harbor_request(key=None))
    assert len(handler.requests) == 1
    assert isinstance(raised.value.__cause__, NemoResponseValidationError)
    assert raised.value.status_code is None
    assert raised.value.request_id == "req-2"
    assert raised.value.idempotency_key == json.loads(handler.requests[0].content)["idempotency_key"]


@pytest.mark.parametrize(
    "request_",
    [
        harbor_request(HarborRunnerTarget(source=RegisteredAgentSource(agent="my-agent"))),
        harbor_request(HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle"))),
        harbor_request(HarborRunnerTarget()),
        run_request(f"default/suite#{DIGEST}", spec_file=None, agent="oracle", dispatcher_profile=None),
    ],
    ids=["registered-agent", "builtin-agent", "all-defaults", "cli-agent"],
)
def test_submit_sends_the_cli_body(make_runs, request_):
    handler = Sequence(ok_submit())
    make_runs(handler).submit(request_)
    body = handler.requests[0].content
    sent = TasksetRunInput.model_validate_json(body)
    assert isinstance(sent.spec.target, HarborRunnerTarget)
    assert request_digest(sent) == request_digest(request_)
    assert body == request_.model_dump_json(exclude_none=True).encode()


def test_submit_sends_a_spec_file_with_publication(make_runs, tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"spec": {"target": {"kind": "harbor"}}, "execution": {}}))
    request_ = run_request(
        f"default/suite#{DIGEST}", spec_file=spec, agent="oracle", dispatcher_profile=None, experiment="exp"
    )
    handler = Sequence(ok_submit())
    make_runs(handler).submit(request_)
    assert handler.requests[0].content == request_.model_dump_json(exclude_none=True).encode()


# --- read methods and cancel --------------------------------------------------------------------------------


def test_retrieve(make_runs):
    handler = Sequence(httpx.Response(200, json=run_status()))
    assert make_runs(handler).retrieve(RUN_ID).phase == "running"
    assert handler.requests[0].url.path == f"{RUNS}/{RUN_ID}"
    with pytest.raises(NotFoundError):
        make_runs(Sequence(error_response(404))).retrieve(RUN_ID)


def test_list_sends_its_filters_and_keeps_the_envelope(make_runs):
    handler = Sequence(httpx.Response(200, json=page([run_status()], sort="-created_at")))
    result = make_runs(handler).list(active=True, mine=True, page=2, page_size=5)
    assert dict(handler.requests[0].url.params) == {"active": "true", "mine": "true", "page": "2", "page_size": "5"}
    assert result.sort == "-created_at" and result.data[0].run_id == RUN_ID


def test_tasks_sends_phase_only_when_set(make_runs):
    handler = Sequence(httpx.Response(200, json=page([row(0)], sort="task_index")))
    runs = make_runs(handler)
    result = runs.tasks(RUN_ID)
    assert "phase" not in handler.requests[0].url.params
    assert result.sort == "task_index" and result.data[0].task == "task-0"
    runs.tasks(RUN_ID, phase="failed")
    assert handler.requests[1].url.params["phase"] == "failed"
    with pytest.raises(NotFoundError):
        make_runs(Sequence(error_response(404))).tasks(RUN_ID)


def test_cancel_retries_a_contended_503_under_the_platform_policy(make_runs):
    handler = Sequence(error_response(503, retry="true"), httpx.Response(200, json=run_status()))
    assert make_runs(handler, retry=FAST_PLATFORM_POLICY).cancel(RUN_ID).run_id == RUN_ID
    assert len(handler.requests) == 2
    assert handler.requests[0].url.path == f"{RUNS}/{RUN_ID}/cancel"


def test_cancel_is_sent_once_without_a_client_policy(make_runs):
    handler = Sequence(error_response(503, retry="true"))
    with pytest.raises(InternalServerError):
        make_runs(handler).cancel(RUN_ID)
    assert len(handler.requests) == 1


@pytest.mark.parametrize("policy", [None, FAST_PLATFORM_POLICY])
def test_cancel_does_not_retry_a_finished_run(make_runs, policy):
    handler = Sequence(error_response(409, retry="false"))
    with pytest.raises(ConflictError):
        make_runs(handler, retry=policy).cancel(RUN_ID)
    assert len(handler.requests) == 1


# --- iter_tasks and results ---------------------------------------------------------------------------------


class Platform:
    """A run, its pinned taskset, its ``/tasks`` rows (in pages) and result records."""

    def __init__(
        self,
        *,
        run: dict[str, Any] | None = None,
        names: tuple[str, ...] = ("hello", "sum", "sort"),
        rows: list[dict[str, Any]] | None = None,
        page_size: int = 1000,
        taskset_status: int = 200,
        results: dict[str, dict[str, Any]] | None = None,
        result_status: dict[str, int] | None = None,
    ) -> None:
        self.run = run or run_status()
        self.names = names
        self.rows = rows if rows is not None else [row(i) for i in range(len(names))]
        self.page_size = page_size
        self.taskset_status = taskset_status
        self.results = results or {}
        self.result_status = result_status or {}
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path == f"{RUNS}/{RUN_ID}":
            return httpx.Response(200, json=self.run) if self.run else error_response(404)
        if path == TASKSET_PATH:
            if self.taskset_status != 200:
                return error_response(self.taskset_status)
            return httpx.Response(200, json=taskset(*self.names))
        if path == f"{RUNS}/{RUN_ID}/tasks":
            number = int(request.url.params["page"])
            chunks = [self.rows[i : i + self.page_size] for i in range(0, len(self.rows), self.page_size)] or [[]]
            return httpx.Response(200, json=page(chunks[number - 1], number=number, pages=len(chunks)))
        if path.endswith("/agent-eval-results"):
            job = request.url.params["filter[job_id]"]
            if job in self.result_status:
                return error_response(self.result_status[job])
            found = [self.results[job]] if job in self.results else []
            return httpx.Response(200, json=page(found))
        if path.endswith(f"/agent-eval-results/{RUN_ID}"):
            return httpx.Response(200, json=result_record(RUN_ID))
        if path.startswith("/apis/jobs/"):
            raise AssertionError("taskset_runs must not read Jobs")
        raise AssertionError(f"unexpected {request.method} {request.url}")

    def count(self, path: str) -> int:
        return self.paths.count(path)


def test_iter_tasks_walks_every_page_with_taskset_names(make_runs):
    platform = Platform(page_size=2)
    tasks = make_runs(platform).iter_tasks(RUN_ID)
    assert [(t.index, t.name, t.task.task) for t in tasks] == [
        (0, "hello", "task-0"),
        (1, "sum", "task-1"),
        (2, "sort", "task-2"),
    ]
    assert platform.count(f"{RUNS}/{RUN_ID}/tasks") == 2
    assert platform.count(TASKSET_PATH) == 1


def test_iter_tasks_is_a_sync_iterator():
    """The ``Runs`` adapter lists whatever comes back; this pins the sync resource's own iterator protocol."""
    client = EvaluatorClient(base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(Platform())))
    tasks = EvaluatorTasksetRunsResource(client).iter_tasks(RUN_ID)
    assert isinstance(tasks, Iterator)
    assert next(tasks).name == "hello"
    assert [t.name for t in tasks] == ["sum", "sort"]


def test_iter_tasks_is_an_async_iterator():
    """The async resource's own protocol: ``async for``, not a coroutine of a list."""

    async def names() -> list[str]:
        transport = httpx.MockTransport(Platform())
        client = AsyncEvaluatorClient(base_url=BASE, http_client=httpx.AsyncClient(transport=transport))
        tasks = AsyncEvaluatorTasksetRunsResource(client).iter_tasks(RUN_ID)
        assert isinstance(tasks, AsyncIterator)
        return [t.name async for t in tasks]

    assert asyncio.run(names()) == ["hello", "sum", "sort"]


def test_iter_tasks_takes_a_fetched_run_without_reading_it_again(make_runs):
    platform = Platform()
    run = TasksetRunStatus.model_validate(run_status())
    assert [t.name for t in make_runs(platform).iter_tasks(run)] == ["hello", "sum", "sort"]
    assert platform.count(f"{RUNS}/{RUN_ID}") == 0


def test_iter_tasks_reads_a_pinned_taskset_once_per_resource(make_runs):
    platform = Platform()
    runs = make_runs(platform)
    runs.iter_tasks(RUN_ID)
    runs.iter_tasks(RUN_ID)
    assert platform.count(TASKSET_PATH) == 1
    make_runs(platform).iter_tasks(RUN_ID)
    assert platform.count(TASKSET_PATH) == 2


def test_iter_tasks_does_not_cache_a_denied_taskset(make_runs):
    platform = Platform(taskset_status=403)
    runs = make_runs(platform)
    assert [t.name for t in runs.iter_tasks(RUN_ID)] == ["task-0", "task-1", "task-2"]
    platform.taskset_status = 200
    assert [t.name for t in runs.iter_tasks(RUN_ID)] == ["hello", "sum", "sort"]


@pytest.mark.parametrize("status", [403, 404])
def test_iter_tasks_falls_back_to_member_names(make_runs, status):
    tasks = make_runs(Platform(taskset_status=status)).iter_tasks(RUN_ID)
    assert [t.name for t in tasks] == ["task-0", "task-1", "task-2"]


def test_iter_tasks_passes_other_taskset_errors_through(make_runs):
    with pytest.raises(InternalServerError):
        make_runs(Platform(taskset_status=500)).iter_tasks(RUN_ID)


def test_iter_tasks_names_a_row_the_taskset_lacks_by_its_member_name(make_runs):
    tasks = make_runs(Platform(names=("hello",), rows=[row(0), row(1)])).iter_tasks(RUN_ID)
    assert [(t.index, t.name) for t in tasks] == [(0, "hello"), (1, "task-1")]


def test_iter_tasks_raises_for_a_missing_run(make_runs):
    platform = Platform()
    platform.run = {}
    with pytest.raises(NotFoundError):
        make_runs(platform).iter_tasks(RUN_ID)


def test_iter_tasks_yields_every_task_before_its_row_exists(make_runs):
    none = make_runs(Platform(rows=[])).iter_tasks(RUN_ID)
    assert [(t.index, t.task) for t in none] == [(0, None), (1, None), (2, None)]
    some = make_runs(Platform(rows=[row(1)])).iter_tasks(RUN_ID)
    assert [t.task is not None for t in some] == [False, True, False]
    assert make_runs(Platform(rows=[], taskset_status=403)).iter_tasks(RUN_ID) == []


def completed(index: int) -> dict[str, Any]:
    return row(index, "succeeded", job=f"{RUN_ID}-task-{index}", job_status="completed")


def test_results_joins_completed_tasks_with_their_records(make_runs):
    platform = Platform(
        rows=[completed(0), row(1, job=f"{RUN_ID}-task-1", job_status="active")],
        results={f"{RUN_ID}-task-0": result_record(f"{RUN_ID}-task-0")},
    )
    results = make_runs(platform).results(RUN_ID)
    assert platform.count(f"{RUNS}/{RUN_ID}") == 1
    assert [(t.name, t.result is not None) for t in results.tasks] == [
        ("hello", True),
        ("sum", False),
        ("sort", False),
    ]
    assert [t.task is None for t in results.tasks] == [False, False, True]
    assert sum(p.endswith("/agent-eval-results") for p in platform.paths) == 1
    assert results.result is None
    assert TasksetRunResults.model_validate(results.model_dump(mode="json")) == results


def test_results_reads_the_merged_record_once_ready(make_runs):
    platform = Platform(run=run_status(phase="completed", result_state="ready"))
    results = make_runs(platform).results(RUN_ID)
    assert results.result is not None and results.result.name == RUN_ID


def test_results_fails_fast_on_a_failed_lookup(make_runs):
    platform = Platform(
        rows=[completed(0), completed(1), completed(2)],
        result_status={f"{RUN_ID}-task-1": 500},
    )
    with pytest.raises(InternalServerError):
        make_runs(platform).results(RUN_ID)
    lookups = [p for p in platform.paths if p.endswith("/agent-eval-results")]
    assert len(lookups) == 2
