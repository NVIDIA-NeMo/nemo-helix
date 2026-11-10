# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Gym FileSet task source, submitted through the real agent-evaluate route.

The Evals, Files, Entities and Jobs services run in-process, so a snapshot meets the real task and
taskset revision rules. The evaluation itself does not run: that needs Gym, and the run path is the
ordinary inline/stored-task path covered elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from nemo_evals.api.fields import MetadataItem, TaskInputs
from nemo_evals.api.schemas import EvaluatorTaskDefinition, TaskInput
from nemo_evals.api.service.task_service import stored_evaluator_task_digest
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec
from nemo_evals.jobs.fileset_tasks import gym_reward_metric
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.service import EvaluatorPluginService
from nemo_helix_plugin.dependencies import get_submission_authorizer
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nhx.core.entities.service import EntitiesService
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

FILE = "default/golden#gym/golden.jsonl"
TASKSET = "golden-suite"
TASK_NAME = re.compile(r"gym-[0-9a-f]{59}")


class _EvalsService(NemoServiceAdapter):
    __test__ = False

    def __init__(self) -> None:
        super().__init__(EvaluatorPluginService())


@pytest.fixture
def ctx(tmp_path) -> Iterator[ClientContext]:
    with create_test_client(
        _EvalsService,
        EntitiesService,
        FilesService,
        JobsService,
        client_type=ClientContext,
        tmp_dir=tmp_path,
        workspace="default",
        projects=[],
        access_log=True,
    ) as ctx:
        FilesClient.from_client(ctx.client).create_fileset(
            workspace="default", body=CreateFilesetRequest(name="golden")
        )
        yield ctx


def _row(question: str) -> str:
    return json.dumps({"responses_create_params": {"input": question}, "expected_answer": "A"})


def _upload(ctx: ClientContext, *lines: str, path: str = "gym/golden.jsonl") -> str:
    body = ("\n".join(lines) + "\n").encode()
    FilesClient.from_client(ctx.client).upload_file(workspace="default", name="golden", path=path, content=body)
    return hashlib.sha256(body).hexdigest()


def _spec(tasks: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": {
            "kind": "gym",
            "source": {"component": "simple_agent", "config": "responses_api_agents/simple_agent/configs/a.yaml"},
            "resources_server": "mcqa",
        },
        "tasks": tasks,
    }


def _tasks(*, snapshot: str | None = TASKSET) -> dict[str, Any]:
    return {"fileset": FILE} if snapshot is None else {"fileset": FILE, "snapshot": snapshot}


def _submit(ctx: ClientContext, tasks: dict[str, Any]) -> AgentEvalSpec:
    """Submit through the job route and return the frozen spec the job holds."""
    response = ctx.test_client.post(
        "/apis/evals/v2/workspaces/default/agent-evaluate/jobs", json={"spec": _spec(tasks), "profile": "default"}
    )
    assert response.status_code < 300, response.text
    job = ctx.test_client.get(f"/apis/evals/v2/workspaces/default/agent-evaluate/jobs/{response.json()['name']}")
    return AgentEvalSpec.model_validate(job.raise_for_status().json()["spec"])


def _refused(ctx: ClientContext, tasks: dict[str, Any]) -> str:
    response = ctx.test_client.post(
        "/apis/evals/v2/workspaces/default/agent-evaluate/jobs", json={"spec": _spec(tasks), "profile": "default"}
    )
    assert response.status_code == 422, response.text
    return response.text


def _writes(ctx: ClientContext, kind: str) -> int:
    assert ctx.access_log is not None
    return sum(
        1
        for r in ctx.access_log.requests
        if r.method == "PUT" and f"/apis/evals/v2/workspaces/default/{kind}/" in r.path
    )


def _taskset_revisions(ctx: ClientContext) -> int:
    return len(Evaluator.from_client(ctx.client).tasksets.list_revisions(TASKSET, workspace="default").data)


# --- Without a snapshot ------------------------------------------------------------------------


def test_rows_run_inline_and_nothing_is_stored(ctx: ClientContext) -> None:
    _upload(ctx, _row("q1"), "", _row("q2"), _row("q1"))  # the repeated row collapses into one task

    spec = _submit(ctx, _tasks(snapshot=None))

    assert len(spec.tasks) == 2
    for task in spec.tasks:
        assert TASK_NAME.fullmatch(task.id)
        assert task.spec.kind == "evaluator" and task.spec.intent == "Gym row"
        assert task.spec.provenance is None
        assert "gym_dataset_path" not in {item.key for item in task.metadata}
    assert _writes(ctx, "tasks") == _writes(ctx, "tasksets") == 0


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["{oops"], r"default/golden#gym/golden\.jsonl:1: malformed JSON"),
        ([_row("q"), "", "[1]"], r"default/golden#gym/golden\.jsonl:3: expected a JSON object"),
        ([json.dumps({"task_idx": 1})], r"golden\.jsonl:1: row has no 'responses_create_params'"),
    ],
)
def test_a_bad_row_is_refused_naming_its_line_and_writes_nothing(
    ctx: ClientContext, lines: list[str], message: str
) -> None:
    _upload(ctx, *lines)
    assert re.search(message, _refused(ctx, _tasks()))
    assert _writes(ctx, "tasks") == _writes(ctx, "tasksets") == 0


def test_a_missing_file_is_refused(ctx: ClientContext) -> None:
    assert "does not exist" in _refused(ctx, _tasks())


def test_files_past_the_limits_are_refused(ctx: ClientContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _upload(ctx, _row("q1"), _row("q2"), _row("q3"))
    monkeypatch.setattr("nemo_evals.jobs.fileset_tasks.MAX_ROWS", 2)
    assert "golden.jsonl:3: more than the 2-row limit" in _refused(ctx, _tasks(snapshot=None))
    monkeypatch.setattr("nemo_evals.jobs.fileset_tasks.MAX_BYTES", 10)
    assert "larger than the 10-byte limit" in _refused(ctx, _tasks(snapshot=None))


# --- With a snapshot ---------------------------------------------------------------------------


def test_a_snapshot_publishes_the_file_and_the_job_runs_exactly_that(ctx: ClientContext) -> None:
    sha = _upload(ctx, _row("q1"), _row("q2"))

    spec = _submit(ctx, _tasks())

    taskset = Evaluator.from_client(ctx.client).tasksets.retrieve(TASKSET, workspace="default")
    assert taskset.revision == 1
    assert f"src-{sha}" in taskset.tags
    assert {item.key: item.value for item in taskset.metadata} == {"source_fileset": FILE, "source_sha256": sha}
    # Each member is pinned to the revision the job's task was frozen from.
    pins = {ref.root for ref in taskset.tasks}
    for task in spec.tasks:
        provenance = task.spec.provenance
        assert provenance is not None
        assert f"{provenance.entity_name}#{provenance.revision_digest}" in pins
    assert len(pins) == 2


def test_resubmitting_an_unchanged_file_writes_no_task(ctx: ClientContext) -> None:
    _upload(ctx, _row("q1"), _row("q2"))
    first = _submit(ctx, _tasks())
    assert _writes(ctx, "tasks") == 2

    second = _submit(ctx, _tasks())

    assert _writes(ctx, "tasks") == 2
    assert _taskset_revisions(ctx) == 1
    assert [t.spec.provenance for t in second.tasks] == [t.spec.provenance for t in first.tasks]


def test_an_edited_row_writes_one_task_and_keeps_the_old_one(ctx: ClientContext) -> None:
    _upload(ctx, _row("q1"), _row("q2"))
    before = _submit(ctx, _tasks())
    _upload(ctx, _row("q1"), _row("q2 edited"))
    after = _submit(ctx, _tasks())

    assert _writes(ctx, "tasks") == 3
    assert _taskset_revisions(ctx) == 2
    dropped = ({t.id for t in before.tasks} - {t.id for t in after.tasks}).pop()
    assert Evaluator.from_client(ctx.client).tasks.retrieve(dropped, workspace="default").name == dropped


def test_a_member_deleted_since_the_last_snapshot_is_written_again(ctx: ClientContext) -> None:
    _upload(ctx, _row("q1"), _row("q2"))
    first = _submit(ctx, _tasks())
    deleted = first.tasks[0].id
    Evaluator.from_client(ctx.client).tasks.delete(deleted, workspace="default")

    second = _submit(ctx, _tasks())

    assert Evaluator.from_client(ctx.client).tasks.retrieve(deleted, workspace="default").name == deleted
    assert [t.spec.provenance for t in second.tasks] == [t.spec.provenance for t in first.tasks]
    assert _taskset_revisions(ctx) == 1
    assert _submit(ctx, _tasks()).tasks == second.tasks


def test_reordering_or_reverting_the_file_publishes_a_revision_but_no_task(ctx: ClientContext) -> None:
    _upload(ctx, _row("a"), _row("b"))
    _submit(ctx, _tasks())
    _upload(ctx, _row("b"), _row("a"))  # new bytes, same tasks
    _submit(ctx, _tasks())
    _upload(ctx, _row("a"), _row("b"))  # back to the first file
    _submit(ctx, _tasks())

    assert _writes(ctx, "tasks") == 2
    assert _taskset_revisions(ctx) == 3


# --- Keyed rows (id_field + id_prefix) ----------------------------------------------------------

KEYED_NAME = re.compile(r"gold-[0-9a-f]{58}")


def _keyed(uid: object, question: str) -> str:
    return json.dumps({"uid": uid, "responses_create_params": {"input": question}, "expected_answer": "A"})


def _keyed_tasks(*, snapshot: str | None = TASKSET) -> dict[str, Any]:
    return {**_tasks(snapshot=snapshot), "id_field": "uid", "id_prefix": "gold"}


def test_keyed_rows_are_named_by_prefix_and_key(ctx: ClientContext) -> None:
    _upload(ctx, _keyed("t1", "q1"), _keyed(2, "q2"))

    spec = _submit(ctx, _keyed_tasks(snapshot=None))

    assert len(spec.tasks) == 2
    assert all(KEYED_NAME.fullmatch(task.id) for task in spec.tasks)


def test_an_edited_keyed_row_is_a_new_revision_of_its_task(ctx: ClientContext) -> None:
    """[t1, t2, t3] then [t1, t2-edited, t4]: one taskset with two revisions over four tasks, t2 with two."""
    tasks = Evaluator.from_client(ctx.client).tasks
    _upload(ctx, _keyed("t1", "q1"), _keyed("t2", "q2"), _keyed("t3", "q3"))
    first = _submit(ctx, _keyed_tasks())
    _upload(ctx, _keyed("t1", "q1"), _keyed("t2", "q2 edited"), _keyed("t4", "q4"))
    second = _submit(ctx, _keyed_tasks())

    t1, t2, t3 = (task.id for task in first.tasks)
    assert [task.id for task in second.tasks][:2] == [t1, t2]
    assert _taskset_revisions(ctx) == 2
    assert _writes(ctx, "tasks") == 5  # three, then the edit and the new row
    assert len(tasks.list(workspace="default", page_size=100).data) == 4
    assert len(tasks.list_revisions(t2, workspace="default").data) == 2
    assert len(tasks.list_revisions(t1, workspace="default").data) == 1
    assert tasks.retrieve(t3, workspace="default").name == t3  # the removed row's task is kept
    # The second job ran the edited revision of t2, not the first.
    assert second.tasks[1].spec.provenance != first.tasks[1].spec.provenance


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([_keyed("t1", "q1"), "", _keyed("t1", "q2")], r"golden\.jsonl:3: uid 't1' repeats line 1"),
        ([_keyed("t1", "q1"), _keyed(None, "q2")], r"golden\.jsonl:2: row has no usable 'uid' value"),
        ([json.dumps({"responses_create_params": {"input": "q"}})], r"golden\.jsonl:1: row has no usable 'uid'"),
        ([_keyed(True, "q1")], r"golden\.jsonl:1: row has no usable 'uid'"),
        ([_keyed("", "q1")], r"golden\.jsonl:1: row has no usable 'uid'"),
        (['{"uid": 1e400, "responses_create_params": {"input": "q"}}'], r"golden\.jsonl:1: row has no usable 'uid'"),
        (['{"uid": NaN, "responses_create_params": {"input": "q"}}'], r"golden\.jsonl:1: row has no usable 'uid'"),
    ],
)
def test_a_bad_key_is_refused_naming_its_line(ctx: ClientContext, lines: list[str], message: str) -> None:
    _upload(ctx, *lines)
    assert re.search(message, _refused(ctx, _keyed_tasks()))
    assert _writes(ctx, "tasks") == _writes(ctx, "tasksets") == 0


def test_a_resubmit_verifies_members_in_batches(ctx: ClientContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_evals.revisions._PIN_BATCH", 5)
    _upload(ctx, *(_row(f"q{i}") for i in range(30)))
    _submit(ctx, _tasks())
    assert ctx.access_log is not None
    ctx.access_log.clear()

    _submit(ctx, _tasks())

    reads = [r for r in ctx.access_log.requests if r.method == "GET" and "/entities/" in r.path]
    # Six batches of five pins at two queries each, plus a handful of lookups, never two per member.
    assert len(reads) < 30, [r.path for r in reads]


def test_a_failing_submission_check_writes_nothing(ctx: ClientContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _upload(ctx, _row("q"))

    def refuse(*_: object, **__: object) -> None:
        raise ValueError("scoring check failed")

    monkeypatch.setattr("nemo_evals.jobs.agent_evaluate.validate_scoring", refuse)
    assert "scoring check failed" in _refused(ctx, _tasks())
    assert _writes(ctx, "tasks") == _writes(ctx, "tasksets") == 0


def _authorize_as(ctx: ClientContext, authorize: Callable[[str, str], Awaitable[None]]) -> None:
    """Make the job route check the submission with ``authorize``, as the platform does with the caller's scopes."""
    app = ctx.test_client.app
    assert isinstance(app, FastAPI)
    app.dependency_overrides[get_submission_authorizer] = lambda: authorize


def _jobs(ctx: ClientContext) -> int:
    listed = ctx.test_client.get("/apis/evals/v2/workspaces/default/agent-evaluate/jobs").raise_for_status()
    return len(listed.json()["data"])


def test_the_route_checks_the_file_read_with_the_callers_scopes(ctx: ClientContext) -> None:
    """A token whose principal may read the file but whose scopes may not gets no copy of it in a job."""
    _upload(ctx, _row("q"))

    async def authorize(method: str, path: str) -> None:
        if path.startswith("/apis/files/"):
            raise HTTPException(status_code=403, detail="token lacks files:read")

    _authorize_as(ctx, authorize)
    response = ctx.test_client.post(
        "/apis/evals/v2/workspaces/default/agent-evaluate/jobs", json={"spec": _spec(_tasks(snapshot=None))}
    )

    assert response.status_code == 403, response.text
    assert "files:read" in response.text
    assert _jobs(ctx) == 0


def test_the_route_checks_every_snapshot_read_and_write_before_any_write(ctx: ClientContext) -> None:
    _upload(ctx, _row("q"))
    checked: list[tuple[str, str]] = []

    async def authorize(method: str, path: str) -> None:
        checked.append((method, path))
        if "/tasksets/" in path:
            raise PermissionError("no taskset access")

    _authorize_as(ctx, authorize)
    response = ctx.test_client.post(
        "/apis/evals/v2/workspaces/default/agent-evaluate/jobs", json={"spec": _spec(_tasks())}
    )

    assert response.status_code == 403, response.text
    methods = {(method, path.split("/")[-2] if "/evals/" in path else path) for method, path in checked}
    assert ("GET", "/apis/files/v2/workspaces/default/filesets/golden/-/gym/golden.jsonl") in methods
    assert ("PUT", "tasks") in methods
    # Reading the previous revision is checked too, before the taskset write.
    assert checked[-1] == ("GET", "/apis/evals/v2/workspaces/default/tasksets/golden-suite")
    assert _writes(ctx, "tasks") == _writes(ctx, "tasksets") == 0
    assert _jobs(ctx) == 0


def test_a_compile_refusal_leaves_the_snapshot_unchanged(ctx: ClientContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _upload(ctx, _row("q1"), _row("q2"))
    _submit(ctx, _tasks())
    _upload(ctx, _row("q1"), _row("q2 edited"))

    async def refuse(*_: object, **__: object) -> None:
        raise HelixJobCompilationError("sandboxing is disabled")

    monkeypatch.setattr(AgentEvalJob, "_compile", refuse)
    assert "sandboxing is disabled" in _refused(ctx, _tasks())

    assert _writes(ctx, "tasks") == 2
    assert _taskset_revisions(ctx) == 1
    assert _jobs(ctx) == 1


def test_the_computed_digest_is_the_one_the_service_publishes(ctx: ClientContext) -> None:
    """A snapshot skips a row whose computed digest the previous revision pins; if the computation
    drifted from the service, every resubmit would rewrite every row."""
    task = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Gym row",
            inputs=TaskInputs(gym_row={"input": "q"}),
            metrics=[gym_reward_metric()],
        ),
        metadata=[MetadataItem(key="gym_row_extras", value={"expected_answer": "A"})],
    )
    stored = Evaluator.from_client(ctx.client).tasks.replace("gym-parity", task=task, workspace="default")
    assert stored.content_hash == stored_evaluator_task_digest(task, workspace="default")


# --- Spec validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ({"fileset": "default/golden"}, "name one dataset file"),
        ({"fileset": "default/golden#gym/*.jsonl"}, "globs are not supported"),
        ({"fileset": FILE, "snapshot": f"{TASKSET}#latest"}, "write destination"),
        ({"fileset": FILE, "id_field": "uid"}, "`id_field` and `id_prefix` go together"),
        ({"fileset": FILE, "id_prefix": "gold"}, "`id_field` and `id_prefix` go together"),
        ({"fileset": FILE, "id_field": "uid", "id_prefix": "Gold"}, "id_prefix"),
        ({"fileset": FILE, "id_field": "uid", "id_prefix": "golden123"}, "id_prefix"),
        ({"fileset": FILE, "id_field": "uid", "id_prefix": "g-x"}, "id_prefix"),
        ({"fileset": FILE, "id_field": "uid", "id_prefix": "1gold"}, "id_prefix"),
        ({"fileset": FILE, "id_field": "uid", "id_prefix": "gym"}, "reserved"),
    ],
)
def test_invalid_sources_are_refused(tasks: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AgentEvalInputSpec.model_validate(_spec(tasks))


def test_a_fileset_source_needs_a_gym_target() -> None:
    with pytest.raises(ValueError, match="needs a `gym` target, not precomputed `trials`"):
        AgentEvalInputSpec.model_validate({"trials": [], "tasks": {"fileset": FILE}})
