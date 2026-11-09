# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A FileSet task source, end to end through agent-evaluate submission against a fake platform.

The fake serves dataset files and implements task and taskset PUTs with the platform's revision rule
(identical content publishes nothing; anything else publishes a revision), so these tests exercise the
behaviour a snapshot relies on, not mocks of it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from nemo_evals.api.schemas import TaskInput
from nemo_evals.api.service.task_service import stored_evaluator_task_digest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec, AgentEvalTaskInput, FilesetTaskSource
from nemo_evals.jobs.fileset_tasks import SourceFile, publish_snapshot
from nemo_helix_plugin.client.client import AsyncNemoClient

BASE = "https://platform.test"
_FILE = re.compile(r"^/apis/files/v2/workspaces/([^/]+)/filesets/([^/]+)/-/(.+)$")
_RESOURCE = re.compile(r"^/apis/evals/v2/workspaces/([^/]+)/(tasks|tasksets)/([^/]+)$")


def _digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@dataclass
class _Record:
    revisions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (digest, body)


@dataclass
class FakePlatform:
    """Files and evals task/taskset routes, enough for a FileSet task source and its snapshot."""

    files: dict[tuple[str, str, str], bytes] = field(default_factory=dict)
    records: dict[tuple[str, str, str], _Record] = field(default_factory=dict)
    denied_files: set[tuple[str, str, str]] = field(default_factory=set)
    denied_task_writes: bool = False
    task_puts: int = 0
    taskset_puts: int = 0
    #: Awaited between the last task write and the first taskset write, to interleave submissions.
    before_taskset_write: asyncio.Event | None = None
    taskset_write_started: asyncio.Event = field(default_factory=asyncio.Event)

    def head(self, kind: str, workspace: str, name: str) -> dict[str, Any] | None:
        record = self.records.get((kind, workspace, name))
        return None if record is None else record.revisions[-1][1]

    def revision_count(self, kind: str, workspace: str, name: str) -> int:
        record = self.records.get((kind, workspace, name))
        return 0 if record is None else len(record.revisions)

    def _respond(self, kind: str, workspace: str, name: str, status: int) -> httpx.Response:
        record = self.records[(kind, workspace, name)]
        digest, body = record.revisions[-1]
        now = datetime.now(UTC).isoformat()
        payload = {
            **body,
            "id": f"{kind}-{name}",
            "name": name,
            "workspace": workspace,
            "revision": len(record.revisions),
            "content_hash": digest,
            "tags": {"latest": len(record.revisions)},
            "created_at": now,
            "updated_at": now,
        }
        return httpx.Response(status, json=payload)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if match := _FILE.match(path):
            key = (match[1], match[2], match[3])
            if key in self.denied_files:
                return httpx.Response(403, json={"detail": "denied"})
            if key not in self.files:
                return httpx.Response(404, json={"detail": "not found"})
            # A stream, as the Files service serves it: the source reads it chunk by chunk under its cap.
            return httpx.Response(200, stream=httpx.ByteStream(self.files[key]))
        if match := _RESOURCE.match(path):
            workspace, kind, name = match[1], match[2], match[3]
            if request.method == "GET":
                if (kind, workspace, name) not in self.records:
                    return httpx.Response(404, json={"detail": "not found"})
                return self._respond(kind, workspace, name, 200)
            if request.method == "PUT":
                body = json.loads(request.content)
                if kind == "tasks":
                    if self.denied_task_writes:
                        return httpx.Response(403, json={"detail": "denied"})
                    self.task_puts += 1
                else:
                    self.taskset_write_started.set()
                    if self.before_taskset_write is not None:
                        await self.before_taskset_write.wait()
                    self.taskset_puts += 1
                content = {key: value for key, value in body.items() if key != "tags"}
                # Tasks digest as the real service does, so a submission's skip-unchanged check meets
                # real digests; tasksets only need a stable digest of their content.
                digest = (
                    stored_evaluator_task_digest(TaskInput.model_validate(content), workspace=workspace)
                    if kind == "tasks"
                    else _digest(content)
                )
                record = self.records.setdefault((kind, workspace, name), _Record())
                if record.revisions and record.revisions[-1][0] == digest:
                    return self._respond(kind, workspace, name, 200)
                record.revisions.append((digest, content))
                return self._respond(kind, workspace, name, 201)
        return httpx.Response(404, json={"detail": f"unhandled {request.method} {path}"})

    def client(self) -> AsyncNemoClient:
        return AsyncNemoClient(
            base_url=BASE,
            workspace="default",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )


def _row(question: str, **extras: object) -> str:
    return json.dumps({"responses_create_params": {"input": question}, "expected_answer": "A", **extras})


def _put_file(platform: FakePlatform, *lines: str, path: str = "gym/golden.jsonl") -> None:
    platform.files[("default", "golden", path)] = ("\n".join(lines) + "\n").encode()


def _input_spec(tasks: dict[str, Any]) -> AgentEvalInputSpec:
    return AgentEvalInputSpec.model_validate(
        {
            "target": {
                "kind": "gym",
                "source": {"component": "simple_agent", "config": "responses_api_agents/simple_agent/configs/a.yaml"},
                "resources_server": "mcqa",
            },
            "tasks": tasks,
        }
    )


def _source(*, snapshot: dict[str, Any] | None = None, **source: Any) -> dict[str, Any]:
    tasks: dict[str, Any] = {"source": {"fileset": "default/golden#gym/golden.jsonl", "format": "gym-jsonl", **source}}
    if snapshot is not None:
        tasks["snapshot"] = snapshot
    return tasks


async def _submit(platform: FakePlatform, tasks: dict[str, Any], **kwargs: Any) -> AgentEvalSpec:
    spec = await AgentEvalJob.to_spec(
        _input_spec(tasks),
        workspace="default",
        entity_client=None,
        async_sdk=platform.client(),
        is_local=False,
        **kwargs,
    )
    assert isinstance(spec, AgentEvalSpec)
    return spec


# --- Without a snapshot ------------------------------------------------------------------------


async def test_rows_become_inline_tasks_with_the_file_recorded() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q1"), _row("q2"))

    spec = await _submit(platform, _source())

    assert len(spec.tasks) == 2
    assert all(re.fullmatch(r"gym-[0-9a-f]{59}", task.id) and task.spec.provenance is None for task in spec.tasks)
    assert {task.spec.intent for task in spec.tasks if isinstance(task.spec, ResolvedEvaluatorTaskDefinition)} == {
        "Gym row"
    }
    assert all(task.spec.kind == "evaluator" for task in spec.tasks)
    assert all("gym_dataset_path" not in {item.key for item in task.metadata} for task in spec.tasks)
    assert spec.task_source is not None
    assert spec.task_source.fileset.root == "default/golden#gym/golden.jsonl"
    assert (
        spec.task_source.file_sha256
        == hashlib.sha256(platform.files[("default", "golden", "gym/golden.jsonl")]).hexdigest()
    )
    assert spec.task_source.task_count == 2
    assert spec.task_source.snapshot is None
    assert platform.task_puts == platform.taskset_puts == 0


async def test_id_field_gives_bare_ids_without_a_snapshot() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q1", uuid="r1"), _row("q2", uuid=2))

    spec = await _submit(platform, _source(id_field="uuid"))

    assert [task.id for task in spec.tasks] == ["r1", "2"]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["{oops"], r"default/golden#gym/golden\.jsonl:1: malformed JSON"),
        ([_row("q"), "[1]"], r"default/golden#gym/golden\.jsonl:2: expected a JSON object"),
        ([json.dumps({"task_idx": 1})], r"golden\.jsonl:1: row has no 'responses_create_params'"),
    ],
)
async def test_a_bad_row_is_refused_naming_its_line(lines: list[str], message: str) -> None:
    platform = FakePlatform()
    _put_file(platform, *lines)
    with pytest.raises(ValueError, match=message):
        await _submit(platform, _source())


async def test_a_missing_or_unreadable_file_is_refused() -> None:
    platform = FakePlatform()
    with pytest.raises(ValueError, match="does not exist"):
        await _submit(platform, _source())
    _put_file(platform, _row("q"))
    platform.denied_files.add(("default", "golden", "gym/golden.jsonl"))
    with pytest.raises(PermissionError, match="access denied"):
        await _submit(platform, _source())


async def test_caps_come_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_evals.config import get_config

    platform = FakePlatform()
    _put_file(platform, _row("q1"), _row("q2"), _row("q3"))
    monkeypatch.setattr(get_config(), "fileset_task_source_max_rows", 2)
    with pytest.raises(ValueError, match=r"has more than the 2-row limit .*fileset_task_source_max_rows"):
        await _submit(platform, _source())
    monkeypatch.setattr(get_config(), "fileset_task_source_max_rows", 50_000)
    monkeypatch.setattr(get_config(), "fileset_task_source_max_bytes", 10)
    with pytest.raises(ValueError, match=r"larger than the 10-byte limit .*fileset_task_source_max_bytes"):
        await _submit(platform, _source())


async def test_the_caller_authorizer_checks_the_file_read() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q"))
    checked: list[tuple[str, str]] = []

    async def authorize(method: str, path: str) -> None:
        checked.append((method, path))

    await _submit(platform, _source(), authorize=authorize)

    assert ("GET", "/apis/files/v2/workspaces/default/filesets/golden/-/gym/golden.jsonl") in checked


# --- With a snapshot ---------------------------------------------------------------------------

_SNAPSHOT = {"taskset": "default/nokia-mcqa"}


async def test_a_first_snapshot_publishes_every_task_and_one_revision_and_runs_it() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q1"), _row("q2"))

    spec = await _submit(platform, _source(snapshot=_SNAPSHOT))

    record = spec.task_source.snapshot if spec.task_source else None
    assert record is not None
    assert platform.revision_count("tasksets", "default", "nokia-mcqa") == 1
    taskset = platform.head("tasksets", "default", "nokia-mcqa")
    assert taskset is not None
    digest = platform.records[("tasksets", "default", "nokia-mcqa")].revisions[-1][0]
    assert record.taskset.root == f"default/nokia-mcqa#{digest}"
    assert record.delta.model_dump() == {"added": 2, "changed": 0, "removed": 0, "unchanged": 0}
    # Members are pinned to the digest each task write returned, never bare names.
    for ref, task in zip(taskset["tasks"], spec.tasks, strict=True):
        task_digest = platform.records[("tasks", "default", task.id)].revisions[-1][0]
        assert ref == f"default/{task.id}#{task_digest}"
        assert task.spec.provenance is not None
        assert task.spec.provenance.entity_name == f"default/{task.id}"
        assert task.spec.provenance.revision_digest == task_digest
    sha = spec.task_source.file_sha256 if spec.task_source else ""
    assert {item["key"]: item["value"] for item in taskset["metadata"]} == {
        "source_fileset": "default/golden#gym/golden.jsonl",
        "source_sha256": sha,
        "format": "gym-jsonl",
    }


async def test_resubmitting_the_same_bytes_publishes_nothing() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q1"), _row("q2"))
    first = await _submit(platform, _source(snapshot=_SNAPSHOT))
    writes = (platform.task_puts, platform.taskset_puts)
    second = await _submit(platform, _source(snapshot=_SNAPSHOT))

    assert writes == (2, 1)
    assert (platform.task_puts, platform.taskset_puts) == writes  # an unchanged file writes nothing

    assert platform.revision_count("tasksets", "default", "nokia-mcqa") == 1
    assert all(platform.revision_count("tasks", "default", task.id) == 1 for task in second.tasks)
    first_snapshot = first.task_source.snapshot if first.task_source else None
    second_snapshot = second.task_source.snapshot if second.task_source else None
    assert first_snapshot is not None and second_snapshot is not None
    assert second_snapshot.taskset == first_snapshot.taskset
    assert second_snapshot.delta.model_dump() == {"added": 0, "changed": 0, "removed": 0, "unchanged": 2}


async def test_reordering_rows_publishes_a_taskset_revision_but_no_task_revisions() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q1"), _row("q2"))
    await _submit(platform, _source(snapshot=_SNAPSHOT))
    _put_file(platform, _row("q2"), _row("q1"))
    spec = await _submit(platform, _source(snapshot=_SNAPSHOT))

    assert (platform.task_puts, platform.taskset_puts) == (2, 2)  # only the taskset, for its new SHA-256

    assert platform.revision_count("tasksets", "default", "nokia-mcqa") == 2  # the file's SHA-256 changed
    assert all(platform.revision_count("tasks", "default", task.id) == 1 for task in spec.tasks)
    assert spec.task_source and spec.task_source.snapshot
    assert spec.task_source.snapshot.delta.unchanged == 2


async def test_hash_mode_edit_is_an_add_and_a_remove_and_the_old_task_survives() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q1"), _row("q2"))
    before = await _submit(platform, _source(snapshot=_SNAPSHOT))
    _put_file(platform, _row("q1"), _row("q2 edited"))
    after = await _submit(platform, _source(snapshot=_SNAPSHOT))

    assert after.task_source and after.task_source.snapshot
    assert after.task_source.snapshot.delta.model_dump() == {"added": 1, "changed": 0, "removed": 1, "unchanged": 1}
    dropped = ({task.id for task in before.tasks} - {task.id for task in after.tasks}).pop()
    assert platform.head("tasks", "default", dropped) is not None  # never deleted


async def test_id_mode_edit_revises_the_same_prefixed_task() -> None:
    platform = FakePlatform()
    snapshot = {**_SNAPSHOT, "task_id_prefix": "nokia-mcqa"}
    _put_file(platform, _row("q1", uuid=1), _row("q2", uuid=2))
    await _submit(platform, _source(id_field="uuid", snapshot=snapshot))
    _put_file(platform, _row("q1", uuid=1), _row("q2 edited", uuid=2))
    spec = await _submit(platform, _source(id_field="uuid", snapshot=snapshot))

    assert platform.task_puts == 3  # two for the first file, then only the edited row

    assert [task.id for task in spec.tasks] == ["nokia-mcqa-1", "nokia-mcqa-2"]
    assert platform.revision_count("tasks", "default", "nokia-mcqa-2") == 2
    assert platform.revision_count("tasks", "default", "nokia-mcqa-1") == 1
    assert spec.task_source and spec.task_source.snapshot
    assert spec.task_source.snapshot.delta.model_dump() == {"added": 0, "changed": 1, "removed": 0, "unchanged": 1}
    metadata = platform.head("tasksets", "default", "nokia-mcqa")
    assert metadata is not None
    assert {"key": "task_id_prefix", "value": "nokia-mcqa"} in metadata["metadata"]


async def test_a_file_reverting_to_an_older_version_publishes_again() -> None:
    platform = FakePlatform()
    for lines in ([_row("a")], [_row("b")], [_row("a")]):
        _put_file(platform, *lines)
        await _submit(platform, _source(snapshot=_SNAPSHOT))
    assert platform.revision_count("tasksets", "default", "nokia-mcqa") == 3


async def test_a_task_name_the_store_cannot_hold_is_refused_before_any_write() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q", uuid="has space"))
    with pytest.raises(ValueError, match="cannot be a stored task name"):
        await _submit(platform, _source(id_field="uuid", snapshot={**_SNAPSHOT, "task_id_prefix": "p"}))
    assert platform.task_puts == platform.taskset_puts == 0


async def test_a_denied_task_write_fails_before_the_taskset_is_written() -> None:
    platform = FakePlatform(denied_task_writes=True)
    _put_file(platform, _row("q"))
    with pytest.raises(Exception, match="denied|403|Forbidden"):
        await _submit(platform, _source(snapshot=_SNAPSHOT))
    assert platform.taskset_puts == 0


async def test_a_failing_submission_check_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q"))

    def refuse(*_: object, **__: object) -> None:
        raise ValueError("scoring check failed")

    monkeypatch.setattr("nemo_evals.jobs.agent_evaluate.validate_scoring", refuse)
    with pytest.raises(ValueError, match="scoring check failed"):
        await _submit(platform, _source(snapshot=_SNAPSHOT))
    assert platform.task_puts == platform.taskset_puts == 0


async def test_the_caller_authorizer_checks_the_snapshot_writes() -> None:
    platform = FakePlatform()
    _put_file(platform, _row("q"))
    checked: list[tuple[str, str]] = []

    async def authorize(method: str, path: str) -> None:
        checked.append((method, path))

    spec = await _submit(platform, _source(snapshot=_SNAPSHOT), authorize=authorize)

    assert ("PUT", f"/apis/evals/v2/workspaces/default/tasks/{spec.tasks[0].id}") in checked
    assert ("PUT", "/apis/evals/v2/workspaces/default/tasksets/nokia-mcqa") in checked


async def test_a_concurrent_submission_cannot_swap_rows_into_another_files_revision() -> None:
    """Two files sync the same id-mode taskset; the second rewrites a shared task between the first's
    task writes and its taskset write. The first revision must still hold the first file's row."""
    platform = FakePlatform()
    snapshot = {**_SNAPSHOT, "task_id_prefix": "p"}
    gate = asyncio.Event()
    platform.before_taskset_write = gate
    _put_file(platform, _row("version A", uuid=1), path="a.jsonl")
    _put_file(platform, _row("version B", uuid=1), path="b.jsonl")

    def submit(path: str) -> asyncio.Task[AgentEvalSpec]:
        tasks = {
            "source": {"fileset": f"default/golden#{path}", "format": "gym-jsonl", "id_field": "uuid"},
            "snapshot": snapshot,
        }
        return asyncio.create_task(_submit(platform, tasks))

    first = submit("a.jsonl")
    await platform.taskset_write_started.wait()  # A's task is written; A now waits at its taskset write.
    a_task_digest = platform.records[("tasks", "default", "p-1")].revisions[-1][0]
    platform.before_taskset_write = None
    second = await submit("b.jsonl")  # B rewrites p-1 and publishes its own revision.
    gate.set()
    spec_a = await first

    assert platform.records[("tasks", "default", "p-1")].revisions[-1][0] != a_task_digest
    a_revision = next(
        body
        for digest, body in platform.records[("tasksets", "default", "nokia-mcqa")].revisions
        if spec_a.task_source
        and spec_a.task_source.snapshot
        and spec_a.task_source.snapshot.taskset.root.endswith(digest)
    )
    assert a_revision["tasks"] == [f"default/p-1#{a_task_digest}"]
    assert spec_a.tasks[0].spec.provenance is not None
    assert spec_a.tasks[0].spec.provenance.revision_digest == a_task_digest
    assert second.tasks[0].spec.provenance is not None
    assert second.tasks[0].spec.provenance.revision_digest != a_task_digest


async def test_publish_snapshot_refuses_resolved_tasks_that_are_not_its_inputs() -> None:
    source = FilesetTaskSource.model_validate(_source(snapshot=_SNAPSHOT))
    assert source.snapshot is not None
    with pytest.raises(ValueError, match="must be the snapshot's inputs"):
        await publish_snapshot(
            source.snapshot,
            [AgentEvalTaskInput(id="row", intent="Gym row")],
            [],
            SourceFile(fileset=FilesetRef("default/golden#gym/golden.jsonl"), body=b"", sha256="0" * 64),
            source,
            workspace="default",
            async_client=FakePlatform().client(),
            authorize=None,
        )


# --- Spec validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        (_source(fileset="default/golden"), "name one dataset file"),
        (_source(fileset="default/golden#gym/*.jsonl"), "globs are not supported"),
        (_source(id_field="uuid", snapshot=_SNAPSHOT), "task_id_prefix` is required with `source.id_field"),
        (_source(snapshot={**_SNAPSHOT, "task_id_prefix": "p"}), "hash ids take none"),
        (_source(snapshot={"taskset": "default/nokia-mcqa#latest"}), "write destination"),
    ],
)
def test_invalid_sources_are_refused(tasks: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _input_spec(tasks)


def test_id_field_without_a_snapshot_needs_no_prefix() -> None:
    assert isinstance(_input_spec(_source(id_field="uuid")).tasks, FilesetTaskSource)


def test_a_fileset_source_needs_a_gym_target() -> None:
    with pytest.raises(ValueError, match="needs a `gym` target, not precomputed `trials`"):
        AgentEvalInputSpec.model_validate({"trials": [], "tasks": _source()})


def test_task_source_cannot_be_submitted() -> None:
    spec = {**_input_spec(_source()).model_dump(mode="json"), "task_source": {}}
    with pytest.raises(ValueError, match="task_source"):
        AgentEvalInputSpec.model_validate(spec)


def test_the_row_cap_counts_only_up_to_the_limit() -> None:
    from nemo_evals.jobs.fileset_tasks import build_task_inputs

    # Blank lines, Unicode whitespace included, don't count; the count stops at the first row past the
    # limit, wherever the file ends.
    body = (_row("q1") + "\n\u2003\u00a0\n\n" + _row("q2") + "\n" + "1\n" * 100_000).encode()
    source_file = SourceFile(fileset=FilesetRef("default/golden#gym/golden.jsonl"), body=body, sha256="0" * 64)
    with pytest.raises(ValueError, match="more than the 2-row limit"):
        build_task_inputs(source_file, id_field=None, task_id_prefix=None, max_rows=2)
    assert (
        len(
            build_task_inputs(
                SourceFile(source_file.fileset, (_row("q1") + "\n\u2003\n").encode(), "0" * 64),
                id_field=None,
                task_id_prefix=None,
                max_rows=1,
            )
        )
        == 1
    )
