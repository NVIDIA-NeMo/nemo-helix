# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill entities, archive storage and the publish service.

The entity store is an in-memory fake that enforces what publishing leans on: parent-scoped name
uniqueness, conflict on duplicate create, the ``db_version`` optimistic lock, and cascade delete.
Files and evals are real typed clients over ``httpx.MockTransport`` backed by in-memory state.
"""

from __future__ import annotations

import builtins
import io
import math
import sys
import tarfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar
from urllib.parse import unquote

import httpx
import pytest
from nemo_agents_plugin.entities import SkillEntity, SkillRevisionEntity
from nemo_agents_plugin.skill_service import (
    LinkedTasksetNotFoundError,
    SkillExistsError,
    SkillRevisionConflictError,
    SkillRevisionContentMismatchError,
    SkillRevisionNotFoundError,
    SkillService,
    SkillValidationError,
    revision_digest,
    revision_name,
)
from nemo_agents_plugin.skill_storage import (
    SKILLS_FILESET_NAME,
    SkillArchiveStore,
    SkillSourceError,
    SkillStorageError,
    archive_object_path,
)
from nemo_helix_plugin.agents.skills import SkillInput, SkillSource
from nemo_helix_plugin.entities import EntityBase, ListResponse, PaginationInfo
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.skill_archive import SkillArchiveLimits, build_skill_archive

_E = TypeVar("_E", bound=EntityBase)
WS = "default"
BASE = "http://test"
SKILL = "my-skill"
SOURCE_REF = f"{WS}/src#skills/{SKILL}"
TASKSET_DIGEST = "a" * 64
TASKSET_REF = f"{WS}/my-taskset#{TASKSET_DIGEST}"


# --------------------------------------------------------------------------- fakes


class FakeStore:
    """In-memory entity store keyed ``(entity_type, workspace, name, parent)``."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str, str, str | None], EntityBase] = {}
        self._next_id = 0
        #: ordinal → content_hash of a competing publisher that wins that ordinal's create.
        self.contend: dict[int, str | None] = {}

    def _key(self, entity_type: type[EntityBase], name: str, workspace: str, parent: str | None):
        return (entity_type.__entity_type__, workspace, name, parent)

    def _stamp(self, entity: EntityBase) -> None:
        self._next_id += 1
        entity._id = f"id-{self._next_id}"
        entity._created_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self._next_id)
        entity._updated_at = entity._created_at

    async def create(self, entity: _E) -> _E:
        key = self._key(type(entity), entity.name, entity.workspace, entity._parent)
        if isinstance(entity, SkillRevisionEntity) and entity.revision in self.contend:
            digest = self.contend.pop(entity.revision)
            winner = entity.model_copy(update={"content_hash": digest or entity.content_hash})
            if digest is not None:
                winner.frontmatter = winner.frontmatter.model_copy(update={"description": "someone else"})
                winner.content_hash = revision_digest(winner)
            winner._parent = entity._parent
            self._stamp(winner)
            self.records[key] = winner
            raise NemoEntityConflictError(f"{entity.name} taken")
        if key in self.records:
            raise NemoEntityConflictError(f"{entity.name} exists")
        self._stamp(entity)
        self.records[key] = entity.model_copy(deep=True)
        self.records[key]._parent = entity._parent
        self.records[key]._id = entity._id
        self.records[key]._created_at = entity._created_at
        return entity

    async def get(self, entity_type: type[_E], *, name: str, workspace: str, parent: str | None = None) -> _E:
        key = self._key(entity_type, name, workspace, parent)
        if key not in self.records:
            raise NemoEntityNotFoundError(name)
        stored = self.records[key]
        assert isinstance(stored, entity_type)
        return _copy(stored)

    async def update(self, entity: _E, *, original_name: str | None = None) -> _E:
        key = self._key(type(entity), original_name or entity.name, entity.workspace, entity._parent)
        stored = self.records.get(key)
        if stored is None:
            raise NemoEntityNotFoundError(entity.name)
        if entity._db_version != stored._db_version:
            raise NemoEntityConflictError("stale")
        saved = _copy(entity)
        saved._db_version = stored._db_version + 1
        saved._created_at = stored._created_at
        self.records[key] = saved
        return _copy(saved)

    async def delete_by_id(self, entity_type: type[_E], entity_id: str, *, expected_db_version: int | None = None):
        for key, entity in list(self.records.items()):
            if isinstance(entity, entity_type) and entity.id == entity_id:
                if expected_db_version is not None and expected_db_version != entity._db_version:
                    raise NemoEntityConflictError("stale")
                del self.records[key]
                for child_key, child in list(self.records.items()):
                    if child.parent == entity_id:
                        del self.records[child_key]
                return entity
        raise NemoEntityNotFoundError(entity_id)

    async def delete(self, entity_type, name, *, workspace, parent=None, expected_db_version=None):
        raise NotImplementedError

    async def list(
        self,
        entity_type: type[_E],
        *,
        workspace: str,
        filter_operation: FilterOperation | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[_E]:
        rows = [
            r
            for (etype, ws, _, _), r in self.records.items()
            if etype == entity_type.__entity_type__ and ws == workspace
        ]
        for comparison in _comparisons(filter_operation):
            assert comparison.operator is FilterOperator.EQ
            if comparison.field == "parent":
                rows = [r for r in rows if r.parent == comparison.value]
            else:
                attribute = comparison.field.removeprefix("data.")
                rows = [r for r in rows if getattr(r, attribute) == comparison.value]
        if sort:
            rows.sort(key=lambda r: getattr(r, sort.lstrip("-")), reverse=sort.startswith("-"))
        start = (page - 1) * page_size
        data = [_copy(r) for r in rows[start : start + page_size]]
        return ListResponse(
            data=data,
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                current_page_size=len(data),
                total_pages=max(1, math.ceil(len(rows) / page_size)),
                total_results=len(rows),
            ),
        )

    def revisions(self) -> builtins.list[SkillRevisionEntity]:
        return sorted(
            (r for r in self.records.values() if isinstance(r, SkillRevisionEntity)), key=lambda r: r.revision
        )

    def head(self, name: str = SKILL) -> SkillEntity:
        stored = self.records[self._key(SkillEntity, name, WS, None)]
        assert isinstance(stored, SkillEntity)
        return stored


def _copy(entity: _E) -> _E:
    copy = entity.model_copy(deep=True)
    copy._parent, copy._id, copy._db_version = entity._parent, entity._id, entity._db_version
    copy._created_at, copy._updated_at = entity._created_at, entity._updated_at
    return copy


def _comparisons(operation: FilterOperation | None) -> list[ComparisonOperation]:
    if operation is None:
        return []
    if isinstance(operation, LogicalOperation):
        return [c for child in operation.operations for c in _comparisons(child)]
    assert isinstance(operation, ComparisonOperation)
    return [operation]


@dataclass
class FakeFiles:
    """In-memory Files service: ``(workspace, fileset) → {path: bytes}``."""

    objects: dict[tuple[str, str], dict[str, bytes]] = field(default_factory=dict)
    uploads: list[str] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    corrupt_uploads: bool = False

    def put_dir(self, fileset: str, prefix: str, files: dict[str, bytes], workspace: str = WS) -> None:
        store = self.objects.setdefault((workspace, fileset), {})
        for path, content in files.items():
            store[f"{prefix}/{path}"] = content

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = unquote(request.url.path)
        parts = url.split("/")
        # /apis/files/v2/workspaces/{ws}/filesets[/{name}[/files | /-/{path}]]
        workspace = parts[5]
        if len(parts) == 7 and request.method == "POST":
            self.objects.setdefault((workspace, SKILLS_FILESET_NAME), {})
            return httpx.Response(200, json=_fileset_json(workspace, SKILLS_FILESET_NAME))
        name = parts[7]
        store = self.objects.get((workspace, name))
        if store is None:
            return httpx.Response(404, json={"detail": "no fileset"})
        if url.endswith("/files"):
            prefix = request.url.params.get("path") or ""
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"file_ref": f"{workspace}/{name}#{p}", "file_url": "x", "path": p, "size": len(c)}
                        for p, c in store.items()
                        if p.startswith(prefix)
                    ]
                },
            )
        path = url.split("/-/", 1)[1]
        if request.method == "PUT":
            content = request.read()
            store[path] = content + b"garbage" if self.corrupt_uploads else content
            self.uploads.append(path)
            return httpx.Response(
                200, json={"file_ref": f"{workspace}/{name}#{path}", "file_url": "x", "path": path, "size": 1}
            )
        self.downloads.append(path)
        if path not in store:
            return httpx.Response(404, json={"detail": "no file"})
        return httpx.Response(200, content=_chunks(store[path]))


async def _chunks(data: bytes) -> AsyncIterator[bytes]:
    """A streamed body, as the real Files service sends; a pre-read body cannot be streamed."""
    yield data


def _fileset_json(workspace: str, name: str) -> dict[str, object]:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": workspace,
        "description": "",
        "purpose": "generic",
        "storage": {"type": "local", "path": f"/data/{name}"},
        "metadata": {},
        "custom_fields": {},
        "project": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@dataclass
class Authorizer:
    denied: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def __call__(self, method: str, path: str) -> None:
        self.calls.append((method, path))
        if any(path.startswith(d) for d in self.denied):
            raise PermissionError(f"denied: {path}")


def skill_files(description: str = "Use it for testing.", body: str = "Do the thing.\n") -> dict[str, bytes]:
    return {
        "SKILL.md": f"---\nname: {SKILL}\ndescription: {description}\n---\n{body}".encode(),
        "scripts/run.sh": b"echo hi\n",
        "evals/evals.json": b'{"secret": "answer"}',
    }


@dataclass
class Harness:
    store: FakeStore
    files: FakeFiles
    authorize: Authorizer
    service: SkillService
    tasksets: set[str]

    def source(self, **kwargs: str) -> None:
        self.files.objects.pop((WS, "src"), None)
        self.files.put_dir("src", f"skills/{SKILL}", skill_files(**kwargs))


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    store = FakeStore()
    files = FakeFiles()
    tasksets: set[str] = set()

    def evals_handler(request: httpx.Request) -> httpx.Response:
        parts = unquote(request.url.path).split("/")
        # /apis/evals/v2/workspaces/{ws}/tasksets/{name}/revisions/{rev}
        key = f"{parts[5]}/{parts[7]}#{parts[9]}"
        if key in tasksets:
            return httpx.Response(200, json={"name": parts[7], "workspace": parts[5]})
        return httpx.Response(404, json={"detail": "not found"})

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(files.handler)) as files_http,
        httpx.AsyncClient(transport=httpx.MockTransport(evals_handler)) as evals_http,
    ):
        authorize = Authorizer()
        service = SkillService(
            store,
            SkillArchiveStore(AsyncFilesClient(base_url=BASE, workspace=WS, http_client=files_http)),
            authorize=authorize,
            evals_client=AsyncEvaluatorClient(base_url=BASE, workspace=WS, http_client=evals_http),
        )
        h = Harness(store, files, authorize, service, tasksets)
        h.source()
        yield h


def _input(*, tags: list[str] | None = None, taskset: str | None = None) -> SkillInput:
    return SkillInput(source=SkillSource(fileset=SOURCE_REF), tags=tags or [], taskset=taskset)


def _archive_members(h: Harness, archive_ref: str) -> list[str]:
    path = archive_ref.partition("#")[2]
    data = h.files.objects[(WS, SKILLS_FILESET_NAME)][path]
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return tar.getnames()


# --------------------------------------------------------------------------- tests


async def test_create_publishes_revision_one_into_the_skills_fileset(harness: Harness) -> None:
    skill = await harness.service.create_skill(WS, SKILL, _input(tags=["v1"]))

    assert skill.revision == 1
    assert skill.tags == {"latest": 1, "v1": 1}
    assert skill.frontmatter.name == SKILL
    assert skill.archive.archive_ref == (
        f"{WS}/{SKILLS_FILESET_NAME}#{archive_object_path(SKILL, skill.archive.files_hash)}"
    )
    assert skill.archive.archive_ref.endswith(f"{SKILL}/{skill.archive.files_hash}/skill_archive")
    # The head holds pointers, plus the latest frontmatter as a search copy for list filters.
    assert harness.store.head().model_dump(
        include=set(SkillEntity.model_fields) - {"name", "workspace", "project"}
    ) == {
        "latest_revision": 1,
        "tags": {"latest": 1, "v1": 1},
        "frontmatter": skill.frontmatter.model_dump(),
    }
    (revision,) = harness.store.revisions()
    assert revision.name == revision_name(1)
    assert revision.content_hash == skill.content_hash == revision_digest(revision)
    members = _archive_members(harness, skill.archive.archive_ref)
    assert f"{SKILL}/SKILL.md" in members
    assert not any("evals" in member for member in members)


async def test_create_refuses_an_existing_name(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    with pytest.raises(SkillExistsError):
        await harness.service.create_skill(WS, SKILL, _input())


async def test_publishing_identical_content_dedups_and_still_applies_tags(harness: Harness) -> None:
    first = await harness.service.create_skill(WS, SKILL, _input())
    again, created = await harness.service.publish_skill(WS, SKILL, _input(tags=["stable"]))

    assert created is False
    assert again.revision == 1
    assert again.content_hash == first.content_hash
    assert len(harness.store.revisions()) == 1
    assert harness.store.head().tags == {"latest": 1, "stable": 1}


async def test_publishing_new_content_cuts_revision_two_and_moves_latest(harness: Harness) -> None:
    first = await harness.service.create_skill(WS, SKILL, _input(tags=["v1"]))
    harness.source(body="Do the other thing.\n")
    second, created = await harness.service.publish_skill(WS, SKILL, _input())

    assert created is True
    assert second.revision == 2
    assert second.content_hash != first.content_hash
    assert second.archive.files_hash != first.archive.files_hash
    latest = await harness.service.get_skill(WS, SKILL)
    assert latest is not None and latest.revision == 2
    assert latest.tags == {"latest": 2, "v1": 1}

    by_tag = await harness.service.get_skill(WS, SKILL, "v1")
    by_digest = await harness.service.get_skill(WS, SKILL, first.content_hash)
    assert by_tag is not None and by_tag.revision == 1 and by_tag.tags == {"v1": 1}
    assert by_digest is not None and by_digest.revision == 1
    pinned = await harness.service.resolve_pinned_ref(f"{WS}/{SKILL}#{first.content_hash}", WS)
    assert pinned.archive == first.archive


async def test_head_search_copy_follows_latest_only(harness: Harness) -> None:
    first = await harness.service.create_skill(WS, SKILL, _input())
    harness.source(description="Use it for PDFs.")
    second, _ = await harness.service.publish_skill(WS, SKILL, _input())
    assert harness.store.head().frontmatter == second.frontmatter

    # Tagging an older revision moves no search copy: only 'latest' does.
    await harness.service.tag_revision(WS, SKILL, "old", first.content_hash)
    head = harness.store.head()
    assert head.frontmatter is not None and head.frontmatter.description == "Use it for PDFs."


async def test_reverting_to_older_content_publishes_forward(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    await harness.service.publish_skill(WS, SKILL, _input())
    harness.source()
    reverted, created = await harness.service.publish_skill(WS, SKILL, _input())

    assert created is True
    assert reverted.revision == 3
    assert harness.store.head().tags["latest"] == 3


async def test_unknown_tag_and_digest_are_not_found(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    with pytest.raises(SkillRevisionNotFoundError):
        await harness.service.get_skill(WS, SKILL, "nope")
    with pytest.raises(SkillRevisionNotFoundError):
        await harness.service.get_skill(WS, SKILL, "b" * 64)
    assert await harness.service.get_skill(WS, "absent") is None


async def test_tag_revision_moves_user_tags_and_refuses_reserved_ones(harness: Harness) -> None:
    first = await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    await harness.service.publish_skill(WS, SKILL, _input())

    tagged = await harness.service.tag_revision(WS, SKILL, "blessed", first.content_hash)
    assert tagged is not None and tagged.revision == 1 and tagged.tags == {"blessed": 1}
    assert harness.store.head().tags == {"latest": 2, "blessed": 1}

    with pytest.raises(SkillValidationError, match="managed automatically"):
        await harness.service.tag_revision(WS, SKILL, "latest", first.content_hash)
    with pytest.raises(SkillValidationError, match="content digest"):
        await harness.service.tag_revision(WS, SKILL, "c" * 64, first.content_hash)
    assert await harness.service.tag_revision(WS, "absent", "blessed", "latest") is None


async def test_publish_rejects_digest_shaped_tags_before_creating_anything(harness: Harness) -> None:
    with pytest.raises(SkillValidationError, match="content digest"):
        await harness.service.create_skill(WS, SKILL, _input(tags=["d" * 64]))
    assert harness.store.records == {}
    assert harness.files.uploads == []


async def test_read_refuses_a_revision_whose_content_was_modified(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    (revision,) = harness.store.revisions()
    revision.frontmatter = revision.frontmatter.model_copy(update={"description": "tampered"})

    with pytest.raises(SkillRevisionContentMismatchError):
        await harness.service.get_skill(WS, SKILL)
    with pytest.raises(SkillRevisionContentMismatchError):
        await harness.service.list_revisions(WS, SKILL)


async def test_archive_that_fails_readback_blocks_the_revision(harness: Harness) -> None:
    harness.files.corrupt_uploads = True

    with pytest.raises(SkillStorageError):
        await harness.service.create_skill(WS, SKILL, _input())
    assert harness.store.records == {}


async def test_existing_object_is_verified_before_reuse(harness: Harness, tmp_path: Path) -> None:
    source = tmp_path / "src"
    for path, content in skill_files().items():
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_bytes(content)
    info = build_skill_archive(source, tmp_path / "archive")
    good = (tmp_path / "archive").read_bytes()
    key = archive_object_path(SKILL, info.files_hash)

    harness.files.objects[(WS, SKILLS_FILESET_NAME)] = {key: good}
    skill = await harness.service.create_skill(WS, SKILL, _input())
    assert skill.archive.files_hash == info.files_hash
    assert harness.files.uploads == []
    assert key in harness.files.downloads

    await harness.service.delete_skill(WS, SKILL)
    harness.files.objects[(WS, SKILLS_FILESET_NAME)] = {key: b"not an archive"}
    harness.files.downloads.clear()
    rewritten = await harness.service.create_skill(WS, SKILL, _input())
    assert harness.files.uploads == [key]
    assert harness.files.objects[(WS, SKILLS_FILESET_NAME)][key] == good
    assert rewritten.archive.files_hash == info.files_hash


async def test_lost_ordinal_race_retries_on_the_next_ordinal(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    harness.store.contend[2] = "different"

    skill, created = await harness.service.publish_skill(WS, SKILL, _input())

    assert created is True
    assert skill.revision == 3
    assert [r.revision for r in harness.store.revisions()] == [1, 2, 3]
    assert harness.store.head().tags["latest"] == 3
    assert harness.store.head().latest_revision == 3


async def test_identical_concurrent_publish_reuses_the_winners_revision(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    harness.store.contend[2] = None  # the winner published exactly this content, pointers not yet moved

    skill, created = await harness.service.publish_skill(WS, SKILL, _input())

    assert created is False  # the winner created it; this call only finished the pointers
    assert skill.revision == 2
    assert [r.revision for r in harness.store.revisions()] == [1, 2]
    assert harness.store.head().tags["latest"] == 2


async def test_retry_after_a_crash_before_pointer_move_finishes_the_publish(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    v2, _ = await harness.service.publish_skill(WS, SKILL, _input())
    assert v2.revision == 2
    # Model a first attempt that created rev.2 and died before moving pointers, then retry it.
    harness.store.head().tags = {"latest": 1}
    harness.store.head().latest_revision = 1
    retried, _ = await harness.service.publish_skill(WS, SKILL, _input())

    assert retried.revision == 2
    assert len(harness.store.revisions()) == 2
    assert harness.store.head().tags["latest"] == 2
    assert harness.store.head().latest_revision == 2


async def test_unauthorized_source_read_is_refused_before_any_io(harness: Harness) -> None:
    harness.authorize.denied.add(f"/apis/files/v2/workspaces/{WS}/filesets/src")

    with pytest.raises(PermissionError):
        await harness.service.create_skill(WS, SKILL, _input())
    assert harness.files.downloads == []
    assert harness.files.uploads == []
    assert harness.store.records == {}


async def test_linked_taskset_must_exist(harness: Harness) -> None:
    with pytest.raises(LinkedTasksetNotFoundError):
        await harness.service.create_skill(WS, SKILL, _input(taskset=TASKSET_REF))
    assert harness.store.records == {}

    harness.tasksets.add(TASKSET_REF)
    skill = await harness.service.create_skill(WS, SKILL, _input(taskset=TASKSET_REF))
    assert skill.taskset == TASKSET_REF
    assert ("GET", f"/apis/evals/v2/workspaces/{WS}/tasksets/my-taskset") in harness.authorize.calls


async def test_linked_taskset_read_is_authorized_as_the_caller(harness: Harness) -> None:
    harness.tasksets.add(TASKSET_REF)
    harness.authorize.denied.add(f"/apis/evals/v2/workspaces/{WS}/tasksets/my-taskset")
    with pytest.raises(PermissionError):
        await harness.service.create_skill(WS, SKILL, _input(taskset=TASKSET_REF))


async def test_unlinked_publish_needs_no_evals_client(harness: Harness) -> None:
    harness.service.evals_client = None
    skill = await harness.service.create_skill(WS, SKILL, _input())
    assert skill.taskset is None
    with pytest.raises(LinkedTasksetNotFoundError, match="not available"):
        await harness.service.publish_skill(WS, SKILL, _input(taskset=TASKSET_REF))


async def test_frontmatter_name_must_match_the_skill_name(harness: Harness) -> None:
    with pytest.raises(SkillValidationError, match="frontmatter name"):
        await harness.service.create_skill(WS, "other-skill", _input())
    assert harness.store.records == {}


async def test_missing_source_and_oversized_source_are_refused(harness: Harness) -> None:
    harness.files.objects.pop((WS, "src"))
    with pytest.raises(SkillSourceError, match="not found"):
        await harness.service.create_skill(WS, SKILL, _input())

    harness.source()
    harness.service.archive_store.limits = SkillArchiveLimits(max_total_payload_bytes=16)
    with pytest.raises(SkillSourceError, match="limit"):
        await harness.service.create_skill(WS, SKILL, _input())
    assert harness.files.downloads == []


async def test_list_and_delete(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    await harness.service.publish_skill(WS, SKILL, _input())

    page = await harness.service.list_skills(WS)
    assert [(s.name, s.revision) for s in page.data] == [(SKILL, 2)]
    revisions = await harness.service.list_revisions(WS, SKILL)
    assert revisions is not None and [s.revision for s in revisions.data] == [2, 1]

    assert await harness.service.delete_skill(WS, SKILL) is True
    assert await harness.service.get_skill(WS, SKILL) is None
    assert harness.store.revisions() == []
    assert await harness.service.delete_skill(WS, SKILL) is False
    # Archives are retained: accepted runs may hold pins to them.
    assert harness.files.objects[(WS, SKILLS_FILESET_NAME)]


async def test_failed_revision_one_rolls_back_the_head(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(harness.store, "update", fail)
    with pytest.raises(RuntimeError, match="store down"):
        await harness.service.create_skill(WS, SKILL, _input())
    assert not any(isinstance(r, SkillEntity) for r in harness.store.records.values())


@pytest.mark.parametrize("name", ["3d-tools", "a", "x" * 64, "ab-", "a--b"])
async def test_names_the_entity_store_would_refuse_are_rejected_before_any_io(harness: Harness, name: str) -> None:
    with pytest.raises(SkillValidationError):
        await harness.service.create_skill(WS, name, _input())
    assert harness.files.downloads == []
    assert harness.authorize.calls == []


async def test_a_63_character_name_is_accepted(harness: Harness) -> None:
    name = "s" + "k" * 62
    harness.files.put_dir("src", f"skills/{name}", {"SKILL.md": f"---\nname: {name}\ndescription: d\n---\n".encode()})

    skill = await harness.service.create_skill(
        WS, name, SkillInput(source=SkillSource(fileset=f"{WS}/src#skills/{name}"))
    )

    assert skill.name == name


async def test_create_resumes_a_head_left_unpublished_by_a_crash(harness: Harness) -> None:
    await harness.store.create(SkillEntity(name=SKILL, workspace=WS))  # head written, revision 1 never was

    skill = await harness.service.create_skill(WS, SKILL, _input())

    assert skill.revision == 1
    assert harness.store.head().tags["latest"] == 1


async def test_publish_finishes_an_unpublished_head(harness: Harness) -> None:
    await harness.store.create(SkillEntity(name=SKILL, workspace=WS))

    skill, created = await harness.service.publish_skill(WS, SKILL, _input())

    assert (skill.revision, created) == (1, True)


async def test_publish_allocates_above_children_whose_pointers_never_landed(harness: Harness) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    head = harness.store.head()
    rev1 = harness.store.revisions()[0]
    # Revisions 2-6 were created but their pointer writes never landed: the head still says latest=1.
    for n in range(2, 7):
        stray = rev1.model_copy(update={"name": revision_name(n), "revision": n})
        stray.frontmatter = stray.frontmatter.model_copy(update={"description": f"stray {n}"})
        stray.content_hash = revision_digest(stray)
        stray._parent = head.id
        await harness.store.create(stray)
    harness.source(body="v2\n")

    skill, created = await harness.service.publish_skill(WS, SKILL, _input())

    # One conflict, then straight past every stray ordinal rather than five failed attempts.
    assert (skill.revision, created) == (7, True)
    assert harness.store.head().tags["latest"] == 7


async def test_retry_refuses_to_write_into_a_recreated_skill(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    await harness.service.create_skill(WS, SKILL, _input())
    harness.source(body="v2\n")
    harness.store.contend[2] = "different"
    real_create = harness.store.create

    async def create_then_recreate_head(entity: EntityBase) -> EntityBase:
        try:
            return await real_create(entity)
        except NemoEntityConflictError:
            harness.store.head()._id = "id-recreated"  # deleted and recreated under the same name
            raise

    monkeypatch.setattr(harness.store, "create", create_then_recreate_head)

    with pytest.raises(SkillRevisionConflictError, match="was deleted"):
        await harness.service.publish_skill(WS, SKILL, _input())


async def test_source_is_read_with_the_callers_client_not_the_service_client() -> None:
    service_files, caller_files = FakeFiles(), FakeFiles()
    caller_files.put_dir("src", f"skills/{SKILL}", skill_files())
    store = FakeStore()
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(service_files.handler)) as service_http,
        httpx.AsyncClient(transport=httpx.MockTransport(caller_files.handler)) as caller_http,
    ):
        service = SkillService(
            store,
            SkillArchiveStore(
                AsyncFilesClient(base_url=BASE, workspace=WS, http_client=service_http),
                source_files_client=AsyncFilesClient(base_url=BASE, workspace=WS, http_client=caller_http),
            ),
            authorize=Authorizer(),
        )
        await service.create_skill(WS, SKILL, _input())

    assert all(path.startswith(f"skills/{SKILL}/") for path in caller_files.downloads)
    assert caller_files.uploads == []
    assert not any(path.startswith("skills/") for path in service_files.downloads)
    assert service_files.uploads  # the archive went to the service-owned fileset


async def test_a_pre_created_skills_fileset_on_external_storage_is_refused(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = _fileset_json

    def external(workspace: str, name: str) -> dict[str, object]:
        return {
            **real(workspace, name),
            "storage": {"type": "s3", "bucket": "mine", "prefix": "", "use_sdk_auth": True},
        }

    monkeypatch.setattr(sys.modules[__name__], "_fileset_json", external)

    with pytest.raises(SkillStorageError, match="reserved"):
        await harness.service.create_skill(WS, SKILL, _input())
