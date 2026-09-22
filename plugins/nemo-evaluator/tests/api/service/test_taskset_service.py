# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib

import pytest
from nemo_evaluator.api.schemas import MetadataItem, TaskRef, Taskset, TasksetInput
from nemo_evaluator.api.service.taskset_service import (
    DuplicateTaskRefError,
    TaskRefNotFoundError,
    TasksetExistsError,
    TasksetService,
)
from nemo_evaluator.revisions import RevisionNotFoundError
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError


class _FakeTaskService:
    """Stands in for TaskService's existence check; ``get_task`` resolves only known (workspace, name)."""

    def __init__(self, existing: set[tuple[str, str]]) -> None:
        self.existing = existing

    async def get_task(self, workspace: str, name: str) -> object | None:
        return object() if (workspace, name) in self.existing else None

    async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str:
        """A stable per-task digest, so pinned membership is deterministic across a test.

        The digest covers the **fragment** as well as the task, because the real
        ``TaskService.resolve_revision`` resolves it: a bare member and a tag-pinned one give
        different digests whenever the tag names an older revision. A fake that ignored the fragment
        would return the same digest either way, and a publish path that dropped the fragment
        entirely would look correct in every assertion here.
        """
        if (workspace, name) not in self.existing:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        return hashlib.sha256(f"{workspace}/{name}#{fragment}".encode()).hexdigest()

    def digest_for(self, workspace: str, name: str, fragment: str = "latest") -> str:
        """The digest a test should expect for a member — same derivation as ``resolve_revision``."""
        return hashlib.sha256(f"{workspace}/{name}#{fragment}".encode()).hexdigest()


def _taskset_input() -> TasksetInput:
    return TasksetInput(
        description="A smoke-test grouping.",
        tasks=[TaskRef("task-a"), TaskRef("default/task-b")],
        metadata=[MetadataItem(key="suite", value="smoke")],
    )


@pytest.fixture
def existing_tasks() -> set[tuple[str, str]]:
    return {("default", "task-a"), ("default", "task-b")}


@pytest.fixture
def service(existing_tasks: set[tuple[str, str]], entity_store) -> TasksetService:
    return TasksetService(entity_store, _FakeTaskService(existing_tasks))


async def test_create_then_get(service: TasksetService) -> None:
    created, _ = await service.create_taskset("ts-1", _taskset_input(), workspace="default")

    assert isinstance(created, Taskset)
    assert created.name == "ts-1"
    assert created.id == "taskset-ts-1"
    assert created.description == "A smoke-test grouping."
    # Members are stored workspace-qualified *and* digest-pinned: a bare "task-a" is resolved on
    # write, so the stored grouping names exact revisions rather than moving targets.
    assert {t.root.split("#")[0] for t in created.tasks} == {"default/task-a", "default/task-b"}
    assert all(len(t.root.split("#")[1]) == 64 for t in created.tasks)
    assert created.created_at is not None

    got = await service.get_taskset("default", "ts-1")
    assert got is not None and got.name == "ts-1"


async def test_a_tag_pinned_member_stores_the_tagged_revision(
    existing_tasks: set[tuple[str, str]], entity_store
) -> None:
    """A member's ``#tag`` must be resolved through the tag, not silently treated as the head.

    This is the whole point of resolving membership on write: the stored ref has to name the
    revision the tag pointed at *then*, so moving the tag afterwards cannot re-point a published
    taskset. Dropping the fragment would still produce a digest-shaped ref, so only comparing
    against the tag-specific digest catches it.
    """
    task_service = _FakeTaskService(existing_tasks)
    service = TasksetService(entity_store, task_service)

    created, _ = await service.create_taskset(
        "ts-1", TasksetInput(tasks=[TaskRef("task-a#blessed"), TaskRef("task-b")]), workspace="default"
    )

    stored = {t.root.split("#")[0]: t.root.split("#")[1] for t in created.tasks}
    assert stored["default/task-a"] == task_service.digest_for("default", "task-a", "blessed")
    assert stored["default/task-b"] == task_service.digest_for("default", "task-b", "latest")
    assert stored["default/task-a"] != task_service.digest_for("default", "task-a", "latest"), (
        "a tag-pinned member must not collapse onto the task's current revision"
    )


async def test_create_validates_missing_task_ref(service: TasksetService) -> None:
    taskset_input = TasksetInput(tasks=[TaskRef("task-a"), TaskRef("nope")])
    with pytest.raises(TaskRefNotFoundError, match="not found"):
        await service.create_taskset("ts-1", taskset_input, workspace="default")


async def test_create_resolves_bare_ref_against_taskset_workspace(
    existing_tasks: set[tuple[str, str]], entity_store
) -> None:
    # A bare "task-a" ref must resolve against the taskset's own workspace ("other"), where it is absent.
    service = TasksetService(entity_store, _FakeTaskService(existing_tasks))
    with pytest.raises(ValueError, match="not found in workspace 'other'"):
        await service.create_taskset("ts-1", TasksetInput(tasks=[TaskRef("task-a")]), workspace="other")


async def test_create_rejects_refs_resolving_to_same_task(service: TasksetService) -> None:
    # "task-a" and "default/task-a" resolve to the same (default, task-a) — rejected even though the
    # ref strings differ (the field validator only catches byte-identical dupes).
    taskset_input = TasksetInput(tasks=[TaskRef("task-a"), TaskRef("default/task-a")])
    with pytest.raises(DuplicateTaskRefError, match="already in this taskset"):
        await service.create_taskset("ts-1", taskset_input, workspace="default")


async def test_create_rejects_duplicate(service: TasksetService) -> None:
    await service.create_taskset("ts-1", _taskset_input(), workspace="default")
    with pytest.raises(TasksetExistsError, match="already exists"):
        await service.create_taskset("ts-1", _taskset_input(), workspace="default")


async def test_create_allows_same_name_in_different_workspaces(service: TasksetService) -> None:
    # Taskset names are unique per workspace, not globally: the same name in another workspace is a
    # distinct taskset and must not raise TasksetExistsError (409). Empty task lists keep this focused
    # on name scoping rather than per-workspace task-ref validation.
    first, _ = await service.create_taskset("ts-1", TasksetInput(), workspace="default")
    second, _ = await service.create_taskset("ts-1", TasksetInput(), workspace="other")

    assert first.name == second.name == "ts-1"
    assert first.workspace == "default"
    assert second.workspace == "other"


async def test_get_returns_none_when_missing(service: TasksetService) -> None:
    assert await service.get_taskset("default", "nope") is None


async def test_list_returns_workspace_tasksets(service: TasksetService) -> None:
    await service.create_taskset("a", _taskset_input(), workspace="default")
    await service.create_taskset("b", _taskset_input(), workspace="default")

    page = await service.list_tasksets(workspace="default")

    assert {t.name for t in page.data} == {"a", "b"}
    assert page.pagination is not None and page.pagination.total_results == 2


async def test_delete(service: TasksetService) -> None:
    await service.create_taskset("ts-1", _taskset_input(), workspace="default")
    assert await service.delete_taskset("default", "ts-1") is True
    assert await service.get_taskset("default", "ts-1") is None


async def test_delete_returns_false_when_missing(service: TasksetService) -> None:
    assert await service.delete_taskset("default", "nope") is False


# --- Failure handling and membership resolution -------------------------------


async def test_create_rolls_back_the_head_when_publishing_fails(
    existing_tasks: set[tuple[str, str]], entity_store
) -> None:
    """Mirrors the task-side rollback: a taskset head with no revision would break the invariant
    that `#latest` always resolves."""
    service = TasksetService(entity_store, _FakeTaskService(existing_tasks))
    real_create = type(entity_store).create

    async def _boom(entity):
        if entity.__entity_type__ == "taskset_revision":
            raise RuntimeError("store unavailable")
        return await real_create(entity_store, entity)

    entity_store.create = _boom

    with pytest.raises(RuntimeError):
        await service.create_taskset("ts-1", _taskset_input(), workspace="default")

    entity_store.create = real_create.__get__(entity_store)
    assert await service.get_taskset("default", "ts-1") is None


async def test_rollback_failure_does_not_mask_the_original_error(
    existing_tasks: set[tuple[str, str]], entity_store
) -> None:
    """If cleanup also fails, the caller must still see *why* the publish failed — otherwise they
    debug the rollback instead of the actual fault."""
    service = TasksetService(entity_store, _FakeTaskService(existing_tasks))
    real_create = type(entity_store).create

    async def _boom(entity):
        if entity.__entity_type__ == "taskset_revision":
            raise RuntimeError("the original failure")
        return await real_create(entity_store, entity)

    async def _delete_also_fails(*args, **kwargs):
        raise RuntimeError("rollback failed too")

    entity_store.create = _boom
    entity_store.delete = _delete_also_fails

    with pytest.raises(RuntimeError, match="the original failure"):
        await service.create_taskset("ts-1", _taskset_input(), workspace="default")


async def test_member_naming_an_unknown_revision_is_rejected(
    existing_tasks: set[tuple[str, str]], entity_store
) -> None:
    """A member pinned to a revision that does not exist is a client error on the submitted body,
    not a server fault — the route maps this to 422."""

    class _NoSuchRevision(_FakeTaskService):
        async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str:
            raise RevisionNotFoundError(f"no revision {fragment!r}")

    service = TasksetService(entity_store, _NoSuchRevision(existing_tasks))

    with pytest.raises(TaskRefNotFoundError, match="no published revision"):
        await service.create_taskset("ts-1", TasksetInput(tasks=[TaskRef(f"task-a#{'c' * 64}")]), workspace="default")


async def test_members_resolve_concurrently(existing_tasks: set[tuple[str, str]], entity_store) -> None:
    """Membership resolution fans out rather than running one member at a time — a Harbor-scale
    dataset names hundreds of tasks, and serial resolution made publish latency linear in size."""
    overlap = {"peak": 0, "current": 0}

    class _Tracking(_FakeTaskService):
        async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str:
            overlap["current"] += 1
            overlap["peak"] = max(overlap["peak"], overlap["current"])
            await asyncio.sleep(0)  # yield, so overlapping calls can interleave
            overlap["current"] -= 1
            return hashlib.sha256(f"{workspace}/{name}".encode()).hexdigest()

    members = {("default", f"task-{i}") for i in range(5)}
    service = TasksetService(entity_store, _Tracking(members))

    await service.create_taskset(
        "ts-1", TasksetInput(tasks=[TaskRef(f"task-{i}") for i in range(5)]), workspace="default"
    )

    assert overlap["peak"] > 1, "members must resolve concurrently, not one at a time"


async def test_membership_is_stored_in_canonical_order(existing_tasks: set[tuple[str, str]], entity_store) -> None:
    """Membership is a set, so it is stored sorted rather than in submission order.

    The digest covers the stored list, so leaving order to the caller would give one grouping two
    identities. Sorting also has to survive concurrent resolution, which completes out of order.
    """
    members = {("default", f"task-{i}") for i in range(5)}
    service = TasksetService(entity_store, _FakeTaskService(members))

    created, _ = await service.create_taskset(
        "ts-1", TasksetInput(tasks=[TaskRef(f"task-{i}") for i in reversed(range(5))]), workspace="default"
    )

    assert [t.root.split("#")[0] for t in created.tasks] == [f"default/task-{i}" for i in range(5)]


async def test_reordering_members_publishes_no_revision(existing_tasks: set[tuple[str, str]], entity_store) -> None:
    """Re-submitting the same members in a different order is not a content change.

    This is the property canonical ordering exists for: without it the two requests hash
    differently, and a caller that merely rebuilt its list from a set would cut a revision whose
    content is indistinguishable from its predecessor's.
    """
    members = {("default", f"task-{i}") for i in range(3)}
    service = TasksetService(entity_store, _FakeTaskService(members))

    created, _ = await service.create_taskset(
        "ts-1", TasksetInput(tasks=[TaskRef(f"task-{i}") for i in range(3)]), workspace="default"
    )
    replaced, published = await service.replace_taskset(
        "ts-1", TasksetInput(tasks=[TaskRef(f"task-{i}") for i in reversed(range(3))]), workspace="default"
    )

    assert not published, "reordered membership is the same set, so nothing should be published"
    assert replaced.revision == created.revision == 1


async def test_tag_revision_returns_none_for_a_missing_taskset(service: TasksetService) -> None:
    assert await service.tag_revision("default", "nope", "blessed", "latest") is None


async def test_list_revisions_returns_none_for_a_missing_taskset(service: TasksetService) -> None:
    assert await service.list_revisions("default", "nope") is None


async def test_replace_applies_project(service: TasksetService) -> None:
    """`project` used to be accepted and silently discarded on replace."""
    await service.create_taskset("ts-1", _taskset_input(), workspace="default", project="proj-a")
    replaced, _ = await service.replace_taskset("ts-1", _taskset_input(), workspace="default", project="proj-b")
    assert replaced.project == "proj-b"


async def test_replace_without_project_leaves_it_unchanged(service: TasksetService) -> None:
    """An omitted query parameter means "leave it alone", not "clear it"."""
    await service.create_taskset("ts-1", _taskset_input(), workspace="default", project="proj-a")
    replaced, _ = await service.replace_taskset("ts-1", _taskset_input(), workspace="default")
    assert replaced.project == "proj-a"


async def test_a_failed_member_cancels_its_siblings(existing_tasks: set[tuple[str, str]], entity_store) -> None:
    """`gather` propagates the first failure but leaves siblings running; one bad member in a large
    grouping would otherwise keep reading long after the request failed."""
    finished: list[str] = []

    class _SlowExceptOne(_FakeTaskService):
        async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str:
            if name == "task-0":
                raise NemoEntityNotFoundError("missing")
            await asyncio.sleep(0.05)
            finished.append(name)
            return hashlib.sha256(name.encode()).hexdigest()

    members = {("default", f"task-{i}") for i in range(5)}
    service = TasksetService(entity_store, _SlowExceptOne(members))

    with pytest.raises(TaskRefNotFoundError):
        await service.create_taskset(
            "ts-1", TasksetInput(tasks=[TaskRef(f"task-{i}") for i in range(5)]), workspace="default"
        )

    await asyncio.sleep(0.1)  # give any un-cancelled sibling time to finish
    assert finished == [], "siblings must be cancelled once a member has failed"
