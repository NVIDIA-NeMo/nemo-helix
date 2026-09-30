# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for taskset-reference resolution on the agent-eval submit path."""

from __future__ import annotations

from typing import TypeVar

import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    HarborTaskDefinition,
    MetadataItem,
    MetricRef,
    TaskInputs,
    TaskRef,
    TasksetRef,
    parse_subentity_ref,
)
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.jobs.agent_spec import AgentEvalSpec, AgentEvalTaskInput, ResolvedTask
from nemo_evaluator.jobs.kinds.evaluator import _to_runtime_task
from nemo_evaluator.jobs.kinds.registry import KIND_ADAPTERS
from nemo_evaluator.jobs.kinds.types import LoadedTask, SubmitContext
from nemo_evaluator.revisions import apply_tag, get_revision, head_digest, is_digest, publish_revision
from nemo_evaluator.task_identity import UnsupportedTaskKindError
from nemo_evaluator.task_refs import (
    _reject_harbor_taskset_fileref,
    _validate_loaded_ids,
    load_tasks,
    snapshot_task,
    validate_execution_support,
)
from nemo_helix_plugin.entities import EntityBase
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.sdk import AsyncNeMoHelix
from pydantic import ValidationError

_EntityT = TypeVar("_EntityT", bound=EntityBase)


def _revision(item: LoadedTask) -> TaskRevisionEntity:
    assert item.stored is not None
    return item.stored[1]


def _loaded_evaluator(item: LoadedTask) -> EvaluatorTaskDefinition:
    definition = _revision(item).spec
    assert isinstance(definition, EvaluatorTaskDefinition)
    return definition


def _provenance(task: ResolvedTask) -> TaskProvenance:
    assert task.spec.provenance is not None
    return task.spec.provenance


def _async_platform() -> AsyncNeMoHelix:
    return AsyncNeMoHelix(base_url="http://platform.test", workspace="default")


def _task(name: str, *, workspace: str = "default", metric: str = "default/m") -> TaskEntity:
    return TaskEntity(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=f"Do {name}.",
            inputs=TaskInputs(instruction=f"instruction for {name}"),
            metrics=[MetricRef(metric)],
        ),
        name=name,
        workspace=workspace,
        metadata=[MetadataItem(key="suite", value="geo")],
    )


def _taskset(name: str, task_refs: list[str], *, workspace: str = "default") -> TasksetEntity:
    return TasksetEntity(name=name, workspace=workspace, tasks=[TaskRef(r) for r in task_refs])


#: Stands in for the digest of a member the test deliberately never created. A published revision
#: must pin every member, so an unresolvable member still needs a digest-shaped fragment to get
#: stored at all — the lookup it is meant to fail on happens later, during expansion.
_ABSENT_MEMBER_DIGEST = "f" * 64


async def test_harbor_suite_submission_rejects_missing_members(entity_store):

    client = await _store(entity_store, _taskset("suite", [f"default/missing#{_ABSENT_MEMBER_DIGEST}"]))
    with pytest.raises(ValueError, match="not found"):
        await load_tasks(
            TasksetRef("default/suite"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_harbor_direct_list_snapshots_selected_revision(entity_store):
    """Change and delete a stored task after submission and verify its serialized snapshot retains the selected
    archive.
    """
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/files#v1/task/files",
                files_hash="a" * 64,
            ),
        ),
    )
    await _store(entity_store, task)
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    compiled = await AgentEvalJob.to_spec(
        AgentEvalInputSpec(tasks=[TaskRef("checkout")], target=HarborRunnerTarget()),
        workspace="default",
        entity_client=entity_store,
        async_sdk=None,
        is_local=False,
    )
    assert isinstance(compiled, AgentEvalSpec) and all(task.spec.kind == "harbor" for task in compiled.tasks)
    snapshot = compiled.model_dump_json()
    assert isinstance(task.spec, HarborTaskDefinition)
    task.spec.source.fileset_ref = "default/files#v2/task/files"
    await entity_store.update(task)
    await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    await entity_store.delete(TaskEntity, task.name, workspace=task.workspace)
    restored = AgentEvalSpec.model_validate_json(snapshot)
    assert all(task.spec.kind == "harbor" for task in restored.tasks)
    members = restored.tasks
    assert members[0].spec.kind == "harbor"
    assert members[0].spec.source.fileset_ref == "default/files#v1/task/files"
    assert _provenance(members[0]).entity_name == "default/checkout"


async def test_direct_alias_duplicates_fail_without_entity_client():

    with pytest.raises(ValueError, match="Duplicate task identity"):
        await load_tasks(
            [TaskRef("checkout"), TaskRef("default/checkout#blessed")],
            SubmitContext(workspace="default", entity_client=None, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_harbor_snapshot_survives_suite_tag_moves_and_deletion(entity_store):
    """Move a taskset tag and delete entities, verifying only future submissions see the newly selected revision."""
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/files#v1/task/files",
                files_hash="a" * 64,
            ),
        ),
    )
    await _store(entity_store, task, _taskset("suite", ["default/checkout"]))
    suite = await entity_store.get(TasksetEntity, name="suite", workspace="default")
    await apply_tag(entity_store, entity_store, TasksetRevisionEntity, suite, "blessed", "latest")
    request = AgentEvalInputSpec.model_validate({"tasks": "suite#blessed", "target": {"kind": "harbor"}})
    before = await AgentEvalJob.to_spec(
        request, workspace="default", entity_client=entity_store, async_sdk=None, is_local=False
    )
    assert isinstance(before, AgentEvalSpec) and all(task.spec.kind == "harbor" for task in before.tasks)
    stored_task = await entity_store.get(TaskEntity, name="checkout", workspace="default")
    assert isinstance(stored_task.spec, HarborTaskDefinition)
    stored_task.spec.source.fileset_ref = "default/files#v2/task/files"
    await entity_store.update(stored_task)
    await publish_revision(entity_store, entity_store, stored_task, TaskRevisionEntity)
    await _republish_with(entity_store, suite, ["default/checkout"])
    await apply_tag(entity_store, entity_store, TasksetRevisionEntity, suite, "blessed", "latest")

    snapshot = before.model_dump_json()
    assert _provenance(before.tasks[0]).revision_digest == head_digest(task)
    assert before.tasks[0].spec.kind == "harbor"
    assert before.tasks[0].spec.source.fileset_ref == "default/files#v1/task/files"
    after = await AgentEvalJob.to_spec(
        request, workspace="default", entity_client=entity_store, async_sdk=None, is_local=False
    )
    assert isinstance(after, AgentEvalSpec) and all(task.spec.kind == "harbor" for task in after.tasks)
    assert after.tasks != before.tasks
    entity_store.entities.clear()
    restored = AgentEvalSpec.model_validate_json(snapshot)
    assert restored == before


async def test_harbor_suite_post_resolves_each_member_with_scoring(entity_store, monkeypatch):
    """Verify submission loads every taskset member and embeds its scoring fields and selected revision digest."""
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    tasks = [
        TaskEntity(
            name=f"task-{i}",
            workspace="default",
            spec=HarborTaskDefinition(
                kind="harbor",
                native_task_id=f"task-{i}",
                harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
                source=HarborArchiveSource(fileset_ref=f"default/files#task-{i}/archive", files_hash="a" * 64),
            ),
        )
        for i in range(32)
    ]
    suite = _taskset("large", [f"default/{task.name}" for task in tasks])
    await _store(entity_store, *tasks, suite)
    original = entity_store.get
    fetched = []

    async def tracked(entity_type, *args, **kwargs):
        fetched.append(entity_type)
        return await original(entity_type, *args, **kwargs)

    monkeypatch.setattr(entity_store, "get", tracked)
    result = await AgentEvalJob.to_spec(
        AgentEvalInputSpec.model_validate({"tasks": "default/large", "target": {"kind": "harbor"}}),
        workspace="default",
        entity_client=entity_store,
        async_sdk=None,
        is_local=False,
    )
    assert fetched.count(TaskEntity) == len(tasks)
    assert isinstance(result, AgentEvalSpec)
    assert all(task.spec.kind == "harbor" for task in result.tasks)
    assert len(result.tasks) == len(tasks)
    assert all(entry.spec.metrics == [] and entry.spec.views == {} for entry in result.tasks)
    assert {f"{_provenance(entry).entity_name}#{_provenance(entry).revision_digest}" for entry in result.tasks} == {
        f"default/{task.name}#{head_digest(task)}" for task in tasks
    }


async def test_submission_rejects_incompatible_suite_member_and_deleted_task(entity_store):
    """Verify a taskset fails submission when a member has an unsupported kind or has been deleted."""
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    task = _task("wrong-kind")
    task.spec.metrics = []
    await _store(entity_store, task, _taskset("suite", ["default/wrong-kind"]))
    with pytest.raises(ValueError, match="evaluator tasks cannot run on a harbor target"):
        loaded = await load_tasks(
            TasksetRef("suite"),
            SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
        )
        snapshots = [
            await snapshot_task(
                item,
                SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
            )
            for item in loaded
        ]
        validate_execution_support(snapshots, target=HarborRunnerTarget(), adapters=KIND_ADAPTERS)
    await entity_store.delete(TaskEntity, task.name, workspace=task.workspace)
    with pytest.raises(ValueError, match="not found"):
        loaded = await load_tasks(
            TasksetRef("suite"),
            SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
        )
        snapshots = [
            await snapshot_task(
                item,
                SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
            )
            for item in loaded
        ]
        validate_execution_support(snapshots, target=HarborRunnerTarget(), adapters=KIND_ADAPTERS)


async def _pin_members(client, taskset: TasksetEntity) -> list[TaskRef]:
    """Resolve a fixture's member refs to digests, the way ``_resolved_content`` does on write.

    Tests name members as ``workspace/task`` because that is what a caller submits; the entity that
    reaches storage always carries ``#<digest>`` instead, and ``TasksetRevisionEntity`` rejects
    anything less. Doing the resolution here keeps the fixtures readable without letting them
    describe a taskset the service could not have produced. A ref the test already pinned by hand is
    left alone — that is the case under test.

    Non-digest fragments go through ``get_revision`` rather than ``head_digest``, matching what
    ``resolve_revision`` does on the write path. The two agree for a bare ref, whose fragment is
    ``latest`` — but a member named ``task#blessed`` must pin the revision that tag points at, which
    is not the head once the tag has been left behind.
    """
    pinned: list[TaskRef] = []
    for ref in taskset.tasks:
        member_workspace, member_name, fragment = parse_subentity_ref(ref.root, taskset.workspace)
        if is_digest(fragment):
            pinned.append(ref)
            continue
        try:
            task = await client.get(TaskEntity, name=member_name, workspace=member_workspace)
            digest = (await get_revision(client, TaskRevisionEntity, task, fragment)).content_hash
        except NemoEntityNotFoundError:
            digest = _ABSENT_MEMBER_DIGEST
        pinned.append(TaskRef(f"{member_workspace}/{member_name}#{digest}"))
    return pinned


async def _create_published(client, entity: EntityBase) -> EntityBase:
    """Insert an entity and publish its first revision, as the service does on create.

    Expansion reads published revisions on both levels — the taskset's own and each member's — so a
    record inserted without one is not a state the services can produce. Every helper here goes
    through this rather than a bare ``create`` so the fixtures stay reachable from the real API.
    """
    if isinstance(entity, TasksetEntity):
        entity.tasks = await _pin_members(client, entity)
    await client.create(entity)
    if isinstance(entity, TaskEntity):
        await publish_revision(client, client, entity, TaskRevisionEntity)
    elif isinstance(entity, TasksetEntity):
        await publish_revision(client, client, entity, TasksetRevisionEntity)
    return entity


async def _store(client, *entities: EntityBase):
    """Build a store in which every task and taskset carries a published revision."""
    for entity in entities:
        await _create_published(client, entity)
    return client


async def test_resolves_taskset_members_to_inline_task_inputs(entity_store) -> None:
    client = await _store(
        entity_store,
        _task("capital-of-france"),
        _task("capital-of-japan"),
        _taskset("geo", ["default/capital-of-france", "default/capital-of-japan"]),
    )

    tasks = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert [KIND_ADAPTERS[t.kind].runtime_id(t) for t in tasks] == ["capital-of-france", "capital-of-japan"]
    assert all(isinstance(t, LoadedTask) for t in tasks)
    # A stored task's refs pass through untouched (resolved to inline later in the metric pass).
    assert _loaded_evaluator(tasks[0]).metrics == [MetricRef("default/m")]
    assert _loaded_evaluator(tasks[0]).intent == "Do capital-of-france."
    assert _loaded_evaluator(tasks[0]).inputs.instruction == "instruction for capital-of-france"
    # A task stored without ground truth expands to an empty reference, not a missing one.
    assert _loaded_evaluator(tasks[0]).reference == {}


async def test_grader_only_reference_survives_taskset_expansion(entity_store) -> None:
    """Held-out ground truth must not be the privilege of inline submissions.

    Expansion projects a stored task onto the inline DTO field by field, so a field added to the
    stored spec and forgotten here silently becomes empty at run time — the agent is then graded
    against nothing, and the run still reports a score. That is the failure this guards.
    """
    task = _task("fix-bug")
    assert isinstance(task.spec, EvaluatorTaskDefinition)
    task.spec.reference = {"expected": "Paris", "held_out_tests": ["test_capital.py"]}
    client = await _store(entity_store, task, _taskset("geo", ["default/fix-bug"]))

    tasks = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert _loaded_evaluator(tasks[0]).reference == {"expected": "Paris", "held_out_tests": ["test_capital.py"]}


async def test_expansion_returns_the_pinned_reference_not_the_current_one(entity_store) -> None:
    """``reference`` is digest-covered, so republishing it cuts a revision the old pin excludes.

    A pin that honoured new ground truth would silently re-grade a "reproducible" dataset.
    """
    task = _task("fix-bug")
    assert isinstance(task.spec, EvaluatorTaskDefinition)
    task.spec.reference = {"expected": "Paris"}
    client = await _store(entity_store, task)
    pinned_digest = head_digest(task)
    await _create_published(client, _taskset("geo", [f"default/fix-bug#{pinned_digest}"]))

    task.spec.reference = {"expected": "Lyon"}
    await client.update(task)
    await publish_revision(client, client, task, TaskRevisionEntity)

    tasks = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert _loaded_evaluator(tasks[0]).reference == {"expected": "Paris"}, (
        "expansion must return the pinned ground truth"
    )


async def test_bare_member_ref_resolves_against_taskset_workspace(entity_store) -> None:
    client = await _store(entity_store, _task("t1", workspace="team"), _taskset("ts", ["t1"], workspace="team"))

    tasks = await load_tasks(
        TasksetRef("team/ts"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert [KIND_ADAPTERS[t.kind].runtime_id(t) for t in tasks] == ["t1"]


async def test_unknown_taskset_raises_clear_error(entity_store) -> None:
    with pytest.raises(ValueError, match="Taskset reference 'default/missing' not found"):
        await load_tasks(
            TasksetRef("default/missing"),
            SubmitContext(
                workspace="default", entity_client=await _store(entity_store), async_sdk=None, adapters=KIND_ADAPTERS
            ),
        )


@pytest.mark.parametrize("missing", ["taskset", "taskset-revision", "member", "member-revision"])
async def test_offline_taskset_translates_resolution_errors(entity_store, missing):
    """Check missing tasksets, members, and revisions produce reference-specific errors while preserving their
    causes.
    """
    from nemo_evaluator.revisions import RevisionNotFoundError

    client = await _store(entity_store, _task("only"))
    ref = TasksetRef("default/geo")
    if missing != "taskset":
        member_ref = (
            f"default/gone#{_ABSENT_MEMBER_DIGEST}" if missing == "member" else f"default/only#{_ABSENT_MEMBER_DIGEST}"
        )
        await _create_published(client, _taskset("geo", [member_ref]))
    if missing == "taskset-revision":
        ref = TasksetRef(f"default/geo#{'c' * 64}")
    message = {
        "taskset": "not found",
        "taskset-revision": "names a revision that does not resolve",
        "member": "names a member that does not resolve",
        "member-revision": "names a member that does not resolve",
    }[missing]
    with pytest.raises(ValueError, match=message) as caught:
        await load_tasks(
            ref, SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS)
        )
    assert f"Taskset reference '{ref.root}'" in str(caught.value)
    expected_cause = RevisionNotFoundError if missing.endswith("revision") else NemoEntityNotFoundError
    assert isinstance(caught.value.__cause__, expected_cause)


async def test_missing_member_task_raises_clear_error(entity_store) -> None:
    client = await _store(entity_store, _taskset("geo", ["default/gone"]))
    with pytest.raises(ValueError, match=r"Task 'default/gone#\w+' referenced by taskset 'default/geo'"):
        await load_tasks(
            TasksetRef("default/geo"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_empty_taskset_raises_clear_error(entity_store) -> None:
    client = await _store(entity_store, _taskset("empty", []))
    with pytest.raises(ValueError, match="has no member tasks"):
        await load_tasks(
            TasksetRef("default/empty"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_duplicate_expanded_task_ids_rejected(entity_store) -> None:
    # Two members from different workspaces share the name 'dup' -> ambiguous task id.
    client = await _store(
        entity_store,
        _task("dup", workspace="a"),
        _task("dup", workspace="b"),
        _taskset("geo", ["a/dup", "b/dup"]),
    )
    with pytest.raises(ValueError, match="more than one task named 'dup'"):
        await load_tasks(
            TasksetRef("default/geo"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_taskset_ref_requires_entity_client(entity_store) -> None:
    with pytest.raises(ValueError, match="requires a platform connection"):
        await load_tasks(
            TasksetRef("default/geo"),
            SubmitContext(workspace="default", entity_client=None, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_canonicalize_agent_eval_tasks_passes_inline_list_through(entity_store) -> None:
    inline = [AgentEvalTaskInput(id="t", intent="x", metrics=[])]
    result = await load_tasks(
        inline, SubmitContext(workspace="default", entity_client=None, async_sdk=None, adapters=KIND_ADAPTERS)
    )
    assert result[0].inline is inline[0]


async def test_canonicalize_agent_eval_tasks_expands_a_taskset_ref(entity_store) -> None:
    client = await _store(entity_store, _task("only"), _taskset("geo", ["default/only"]))
    result = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )
    assert all(t.kind == "evaluator" for t in result)
    assert [KIND_ADAPTERS[t.kind].runtime_id(t) for t in result] == ["only"]


async def test_expansion_uses_the_pinned_revision_not_current_content(entity_store) -> None:
    """The property the whole pinning design exists for: an evaluation re-run expands to the same
    content even after a member task has published newer content.

    Before this was wired, expansion read the member's *head*, silently defeating the pin — the
    taskset looked reproducible and wasn't.
    """
    task = _task("capital-of-france")
    client = await _store(entity_store, task)
    pinned_digest = head_digest(task)

    await _create_published(client, _taskset("geo", [f"default/capital-of-france#{pinned_digest}"]))

    # The member publishes newer content after the taskset was pinned.
    assert isinstance(task.spec, EvaluatorTaskDefinition)
    task.spec.intent = "Something else entirely."
    await client.update(task)
    await publish_revision(client, client, task, TaskRevisionEntity)

    tasks = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert _loaded_evaluator(tasks[0]).intent == "Do capital-of-france.", "expansion must return the pinned content"


async def test_a_tag_pinned_member_stores_the_tagged_revision_not_the_head(entity_store) -> None:
    """A member named ``task#blessed`` resolves through the tag at write time, like any other pin.

    Resolution happens once, on write: the stored ref carries the digest the tag pointed at then, so
    moving the tag afterwards cannot re-point published membership. Pinning to the *head* instead
    would look right whenever the tag happens to name the newest revision and be wrong exactly when
    it does not.
    """
    task = _task("capital-of-france")
    client = await _store(entity_store, task)
    blessed_digest = head_digest(task)

    # 'blessed' stays on revision 1 while the task moves on to revision 2. Work from the head
    # ``apply_tag`` hands back — tagging bumps the record's version, so the original object is stale.
    stored_task = await client.get(TaskEntity, name="capital-of-france", workspace="default")
    tagged = await apply_tag(client, client, TaskRevisionEntity, stored_task, "blessed", "latest")
    assert isinstance(tagged.spec, EvaluatorTaskDefinition)
    tagged.spec.intent = "Something else entirely."
    await client.update(tagged)
    await publish_revision(client, client, tagged, TaskRevisionEntity)

    await _create_published(client, _taskset("geo", ["default/capital-of-france#blessed"]))

    stored_taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    assert stored_taskset.tasks[0].root == f"default/capital-of-france#{blessed_digest}", (
        "membership must pin the revision the tag named, not the head"
    )

    tasks = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )
    assert _loaded_evaluator(tasks[0]).intent == "Do capital-of-france."


async def _republish_with(client, taskset: TasksetEntity, task_refs: list[str]) -> None:
    """Change a stored taskset's membership and publish it, as ``replace_taskset`` does."""
    taskset.tasks = [TaskRef(ref) for ref in task_refs]
    taskset.tasks = await _pin_members(client, taskset)
    await client.update(taskset)
    await publish_revision(client, client, taskset, TasksetRevisionEntity)


async def test_bare_taskset_ref_expands_the_current_revision(entity_store) -> None:
    """An absent fragment means ``latest``, so a bare ref keeps tracking the taskset's tip."""
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    await _republish_with(client, taskset, ["default/a", "default/b"])

    tasks = await load_tasks(
        TasksetRef("default/geo"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert [KIND_ADAPTERS[t.kind].runtime_id(t) for t in tasks] == ["a", "b"], (
        "a bare ref must follow the taskset forward"
    )


async def test_digest_pinned_taskset_ref_expands_the_pinned_membership(entity_store) -> None:
    """The gap this closes: member *content* was already pinned, but membership was not.

    Replacing a taskset changed what a re-submitted spec evaluated, because the ref could only ever
    name the head. A digest-pinned ref holds the whole grouping still.
    """
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    pinned = head_digest(taskset)

    await _republish_with(client, taskset, ["default/a", "default/b"])

    tasks = await load_tasks(
        TasksetRef(f"default/geo#{pinned}"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert [KIND_ADAPTERS[t.kind].runtime_id(t) for t in tasks] == ["a"], (
        "a pinned ref must ignore members added after it was taken"
    )


async def test_tag_pinned_taskset_ref_resolves_through_the_tag(entity_store) -> None:
    """Tags are resolution inputs, so a tag-pinned ref resolves at expansion time, not at write."""
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    await apply_tag(client, client, TasksetRevisionEntity, taskset, "blessed", "latest")

    await _republish_with(client, taskset, ["default/a", "default/b"])

    tasks = await load_tasks(
        TasksetRef("default/geo#blessed"),
        SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
    )

    assert [KIND_ADAPTERS[t.kind].runtime_id(t) for t in tasks] == ["a"], "'blessed' still names revision 1"


async def test_pinned_taskset_ref_survives_a_replace(entity_store) -> None:
    """The reproducibility property end to end: same ref, same membership, across a replace."""
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    ref = TasksetRef(f"default/geo#{head_digest(taskset)}")

    before = await load_tasks(
        ref, SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS)
    )
    await _republish_with(client, taskset, ["default/b"])
    after = await load_tasks(
        ref, SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS)
    )

    assert (
        [KIND_ADAPTERS[t.kind].runtime_id(t) for t in before]
        == [KIND_ADAPTERS[t.kind].runtime_id(t) for t in after]
        == ["a"]
    )
    assert [_loaded_evaluator(t).intent for t in before] == [_loaded_evaluator(t).intent for t in after]


async def test_unresolvable_taskset_fragment_raises_a_clear_error(entity_store) -> None:
    """A pin that cannot be honoured stops the evaluation rather than falling back to the head."""
    client = await _store(entity_store, _task("a"), _taskset("geo", ["default/a"]))

    with pytest.raises(ValueError, match="names a revision that does not resolve"):
        await load_tasks(
            TasksetRef(f"default/geo#{'c' * 64}"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )

    with pytest.raises(ValueError, match="names a revision that does not resolve"):
        await load_tasks(
            TasksetRef("default/geo#nonesuch"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )


def test_taskset_ref_accepts_a_fragment_and_rejects_a_malformed_one() -> None:
    """The field pattern is what admits a pin at all, so assert it directly."""
    assert TasksetRef(f"default/geo#{'a' * 64}").root.endswith("a" * 64)
    assert TasksetRef("geo#blessed").root == "geo#blessed"

    with pytest.raises(ValidationError):
        TasksetRef("default/geo#bad fragment")


async def test_expansion_fails_loudly_when_a_pin_no_longer_resolves(entity_store) -> None:
    """Verify-on-read at the point it matters most: a pin that cannot be honoured must stop the
    evaluation rather than quietly substituting whatever is current."""
    client = await _store(entity_store, _task("only"))
    await _create_published(client, _taskset("geo", [f"default/only#{'c' * 64}"]))

    with pytest.raises(ValueError, match="no longer resolves"):
        await load_tasks(
            TasksetRef("default/geo"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )


async def test_expansion_rejects_a_task_whose_runner_the_target_cannot_run(entity_store) -> None:
    """A Harbor task's content is a directory of files, not fields. Projecting it onto an inline
    agent-eval task would silently produce a task with no intent and no metrics — an evaluation that
    runs and scores nothing. Refused instead, before the run starts."""
    harbor_task = TaskEntity(
        name="fix-test",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/harbor#packages/o-n/abc/files",
                files_hash="a" * 64,
            ),
        ),
    )
    client = await _store(entity_store, harbor_task)
    await _create_published(client, _taskset("mixed", [f"default/fix-test#{head_digest(harbor_task)}"]))

    with pytest.raises(UnsupportedTaskKindError, match="harbor"):
        loaded = await load_tasks(
            TasksetRef("default/mixed"),
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )
        snapshots = [
            await snapshot_task(
                item, SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS)
            )
            for item in loaded
        ]
        validate_execution_support(snapshots, target=_direct_target("fabric"), adapters=KIND_ADAPTERS)


async def test_incompatible_target_error_names_requested_target(entity_store):
    """Verify rejecting a stored Harbor task for a Fabric run identifies the requested target in the error."""
    from nemo_evaluator.jobs.agent_spec import FabricRunnerTarget

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/files#task/files",
                files_hash="a" * 64,
            ),
        ),
    )
    await _store(entity_store, task)
    with pytest.raises(UnsupportedTaskKindError, match="fabric target"):
        loaded = await load_tasks(
            [TaskRef("checkout")],
            SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
        )
        snapshots = [
            await snapshot_task(
                item,
                SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
            )
            for item in loaded
        ]
        validate_execution_support(snapshots, target=FabricRunnerTarget(config={}), adapters=KIND_ADAPTERS)


def _direct_target(kind):
    from nemo_evaluator.jobs.agent_spec import AgentTarget, FabricRunnerTarget, GymRunnerTarget, ModelTarget
    from nemo_evaluator_sdk.values import GenericAgent, Model

    return {
        "model": lambda: ModelTarget(model=Model(name="test", url="http://localhost/model")),
        "agent": lambda: AgentTarget(
            agent=GenericAgent(name="test", url="http://localhost/agent", body={}, response_path="$.output")
        ),
        "fabric": lambda: FabricRunnerTarget(
            config={"metadata": {"name": "test"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}
        ),
        "gym": lambda: GymRunnerTarget(agent="simple_agent", resources_server="mcqa", agent_config="simple.yaml"),
        "offline": lambda: None,
    }[kind]()


@pytest.mark.parametrize("kind", ["model", "agent", "fabric", "gym", "offline"])
async def test_direct_evaluator_references_resolve_and_compile(kind, entity_store, monkeypatch, tmp_path):
    """Resolve mixed revision selectors, compile snapshots, and verify supported runtimes preserve task data and
    score attribution.
    """
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec
    from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
    from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
    from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
    from nemo_evaluator_sdk.metrics.runner_rewards import GymRewardMetric
    from nemo_helix_plugin.jobs.spec import HelixJobSpec

    metric = (
        GymRewardMetric() if kind == "gym" else ExactMatchMetric(reference="DONE", candidate="{{sample.output_text}}")
    )
    bundle = bundle_metric(metric, CloudpickleMetricBundlePackager())
    resolved_metrics = []

    async def resolve_metric(ref, **kwargs):
        resolved_metrics.append(ref.root)
        return bundle

    monkeypatch.setattr("nemo_evaluator.metric_refs.resolve_metric_ref", resolve_metric)
    from nemo_evaluator_sdk.agent_eval.runtimes.gym.dataset import discover_gym_tasks

    dataset = tmp_path / "source.jsonl"
    dataset.write_text('{"responses_create_params": {}}\n')
    discovered = discover_gym_tasks(dataset, metrics=[])[0]
    tasks = [_task(name) for name in ["first", "second", "third"]]
    for task in tasks:
        assert isinstance(task.spec, EvaluatorTaskDefinition)
        task.spec.inputs = TaskInputs.model_validate(
            {**discovered.inputs, "instruction": task.name, "files": {"seed.txt": task.name}}
        )
        task.spec.views = {
            "quality": SemanticView.model_validate(
                {"reducer": "single", "signals": [{"metric": bundle.metric_type, "output": bundle.outputs[0].name}]}
            )
        }
        task.spec.reference = {"answer": "DONE"}
        task.metadata.append(MetadataItem(key="gym_row_extras", value=discovered.metadata["gym_row_extras"]))
    await _store(entity_store, *tasks)
    await apply_tag(entity_store, entity_store, TaskRevisionEntity, tasks[1], "blessed", "latest")
    digest = head_digest(tasks[0])
    assert isinstance(tasks[0].spec, EvaluatorTaskDefinition)
    tasks[0].spec.intent = "changed after selected revision"
    await entity_store.update(tasks[0])
    await publish_revision(entity_store, entity_store, tasks[0], TaskRevisionEntity)
    target = _direct_target(kind)
    request = AgentEvalInputSpec(
        tasks=[TaskRef("third"), TaskRef("second#blessed"), TaskRef(f"first#{digest}")],
        target=target,
        trials=[] if target is None else None,
    )
    spec = await AgentEvalJob.to_spec(
        request, workspace="default", entity_client=entity_store, async_sdk=_async_platform(), is_local=False
    )
    assert isinstance(spec, AgentEvalSpec) and all(task.spec.kind == "evaluator" for task in spec.tasks)
    assert [task.id for task in spec.tasks] == ["third", "second", "first"]
    assert spec.tasks[-1].spec.kind == "evaluator"
    assert spec.tasks[-1].spec.intent == "Do first."
    assert resolved_metrics == ["default/m"] * 3
    for task in spec.tasks:
        runtime = _to_runtime_task(task)
        assert runtime.inputs["instruction"] == task.id
        assert runtime.inputs["gym_row"] == {}
        assert runtime.metadata["gym_row_extras"] == {}
        assert runtime.reference == {"answer": "DONE"}
        assert runtime.views == task.spec.views
        assert len(runtime.metrics) == 1
    if kind == "gym":
        from types import SimpleNamespace

        class Jobs:
            async def get_execution_profiles(self):
                return SimpleNamespace(data=lambda: [])

        monkeypatch.setattr("nemo_evaluator.jobs.agent_evaluate.client_from_platform", lambda *_: Jobs())
    compiled = HelixJobSpec.model_validate(
        await AgentEvalJob.compile(
            workspace="default",
            spec=spec,
            entity_client=None,
            job_name=None,
            async_sdk=_async_platform(),
        )
    )
    assert compiled.steps[0].config["tasks"] == spec.model_dump(mode="json")["tasks"]
    if kind in {"model", "agent"}:
        return  # These endpoints execute in the platform integration matrix.

    import json
    import runpy
    from pathlib import Path

    from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
    from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig
    from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput

    runtime_tasks = [_to_runtime_task(task) for task in spec.tasks]
    trials = None
    runner = None
    if kind == "fabric":
        # Reuse the SDK runtime suite's fake native client rather than fake the runtime adapter.
        harness = runpy.run_path(
            str(
                Path(__file__).resolve().parents[3]
                / "packages/nemo_evaluator_sdk/tests/agent_eval/test_fabric_runtime.py"
            )
        )

        def handler(agent, kwargs):
            assert (Path(agent.environment.workspace) / "seed.txt").read_text() == kwargs["request"].request_id
            return harness["_FakeResult"](status="succeeded", output={"response": "DONE"})

        client_type = harness["_install_fake_fabric"](monkeypatch, handler)
        runner = FabricAgentRuntime(config=target.config, work_root=tmp_path / "fabric")
    elif kind == "gym":
        runner = GymAgentTaskRunner(
            config=GymRuntimeConfig(agent="simple_agent", resources_server="mcqa", agent_config="simple.yaml")
        )

        async def collect(input_path, output_path, work_dir):
            """Return fake Gym rollouts in reverse order to test attribution by task index."""
            rows = [json.loads(line) for line in input_path.read_text().splitlines()]
            assert [row["_ng_task_index"] for row in rows] == [0, 1, 2]
            assert all(row["responses_create_params"] == {} for row in rows)
            # Reverse rollout order to prove index-based attribution, not positional zipping.
            output_path.write_text(
                "".join(
                    json.dumps(
                        {"_ng_task_index": index, "_ng_rollout_index": 0, "reward": 1.0, "response": {"output": "DONE"}}
                    )
                    + "\n"
                    for index in [2, 1, 0]
                )
            )

        monkeypatch.setattr(runner, "_run_two_step", collect)
    else:
        trials = [
            AgentEvalTrial(
                id=f"trial-{task.id}",
                task_id=task.id,
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="DONE"),
            )
            for task in runtime_tasks
        ]
    result = await AgentEvaluator().run(
        tasks=runtime_tasks, target=runner, trials=trials, config=AgentEvalRunConfig(work_dir=tmp_path)
    )
    assert {trial.task_id for trial in result.trials} == {task.id for task in runtime_tasks}
    assert len(result.scores) == 3
    assert all(score.outputs for score in result.scores)
    assert all(output.value == 1.0 for score in result.scores for output in score.outputs)
    if kind == "fabric":
        assert {call["request"].request_id for call in client_type.recorded} == {task.id for task in runtime_tasks}
    if kind == "offline":
        assert trials is not None
        for invalid_trials, message in [
            (trials[:-1], "no trials produced"),
            (
                [
                    AgentEvalTrial(
                        id="extra",
                        task_id="unknown",
                        status=AgentEvalTrialStatus.COMPLETED,
                        output=AgentOutput(output_text="DONE"),
                    )
                ],
                "unknown task",
            ),
        ]:
            with pytest.raises(ValueError, match=message):
                await AgentEvaluator().run(tasks=runtime_tasks, trials=invalid_trials)


@pytest.mark.parametrize("source", ["inline", "refs", "taskset"])
@pytest.mark.parametrize(
    "field,value", [("gym_row", None), ("gym_row", []), ("gym_row_extras", None), ("gym_row_extras", "invalid")]
)
async def test_gym_content_rejected_before_environment_resolution(source, field, value, entity_store, monkeypatch):
    """Verify malformed Gym rows fail for every input form before environment resolution starts."""
    from unittest.mock import AsyncMock

    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, GymRunnerTarget

    task = _task("invalid")
    assert isinstance(task.spec, EvaluatorTaskDefinition)
    task.spec.metrics = []
    task.spec.inputs = TaskInputs.model_validate({"gym_row": value if field == "gym_row" else {}})
    task.metadata = [MetadataItem(key="gym_row_extras", value=value if field == "gym_row_extras" else {})]
    await _store(entity_store, task, _taskset("suite", ["invalid"]))
    sources = {
        "inline": [
            AgentEvalTaskInput(id=task.name, intent=task.spec.intent, inputs=task.spec.inputs, metadata=task.metadata)
        ],
        "refs": [TaskRef("invalid")],
        "taskset": TasksetRef("suite"),
    }
    resolve_environment = AsyncMock()
    monkeypatch.setattr("nemo_evaluator.jobs.agent_evaluate._resolve_gym_environment", resolve_environment)
    with pytest.raises(ValueError, match="task 'invalid'.*gym_row.*gym_row_extras"):
        await AgentEvalJob.to_spec(
            AgentEvalInputSpec(
                tasks=sources[source],
                target=GymRunnerTarget(agent="simple_agent", resources_server="mcqa", agent_config="simple.yaml"),
            ),
            workspace="default",
            entity_client=entity_store,
            async_sdk=None,
            is_local=False,
        )
    resolve_environment.assert_not_called()


async def test_direct_evaluator_refs_reject_cross_workspace_id_collision(entity_store):
    tasks = [_task("same", workspace=workspace) for workspace in ["default", "other"]]
    await _store(entity_store, *tasks)
    with pytest.raises(ValueError, match="task ids must be unique"):
        await load_tasks(
            [TaskRef("default/same"), TaskRef("other/same")],
            SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
        )


@pytest.mark.parametrize("kind", ["model", "agent", "fabric", "gym", "offline"])
@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong-kind"])
async def test_direct_reference_failures_for_every_evaluator_target(kind, failure, entity_store):
    """Check missing, duplicate, and incompatible task references across evaluator targets and offline scoring."""
    refs = [TaskRef("missing")]
    match = "not found"
    if failure == "duplicate":
        refs = [TaskRef("same"), TaskRef("default/same#latest")]
        match = "Duplicate task identity"
    elif failure == "wrong-kind":
        task = TaskEntity(
            name="harbor",
            workspace="default",
            spec=HarborTaskDefinition(
                kind="harbor",
                native_task_id="task",
                harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
                source=HarborArchiveSource(
                    fileset_ref="default/files#task/files",
                    files_hash="a" * 64,
                ),
            ),
        )
        await _store(entity_store, task)
        refs = [TaskRef("harbor")]
        match = "harbor tasks cannot run"
        if kind == "offline":
            resolved = await load_tasks(
                refs,
                SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
            )
            assert all(t.kind == "harbor" for t in resolved)
            return
    with pytest.raises(ValueError, match=match):
        loaded = await load_tasks(
            refs, SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS)
        )
        snapshots = [
            await snapshot_task(
                item,
                SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
            )
            for item in loaded
        ]
        validate_execution_support(snapshots, target=_direct_target(kind), adapters=KIND_ADAPTERS)


@pytest.mark.parametrize("offline", [False, True])
async def test_evaluator_taskset_shared_files_remains_supported(entity_store, offline):
    """Verify shared taskset files do not prevent evaluator task submission for live or offline runs."""
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    task = _task("shared")
    task.spec.metrics = []
    suite = _taskset("suite", ["shared"])
    suite.files_ref = "default/shared-files#scoring"
    await _store(entity_store, task, suite)
    spec = await AgentEvalJob.to_spec(
        AgentEvalInputSpec(
            tasks=TasksetRef("suite"),
            target=None if offline else _direct_target("model"),
            trials=[] if offline else None,
        ),
        workspace="default",
        entity_client=entity_store,
        async_sdk=_async_platform(),
        is_local=False,
    )
    assert isinstance(spec, AgentEvalSpec)
    assert spec.tasks[0].spec.kind == "evaluator"
    assert _provenance(spec.tasks[0]).entity_name == "default/shared"


@pytest.mark.parametrize("source_adapter", [False, True])
async def test_harbor_taskset_shared_files_rejected_on_both_paths(entity_store, source_adapter):
    """Reject Harbor tasksets with shared files in both job submission and source-adapter resolution."""
    from nemo_evaluator.harbor.resolution import resolve_harbor_taskset

    task = TaskEntity(
        name="harbor",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="test"),
        ),
    )
    suite = _taskset("suite", ["harbor"])
    suite.files_ref = "default/shared-files#scoring"
    await _store(entity_store, task, suite)
    with pytest.raises(ValueError, match="do not support shared files_ref"):
        if source_adapter:
            await resolve_harbor_taskset(TasksetRef("suite"), entity_client=entity_store)
        else:
            await load_tasks(
                TasksetRef("suite"), SubmitContext("default", entity_store, _async_platform(), KIND_ADAPTERS)
            )


async def test_inline_casefold_ids_rejected_during_loading():
    with pytest.raises(ValueError, match="task ids must be unique"):
        await load_tasks(
            [AgentEvalTaskInput(id="task", intent="First"), AgentEvalTaskInput(id="TASK", intent="Second")],
            SubmitContext("default", None, _async_platform(), KIND_ADAPTERS),
        )


def _inline(task_id: str) -> LoadedTask:
    return LoadedTask(inline=AgentEvalTaskInput(id=task_id, intent=task_id))


def _stored_evaluator(name: str, *, workspace: str = "default") -> LoadedTask:
    task = _task(name, workspace=workspace)
    revision = TaskRevisionEntity(name="rev", revision=1, workspace=workspace, spec=task.spec, content_hash="a" * 64)
    return LoadedTask(stored=(task, revision))


def _stored_harbor(name: str, native_task_id: str) -> LoadedTask:
    spec = HarborTaskDefinition(
        kind="harbor",
        native_task_id=native_task_id,
        source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="a" * 64),
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="test"),
    )
    task = TaskEntity(name=name, workspace="default", spec=spec)
    revision = TaskRevisionEntity(name="rev", revision=1, workspace="default", spec=spec, content_hash="c" * 64)
    return LoadedTask(stored=(task, revision))


def _taskset_revision(files_ref: str | None) -> TasksetRevisionEntity:
    return TasksetRevisionEntity(
        name="rev", revision=1, workspace="default", content_hash="d" * 64, files_ref=files_ref
    )


def test_harbor_taskset_shared_files_are_rejected():
    with pytest.raises(ValueError, match="do not support shared files_ref"):
        _reject_harbor_taskset_fileref(
            _taskset_revision("default/shared-files#scoring"), [_stored_harbor("alpha", "alpha")]
        )


@pytest.mark.parametrize(
    ("revision", "tasks"),
    [
        (None, [_stored_harbor("alpha", "alpha")]),
        (_taskset_revision(None), [_stored_harbor("alpha", "alpha")]),
        (_taskset_revision("default/shared-files#scoring"), [_stored_evaluator("shared")]),
    ],
)
def test_shared_files_are_kept_unless_a_harbor_taskset_carries_them(revision, tasks):
    _reject_harbor_taskset_fileref(revision, tasks)


def test_validate_loaded_ids_rejects_an_empty_list():
    with pytest.raises(ValueError, match="Expected at least one task"):
        _validate_loaded_ids([], KIND_ADAPTERS, None)


def test_validate_loaded_ids_accepts_distinct_ids():
    _validate_loaded_ids([_inline("one"), _inline("two")], KIND_ADAPTERS, None)
    _validate_loaded_ids([_stored_evaluator("one"), _stored_evaluator("two", workspace="other")], KIND_ADAPTERS, None)
    _validate_loaded_ids([_stored_harbor("alpha", "alpha"), _stored_harbor("beta", "beta")], KIND_ADAPTERS, None)


@pytest.mark.parametrize(
    ("tasks", "taskset", "match"),
    [
        ([_inline("task"), _inline("TASK")], None, "^task ids must be unique within an evaluation$"),
        (
            [_inline("task"), _inline("Task")],
            TasksetRef("suite"),
            "Taskset 'suite' expands to more than one task named 'Task'",
        ),
        ([_stored_evaluator("Same"), _stored_evaluator("same", workspace="other")], None, "task ids must be unique"),
        ([_stored_harbor("alpha", "Task"), _stored_harbor("beta", "task")], None, "task ids must be unique"),
    ],
)
def test_validate_loaded_ids_rejects_casefold_collisions(tasks, taskset, match):
    with pytest.raises(ValueError, match=match):
        _validate_loaded_ids(tasks, KIND_ADAPTERS, taskset)
