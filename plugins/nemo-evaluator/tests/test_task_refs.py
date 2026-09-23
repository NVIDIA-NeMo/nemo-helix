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
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput
from nemo_evaluator.revisions import apply_tag, get_revision, head_digest, is_digest, publish_revision
from nemo_evaluator.task_refs import (
    UnsupportedTaskKindError,
    resolve_agent_eval_tasks,
    resolve_taskset_ref,
)
from nemo_helix_plugin.entities import EntityBase
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from pydantic import ValidationError

_EntityT = TypeVar("_EntityT", bound=EntityBase)


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

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)

    assert [t.id for t in tasks] == ["capital-of-france", "capital-of-japan"]
    assert all(isinstance(t, AgentEvalTaskInput) for t in tasks)
    # A stored task's refs pass through untouched (resolved to inline later in the metric pass).
    assert tasks[0].metrics == [MetricRef("default/m")]
    assert tasks[0].intent == "Do capital-of-france."
    assert tasks[0].inputs.instruction == "instruction for capital-of-france"
    # A task stored without ground truth expands to an empty reference, not a missing one.
    assert tasks[0].reference == {}


async def test_grader_only_reference_survives_taskset_expansion(entity_store) -> None:
    """Held-out ground truth must not be the privilege of inline submissions.

    Expansion projects a stored task onto the inline DTO field by field, so a field added to the
    stored spec and forgotten here silently becomes empty at run time — the agent is then graded
    against nothing, and the run still reports a score. That is the failure this guards.
    """
    task = _task("fix-bug")
    task.spec.reference = {"expected": "Paris", "held_out_tests": ["test_capital.py"]}
    client = await _store(entity_store, task, _taskset("geo", ["default/fix-bug"]))

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)

    assert tasks[0].reference == {"expected": "Paris", "held_out_tests": ["test_capital.py"]}


async def test_expansion_returns_the_pinned_reference_not_the_current_one(entity_store) -> None:
    """``reference`` is digest-covered, so republishing it cuts a revision the old pin excludes.

    A pin that honoured new ground truth would silently re-grade a "reproducible" dataset.
    """
    task = _task("fix-bug")
    task.spec.reference = {"expected": "Paris"}
    client = await _store(entity_store, task)
    pinned_digest = head_digest(task)
    await _create_published(client, _taskset("geo", [f"default/fix-bug#{pinned_digest}"]))

    task.spec.reference = {"expected": "Lyon"}
    await client.update(task)
    await publish_revision(client, client, task, TaskRevisionEntity)

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)

    assert tasks[0].reference == {"expected": "Paris"}, "expansion must return the pinned ground truth"


async def test_bare_member_ref_resolves_against_taskset_workspace(entity_store) -> None:
    client = await _store(entity_store, _task("t1", workspace="team"), _taskset("ts", ["t1"], workspace="team"))

    tasks = await resolve_taskset_ref(TasksetRef("team/ts"), workspace="default", entity_client=client)

    assert [t.id for t in tasks] == ["t1"]


async def test_unknown_taskset_raises_clear_error(entity_store) -> None:
    with pytest.raises(ValueError, match="Taskset reference 'default/missing' not found"):
        await resolve_taskset_ref(
            TasksetRef("default/missing"), workspace="default", entity_client=await _store(entity_store)
        )


async def test_missing_member_task_raises_clear_error(entity_store) -> None:
    client = await _store(entity_store, _taskset("geo", ["default/gone"]))
    with pytest.raises(ValueError, match=r"Task 'default/gone#\w+' referenced by taskset 'default/geo'"):
        await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)


async def test_empty_taskset_raises_clear_error(entity_store) -> None:
    client = await _store(entity_store, _taskset("empty", []))
    with pytest.raises(ValueError, match="has no member tasks"):
        await resolve_taskset_ref(TasksetRef("default/empty"), workspace="default", entity_client=client)


async def test_duplicate_expanded_task_ids_rejected(entity_store) -> None:
    # Two members from different workspaces share the name 'dup' -> ambiguous task id.
    client = await _store(
        entity_store,
        _task("dup", workspace="a"),
        _task("dup", workspace="b"),
        _taskset("geo", ["a/dup", "b/dup"]),
    )
    with pytest.raises(ValueError, match="more than one task named 'dup'"):
        await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)


async def test_taskset_ref_requires_entity_client(entity_store) -> None:
    with pytest.raises(ValueError, match="requires a platform connection"):
        await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=None)


async def test_resolve_agent_eval_tasks_passes_inline_list_through(entity_store) -> None:
    inline = [AgentEvalTaskInput(id="t", intent="x", metrics=[])]
    result = await resolve_agent_eval_tasks(inline, workspace="default", entity_client=None)
    assert result is inline


async def test_resolve_agent_eval_tasks_expands_a_taskset_ref(entity_store) -> None:
    client = await _store(entity_store, _task("only"), _taskset("geo", ["default/only"]))
    result = await resolve_agent_eval_tasks(TasksetRef("default/geo"), workspace="default", entity_client=client)
    assert [t.id for t in result] == ["only"]


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
    task.spec.intent = "Something else entirely."
    await client.update(task)
    await publish_revision(client, client, task, TaskRevisionEntity)

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)

    assert tasks[0].intent == "Do capital-of-france.", "expansion must return the pinned content"


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
    tagged.spec.intent = "Something else entirely."
    await client.update(tagged)
    await publish_revision(client, client, tagged, TaskRevisionEntity)

    await _create_published(client, _taskset("geo", ["default/capital-of-france#blessed"]))

    stored_taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    assert stored_taskset.tasks[0].root == f"default/capital-of-france#{blessed_digest}", (
        "membership must pin the revision the tag named, not the head"
    )

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)
    assert tasks[0].intent == "Do capital-of-france."


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

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)

    assert [t.id for t in tasks] == ["a", "b"], "a bare ref must follow the taskset forward"


async def test_digest_pinned_taskset_ref_expands_the_pinned_membership(entity_store) -> None:
    """The gap this closes: member *content* was already pinned, but membership was not.

    Replacing a taskset changed what a re-submitted spec evaluated, because the ref could only ever
    name the head. A digest-pinned ref holds the whole grouping still.
    """
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    pinned = head_digest(taskset)

    await _republish_with(client, taskset, ["default/a", "default/b"])

    tasks = await resolve_taskset_ref(TasksetRef(f"default/geo#{pinned}"), workspace="default", entity_client=client)

    assert [t.id for t in tasks] == ["a"], "a pinned ref must ignore members added after it was taken"


async def test_tag_pinned_taskset_ref_resolves_through_the_tag(entity_store) -> None:
    """Tags are resolution inputs, so a tag-pinned ref resolves at expansion time, not at write."""
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    await apply_tag(client, client, TasksetRevisionEntity, taskset, "blessed", "latest")

    await _republish_with(client, taskset, ["default/a", "default/b"])

    tasks = await resolve_taskset_ref(TasksetRef("default/geo#blessed"), workspace="default", entity_client=client)

    assert [t.id for t in tasks] == ["a"], "'blessed' still names revision 1"


async def test_pinned_taskset_ref_survives_a_replace(entity_store) -> None:
    """The reproducibility property end to end: same ref, same membership, across a replace."""
    client = await _store(entity_store, _task("a"), _task("b"), _taskset("geo", ["default/a"]))
    taskset = await client.get(TasksetEntity, name="geo", workspace="default")
    ref = TasksetRef(f"default/geo#{head_digest(taskset)}")

    before = await resolve_taskset_ref(ref, workspace="default", entity_client=client)
    await _republish_with(client, taskset, ["default/b"])
    after = await resolve_taskset_ref(ref, workspace="default", entity_client=client)

    assert [t.id for t in before] == [t.id for t in after] == ["a"]
    assert [t.intent for t in before] == [t.intent for t in after]


async def test_unresolvable_taskset_fragment_raises_a_clear_error(entity_store) -> None:
    """A pin that cannot be honoured stops the evaluation rather than falling back to the head."""
    client = await _store(entity_store, _task("a"), _taskset("geo", ["default/a"]))

    with pytest.raises(ValueError, match="names a revision that does not resolve"):
        await resolve_taskset_ref(TasksetRef(f"default/geo#{'c' * 64}"), workspace="default", entity_client=client)

    with pytest.raises(ValueError, match="names a revision that does not resolve"):
        await resolve_taskset_ref(TasksetRef("default/geo#nonesuch"), workspace="default", entity_client=client)


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
        await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)


async def test_expansion_rejects_a_task_whose_runner_the_target_cannot_run(entity_store) -> None:
    """A Harbor task's content is a directory of files, not fields. Projecting it onto an inline
    agent-eval task would silently produce a task with no intent and no metrics — an evaluation that
    runs and scores nothing. Refused instead, before the run starts."""
    harbor_task = TaskEntity(
        name="fix-test",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor", archive_ref="default/harbor#packages/o-n/abc/dist.tar.gz", archive_digest="a" * 64
        ),
    )
    client = await _store(entity_store, harbor_task)
    await _create_published(client, _taskset("mixed", [f"default/fix-test#{head_digest(harbor_task)}"]))

    with pytest.raises(UnsupportedTaskKindError, match="harbor"):
        await resolve_taskset_ref(TasksetRef("default/mixed"), workspace="default", entity_client=client)
