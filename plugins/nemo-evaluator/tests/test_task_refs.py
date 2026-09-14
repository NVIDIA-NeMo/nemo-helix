# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for taskset-reference resolution on the agent-eval submit path."""

from __future__ import annotations

from typing import TypeVar, cast

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
from nemo_evaluator.api.task_definitions.harbor import HarborTreeSource
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.harbor.tasks import PinnedHarborTaskList, PinnedHarborTaskset
from nemo_evaluator.jobs.agent_spec import AgentEvalSpec, AgentEvalTaskInput
from nemo_evaluator.revisions import apply_tag, get_revision, head_digest, is_digest, publish_revision
from nemo_evaluator.task_refs import (
    UnsupportedTaskKindError,
    resolve_agent_eval_tasks,
    resolve_taskset_ref,
)
from nemo_platform_plugin.entities import EntityBase, SyncEntityClient
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError
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


async def test_harbor_suite_submission_defers_missing_member_resolution(entity_store):
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    client = await _store(entity_store, _taskset("suite", [f"default/missing#{_ABSENT_MEMBER_DIGEST}"]))
    source = await resolve_agent_eval_tasks(
        TasksetRef("default/suite"), workspace="default", entity_client=client, target=HarborRunnerTarget()
    )
    assert isinstance(source, PinnedHarborTaskset)
    assert source.kind == "harbor-taskset"
    assert source.taskset_ref.root.startswith("default/suite#")
    assert "members" not in source.model_dump()


async def test_harbor_direct_list_pins_and_worker_resolves_selected_revision(entity_store):
    from nemo_evaluator.harbor.resolution import resolve_harbor_source
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            tree=HarborTreeSource(
                root_ref="default/files#v1/task/files",
                manifest_ref="default/files#v1/task/files.manifest.json",
                tree_digest="a" * 64,
                manifest_digest="c" * 64,
            ),
        ),
    )
    await _store(entity_store, task)
    source = await resolve_agent_eval_tasks(
        [TaskRef("checkout")], workspace="default", entity_client=entity_store, target=HarborRunnerTarget()
    )
    assert isinstance(source, PinnedHarborTaskList)
    assert isinstance(task.spec, HarborTaskDefinition)
    task.spec.tree.root_ref = "default/files#v2/task/files"
    await entity_store.update(task)
    await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    members = await resolve_harbor_source(source, entity_client=entity_store)
    assert members[0].definition.tree.root_ref == "default/files#v1/task/files"
    assert members[0].entity_name == "default/checkout"


async def test_direct_alias_duplicates_fail_without_entity_client():
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    with pytest.raises(ValueError, match="Duplicate task identity"):
        await resolve_agent_eval_tasks(
            [TaskRef("checkout"), TaskRef("default/checkout#blessed")],
            workspace="default",
            entity_client=None,
            target=HarborRunnerTarget(),
        )


async def test_harbor_worker_keeps_compiled_suite_pin_after_tag_moves(entity_store, monkeypatch):
    from nemo_evaluator.harbor.resolution import resolve_harbor_source
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            tree=HarborTreeSource(
                root_ref="default/files#v1/task/files",
                manifest_ref="default/files#v1/task/files.manifest.json",
                tree_digest="a" * 64,
                manifest_digest="c" * 64,
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
    assert isinstance(before, AgentEvalSpec) and isinstance(before.tasks, PinnedHarborTaskset)
    stored_task = await entity_store.get(TaskEntity, name="checkout", workspace="default")
    assert isinstance(stored_task.spec, HarborTaskDefinition)
    stored_task.spec.tree.root_ref = "default/files#v2/task/files"
    await entity_store.update(stored_task)
    await publish_revision(entity_store, entity_store, stored_task, TaskRevisionEntity)
    await _republish_with(entity_store, suite, ["default/checkout"])
    await apply_tag(entity_store, entity_store, TasksetRevisionEntity, suite, "blessed", "latest")

    selected = []
    import nemo_evaluator.harbor.resolution as resolution

    original = resolution.get_revision

    async def tracked(client, revision_type, head, fragment):
        selected.append(fragment)
        return await original(client, revision_type, head, fragment)

    monkeypatch.setattr(resolution, "get_revision", tracked)
    members = await resolve_harbor_source(before.tasks, entity_client=entity_store)
    assert selected == [before.tasks.taskset_ref.root.split("#")[1], head_digest(task)]
    assert members[0].definition.tree.root_ref == "default/files#v1/task/files"
    after = await AgentEvalJob.to_spec(
        request, workspace="default", entity_client=entity_store, async_sdk=None, is_local=False
    )
    assert isinstance(after, AgentEvalSpec) and isinstance(after.tasks, PinnedHarborTaskset)
    assert after.tasks != before.tasks


async def test_large_harbor_suite_post_does_not_fetch_member_definitions(entity_store, monkeypatch):
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    suite = _taskset("large", [f"default/task-{i}#{_ABSENT_MEMBER_DIGEST}" for i in range(2000)])
    await _store(entity_store, suite)
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
    assert TaskEntity not in fetched and TaskRevisionEntity not in fetched
    assert isinstance(result, AgentEvalSpec)
    assert isinstance(result.tasks, PinnedHarborTaskset)
    assert result.tasks.model_dump() == {"kind": "harbor-taskset", "taskset_ref": f"default/large#{head_digest(suite)}"}
    assert len(result.model_dump_json()) < 2000


async def test_worker_rejects_incompatible_suite_member_and_deleted_revision(entity_store):
    from nemo_evaluator.harbor.resolution import resolve_harbor_source
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    task = _task("wrong-kind")
    await _store(entity_store, task, _taskset("suite", ["default/wrong-kind"]))
    source = await resolve_agent_eval_tasks(
        TasksetRef("suite"), workspace="default", entity_client=entity_store, target=HarborRunnerTarget()
    )
    assert isinstance(source, PinnedHarborTaskset)
    with pytest.raises(ValueError, match="stored kind 'evaluator'.*target 'harbor'"):
        await resolve_harbor_source(source, entity_client=entity_store)
    await entity_store.delete(TaskEntity, task.name, workspace=task.workspace)
    with pytest.raises(NemoEntityNotFoundError):
        await resolve_harbor_source(source, entity_client=entity_store)


async def test_sync_worker_resolves_same_exact_pins(entity_store):
    from nemo_evaluator.harbor.resolution import resolve_harbor_source, resolve_harbor_source_sync
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget
    from nemo_evaluator_sdk.execution.metric_execution import run_sync

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            tree=HarborTreeSource(
                root_ref="default/files#v1/task/files",
                manifest_ref="default/files#v1/task/files.manifest.json",
                tree_digest="a" * 64,
                manifest_digest="c" * 64,
            ),
        ),
    )
    await _store(entity_store, task, _taskset("suite", ["default/checkout"]))
    source = await resolve_agent_eval_tasks(
        TasksetRef("suite"), workspace="default", entity_client=entity_store, target=HarborRunnerTarget()
    )

    class SyncStore:
        def get(self, *args, **kwargs):
            return run_sync(lambda: entity_store.get(*args, **kwargs))

        def list(self, *args, **kwargs):
            return run_sync(lambda: entity_store.list(*args, **kwargs))

    assert isinstance(source, PinnedHarborTaskset)
    assert resolve_harbor_source_sync(
        source, entity_client=cast(SyncEntityClient, SyncStore())
    ) == await resolve_harbor_source(source, entity_client=entity_store)


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
    assert isinstance(task.spec, EvaluatorTaskDefinition)
    task.spec.reference = {"expected": "Paris", "held_out_tests": ["test_capital.py"]}
    client = await _store(entity_store, task, _taskset("geo", ["default/fix-bug"]))

    tasks = await resolve_taskset_ref(TasksetRef("default/geo"), workspace="default", entity_client=client)

    assert tasks[0].reference == {"expected": "Paris", "held_out_tests": ["test_capital.py"]}


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
    assert isinstance(result, list)
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
    assert isinstance(task.spec, EvaluatorTaskDefinition)
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
    assert isinstance(tagged.spec, EvaluatorTaskDefinition)
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
            kind="harbor",
            tree=HarborTreeSource(
                root_ref="default/harbor#packages/o-n/abc/files",
                manifest_ref="default/harbor#packages/o-n/abc/files.manifest.json",
                tree_digest="a" * 64,
                manifest_digest="c" * 64,
            ),
        ),
    )
    client = await _store(entity_store, harbor_task)
    await _create_published(client, _taskset("mixed", [f"default/fix-test#{head_digest(harbor_task)}"]))

    with pytest.raises(UnsupportedTaskKindError, match="harbor"):
        await resolve_taskset_ref(TasksetRef("default/mixed"), workspace="default", entity_client=client)


async def test_incompatible_target_error_names_requested_target(entity_store):
    from nemo_evaluator.jobs.agent_spec import FabricRunnerTarget

    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            tree=HarborTreeSource(
                root_ref="default/files#task/files",
                manifest_ref="default/files#task/files.manifest.json",
                tree_digest="a" * 64,
                manifest_digest="c" * 64,
            ),
        ),
    )
    await _store(entity_store, task)
    with pytest.raises(UnsupportedTaskKindError, match="target 'fabric'"):
        await resolve_agent_eval_tasks(
            [TaskRef("checkout")], workspace="default", entity_client=entity_store, target=FabricRunnerTarget(config={})
        )


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
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob, _to_runtime_task
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec
    from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
    from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
    from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
    from nemo_evaluator_sdk.metrics.runner_rewards import GymRewardMetric
    from nemo_platform_plugin.jobs.spec import PlatformJobSpec

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
        request, workspace="default", entity_client=entity_store, async_sdk=None, is_local=False
    )
    assert isinstance(spec, AgentEvalSpec) and isinstance(spec.tasks, list)
    assert [task.id for task in spec.tasks] == ["third", "second", "first"]
    assert spec.tasks[-1].intent == "Do first."
    assert resolved_metrics == ["default/m"] * 3
    for task in spec.tasks:
        runtime = _to_runtime_task(task)
        assert runtime.inputs["instruction"] == task.id
        assert runtime.inputs["gym_row"] == {}
        assert runtime.metadata["gym_row_extras"] == {}
        assert runtime.reference == {"answer": "DONE"}
        assert runtime.views == task.views
        assert len(runtime.metrics) == 1
    compiled = PlatformJobSpec.model_validate(
        await AgentEvalJob.compile(workspace="default", spec=spec, entity_client=None, job_name=None, async_sdk=None)
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
        await resolve_agent_eval_tasks(
            [TaskRef("default/same"), TaskRef("other/same")], workspace="default", entity_client=entity_store
        )


@pytest.mark.parametrize("kind", ["model", "agent", "fabric", "gym", "offline"])
@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong-kind"])
async def test_direct_reference_failures_for_every_evaluator_target(kind, failure, entity_store):
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
                tree=HarborTreeSource(
                    root_ref="default/files#task/files",
                    manifest_ref="default/files#task/manifest.json",
                    tree_digest="a" * 64,
                    manifest_digest="b" * 64,
                ),
            ),
        )
        await _store(entity_store, task)
        refs = [TaskRef("harbor")]
        match = "stored kind 'harbor'"
    with pytest.raises(NemoEntityNotFoundError if failure == "missing" else ValueError, match=match):
        await resolve_agent_eval_tasks(
            refs, workspace="default", entity_client=entity_store, target=_direct_target(kind)
        )
