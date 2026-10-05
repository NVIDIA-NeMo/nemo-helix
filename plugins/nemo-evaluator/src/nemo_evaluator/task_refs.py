# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load generic selectors, snapshot definitions, and validate execution policy."""

from collections.abc import Mapping, Sequence

from nemo_evaluator.api.schemas import TaskRef, TasksetRef, parse_subentity_ref
from nemo_evaluator.entities import TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.harbor.resolution import map_with_limited_concurrency, task_revision
from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput, ResolvedTask, Target, validate_single_kind
from nemo_evaluator.jobs.kinds.registry import get_adapter
from nemo_evaluator.jobs.kinds.types import LoadedTask, SubmitContext, TaskKindAdapter
from nemo_evaluator.revisions import RevisionNotFoundError, get_revision
from nemo_evaluator.task_identity import UnsupportedTaskKindError, qualified_task_refs
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError


async def _resolve_task_ref(ref: TaskRef, client: EntityClient, *, taskset: TasksetRef | None) -> LoadedTask:
    """Load one stored task's head and the revision named by its ref.

    A missing task or revision raises ``ValueError``. When ``taskset`` is set, the message names that
    taskset, because the missing member was referenced by it.
    """
    try:
        head, revision = await task_revision(ref, client)
        return LoadedTask(stored=(head, revision))
    except NemoEntityNotFoundError as exc:
        if taskset is not None:
            raise ValueError(
                f"Taskset reference '{taskset.root}' names a member that does not resolve: "
                f"Task '{ref.root}' referenced by taskset '{taskset.root}' was not found; "
                "the stored task may have been deleted."
            ) from exc
        raise ValueError(f"Task '{ref.root}' was not found; the stored task may have been deleted.") from exc
    except RevisionNotFoundError as exc:
        if taskset is not None:
            raise ValueError(
                f"Taskset reference '{taskset.root}' names a member that does not resolve: "
                f"Task '{ref.root}' names a revision that no longer resolves: {exc}"
            ) from exc
        raise ValueError(f"Task '{ref.root}' names a revision that no longer resolves: {exc}") from exc


async def load_tasks(
    tasks: TasksetRef | Sequence[AgentEvalTaskInput] | Sequence[TaskRef], ctx: SubmitContext
) -> list[LoadedTask]:
    """Load the tasks from entity store. Inline tasks are not loaded but validated.

    A stored ref loads one task record and the one revision its fragment names. A missing fragment is
    ``latest`` tag that is resolved to the latest revision at the time of the submission.

    - Tasksets: a taskset ref resolves to its revision, then to that revision's member task refs.
    - Inline inputs: become ``LoadedTask.inline`` values. Mixing them with stored refs is rejected.
    - TaskRefs: each ref is rewritten to ``workspace/name#fragment``. A missing fragment becomes ``latest``.
      The load fetches that one ``TaskEntity`` and ``TaskRevisionEntity``.
    - Empty inputs, duplicate ids, and Harbor tasksets with ``files_ref`` are rejected.
    """
    taskset_revision = None
    if isinstance(tasks, TasksetRef):
        if ctx.entity_client is None:
            raise ValueError("A TasksetRef requires a platform connection (entity store)")
        workspace, name, fragment = parse_subentity_ref(tasks.root, ctx.workspace)
        try:
            head = await ctx.entity_client.get(TasksetEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise ValueError(f"Taskset reference '{tasks.root}' not found in workspace '{workspace}'") from exc
        try:
            taskset_revision = await get_revision(ctx.entity_client, TasksetRevisionEntity, head, fragment)
        except RevisionNotFoundError as exc:
            raise ValueError(f"Taskset reference '{tasks.root}' names a revision that does not resolve: {exc}") from exc
        if not taskset_revision.tasks:
            raise ValueError(
                f"Taskset '{tasks.root}' has no member tasks; an agent evaluation needs at least one task."
            )
        refs = qualified_task_refs(taskset_revision.tasks, taskset_revision.workspace)
    # handle inline tasks and TaskRefs as input
    else:
        # handle TaskRefs case
        refs = [task for task in tasks if isinstance(task, TaskRef)]
        if refs and len(refs) != len(tasks):
            raise ValueError("Cannot mix inline tasks and stored task references")
        # handle AgentEvalTaskInput case
        if not refs:
            loaded = []
            for task in tasks:
                if not isinstance(task, AgentEvalTaskInput):
                    raise ValueError("Expected inline tasks or stored task references")
                loaded.append(LoadedTask(inline=task))
            _validate_loaded_ids(loaded, ctx.adapters, None)
            return loaded
        refs = qualified_task_refs(refs, ctx.workspace)
    if ctx.entity_client is None:
        raise ValueError("TaskRef inputs require a platform connection (entity store)")
    client = ctx.entity_client
    # load the tasks from the entity store
    taskset = tasks if isinstance(tasks, TasksetRef) else None
    loaded = await map_with_limited_concurrency(
        lambda ref: _resolve_task_ref(ref, client, taskset=taskset),
        refs,
    )
    _reject_harbor_taskset_fileref(taskset_revision, loaded)
    _validate_loaded_ids(loaded, ctx.adapters, taskset)
    return loaded


def _reject_harbor_taskset_fileref(revision: TasksetRevisionEntity | None, tasks: Sequence[LoadedTask]) -> None:
    """Reject a Harbor taskset that carries ``files_ref``.

    That field is a fileset directory owned by the taskset and shared by its members. Harbor
    materializes only each task's own archive and never mounts this directory, so the shared files
    would be stored and then ignored which might be confusing to a user.
    """
    if revision is not None and revision.files_ref is not None and any(task.kind == "harbor" for task in tasks):
        raise ValueError("Stored Harbor tasksets do not support shared files_ref")


def _validate_loaded_ids(
    tasks: Sequence[LoadedTask], adapters: Mapping[str, TaskKindAdapter], taskset: TasksetRef | None
) -> None:
    """Reject an empty task list and runtime ids that collide, ignoring case.

    The id comes from the kind adapter: the inline task id, the stored task name, or a Harbor
    ``native_task_id``. When ``taskset`` is set, a collision names that taskset in the error.
    """
    if not tasks:
        raise ValueError("Expected at least one task")
    seen: set[str] = set()
    for task in tasks:
        # Kinds may be mixed in the future, so we resolve kind-specific adapter per task here
        # instead of passing a kind-specific adapter as an arg.
        task_id = get_adapter(task.kind, adapters).runtime_id(task)
        if task_id.casefold() in seen:
            prefix = f"Taskset '{taskset.root}' expands to more than one task named '{task_id}'; " if taskset else ""
            raise ValueError(prefix + "task ids must be unique within an evaluation")
        seen.add(task_id.casefold())


async def snapshot_task(loaded_task: LoadedTask, ctx: SubmitContext) -> ResolvedTask:
    """Create a submission-time snapshot with resolved metrics and pinned stored provenance.

    The kind adapter returns the resolved definition, including provenance for stored tasks.
    Both stored and inline tasks copy their metadata into the resulting snapshot.
    """
    adapter = get_adapter(loaded_task.kind, ctx.adapters)
    if loaded_task.inline is not None:
        metadata = loaded_task.inline.metadata
    else:
        assert loaded_task.stored is not None
        _, revision = loaded_task.stored
        metadata = revision.metadata
    return ResolvedTask(
        id=adapter.runtime_id(loaded_task),
        spec=await adapter.resolve(loaded_task, ctx),
        metadata=[item.model_copy(deep=True) for item in metadata],
    )


def groupby_kind(tasks: Sequence[ResolvedTask]) -> list[tuple[str, list[ResolvedTask]]]:
    """Preserve first-seen kind order and input order inside each group."""
    groups: dict[str, list[ResolvedTask]] = {}
    for task in tasks:
        groups.setdefault(task.spec.kind, []).append(task)
    return list(groups.items())


def validate_execution_support(
    tasks: Sequence[ResolvedTask], *, target: Target | None, adapters: Mapping[str, TaskKindAdapter]
) -> None:
    """Reject tasks that the selected execution target cannot run.

    A single evaluation must contain one task kind. After enforcing that constraint, the kind's
    adapter decides whether the target, including a trials-only run represented by ``None``, is
    supported.
    """
    kind = validate_single_kind(tasks)
    if not get_adapter(kind, adapters).accepts_target(target, tasks):
        raise UnsupportedTaskKindError(
            f"{kind} tasks cannot run on a {target.kind if target else 'trials-only'} target"
        )


def validate_scoring(
    tasks: Sequence[ResolvedTask],
    *,
    target: Target | None,
    trials: Sequence[AgentEvalTrial] | None,
    adapters: Mapping[str, TaskKindAdapter],
) -> None:
    """Validate each task kind's scoring requirements for the requested run.

    Tasks are grouped without changing their order, then each kind's adapter validates the target
    and any supplied trials for that group.
    """
    for kind, group in groupby_kind(tasks):
        get_adapter(kind, adapters).validate_scoring(group, target=target, trials=trials)
