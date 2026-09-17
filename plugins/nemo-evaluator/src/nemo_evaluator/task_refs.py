# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""References to persisted tasksets and their resolution into inline tasks.

An agent-eval submission carries ``tasks`` as either an inline list of
:class:`~nemo_evaluator.jobs.agent_spec.AgentEvalTaskInput` or a
:class:`~nemo_evaluator.api.schemas.TasksetRef` pointing at a stored taskset. During spec resolution
(``AgentEvalJob.to_spec``) a taskset reference is loaded from storage and its member tasks are
expanded into the same inline task DTOs, so the rest of the pipeline (metric-ref resolution, the
canonical :class:`~nemo_evaluator.jobs.agent_spec.AgentEvalSpec`) only ever sees inline tasks.

This mirrors :mod:`nemo_evaluator.metric_refs`: references are loaded here, next to the entity types,
so the job's ``to_spec`` stays a thin orchestration over ref-resolution helpers.
"""

from __future__ import annotations

from typing import cast

from nemo_evaluator.api.schemas import EvaluatorTaskDefinition, TasksetRef, parse_subentity_ref
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput
from nemo_evaluator.revisions import RevisionNotFoundError, get_revision
from nemo_platform_plugin.entities import EntityClientProtocol
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError


class UnsupportedTaskKindError(ValueError):
    """A taskset member's runner kind cannot be executed by the requested target.

    A taskset may group tasks of different kinds — that is the point of managing every evaluation
    unit in one place — but a single run has one target, so expansion is where a mismatch surfaces.

    Raised during ``to_spec``, which the job submit path wraps: any exception there becomes a 422
    carrying this message (``_apply_transformer`` in ``nemo_platform_plugin.jobs.api_factory``). It
    subclasses ``ValueError`` for local callers that catch it deliberately, not to obtain that
    mapping — the mapping is a catch-all and would apply to any exception type.
    """


def _entity_to_task_input(entity: TaskEntity, revision: TaskRevisionEntity) -> AgentEvalTaskInput:
    """Project a stored task's *published revision* onto the submitter-facing inline task DTO.

    Identity (``id``) comes from the head record — it is the same task — while every content field
    comes from the revision the taskset pinned. That split is what makes a taskset-driven evaluation
    reproducible: re-running it expands to the same content even if the member task has published
    since.

    A stored task holds metric *references* (inline metrics were normalized to derived stored
    metrics on create); those resolve to inline bundles in the shared metric-ref pass that runs
    after expansion. The grader-only ``reference`` comes from the revision too, so a taskset-driven
    run grades against the ground truth that revision pinned — held-out data is not the privilege of
    inline submissions.
    """
    spec = revision.spec
    if not isinstance(spec, EvaluatorTaskDefinition):
        # A Harbor task's content is a *directory of files*, not fields — the runner needs the
        # archive materialized on disk, which this pure projection cannot do. Rejecting here means a
        # mismatched taskset fails before the run rather than silently evaluating an empty task.
        #
        # Deliberately does *not* suggest picking a different target: no target can run a stored
        # task of this kind yet, so pointing at one would send the reader in circles. Storing the
        # kind landed ahead of the execution bridge (AALGO-481).
        raise UnsupportedTaskKindError(
            f"Task '{entity.workspace}/{entity.name}' is a {spec.kind!r} task. Running a stored "
            f"{spec.kind!r} task is not supported yet — no target can execute one, so this taskset "
            "cannot be evaluated until that lands. Remove the member, or submit an "
            "'evaluator'-kind taskset."
        )
    return AgentEvalTaskInput(
        id=entity.name,
        intent=spec.intent,
        inputs=spec.inputs,
        reference=dict(spec.reference),
        metrics=list(spec.metrics),
        views=spec.views,
        metadata=revision.metadata,
    )


#: Expanding a taskset reads four entity types through one client — the taskset head, its pinned
#: revision, each member task's head, and the pinned revision of each member. Python has no
#: intersection types, so the parameter is annotated at one of them and the rest are taken as typed
#: views of the same object; the concrete client's methods are generic over the entity type and
#: satisfy all four.
TasksetStoreProtocol = EntityClientProtocol[TasksetEntity]


async def resolve_taskset_ref(
    ref: TasksetRef,
    *,
    workspace: str,
    entity_client: TasksetStoreProtocol | None,
) -> list[AgentEvalTaskInput]:
    """Load a stored taskset and expand its members into inline task DTOs.

    Loading needs only the entity store (metrics stay as refs, resolved downstream), so unlike
    metric-ref resolution this does not require an async SDK / file I/O.

    The ref may pin a taskset revision (``suite#<tag-or-digest>``); an absent fragment means
    ``latest``. Both paths go through :func:`get_revision` rather than reading the head's own
    ``tasks``, because a head and its ``latest`` revision are guaranteed to agree and resolving one
    way for pinned refs and another way for bare ones would make the two drift apart on the next
    bug. It also buys content verification for the bare case for free.
    """
    if entity_client is None:
        raise ValueError(
            "A TasksetRef requires a platform connection (entity store) to resolve; pass an inline task list instead."
        )
    task_store = cast(EntityClientProtocol[TaskEntity], entity_client)
    revision_store = cast(EntityClientProtocol[TaskRevisionEntity], entity_client)
    taskset_revision_store = cast(EntityClientProtocol[TasksetRevisionEntity], entity_client)

    ref_workspace, name, taskset_fragment = parse_subentity_ref(ref.root, workspace)
    try:
        taskset = await entity_client.get(TasksetEntity, name=name, workspace=ref_workspace)
    except NemoEntityNotFoundError as exc:
        raise ValueError(
            f"Taskset reference '{ref.root}' not found. "
            f"Ensure a stored taskset named '{name}' exists in workspace '{ref_workspace}', "
            "or pass an inline task list instead."
        ) from exc

    # Expand the taskset revision the ref names. Members are digest-pinned inside a revision, so a
    # member republishing on its own never moves this. A bare ref still follows the taskset's own
    # revisions, and a ``replace`` re-resolves members on write — so it can change both which members
    # are named and what they resolve to. Only a pinned ref holds both steady.
    try:
        taskset_revision = await get_revision(taskset_revision_store, TasksetRevisionEntity, taskset, taskset_fragment)
    except RevisionNotFoundError as exc:
        raise ValueError(f"Taskset reference '{ref.root}' names a revision that does not resolve: {exc}") from exc

    if not taskset_revision.tasks:
        raise ValueError(f"Taskset '{ref.root}' has no member tasks; an agent evaluation needs at least one task.")

    tasks: list[AgentEvalTaskInput] = []
    seen_ids: set[str] = set()
    for task_ref in taskset_revision.tasks:
        task_workspace, task_name, fragment = parse_subentity_ref(task_ref.root, ref_workspace)
        try:
            entity = await task_store.get(TaskEntity, name=task_name, workspace=task_workspace)
        except NemoEntityNotFoundError as exc:
            raise ValueError(
                f"Task '{task_ref.root}' referenced by taskset '{ref.root}' was not found; "
                "the stored task may have been deleted after the taskset was created."
            ) from exc
        # Expand the *pinned* revision, not the task's current content. A published taskset names
        # exact revisions; resolving to whatever is current would silently defeat the pinning and
        # make an evaluation irreproducible the moment a member republished.
        try:
            revision = await get_revision(revision_store, TaskRevisionEntity, entity, fragment)
        except RevisionNotFoundError as exc:
            raise ValueError(
                f"Task '{task_ref.root}' referenced by taskset '{ref.root}' names a revision that no "
                f"longer resolves: {exc}"
            ) from exc
        # Agent-eval task ids must be unique within a run. Member refs are unique per (workspace,
        # name), but refs from different workspaces can share a name — surface that as a clear error
        # rather than letting the SDK evaluator reject duplicate ids deeper in the run.
        if entity.name in seen_ids:
            raise ValueError(
                f"Taskset '{ref.root}' expands to more than one task named '{entity.name}'; "
                "task ids must be unique within an evaluation."
            )
        seen_ids.add(entity.name)
        tasks.append(_entity_to_task_input(entity, revision))
    return tasks


async def resolve_agent_eval_tasks(
    tasks: TasksetRef | list[AgentEvalTaskInput],
    *,
    workspace: str,
    entity_client: TasksetStoreProtocol | None,
) -> list[AgentEvalTaskInput]:
    """Normalize an agent-eval ``tasks`` field to an inline task list.

    An inline list passes through unchanged; a :class:`TasksetRef` is loaded and expanded.
    """
    if isinstance(tasks, TasksetRef):
        return await resolve_taskset_ref(tasks, workspace=workspace, entity_client=entity_client)
    return tasks
