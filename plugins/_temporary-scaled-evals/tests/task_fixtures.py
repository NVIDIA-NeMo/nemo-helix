# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nemo_scaled_evals_plugin.entities import ScaledEvalsBenchmark, ScaledEvalsTask
from scaled_evals.api import entity_store
from scaled_evals.api.benchmark_store import BenchmarkStore, TaskRef
from scaled_evals.api.task_store import TaskStore


def seed_task_revision(
    task_id: str,
    revision: int = 1,
    *,
    slug: str | None = None,
    name: str | None = None,
    owner_id: str | None = None,
    **fields: Any,
) -> None:
    """Store `task_id` with revisions up to `revision`, setting `fields` on that one.

    The revision has no task-pack object unless `tarball_object_key` is given.
    """
    fields.setdefault("tarball_object_key", None)
    store = TaskStore()
    if store.get(task_id) is None:
        store.create_with_initial_revision(
            task_id,
            name=name or task_id,
            slug=slug or task_id.replace("_", "-"),
            description=None,
            visibility="private",
            object_key="k",
            owner_id=owner_id,
        )
    while stored_task(task_id, detail=True)["revision"] < revision:
        store.create_next_revision(task_id)
    client = entity_store.entity_client()
    entity = client.get(ScaledEvalsTask, task_id, workspace=entity_store.private_workspace())
    target = next(rev for rev in entity.revisions if rev.revision == revision)
    for field, value in fields.items():
        setattr(target, field, value)
    client.update(entity)


def stored_task(task_id: str, *, detail: bool = False) -> dict[str, Any]:
    task = TaskStore().get_detail(task_id) if detail else TaskStore().get(task_id)
    assert task is not None, task_id
    return task


def stored_revision(task_id: str, revision: int, **kwargs: Any) -> dict[str, Any]:
    view = TaskStore().revision(task_id, revision, **kwargs)
    assert view is not None, (task_id, revision)
    return view


def seed_benchmark(
    benchmark_id: str = "bm_x",
    tasks: tuple[tuple[str, int | None], ...] = (),
    *,
    name: str = "Suite",
    slug: str | None = None,
    **fields: Any,
) -> None:
    """Store a benchmark whose revision 1 has `(task_id, task_revision)` members, setting `fields` on it."""
    BenchmarkStore().create_with_initial_revision(
        benchmark_id,
        name=name,
        slug=slug or benchmark_id.replace("_", "-"),
        description=None,
        visibility="private",
        tasks=[TaskRef(task_id, task_revision) for task_id, task_revision in tasks],
    )
    if fields:
        client = entity_store.entity_client()
        entity = client.get(ScaledEvalsBenchmark, benchmark_id, workspace=entity_store.private_workspace())
        for field, value in fields.items():
            setattr(entity, field, value)
        client.update(entity)
