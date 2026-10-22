# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

from nemo_scaled_evals_plugin.entities import (
    ScaledEvalsBenchmark,
    ScaledEvalsBenchmarkRun,
    ScaledEvalsEvaluation,
    ScaledEvalsTask,
)
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


def _seed(entity_type: Any, row: Mapping[str, Any], renames: Mapping[str, str]) -> None:
    fields = {renames.get(key, key): value for key, value in row.items()}
    fields = {key: value for key, value in fields.items() if key in entity_type.model_fields}
    client = entity_store.entity_client()
    entity = entity_type(workspace=entity_store.private_workspace(), **fields)
    client.create(entity)
    if row.get("created_at") is not None:
        cast(Any, client).stored[(entity_type.__entity_type__, entity.workspace, entity.name)]._created_at = row[
            "created_at"
        ]


_EVALUATION_DEFAULTS = {
    "framework": "harbor",
    "task_id": "task_x",
    "task_revision": 1,
    "runtime": "sandbox_k8s",
    "parallelism": 1,
    "status": "queued",
    "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
}


def seed_evaluation(row: Mapping[str, Any]) -> None:
    """Store an evaluation from a column-named row; columns the entity lacks are ignored."""
    row = {**_EVALUATION_DEFAULTS, "name": row["id"], **{key: value for key, value in row.items() if value is not None}}
    _seed(ScaledEvalsEvaluation, row, {"id": "name", "name": "display_name", "updated_at": "changed_at"})


def seed_benchmark_run(row: Mapping[str, Any]) -> None:
    """Store a benchmark run from a column-named row; columns the entity lacks are ignored."""
    _seed(ScaledEvalsBenchmarkRun, row, {"id": "name", "name": "display_name"})
