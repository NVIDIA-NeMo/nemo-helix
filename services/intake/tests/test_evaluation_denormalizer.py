# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the debounced evaluation name-facet refresher."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import cast

import pytest
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.intake.entities.experiments import Experiment
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.repository.evaluation_rollup import EvaluationRollup, EvaluationRollupRepository


def _sample_rollup(evaluation_name: str) -> EvaluationRollup:
    return EvaluationRollup(
        evaluation_name=evaluation_name,
        model_names=["provider/model-a", "provider/model-b"],
        agent_names=["agent-x"],
        agent_versions=["1.0", "1.1"],
    )


def _bounded_rollup(evaluation_name: str) -> EvaluationRollup:
    """The rollup a recreated evaluation should see: only runs published after its own creation."""
    return EvaluationRollup(
        evaluation_name=evaluation_name,
        model_names=["provider/model-new"],
        agent_names=["agent-y"],
        agent_versions=["2.0"],
    )


class _FakeRollupRepo:
    def __init__(self, *, error: Exception | None = None, bounded: bool = False) -> None:
        self.calls: list[tuple[str, list[str], dict[str, datetime | None]]] = []
        self._error = error
        self._bounded = bounded

    async def get_rollups(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
        created_after_by_name: dict[str, datetime | None],
    ) -> dict[str, EvaluationRollup]:
        self.calls.append((workspace, list(evaluation_names), dict(created_after_by_name)))
        if self._error is not None:
            raise self._error
        factory = _bounded_rollup if self._bounded else _sample_rollup
        return {evaluation_name: factory(evaluation_name) for evaluation_name in evaluation_names}


class _FakeEntityClient:
    def __init__(
        self,
        *,
        get_error: Exception | None = None,
        update_error: Exception | None = None,
        existing: Experiment | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.updated: list[Experiment] = []
        self.get_calls: list[tuple[str, str]] = []
        self._get_error = get_error
        self._update_error = update_error
        self._existing = existing
        self._created_at = created_at

    async def get(self, entity_type: type[Experiment], *, name: str, workspace: str) -> Experiment:
        self.get_calls.append((name, workspace))
        if self._get_error is not None:
            raise self._get_error
        if self._existing is not None:
            return self._existing
        entity = Experiment(name=name, workspace=workspace, experiment_ids=["grp"], dataset_name="ds")
        entity._created_at = self._created_at
        return entity

    async def update(self, entity: Experiment) -> Experiment:
        if self._update_error is not None:
            raise self._update_error
        self.updated.append(entity)
        return entity


def _refresher(repo: _FakeRollupRepo, entity_client: _FakeEntityClient) -> EvaluationDenormalizer:
    return EvaluationDenormalizer(
        rollup_repository=cast(EvaluationRollupRepository, repo),
        entity_client=cast(EntityClient, entity_client),
        interval_seconds=999,
    )


@pytest.mark.asyncio
async def test_flush_writes_denormalized_facets() -> None:
    repo = _FakeRollupRepo()
    entity_client = _FakeEntityClient()
    refresher = _refresher(repo, entity_client)

    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    refresher.mark_dirty(workspace="default", evaluation_name="eval-b")
    await refresher.flush()

    # One batched rollup query for the workspace, covering both dirty evaluations and
    # carrying each live entity's created_at as the rollup bound.
    assert len(repo.calls) == 1
    workspace, ids, created_after_by_name = repo.calls[0]
    assert workspace == "default"
    assert set(ids) == {"eval-a", "eval-b"}
    assert created_after_by_name == {"eval-a": None, "eval-b": None}

    # Each evaluation got its name facets written, and the dirty set is drained.
    assert {entity.name for entity in entity_client.updated} == {"eval-a", "eval-b"}
    written = entity_client.updated[0]
    assert written.agent_names == ["agent-x"]
    assert written.agent_versions == ["1.0", "1.1"]
    assert written.model_names == ["provider/model-a", "provider/model-b"]
    assert refresher.pending() == set()


@pytest.mark.asyncio
async def test_unchanged_facets_skip_write() -> None:
    # The stored entity already carries exactly the rollup's names -> no update, no wasted write.
    existing = Experiment(
        name="eval-a",
        workspace="default",
        experiment_ids=["grp"],
        dataset_name="ds",
        agent_names=["agent-x"],
        agent_versions=["1.0", "1.1"],
        model_names=["provider/model-a", "provider/model-b"],
    )
    entity_client = _FakeEntityClient(existing=existing)
    refresher = _refresher(_FakeRollupRepo(), entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()
    assert entity_client.updated == []
    assert refresher.pending() == set()


def test_mark_dirty_dedupes() -> None:
    refresher = _refresher(_FakeRollupRepo(), _FakeEntityClient())
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    assert refresher.pending() == {("default", "eval-a")}


@pytest.mark.asyncio
async def test_flush_noop_when_clean() -> None:
    repo = _FakeRollupRepo()
    await _refresher(repo, _FakeEntityClient()).flush()
    assert repo.calls == []


@pytest.mark.asyncio
async def test_flush_bounds_rollups_by_each_entity_created_at() -> None:
    created_at = datetime(2026, 3, 1, tzinfo=timezone.utc)
    repo = _FakeRollupRepo()
    entity_client = _FakeEntityClient(created_at=created_at)
    refresher = _refresher(repo, entity_client)

    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()

    # The rollup query is bounded to the live entity's own created_at, so a recreated
    # evaluation never sees the deleted namesake's runs.
    assert repo.calls == [("default", ["eval-a"], {"eval-a": created_at})]


@pytest.mark.asyncio
async def test_recreated_entity_facets_come_from_the_bounded_rollup() -> None:
    # The fake returns the recreated evaluation's own runs only when the call is bounded;
    # the facets written onto the entity must come from that bounded rollup.
    created_at = datetime(2026, 3, 1, tzinfo=timezone.utc)
    repo = _FakeRollupRepo(bounded=True)
    entity_client = _FakeEntityClient(created_at=created_at)
    refresher = _refresher(repo, entity_client)

    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()

    assert repo.calls[0][2] == {"eval-a": created_at}
    written = entity_client.updated[0]
    assert written.agent_names == ["agent-y"]
    assert written.agent_versions == ["2.0"]
    assert written.model_names == ["provider/model-new"]


@pytest.mark.asyncio
async def test_missing_evaluation_is_skipped() -> None:
    entity_client = _FakeEntityClient(get_error=EntityNotFoundError("gone"))
    repo = _FakeRollupRepo()
    refresher = _refresher(repo, entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()
    # The missing entity is skipped before the rollup query, so no unbounded rollup runs
    # and no facets are written.
    assert repo.calls == []
    assert entity_client.updated == []
    assert refresher.pending() == set()  # not re-queued; the evaluation no longer exists


@pytest.mark.asyncio
async def test_deleted_evaluation_is_skipped() -> None:
    deleted = Experiment(name="eval-a", workspace="default", experiment_ids=["grp"], dataset_name="ds")
    deleted.is_deleted = True
    repo = _FakeRollupRepo()
    entity_client = _FakeEntityClient(existing=deleted)
    refresher = _refresher(repo, entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()
    # A deleted entity writes no facets; its name may already belong to a new evaluation.
    assert repo.calls == []
    assert entity_client.updated == []
    assert refresher.pending() == set()


@pytest.mark.asyncio
async def test_update_conflict_requeues() -> None:
    entity_client = _FakeEntityClient(update_error=EntityConflictError("version mismatch"))
    refresher = _refresher(_FakeRollupRepo(), entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()
    # A concurrent edit won the optimistic lock; the evaluation is re-queued for the next cycle.
    assert refresher.pending() == {("default", "eval-a")}


@pytest.mark.asyncio
async def test_stop_flushes_pending_without_loss() -> None:
    entity_client = _FakeEntityClient()
    refresher = _refresher(_FakeRollupRepo(), entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    refresher.start()
    # stop() signals the loop to exit and lets it run a final drain — no mid-flush cancellation.
    await refresher.stop()
    assert {entity.name for entity in entity_client.updated} == {"eval-a"}
    assert refresher.pending() == set()


@pytest.mark.asyncio
async def test_rollup_query_failure_requeues() -> None:
    repo = _FakeRollupRepo(error=RuntimeError("clickhouse down"))
    refresher = _refresher(repo, _FakeEntityClient())
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()
    assert refresher.pending() == {("default", "eval-a")}


@pytest.mark.asyncio
async def test_stop_bounded_drain_does_not_hang_on_persistent_requeue() -> None:
    # A persistent optimistic-lock conflict re-queues the evaluation on every flush. stop() must
    # terminate (bounded drain) rather than loop forever, leaving the key queued.
    entity_client = _FakeEntityClient(update_error=EntityConflictError("version mismatch"))
    refresher = _refresher(_FakeRollupRepo(), entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    refresher.start()
    # wait_for turns a (regressed) infinite drain into a failure instead of a hung test.
    await asyncio.wait_for(refresher.stop(), timeout=5)
    assert refresher.pending() == {("default", "eval-a")}
