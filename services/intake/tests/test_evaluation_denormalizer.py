# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the debounced evaluation name-facet refresher."""

from __future__ import annotations

import asyncio
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


class _FakeRollupRepo:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._error = error

    async def get_rollups(self, *, workspace: str, evaluation_names: list[str]) -> dict[str, EvaluationRollup]:
        self.calls.append((workspace, list(evaluation_names)))
        if self._error is not None:
            raise self._error
        return {evaluation_name: _sample_rollup(evaluation_name) for evaluation_name in evaluation_names}


class _FakeEntityClient:
    def __init__(
        self,
        *,
        get_error: Exception | None = None,
        update_error: Exception | None = None,
        existing: Experiment | None = None,
    ) -> None:
        self.updated: list[Experiment] = []
        self.get_calls: list[tuple[str, str]] = []
        self._get_error = get_error
        self._update_error = update_error
        self._existing = existing

    async def get(self, entity_type: type[Experiment], *, name: str, workspace: str) -> Experiment:
        self.get_calls.append((name, workspace))
        if self._get_error is not None:
            raise self._get_error
        if self._existing is not None:
            return self._existing
        return Experiment(name=name, workspace=workspace, experiment_ids=["grp"], dataset_name="ds")

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

    # One batched rollup query for the workspace, covering both dirty evaluations.
    assert len(repo.calls) == 1
    workspace, ids = repo.calls[0]
    assert workspace == "default"
    assert set(ids) == {"eval-a", "eval-b"}

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
async def test_missing_evaluation_is_skipped() -> None:
    entity_client = _FakeEntityClient(get_error=EntityNotFoundError("gone"))
    refresher = _refresher(_FakeRollupRepo(), entity_client)
    refresher.mark_dirty(workspace="default", evaluation_name="eval-a")
    await refresher.flush()
    assert entity_client.updated == []
    assert refresher.pending() == set()  # not re-queued; the evaluation no longer exists


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
