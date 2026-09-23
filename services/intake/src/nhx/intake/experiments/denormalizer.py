# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Denormalizes agent/model name fields from ClickHouse onto Evaluation entities.

Ingest marks ``(workspace, evaluation_name)`` dirty; a background loop recomputes each touched
evaluation's rollup and writes the distinct ``agent_names``/``agent_versions``/``model_names`` sets onto
the (system-managed) fields of its Evaluation entity. That lets the Evaluations list filter by
agent/model name against the entity store (``$contains``) instead of scanning the ClickHouse session
table on every request.

Only the name fields are denormalized, never the computed metric rollups: names are raw observed
strings with no formula, so a change to how an aggregate is computed can never invalidate them.

The start/stop lifecycle comes from :class:`nhx.intake.background_worker.BackgroundWorker`; the
debouncing (a coalescing dirty set drained on a fixed cadence) lives here because it is specific to how
this worker refreshes evaluations.
"""

from __future__ import annotations

import asyncio
import logging

from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.intake.background_worker import BackgroundWorker
from nhx.intake.entities.experiments import Experiment
from nhx.intake.repository.evaluation_rollup import EvaluationRollup, EvaluationRollupRepository

logger = logging.getLogger(__name__)

# stop()'s final flush can itself re-queue keys (a transient failure/conflict that would clear on
# retry), so it drains up to this many passes; anything still queued afterwards is logged.
_STOP_DRAIN_PASSES = 3


class EvaluationDenormalizer(BackgroundWorker):
    """Coalesces dirty evaluation names and refreshes their denormalized name fields on a fixed cadence.

    A burst of :meth:`mark_dirty` calls for one evaluation within an interval collapses to a single
    refresh, and marking is a plain, non-blocking set add — so the ingest and read hot paths are never
    gated on the refresh work.
    """

    def __init__(
        self,
        *,
        rollup_repository: EvaluationRollupRepository,
        entity_client: EntityClient,
        interval_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self._rollup_repository = rollup_repository
        self._entity_client = entity_client
        self._interval_seconds = interval_seconds
        self._dirty: set[tuple[str, str]] = set()

    def mark_dirty(self, *, workspace: str, evaluation_name: str) -> None:
        """Queue an evaluation for refresh. Cheap and non-blocking; safe to call from the ingest path."""
        self._dirty.add((workspace, evaluation_name))

    def pending(self) -> set[tuple[str, str]]:
        """Return a copy of the currently-queued ``(workspace, evaluation_name)`` pairs (observability/tests)."""
        return set(self._dirty)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                # Interruptible sleep: wakes early when stop() sets the event so shutdown is prompt.
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval_seconds)
            except asyncio.TimeoutError:
                pass  # interval elapsed; time for a periodic flush
            try:
                await self.flush()
            except Exception:
                logger.exception("Evaluation denormalization cycle failed")

    async def stop(self) -> None:
        await super().stop()
        # The loop may have exited before a final flush, and a flush can itself re-queue keys (a
        # transient failure/conflict), so drain what's left — bounded, so a persistent failure can't
        # hang shutdown.
        for _ in range(_STOP_DRAIN_PASSES):
            if not self._dirty:
                return
            await self.flush()
        if self._dirty:
            logger.warning(
                "Evaluation denormalizer stopped with %d evaluation(s) still queued after %d drain "
                "passes; not retried before shutdown",
                len(self._dirty),
                _STOP_DRAIN_PASSES,
            )

    async def flush(self) -> None:
        """Drain the dirty set and refresh each evaluation's name fields. Directly callable for tests."""
        if not self._dirty:
            return
        batch = self._dirty
        self._dirty = set()
        by_workspace: dict[str, list[str]] = {}
        for workspace, evaluation_name in batch:
            by_workspace.setdefault(workspace, []).append(evaluation_name)
        for workspace, evaluation_names in by_workspace.items():
            try:
                await self._refresh_workspace(workspace, evaluation_names)
            except Exception:
                # Re-queue the whole workspace batch for the next cycle (e.g. ClickHouse unavailable).
                logger.exception("Failed to refresh evaluation names for workspace %s; re-queuing", workspace)
                for evaluation_name in evaluation_names:
                    self.mark_dirty(workspace=workspace, evaluation_name=evaluation_name)

    async def _refresh_workspace(self, workspace: str, evaluation_names: list[str]) -> None:
        rollups = await self._rollup_repository.get_rollups(workspace=workspace, evaluation_names=evaluation_names)
        for evaluation_name in evaluation_names:
            rollup = rollups.get(evaluation_name)
            if rollup is None:
                continue
            await self._write_names(workspace, evaluation_name, rollup)

    async def _write_names(self, workspace: str, evaluation_name: str, rollup: EvaluationRollup) -> None:
        try:
            evaluation = await self._entity_client.get(Experiment, name=evaluation_name, workspace=workspace)
        except EntityNotFoundError:
            # Deleted between ingest and refresh; nothing to update.
            return
        # Skip the write when nothing changed, so a burst of re-ingests that add no new names doesn't
        # churn the entity store (and doesn't lose an optimistic-lock race for no reason).
        if (
            evaluation.agent_names == rollup.agent_names
            and evaluation.agent_versions == rollup.agent_versions
            and evaluation.model_names == rollup.model_names
        ):
            return
        evaluation.agent_names = rollup.agent_names
        evaluation.agent_versions = rollup.agent_versions
        evaluation.model_names = rollup.model_names
        try:
            await self._entity_client.update(evaluation)
        except EntityConflictError:
            # A concurrent user edit won the optimistic lock; re-queue for the next cycle.
            self.mark_dirty(workspace=workspace, evaluation_name=evaluation_name)
