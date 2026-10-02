# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-heal of denormalized name facets on read.

Reading an evaluation compares its live rollup names against the entity's stored facets and queues a
refresh when they differ. This backfills evaluations ingested before the facets existed the first time
they're read, with no per-instance migration to run.
"""

from __future__ import annotations

from typing import cast

from nhx.intake.api.v2.experiments.endpoints import _enqueue_stale_denormalization
from nhx.intake.entities.experiments import Experiment
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.experiments.read_service import EvaluationRead
from nhx.intake.repository.evaluation_rollup import EvaluationRollup


class _CapturingRefresher:
    def __init__(self) -> None:
        self.marked: list[tuple[str, str]] = []

    def mark_dirty(self, *, workspace: str, evaluation_name: str) -> None:
        self.marked.append((workspace, evaluation_name))


def _entity(
    name: str,
    *,
    agent_names: list[str] | None = None,
    agent_versions: list[str] | None = None,
    model_names: list[str] | None = None,
) -> Experiment:
    return Experiment(
        name=name,
        workspace="default",
        experiment_ids=["grp"],
        dataset_name="ds",
        agent_names=agent_names or [],
        agent_versions=agent_versions or [],
        model_names=model_names or [],
    )


def _rollup(
    name: str, *, agent_names: list[str], agent_versions: list[str], model_names: list[str]
) -> EvaluationRollup:
    return EvaluationRollup(
        evaluation_name=name,
        agent_names=agent_names,
        agent_versions=agent_versions,
        model_names=model_names,
    )


def _heal(reads: list[EvaluationRead]) -> list[tuple[str, str]]:
    refresher = _CapturingRefresher()
    _enqueue_stale_denormalization(cast(EvaluationDenormalizer, refresher), workspace="default", reads=reads)
    return refresher.marked


def test_enqueues_when_stored_facets_are_empty_but_rollup_has_names() -> None:
    read = EvaluationRead(
        entity=_entity("eval-a"),
        rollup=_rollup("eval-a", agent_names=["agent-x"], agent_versions=["1.0"], model_names=["m"]),
    )
    assert _heal([read]) == [("default", "eval-a")]


def test_does_not_enqueue_when_facets_match() -> None:
    names = dict(agent_names=["agent-x"], agent_versions=["1.0"], model_names=["m"])
    read = EvaluationRead(entity=_entity("eval-a", **names), rollup=_rollup("eval-a", **names))
    assert _heal([read]) == []


def test_enqueues_on_drift_in_any_single_field() -> None:
    # Agent names/versions match, but the model set drifted -> still a discrepancy.
    read = EvaluationRead(
        entity=_entity("eval-a", agent_names=["agent-x"], agent_versions=["1.0"], model_names=["m-old"]),
        rollup=_rollup("eval-a", agent_names=["agent-x"], agent_versions=["1.0"], model_names=["m-old", "m-new"]),
    )
    assert _heal([read]) == [("default", "eval-a")]


def test_skips_when_rollup_missing() -> None:
    # No live rollup (e.g. ClickHouse unavailable) -> nothing to compare against, leave facets as-is.
    read = EvaluationRead(entity=_entity("eval-a"), rollup=None)
    assert _heal([read]) == []


def test_no_refresher_is_a_noop() -> None:
    read = EvaluationRead(
        entity=_entity("eval-a"),
        rollup=_rollup("eval-a", agent_names=["agent-x"], agent_versions=["1.0"], model_names=["m"]),
    )
    # Must not raise when the refresher is absent (startup didn't create one).
    _enqueue_stale_denormalization(None, workspace="default", reads=[read])
