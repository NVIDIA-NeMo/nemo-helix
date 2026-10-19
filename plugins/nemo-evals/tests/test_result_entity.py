# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialization round-trip tests for the eval-result entities.

The entity store persists an entity's custom fields with
``model_dump(exclude=base, mode="json")`` into a JSON column and rebuilds it with
``model_validate``. These tests exercise that exact round-trip (which the in-memory fakes elsewhere
bypass) to guard the aggregated ``scores`` rollup and the row-eval input refs that carry nested types.
"""

from __future__ import annotations

import json
from typing import TypeVar

import pytest
from nemo_evals.api.schemas import AgentEvalResultSummary
from nemo_evals.entities import AgentEvalResultEntity, EvaluateResultEntity
from nhx_evals_sdk.agent_eval.results import AgentEvalMetricOutputCoverage
from nhx_evals_sdk.values.results import AggregatedMetricResult, AggregateRangeScore
from pydantic import ValidationError

# Constrained (not bound) so _E resolves to a concrete entity — which has EntityBase's name/workspace/
# __base_fields__ plus its own result fields — rather than the abstract _EvalResultCommon mixin.
_E = TypeVar("_E", AgentEvalResultEntity, EvaluateResultEntity)


def _scores() -> AggregatedMetricResult:
    return AggregatedMetricResult(scores=[AggregateRangeScore(name="accuracy", count=10, nan_count=0, mean=0.9)])


def _roundtrip(entity: _E) -> _E:
    """Mirror the entity store: dump custom fields to JSON, prove it's JSON-safe, then rebuild."""
    cls = type(entity)
    data = entity.model_dump(exclude=cls.__base_fields__, exclude_computed_fields=True, mode="json")
    data = json.loads(json.dumps(data))
    return cls.model_validate({"name": entity.name, "workspace": entity.workspace, **data})


def test_agent_eval_result_roundtrip_reads_a_retired_runner_kind() -> None:
    """A result stored before its runner was removed still loads: ``target_kind`` is a free-form
    trait, not a live discriminator, so retiring a runner never strands the rows it wrote.

    Built from the raw string rather than a target class on purpose — the class is gone, and a test
    that needed it could not express this.
    """
    entity = AgentEvalResultEntity(
        name="job-legacy",
        workspace="default",
        job_id="job-legacy",
        target_kind="codex",
        target_name="gpt-5.5",
        target_url=None,
        scores=_scores(),
        bundle_ref="fileset://default/agent-eval-results#legacy",
    )

    restored = _roundtrip(entity)

    assert restored.target_kind == "codex"
    assert restored.target_name == "gpt-5.5"
    assert restored.scores == entity.scores


def test_agent_eval_result_roundtrip_preserves_scores_and_target() -> None:
    entity = AgentEvalResultEntity(
        name="job-123",
        workspace="default",
        job_id="job-123",
        target_kind="fabric",
        target_name="openai/gpt-5.4",
        target_url=None,
        scores=_scores(),
        bundle_ref="fileset://default/agent-eval-results#bundle",
        summary=AgentEvalResultSummary(
            task_count=5,
            trial_count=10,
            score_count=10,
            error_count=2,
            metric_coverage={"accuracy": {"score": AgentEvalMetricOutputCoverage(total=10, scored=8, failed=2)}},
        ),
    )

    restored = _roundtrip(entity)

    assert restored.job_id == "job-123"
    assert restored.summary == entity.summary
    assert restored.target_kind == "fabric"
    assert restored.target_name == "openai/gpt-5.4"
    assert restored.target_url is None
    assert restored.bundle_ref == entity.bundle_ref
    # The nested AggregatedMetricResult must survive the JSON column intact.
    assert restored.scores == entity.scores
    assert restored.scores.scores[0].mean == 0.9


def test_evaluate_result_roundtrip_preserves_dataset_and_metric_types() -> None:
    entity = EvaluateResultEntity(
        name="job-456",
        workspace="default",
        job_id="job-456",
        target_kind="model",
        target_name="my-model",
        target_url="https://model.test/v1/chat/completions",
        scores=_scores(),
        bundle_ref="fileset://default/eval-results#bundle",
        dataset_ref="default/my-dataset",
        metric_types=["exact_match", "string_check"],
        row_count=10,
        error_row_count=1,
    )

    restored = _roundtrip(entity)

    assert restored.dataset_ref == "default/my-dataset"
    assert (restored.row_count, restored.error_row_count) == (10, 1)
    assert restored.metric_types == ["exact_match", "string_check"]
    assert restored.target_url == "https://model.test/v1/chat/completions"
    assert restored.scores == entity.scores


def test_records_persisted_before_the_counts_still_load() -> None:
    restored = AgentEvalResultEntity.model_validate(
        {
            "name": "job-old",
            "workspace": "default",
            "job_id": "job-old",
            "target_kind": "fabric",
            "target_name": "gpt-5.5",
            "target_url": None,
            "scores": _scores().model_dump(mode="json"),
            "bundle_ref": "fileset://default/agent-eval-results#old",
        }
    )

    assert restored.summary is None


def test_entity_types_are_distinct() -> None:
    # Distinct __entity_type__ keeps the two collections from colliding in the store.
    assert AgentEvalResultEntity.__entity_type__ == "agent_eval_result"
    assert EvaluateResultEntity.__entity_type__ == "evaluate_result"
    assert AgentEvalResultEntity.__entity_type__ != EvaluateResultEntity.__entity_type__


def test_shared_fields_are_required() -> None:
    # A result is only persisted once the run produced all of it — no schema defaults papering over
    # missing data. job_id (and the rest of the shared record) must be supplied by the caller.
    with pytest.raises(ValidationError):
        AgentEvalResultEntity(name="x", workspace="default", scores=_scores())  # ty: ignore[missing-argument]
