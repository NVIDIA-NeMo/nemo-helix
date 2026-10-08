# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample payload adapters for evals SDK execution."""

from collections.abc import Mapping
from typing import Any

from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.metrics.utils import as_finite_float

_CANDIDATE_SAMPLE_FIELDS = frozenset({"output_text", "response", "trajectory", "evidence"})

SAMPLE_RUNTIME_SEC_KEY = "runtime_sec"
_SAMPLE_FIELDS_HIDDEN_FROM_METRICS = frozenset({SAMPLE_RUNTIME_SEC_KEY})


def build_offline_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Build the sample payload for an offline row.

    Field mapping can normalize an offline prediction into the canonical
    ``output`` row field. Surface that value as ``sample.output_text`` so
    protocol metrics see the same candidate location as online evaluations.
    """
    output = row.get("output")
    if isinstance(output, str):
        return {"output_text": output}
    return {}


def build_retrieval_sample(row: dict[str, Any], rankings: dict[str, dict[str, float]] | None) -> dict[str, Any]:
    """Attach dense-search (or reranked) scores for one query row."""
    query_id = row.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("retrieval evaluation requires a query_id on each dataset row")
    if rankings is None or query_id not in rankings:
        raise ValueError(f"missing retrieval rankings for query {query_id!r}")
    return {"retrieval_scores": rankings[query_id]}


def build_metric_input(row: dict[str, Any], sample: dict[str, Any], index: int | None = None) -> MetricInput:
    """Build the metric protocol input from dataset row and generated sample payloads."""
    output_text = sample.get("output_text")
    metadata = {
        key: value
        for key, value in sample.items()
        if key not in _SAMPLE_FIELDS_HIDDEN_FROM_METRICS
        and (key not in _CANDIDATE_SAMPLE_FIELDS or (key == "output_text" and not isinstance(output_text, str)))
    }
    return MetricInput(
        row=DatasetRow(row_index=index, data=row),
        candidate=CandidateOutput(
            output_text=output_text if isinstance(output_text, str) else None,
            response=sample.get("response"),
            trajectory=sample.get("trajectory"),
            evidence=sample.get("evidence"),
            metadata=metadata,
        ),
    )


def sample_runtime_sec(sample: Mapping[str, Any]) -> float | None:
    """Seconds the target spent on its successful attempt, excluding queue wait, failed attempts, and backoff."""
    seconds = as_finite_float(sample.get(SAMPLE_RUNTIME_SEC_KEY))
    return seconds if seconds is not None and seconds >= 0 else None
