# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: `_persist_trial` against a real Intake service and ClickHouse.

The unit tests for these paths drive hand-written doubles, so they would keep passing if
the generated SDK method were renamed or changed shape. This exercises the whole chain —
backend, typed SDK, Intake, ClickHouse — and reads the trace back through the SDK.

Run directly::

    uv run pytest plugins/nemo-experimentalist/tests/integration/test_persist_trial_to_intake.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import pytest
from nemo_experimentalist_plugin.entities import ResourceRef, TrialResult
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import EvaluationCreateRequest, ExperimentCreateRequest

pytestmark = pytest.mark.integration

WORKSPACE = "default"


def _otlp_trace(tmp_path: Path, span_name: str) -> tuple[ResourceRef, str]:
    """One OTLP span on disk, in the jsonl shape the experimentalist's recorders emit.

    Ids are minted per call: the ClickHouse database is shared across the session with no
    truncation between tests, so fixed ids would make each test's reads depend on which
    others had already run.
    """
    trace_id = uuid4().hex
    span_id = uuid4().hex[:16]
    # Intake's spans table is TTL'd at 90 days on start_time, so a fixed past timestamp
    # would be dropped on write and the ingest would look successful but store nothing.
    now_ns = time.time_ns()
    line = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": span_name,
                                "startTimeUnixNano": str(now_ns),
                                "endTimeUnixNano": str(now_ns + 5_000_000),
                            }
                        ]
                    }
                ]
            }
        ]
    }
    path = tmp_path / f"trace-{trace_id}.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    ref = ResourceRef(uri=f"file://{path}", description="", metadata={"trace_format": "otlp"})
    return ref, trace_id


async def _ensure_evaluation(platform: AsyncNemoClient, name: str) -> None:
    """Register the Evaluation these spans name, or Intake drops them."""
    intake = AsyncIntakeClient.from_client(platform)
    group = (await intake.create_experiment(workspace=WORKSPACE, body=ExperimentCreateRequest(name=name))).data()
    await intake.create_evaluation(
        workspace=WORKSPACE,
        body=EvaluationCreateRequest(
            name=name,
            experiment_ids=[group.id],
            dataset_name=name,
            dataset_version="v1",
        ),
    )


async def test_persist_trial_uploads_a_local_otlp_trace_and_repoints_it(
    platform: AsyncNemoClient, tmp_path: Path
) -> None:
    backend = LocalExperimentalistBackend(client=platform, path=tmp_path / "backend")
    trace, trace_id = _otlp_trace(tmp_path, "it-span")
    trial = TrialResult(id="trial-it", task_id="case-it", status="completed", trace=trace)
    evaluation_name = f"exp-it-{uuid4().hex}"
    await _ensure_evaluation(platform, evaluation_name)

    await backend._persist_trial(trial, workspace=WORKSPACE, evaluation_name=evaluation_name, agent_attrs={})

    # The span reached ClickHouse through the typed SDK and is readable back through it.
    response = await AsyncIntakeClient.from_client(platform).list_spans(
        workspace=WORKSPACE, query_params={"filter": {"trace_id": trace_id}}
    )
    assert [span.name async for span in response.items()] == ["it-span"]

    # The trial now points at Intake, with the local file kept for reference.
    assert trial.trace is not None
    assert trial.trace.uri == f"intake://traces/{trace_id}"
    assert str(trial.trace.metadata["local_uri"]).startswith("file://")


async def test_persist_trial_stamps_evaluation_identity_onto_the_uploaded_spans(
    platform: AsyncNemoClient, tmp_path: Path
) -> None:
    backend = LocalExperimentalistBackend(client=platform, path=tmp_path / "backend")
    trace, _ = _otlp_trace(tmp_path, "s")
    trial = TrialResult(id="trial-attrs", task_id="case-attrs", status="completed", trace=trace)
    # Unique for the same reason the ids are: this is the value the read filters on.
    evaluation_name = f"exp-attrs-{uuid4().hex}"
    await _ensure_evaluation(platform, evaluation_name)

    await backend._persist_trial(trial, workspace=WORKSPACE, evaluation_name=evaluation_name, agent_attrs={})

    # evaluation_name is the attribute Intake indexes and the experiments rollup groups on,
    # so it has to survive the round trip rather than merely be sent.
    response = await AsyncIntakeClient.from_client(platform).list_spans(
        workspace=WORKSPACE, query_params={"filter": {"evaluation_name": evaluation_name}}
    )
    assert [span.name async for span in response.items()] == ["s"]
