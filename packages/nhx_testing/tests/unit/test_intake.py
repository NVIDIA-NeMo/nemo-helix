# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory Intake service."""

import httpx
import pytest
from nemo_helix_plugin.client.errors import NemoTransportError, NotFoundError
from nemo_helix_plugin.intake.types import AtifCreateRequest, EvaluationPatchRequest
from nhx.testing.intake import InMemoryIntakeService


def _atif(session_id: str) -> AtifCreateRequest:
    return AtifCreateRequest.model_validate(
        {"schema_version": "ATIF-v1.7", "session_id": session_id, "agent": {"name": "a", "version": "1"}, "steps": []}
    )


@pytest.mark.asyncio
async def test_records_ingest_and_resolves_root_spans() -> None:
    service = InMemoryIntakeService()
    intake = service.client()

    await intake.create_atif(body=_atif("s-1"))
    traces = [
        trace async for trace in (await intake.list_traces(query_params={"filter": {"session_id": "s-1"}})).items()
    ]

    assert [(body["session_id"], body["workspace"]) for body in service.atif_requests] == [("s-1", "default")]
    assert [trace.root_span_id for trace in traces] == ["span:s-1"]
    assert service.trace_lookups == ["s-1"]


@pytest.mark.asyncio
async def test_traces_exist_only_for_ingested_sessions() -> None:
    service = InMemoryIntakeService()
    intake = service.client()
    await intake.create_atif(body=_atif("s-1"))

    traces = [
        trace async for trace in (await intake.list_traces(query_params={"filter": {"session_id": "s-2"}})).items()
    ]

    assert traces == []


@pytest.mark.asyncio
async def test_evaluation_lookup_patch_and_configured_failures() -> None:
    service = InMemoryIntakeService()
    service.evaluation_metadata = {"k": "v"}
    intake = service.client()

    assert (await intake.get_evaluation(name="eval-1")).data().metadata == {"k": "v"}
    await intake.patch_evaluation(name="eval-1", body=EvaluationPatchRequest(metadata={"k": "w"}))
    assert service.evaluation_patches == [{"name": "eval-1", "metadata": {"k": "w"}}]
    assert (await intake.get_evaluation(name="eval-1")).data().metadata == {"k": "w"}
    await intake.patch_evaluation(name="eval-1", body=EvaluationPatchRequest(metadata=None))
    assert (await intake.get_evaluation(name="eval-1")).data().metadata == {}

    service.evaluation_exists = False
    with pytest.raises(NotFoundError):
        await intake.get_evaluation(name="eval-1")

    service.atif_failure = httpx.ConnectError("ingest failed")
    with pytest.raises(NemoTransportError):
        await intake.create_atif(body=_atif("s-1"))
