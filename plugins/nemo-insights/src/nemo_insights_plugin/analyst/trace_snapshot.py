# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load trace-intel's canonical snapshot through authenticated Platform clients."""

from datetime import datetime, timezone

from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import TraceFilterParam
from trace_ingest.loaders.intake import (
    _EvaluatorResult,
    _IntakeSpan,
    _IntakeTrace,
    _normalize_trace,
)
from trace_ingest.models import Trace, TraceSnapshot


async def load_trace_snapshot(
    client: AsyncIntakeClient,
    *,
    workspace: str,
    agent: str,
    base_url: str,
    since: datetime | None = None,
    evaluation_id: str | None = None,
) -> TraceSnapshot:
    """Select complete traces, retaining SDK authentication and token refresh.

    The Git-pinned trace-ingest normalizer owns the conversion to canonical
    traces. Platform owns HTTP access because its clients refresh credentials
    and support evaluation IDs in addition to the package's name-based query.
    Time bounds select traces; child spans are loaded in full.
    """
    bounds = {"$lte": datetime.now(timezone.utc).isoformat()}
    if since is not None:
        bounds["$gte"] = since.isoformat()
    selection: TraceFilterParam = {"agent_name": agent, "started_at": bounds}
    if evaluation_id is not None:
        selection["evaluation_id"] = evaluation_id
    response = await client.list_traces(
        workspace=workspace,
        query_params={"filter": selection, "sort": "started_at", "mode": "detailed", "page_size": 100},
    )
    traces: list[Trace] = []
    async for trace in response.items():
        spans = await client.list_spans(
            workspace=workspace,
            query_params={
                "filter": {"trace_id": trace.id},
                "sort": "started_at",
                "mode": "detailed",
                "page_size": 100,
            },
        )
        results = await client.list_evaluator_results(
            workspace=workspace,
            query_params={"filter": {"session_id": trace.session_id}, "page_size": 100},
        )
        traces.append(
            _normalize_trace(
                _IntakeTrace.model_validate(trace.model_dump(mode="json")),
                [_IntakeSpan.model_validate(span.model_dump(mode="json")) async for span in spans.items()],
                [_EvaluatorResult.model_validate(result.model_dump(mode="json")) async for result in results.items()],
                base_url=base_url,
                workspace=workspace,
            )
        )
    return TraceSnapshot(traces)
