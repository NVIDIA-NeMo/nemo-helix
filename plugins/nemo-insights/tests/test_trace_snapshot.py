# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated Intake pagination and canonical trace conversion."""

import json
from datetime import datetime, timezone

import httpx
import pytest
from nemo_insights_plugin.analyst.trace_snapshot import load_trace_snapshot
from nemo_platform_plugin.intake.client import AsyncIntakeClient


def _page(data: list[dict], page: int = 1, total: int = 1) -> dict:
    return {
        "data": data,
        "pagination": {
            "page": page,
            "page_size": 1,
            "current_page_size": len(data),
            "total_pages": total,
            "total_results": total,
        },
    }


@pytest.mark.asyncio
async def test_loads_all_pages_and_preserves_complete_traces() -> None:
    requests: list[httpx.Request] = []
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        assert "/workspaces/test/" in request.url.path
        page = int(request.url.params.get("page", "1"))
        selection = json.loads(request.url.params["filter"])
        if request.url.path.endswith("/traces"):
            assert selection["agent_name"] == "target"
            assert selection["evaluation_id"] == "eval-id"
            assert selection["started_at"]["$gte"] == since.isoformat()
            assert "$lte" in selection["started_at"]
            return httpx.Response(
                200,
                json=_page(
                    [
                        {
                            "id": f"trace-{page}",
                            "session_id": "session",
                            "workspace": "test",
                            "agent_name": "target",
                            "started_at": since.isoformat(),
                            "status": "error",
                        }
                    ],
                    page,
                    2,
                ),
            )
        if request.url.path.endswith("/spans"):
            # Selection bounds must not clip the selected trace's child spans.
            assert "started_at" not in selection
            assert "evaluation_id" not in selection
            trace_id = selection["trace_id"]
            return httpx.Response(
                200,
                json=_page(
                    [
                        {
                            "span_id": f"{trace_id}-span-{page}",
                            "trace_id": trace_id,
                            "parent_span_id": f"{trace_id}-span-1" if page == 2 else None,
                            "session_id": "session",
                            "workspace": "test",
                            "kind": "TOOL",
                            "source": "otlp",
                            "tool_name": "search",
                            "ingested_at": since.isoformat(),
                            "started_at": since.isoformat(),
                            "status": "error",
                            "error_message": "search timed out",
                        }
                    ],
                    page,
                    2,
                ),
            )
        assert request.url.path.endswith("/evaluator-results")
        assert selection["session_id"] == "session"
        # Another evaluation's trace shares this session but was not selected.
        trace_id = f"trace-{page}" if page < 3 else "unselected-trace"
        return httpx.Response(
            200,
            json=_page(
                [
                    {
                        "evaluator_result_id": f"result-{page}",
                        "span_id": f"{trace_id}-span-1",
                        "session_id": "session",
                        "workspace": "test",
                        "name": "accuracy",
                        "data_type": "NUMERIC",
                        "value": page / 10,
                        "created_at": since.isoformat(),
                        "ingested_at": since.isoformat(),
                    }
                ],
                page,
                3,
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIntakeClient(
            base_url="https://platform.example",
            http_client=http_client,
            auth="test-token",
        )
        snapshot = await load_trace_snapshot(
            client,
            workspace="test",
            agent="target",
            base_url="https://platform.example",
            since=since,
            evaluation_id="eval-id",
        )
    assert len(snapshot) == 2
    first = snapshot.get_trace_by_id("trace-1")
    assert first.root_spans[0].children[0].error == "search timed out"
    pointer = first.attributes["source_pointer"]
    assert isinstance(pointer, dict)
    assert pointer["workspace"] == "test"
    for index in (1, 2):
        trace = snapshot.get_trace_by_id(f"trace-{index}")
        assert set(trace.evaluator_results) == {"accuracy"}
        result = trace.evaluator_results["accuracy"]
        assert isinstance(result, dict)
        assert result["evaluator_result_id"] == f"result-{index}"
        assert result["value"] == index / 10
    assert len(requests) == 12


@pytest.mark.asyncio
async def test_empty_selection_is_an_empty_snapshot() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_page([], total=0))),
    ) as http_client:
        client = AsyncIntakeClient(base_url="https://platform.example", http_client=http_client)
        snapshot = await load_trace_snapshot(
            client,
            workspace="test",
            agent="target",
            base_url="https://platform.example",
        )
    assert len(snapshot) == 0
