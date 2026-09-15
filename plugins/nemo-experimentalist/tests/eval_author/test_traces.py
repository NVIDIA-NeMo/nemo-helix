# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Eval Author's Intake trace tools.

Only the logic this module owns: what the raw queries send and return, how the
composite dedupes and merges, ref normalization, and the error messages. Reading and
diagnosing a trace belong to Experimentalist, so those seams are faked.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_experimentalist_plugin.eval_author import traces
from nemo_experimentalist_plugin.eval_author.agent import EvalAuthor
from nemo_platform_plugin.client.errors import NemoHTTPError, NemoTransportError, raise_for_status

_BASE = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _at(minute: int) -> str:
    return (_BASE + timedelta(minutes=minute)).isoformat()


class _Row:
    """A stand-in for an SDK model. Only ``model_dump`` is used."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {name: value for name, value in self._fields.items() if value is not None}


def _span(trace_id: str | None, *, minute: int, **fields: Any) -> _Row:
    return _Row(trace_id=trace_id, started_at=_at(minute), span_id=f"s-{minute}", **fields)


def _group(trace_id: str, *, span_count: int = 1) -> _Row:
    """One row of ``spans.groups.list`` grouped by trace id."""
    return _Row(group={"trace_id": trace_id}, span_count=span_count)


def _summary(trace_id: str, *, minute: int, **fields: Any) -> _Row:
    return _Row(
        id=trace_id,
        started_at=_at(minute),
        status=fields.pop("status", "success"),
        span_count=fields.pop("span_count", 1),
        error_count=fields.pop("error_count", 0),
        **fields,
    )


def _pages(items: list[Any], error: Exception | None) -> Any:
    async def stream() -> Any:
        if error is not None:
            raise error
        for item in items:
            yield item

    return stream()


class _FakeClient:
    """Stands in for the three Intake reads these tools make."""

    def __init__(
        self,
        spans: list[_Row],
        summaries: list[_Row],
        groups: list[_Row],
        error: Exception | None,
    ) -> None:
        self._spans = spans
        self._summaries = summaries
        self._groups = groups
        self._error = error
        self.span_calls: list[dict[str, Any]] = []
        self.trace_calls: list[dict[str, Any]] = []
        self.group_calls: list[dict[str, Any]] = []

    async def list_spans(self, *, workspace: str, query_params: dict[str, Any] | None = None) -> Any:
        del workspace
        self.span_calls.append(query_params or {})
        return SimpleNamespace(items=lambda: _pages(self._spans, self._error))

    async def list_span_groups(self, *, workspace: str, query_params: dict[str, Any] | None = None) -> Any:
        del workspace
        self.group_calls.append(query_params or {})
        return SimpleNamespace(items=lambda: _pages(self._groups, self._error))

    async def list_traces(self, *, workspace: str, query_params: dict[str, Any] | None = None) -> Any:
        del workspace
        params = query_params or {}
        self.trace_calls.append(params)
        rows = self._summaries
        wanted = params.get("filter", {}).get("id", {}).get("$in")
        if wanted is not None:
            rows = [row for row in rows if row.model_dump()["id"] in set(wanted)]
        return SimpleNamespace(items=lambda: _pages(rows, self._error))


def _client(
    spans: list[_Row] | None = None,
    summaries: list[_Row] | None = None,
    groups: list[_Row] | None = None,
    error: Exception | None = None,
) -> Any:
    """A fake platform client, typed ``Any`` so call sites need no cast."""
    return _FakeClient(spans or [], summaries or [], groups or [], error)


@pytest.fixture(autouse=True)
def fake_intake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(traces.AsyncIntakeClient, "from_client", lambda client: client)


def _status_error(code: int) -> NemoHTTPError:
    request = httpx.Request("GET", "https://example.invalid/spans")
    response = httpx.Response(code, request=request, json={"detail": "boom"})
    try:
        raise_for_status(response)
    except NemoHTTPError as exc:
        return exc
    raise AssertionError(f"HTTP {code} did not raise")


# --- raw queries -------------------------------------------------------------------


async def test_query_spans_sends_the_filter_and_returns_plain_rows() -> None:
    client = _client(spans=[_span("t-1", minute=5, tool_name="search")])

    result = await traces.query_spans(
        client,
        workspace="ws",
        filter={"agent_name": "aut", "tool_name": "search", "status": "error"},
        mode="detailed",
        limit=10,
    )

    call = client.span_calls[0]
    assert call["filter"] == {"agent_name": "aut", "tool_name": "search", "status": "error"}
    assert call["mode"] == "detailed"
    assert call["sort"] == "-started_at"
    assert result["count"] == 1
    assert result["truncated"] is False
    assert result["spans"][0]["tool_name"] == "search"


async def test_query_spans_omits_an_absent_filter() -> None:
    # The SDK uses an omit sentinel, so a None filter must not be sent as null.
    client = _client()

    await traces.query_spans(client, workspace="ws")

    assert "filter" not in client.span_calls[0]


async def test_query_spans_truncates_at_the_limit() -> None:
    client = _client(spans=[_span(f"t-{index}", minute=index) for index in range(10)])

    result = await traces.query_spans(client, workspace="ws", limit=4)

    assert result["count"] == 4
    assert result["truncated"] is True


async def test_query_spans_groups_server_side() -> None:
    client = _client(groups=[_Row(group={"trace_id": "t-1"}, span_count=12)])

    result = await traces.query_spans(client, workspace="ws", filter={"tool_name": "search"}, group_by="trace_id")

    assert client.group_calls[0]["by"] == "trace_id"
    assert client.group_calls[0]["filter"] == {"tool_name": "search"}
    assert client.span_calls == []
    assert result["grouped_by"] == "trace_id"
    assert result["groups"][0]["span_count"] == 12


async def test_grouped_queries_take_a_sort_of_their_own() -> None:
    """Groups sort by size as well as by time, so a caller's choice must reach the server."""
    client = _client(groups=[_group("t-1", span_count=12)])

    await traces.query_spans(client, workspace="ws", group_by="trace_id", sort="-span_count")

    assert client.group_calls[0]["sort"] == "-span_count"


async def test_every_span_row_carries_the_way_back_to_its_trace() -> None:
    """A span read alone loses the trajectory around it, so the ref home travels with it."""
    client = _client(spans=[_span("t-1", minute=5, tool_name="search")])

    result = await traces.query_spans(client, workspace="ws")

    assert result["spans"][0]["trace_ref"] == "intake://t-1"


async def test_every_trace_group_carries_the_way_back_to_its_trace() -> None:
    client = _client(groups=[_Row(group={"trace_id": "t-1"}, span_count=12)])

    result = await traces.query_spans(client, workspace="ws", group_by="trace_id")

    assert result["groups"][0]["trace_ref"] == "intake://t-1"


async def test_a_session_group_has_no_trace_to_point_at() -> None:
    """Grouping by session spans many traces, so there is no single trace to open."""
    client = _client(groups=[_Row(group={"session_id": "s-1"}, span_count=12)])

    result = await traces.query_spans(client, workspace="ws", group_by="session_id")

    assert "trace_ref" not in result["groups"][0]


async def test_query_traces_defaults_to_preview_for_the_rollups() -> None:
    client = _client(summaries=[_summary("t-1", minute=5, span_count=40, error_count=3)])

    result = await traces.query_traces(client, workspace="ws", filter={"id": {"$in": ["t-1"]}})

    assert client.trace_calls[0]["mode"] == "preview"
    assert client.trace_calls[0]["sort"] == "-started_at"
    assert result["traces"][0]["span_count"] == 40
    assert result["traces"][0]["error_count"] == 3


# --- the composite -----------------------------------------------------------------


async def test_find_agent_traces_asks_the_server_to_group_and_sort() -> None:
    # The whole point of the grouped call: one request returns the agent's recent
    # traces. Draining spans to collect ids only ever found the traces that fit
    # inside the scanned window, so the span endpoint must go untouched.
    client = _client(groups=[_group("t-1")], summaries=[_summary("t-1", minute=10)])

    await traces.find_agent_traces(client, agent="aut", workspace="ws")

    assert client.group_calls[0]["by"] == "trace_id"
    assert client.group_calls[0]["sort"] == "-started_at"
    assert client.group_calls[0]["filter"] == {"agent_name": "aut"}
    assert client.span_calls == []


async def test_find_agent_traces_orders_newest_first() -> None:
    client = _client(
        groups=[_group("t-new"), _group("t-old")],
        summaries=[_summary("t-new", minute=40), _summary("t-old", minute=10)],
    )

    result = await traces.find_agent_traces(client, agent="aut", workspace="ws")

    assert [trace["trace_ref"] for trace in result["traces"]] == ["intake://t-new", "intake://t-old"]
    assert result["traces"][0]["trace_id"] == "t-new"
    assert result["count"] == 2
    assert result["truncated"] is False


async def test_find_agent_traces_uses_the_server_summary() -> None:
    # A group counts only the spans that matched the filter, so it reports one span and
    # no errors. The server knows the trace holds 40 spans and three errors. The group
    # alone would report a healthy trace.
    client = _client(
        groups=[_group("t-1")],
        summaries=[_summary("t-1", minute=10, status="error", span_count=40, error_count=3, name="run")],
    )

    result = await traces.find_agent_traces(client, agent="aut", workspace="ws")

    trace = result["traces"][0]
    assert trace["status"] == "error"
    assert trace["span_count"] == 40
    assert trace["error_count"] == 3
    assert trace["name"] == "run"
    assert trace["started_at"] == _at(10)
    assert client.trace_calls[0]["filter"] == {"id": {"$in": ["t-1"]}}


async def test_find_agent_traces_keeps_a_trace_with_no_summary() -> None:
    # A trace whose root span was never ingested has no traces.list row. It must still
    # be returned, and its unknown fields must not read as zero.
    client = _client(groups=[_group("t-1")], summaries=[])

    result = await traces.find_agent_traces(client, agent="aut", workspace="ws")

    trace = result["traces"][0]
    assert trace["trace_ref"] == "intake://t-1"
    assert trace["status"] == "unknown"
    assert trace["span_count"] is None
    assert trace["error_count"] is None


async def test_find_agent_traces_chunks_trace_ids() -> None:
    client = _client(
        groups=[_group(f"t-{index}") for index in range(120)],
        summaries=[_summary(f"t-{index}", minute=index) for index in range(120)],
    )

    result = await traces.find_agent_traces(client, agent="aut", workspace="ws", limit=120)

    assert result["count"] == 120
    assert [len(call["filter"]["id"]["$in"]) for call in client.trace_calls] == [50, 50, 20]


async def test_find_agent_traces_truncates_on_the_trace_limit() -> None:
    client = _client(groups=[_group(f"t-{index}") for index in range(10)])

    result = await traces.find_agent_traces(client, agent="aut", workspace="ws", limit=3)

    assert result["count"] == 3
    assert result["truncated"] is True


async def test_find_agent_traces_sends_since() -> None:
    client = _client()
    since = _BASE + timedelta(minutes=30)

    await traces.find_agent_traces(client, agent="aut", workspace="ws", since=since)

    assert client.group_calls[0]["filter"] == {"agent_name": "aut", "started_at": {"gte": since.isoformat()}}


async def test_find_agent_traces_empty_result_is_not_an_error() -> None:
    client = _client()

    result = await traces.find_agent_traces(client, agent="typo-agent", workspace="ws")

    assert result["count"] == 0
    assert "typo-agent" in result["note"]
    assert "ws" in result["note"]
    # Nothing to summarize, so the second call must not be made.
    assert client.trace_calls == []


# --- errors ------------------------------------------------------------------------


async def test_auth_failure_says_how_to_recover() -> None:
    client = _client(error=_status_error(401))

    with pytest.raises(traces.TraceQueryError, match="nemo auth login"):
        await traces.find_agent_traces(client, agent="aut", workspace="ws")


async def test_bad_filter_points_at_the_vocabulary() -> None:
    client = _client(error=_status_error(400))

    with pytest.raises(traces.TraceQueryError, match="filter fields and operators"):
        await traces.query_spans(client, workspace="ws", filter={"nonsense": 1})


async def test_an_unexpected_status_still_names_itself() -> None:
    # No status gets swallowed. A 500 is Intake's fault rather than the caller's, so
    # there is no corrective action to offer, but the code has to reach the agent.
    client = _client(error=_status_error(500))

    with pytest.raises(traces.TraceQueryError, match="HTTP 500"):
        await traces.query_spans(client, workspace="ws")


async def test_unreachable_platform_names_the_base_url() -> None:
    request = httpx.Request("GET", "https://example.invalid/spans")
    client = _client(error=NemoTransportError(httpx.ConnectError("unreachable", request=request)))

    with pytest.raises(traces.TraceQueryError, match="NMP_BASE_URL"):
        await traces.query_traces(client, workspace="ws")


# --- reading and analysis ----------------------------------------------------------


@pytest.mark.parametrize("ref", ["t-1", "intake://t-1", "intake://traces/t-1"])
async def test_all_three_ref_spellings_reach_the_explorer(monkeypatch: pytest.MonkeyPatch, ref: str) -> None:
    seen: list[str] = []

    async def _from_ref(resource_ref: Any, client: Any, workspace: str) -> str:
        seen.append(resource_ref.uri)
        return "explorer"

    monkeypatch.setattr(traces.TraceExplorer, "from_ref", _from_ref)

    assert await traces.read_trace(_client(), ref, workspace="ws") == "explorer"  # type: ignore[comparison-overlap]
    # from_ref strips both prefixes itself, so a bare id only needs the scheme added.
    assert seen == [ref if ref.startswith("intake://") else "intake://t-1"]


async def test_missing_trace_names_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _from_ref(resource_ref: Any, client: Any, workspace: str) -> str:
        raise ValueError("No spans found in Intake for trace: t-gone")

    monkeypatch.setattr(traces.TraceExplorer, "from_ref", _from_ref)

    with pytest.raises(traces.TraceQueryError, match="No spans found in Intake for trace: t-gone") as caught:
        await traces.read_trace(_client(), "t-gone", workspace="ws")
    assert "ws" in str(caught.value)


async def test_read_trace_404_names_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _from_ref(resource_ref: Any, client: Any, workspace: str) -> str:
        raise _status_error(404)

    monkeypatch.setattr(traces.TraceExplorer, "from_ref", _from_ref)

    with pytest.raises(traces.TraceQueryError, match="workspace 'ws' exists"):
        await traces.read_trace(_client(), "t-gone", workspace="ws")


async def test_synthesized_task_starts_no_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeAnalyzer:
        def __init__(self, *, experiment_dir: Path) -> None:
            captured["experiment_dir"] = experiment_dir

        async def run(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "diagnostic"

    monkeypatch.setattr(traces, "TraceAnalyzer", _FakeAnalyzer)

    result = await traces.analyze_trace(
        _client(),
        "intake://traces/t-1",
        workspace="ws",
        experiment_dir=Path("/tmp/experiment"),
        agent_path=Path("/tmp/agent"),
    )

    assert result == "diagnostic"
    assert captured["insight"] is None
    assert captured["trial"].trace.uri == "intake://t-1"
    # A production trace has no evaluator task, so the synthetic one must start nothing.
    async with captured["task"].start_deps() as runtime:
        assert runtime is None


# --- agent surface -----------------------------------------------------------------


def _agent(client: Any, workspace: str | None) -> Any:
    """An EvalAuthor with only the trace-tool attributes set.

    ``object.__new__`` skips ``__init__``, which builds an LLM client the trace tools
    do not touch.
    """
    eval_author = object.__new__(EvalAuthor)
    eval_author.client = client
    eval_author.workspace = workspace
    return eval_author


async def test_agent_trace_tools_say_what_is_missing() -> None:
    eval_author = _agent(None, None)

    with pytest.raises(traces.TraceQueryError, match="EvalAuthor.client"):
        await eval_author.find_agent_traces("aut")


async def test_agent_trace_tools_delegate() -> None:
    client = _client(
        spans=[_span("t-1", minute=10)],
        groups=[_group("t-1")],
        summaries=[_summary("t-1", minute=10, span_count=7)],
    )
    eval_author = _agent(client, "ws")

    found = await eval_author.find_agent_traces("aut")
    queried = await eval_author.query_spans(filter={"trace_id": "t-1"}, limit=5)

    assert found["traces"][0]["span_count"] == 7
    assert client.span_calls[-1]["filter"] == {"trace_id": "t-1"}
    assert queried["count"] == 1
