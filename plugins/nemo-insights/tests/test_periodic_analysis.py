# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for periodic insights analysis plumbing."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generic, TypeVar
from zoneinfo import ZoneInfo

import httpx
import pytest
import yaml
from nemo_insights_plugin.analyst.analyst_backend import (
    InsightsFileStore,
    LocalAnalystBackend,
    RemoteAnalystBackend,
    _merge_eval_filter,
    _merge_since_filter,
    make_analyst_backend,
)
from nemo_insights_plugin.analyst.result import AnalystResult, InsightUpdate, NewInsight
from nemo_insights_plugin.config import (
    AnalystSchedulerConfig,
    Frequency,
    InsightsConfig,
    Weekday,
)
from nemo_insights_plugin.controller import InsightsAnalysisController, _job_name
from nemo_insights_plugin.entities import (
    AnalysisConfig,
    AnalysisConfigStatus,
    AnalysisRunStatus,
    Insight,
    InsightStatus,
)
from nemo_insights_plugin.jobs.analyze import AnalyzeJob, AnalyzeSpec
from nemo_insights_plugin.schedule import is_due, previous_scheduled
from nemo_insights_plugin.schema import UpdateAnalysisRunStatusRequest
from nemo_insights_plugin.sdk_resources.analysis_jobs import (
    AnalysisJob,
    AsyncAnalysisJobsClient,
    CreateAnalysisJobRequest,
    ListAnalysisJobsQueryParams,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import SpanFilterParam, SpanGroup, SpanGroupsPage
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import JobResults, ResultRef
from nemo_platform_plugin.jobs.constants import (
    DEFAULT_JOB_STORAGE_PATH,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.nooa_model_client import ConfiguredModelRefs
from nemo_platform_plugin.schema import PaginationData
from pydantic import JsonValue, ValidationError

_BASE_URL = "https://example.com"
_T = TypeVar("_T")


def _async_platform() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(base_url=_BASE_URL)


def _http_not_found_error() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", _BASE_URL)
    response = httpx.Response(404, request=request)
    return httpx.HTTPStatusError("not found", request=request, response=response)


@dataclass
class _TypedResponse(Generic[_T]):
    body: _T

    def data(self) -> _T:
        return self.body


class _AsyncItems(Generic[_T]):
    def __init__(self, items: list[_T]) -> None:
        self._items = items

    async def items(self) -> AsyncIterator[_T]:
        for item in self._items:
            yield item


def test_merge_since_filter_adds_lower_bound() -> None:
    since = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)

    result = _merge_since_filter({"agent_name": "research-agent"}, since=since)

    assert result == {
        "agent_name": "research-agent",
        "started_at": {"gte": "2026-06-04T12:00:00+00:00"},
    }


def test_merge_since_filter_keeps_later_existing_lower_bound() -> None:
    since = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)

    result = _merge_since_filter(
        {"started_at": {"gte": "2026-06-04T13:00:00+00:00"}},
        since=since,
    )

    assert result == {"started_at": {"gte": "2026-06-04T13:00:00+00:00"}}


def test_merge_since_filter_compares_equivalent_iso_representations() -> None:
    since = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)
    current = "2026-06-04T07:00:00-05:00"

    result = _merge_since_filter({"started_at": {"gte": current}}, since=since)

    assert result == {"started_at": {"gte": current}}


_STAMP = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)


class _RecordingInsights:
    def __init__(self) -> None:
        self.rows: dict[str, Insight] = {}

    async def create(
        self,
        *,
        workspace: str,
        title: str,
        agent: str,
        description: str,
        status: InsightStatus | str = InsightStatus.OPEN,
        trace_refs: list[str] | None = None,
    ) -> Insight:
        insight_id = f"insight-remote-{len(self.rows) + 1}"
        row = Insight(
            workspace=workspace,
            name=f"{insight_id}-slug",
            title=title,
            agent=agent,
            description=description,
            status=InsightStatus(status),
            trace_refs=list(trace_refs or []),
        )
        row._id = insight_id
        row._created_at = _STAMP
        row._updated_at = _STAMP
        row._db_version = 1
        self.rows[insight_id] = row
        return row

    async def get(self, *, workspace: str, insight_id: str) -> Insight:
        del workspace
        row = self.rows.get(insight_id)
        if row is None:
            raise _http_not_found_error()
        return row

    async def update(
        self,
        *,
        workspace: str,
        insight_id: str,
        agent: str | None = None,
        description: str | None = None,
        status: InsightStatus | str | None = None,
        trace_refs: list[str] | None = None,
    ) -> Insight:
        del workspace
        row = self.rows.get(insight_id)
        if row is None:
            raise _http_not_found_error()
        if agent is not None:
            row.agent = agent
        if description is not None:
            row.description = description
        if status is not None:
            row.status = InsightStatus(status)
        if trace_refs is not None:
            row.trace_refs = list(trace_refs)
        return row


@asynccontextmanager
async def _remote_backend_with_mirror(
    path: Path | None,
    monkeypatch: pytest.MonkeyPatch,
    *,
    insights: _RecordingInsights | None = None,
) -> AsyncIterator[tuple[RemoteAnalystBackend, _RecordingInsights]]:
    async with _async_platform() as sdk:
        insights_resource = sdk.insights.insights
        recording = insights or _RecordingInsights()
        monkeypatch.setattr(insights_resource, "create", recording.create)
        monkeypatch.setattr(insights_resource, "get", recording.get)
        monkeypatch.setattr(insights_resource, "update", recording.update)
        mirror = InsightsFileStore(path) if path is not None else None
        yield RemoteAnalystBackend(sdk, mirror=mirror), recording


@pytest.mark.asyncio
async def test_remote_persist_validates_updates_without_trace_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    result = AnalystResult(
        summary="Nothing new.",
        updated_insights=[InsightUpdate(id="missing-insight")],
    )

    async with _remote_backend_with_mirror(None, monkeypatch) as (backend, _):
        report = await backend.persist_result(workspace="default", agent="research-agent", result=result)

    assert "- skipped (insight not found): missing-insight" in report
    assert "- updated: missing-insight" not in report


@pytest.mark.asyncio
async def test_remote_persist_mirrors_platform_records_to_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "insights.yaml"
    result = AnalystResult(
        summary="Found one.",
        new_insights=[NewInsight(title="Retrieval drops context", description="Long inputs.", trace_refs=["t1"])],
    )

    async with _remote_backend_with_mirror(path, monkeypatch) as (backend, _):
        report = await backend.persist_result(workspace="default", agent="research-agent", result=result)

    records = yaml.safe_load(path.read_text(encoding="utf-8"))["insights"]
    assert [r["id"] for r in records] == ["insight-remote-1"]
    assert records[0]["trace_refs"] == ["t1"]
    assert records[0]["created_at"] == _STAMP.isoformat()
    assert f"- mirrored 1 insight(s) to {path}" in report


@pytest.mark.asyncio
async def test_mirror_updates_match_platform_ids_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "insights.yaml"
    async with _remote_backend_with_mirror(path, monkeypatch) as (backend, _):
        await backend.persist_result(
            workspace="default",
            agent="research-agent",
            result=AnalystResult(
                summary="Found one.",
                new_insights=[
                    NewInsight(title="Retrieval drops context", description="Long inputs.", trace_refs=["t1"])
                ],
            ),
        )

        await backend.persist_result(
            workspace="default",
            agent="research-agent",
            result=AnalystResult(
                summary="More evidence.",
                updated_insights=[InsightUpdate(id="insight-remote-1", trace_refs=["t2"])],
            ),
        )

    records = yaml.safe_load(path.read_text(encoding="utf-8"))["insights"]
    assert len(records) == 1, "the second run must update the mirrored record, not append a second one"
    assert records[0]["trace_refs"] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_mirror_write_failure_warns_without_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    result = AnalystResult(
        summary="Found one.",
        new_insights=[NewInsight(title="Retrieval drops context", description="Long inputs.")],
    )

    async with _remote_backend_with_mirror(blocked / "insights.yaml", monkeypatch) as (backend, insights):
        report = await backend.persist_result(workspace="default", agent="research-agent", result=result)

    assert "- created: Retrieval drops context" in report
    assert "could not be written" in report
    assert list(insights.rows) == ["insight-remote-1"], "the platform write is the source of truth and must stand"


@pytest.mark.asyncio
async def test_make_analyst_backend_always_writes_to_the_platform(tmp_path: Path) -> None:
    """An output path adds a mirror; it never diverts writes off the platform."""
    async with _async_platform() as client:
        plain = make_analyst_backend(client=client, insights_output=None)
        mirrored = make_analyst_backend(client=client, insights_output=str(tmp_path / "insights.yaml"))

    assert isinstance(plain, RemoteAnalystBackend) and plain.mirror is None
    assert isinstance(mirrored, RemoteAnalystBackend)
    assert mirrored.mirror is not None and mirrored.mirror.path == tmp_path / "insights.yaml"


@pytest.mark.asyncio
async def test_local_only_is_evaluation_plumbing_and_requires_a_path(tmp_path: Path) -> None:
    """``local_only`` stays reachable for the evaluation, which no CLI flag sets."""
    async with _async_platform() as client:
        local = make_analyst_backend(
            client=client,
            insights_output=str(tmp_path / "insights.yaml"),
            local_only=True,
        )
        assert isinstance(local, LocalAnalystBackend)

        with pytest.raises(ValueError, match="requires an insights output path"):
            make_analyst_backend(client=client, insights_output=None, local_only=True)


@pytest.mark.asyncio
async def test_local_backend_reads_and_writes_insights_file_with_explicit_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_calls: list[dict[str, object]] = []
    write_calls: list[dict[str, object]] = []
    original_read_text = Path.read_text
    original_write_text = Path.write_text

    def spy_read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        read_calls.append({"encoding": encoding, "errors": errors})
        return original_read_text(self, encoding=encoding, errors=errors)

    def spy_write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        write_calls.append({"encoding": encoding, "errors": errors, "newline": newline})
        return original_write_text(self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "read_text", spy_read_text)
    monkeypatch.setattr(Path, "write_text", spy_write_text)

    async with _async_platform() as client:
        backend = LocalAnalystBackend(client=client, path=tmp_path / "insights.yaml")
        backend.store.write_records([])
        backend.store.read_records()

    assert write_calls[-1].get("encoding") == "utf-8"
    assert read_calls[-1].get("encoding") == "utf-8"


@pytest.mark.asyncio
async def test_local_backend_write_preserves_other_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "insights.yaml"
    path.write_text("metadata: retained\ninsights:\n- id: stale\n", encoding="utf-8")

    async with _async_platform() as client:
        backend = LocalAnalystBackend(client=client, path=path)
        backend.store.write_records([{"id": "insight-1"}])

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "metadata": "retained",
        "insights": [{"id": "insight-1"}],
    }


@pytest.mark.asyncio
async def test_local_backend_list_preserves_entity_metadata(tmp_path: Path) -> None:
    async with _async_platform() as client:
        backend = LocalAnalystBackend(client=client, path=tmp_path / "insights.yaml")
        backend.store.write_records(
            [
                {
                    "id": "insight-local-1",
                    "workspace": "default",
                    "title": "Repeated failure",
                    "description": "The agent repeats the same failure.",
                    "agent": "research-agent",
                    "status": "open",
                    "trace_refs": ["trace-1"],
                    "created_at": _STAMP.isoformat(),
                    "updated_at": _STAMP.isoformat(),
                }
            ]
        )
        page = await backend.list_insights(
            workspace="default",
            page=1,
            page_size=10,
            agent=None,
            status=None,
        )

    assert page.data[0].id == "insight-local-1"
    assert page.data[0].created_at == _STAMP
    assert page.data[0].updated_at == _STAMP


def test_merge_eval_filter_pins_evaluation_id() -> None:
    assert _merge_eval_filter({"agent_name": "a"}, evaluation_id="run-1") == {
        "agent_name": "a",
        "evaluation_id": "run-1",
    }


def test_merge_eval_filter_none_is_noop() -> None:
    assert _merge_eval_filter({"agent_name": "a"}, evaluation_id=None) == {"agent_name": "a"}
    assert _merge_eval_filter(None, evaluation_id=None) is None


def test_merge_eval_filter_overwrites_model_supplied_scope() -> None:
    assert _merge_eval_filter({"evaluation_id": "sneaky"}, evaluation_id="run-1") == {
        "evaluation_id": "run-1",
    }


@dataclass
class _SpanGroupsCall:
    workspace: str | None
    by: str
    filter: SpanFilterParam | dict[str, JsonValue] | None
    page: int | None
    page_size: int | None
    sort: str | None


class _SpanGroups:
    def __init__(self, *, data: list[SpanGroup], total: int) -> None:
        self.data = data
        self.total = total
        self.calls: list[_SpanGroupsCall] = []

    async def list(
        self,
        *,
        workspace: str | None = None,
        by: str,
        filter: SpanFilterParam | dict[str, JsonValue] | None,
        page: int | None,
        page_size: int | None,
        sort: str | None,
    ) -> SpanGroupsPage:
        self.calls.append(
            _SpanGroupsCall(
                workspace=workspace,
                by=by,
                filter=filter,
                page=page,
                page_size=page_size,
                sort=sort,
            )
        )
        current_page = page or 1
        size = page_size or len(self.data)
        return SpanGroupsPage(
            data=self.data,
            grouped_by=[by],
            pagination=PaginationData(
                page=current_page,
                page_size=size,
                current_page_size=len(self.data),
                total_pages=1,
                total_results=self.total,
            ),
        )


class _SpansWithGroups:
    def __init__(self, groups: _SpanGroups) -> None:
        self.groups = groups


class _IntakeWithGroups:
    def __init__(self, groups: _SpanGroups) -> None:
        self.spans = _SpansWithGroups(groups)


def _patch_intake_groups(monkeypatch: pytest.MonkeyPatch, groups: _SpanGroups) -> None:
    def fake_client_from_platform(platform: AsyncNeMoPlatform, client_cls: type[object]) -> object:
        del platform
        if client_cls is AsyncIntakeClient:
            return _IntakeWithGroups(groups)
        raise AssertionError(f"unexpected client type: {client_cls!r}")

    monkeypatch.setattr("nemo_insights_plugin.analyst.analyst_backend.client_from_platform", fake_client_from_platform)


@pytest.mark.asyncio
async def test_count_agent_sessions_uses_server_side_session_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    since = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)
    groups = _SpanGroups(data=[SpanGroup(group={"session_id": "session-1"}, span_count=3, started_at=_STAMP)], total=7)

    async with _async_platform() as client:
        _patch_intake_groups(monkeypatch, groups)
        backend = RemoteAnalystBackend(client)
        count = await backend.count_agent_sessions(
            agent="research-agent",
            workspace="default",
            since=since,
        )

    assert count == 7
    assert groups.calls == [
        _SpanGroupsCall(
            workspace="default",
            by="session_id",
            filter={
                "agent_name": "research-agent",
                "started_at": {"gte": "2026-06-04T12:00:00+00:00"},
            },
            page=1,
            page_size=1,
            sort="-span_count",
        )
    ]


@pytest.mark.asyncio
async def test_list_span_groups_fans_out_over_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    since = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)
    groups = _SpanGroups(
        data=[
            SpanGroup(group={"session_id": "session-1"}, span_count=12, started_at=_STAMP),
            SpanGroup(group={"session_id": "session-2"}, span_count=5, started_at=_STAMP),
        ],
        total=37,
    )

    async with _async_platform() as client:
        _patch_intake_groups(monkeypatch, groups)
        backend = RemoteAnalystBackend(client)
        result = await backend.list_span_groups(
            workspace="default",
            filter={"agent_name": "research-agent"},
            group_by="session_id",
            limit=100,
            since=since,
        )

    stamp = _STAMP.isoformat().replace("+00:00", "Z")
    assert result == {
        "groups": [
            {"group": {"session_id": "session-1"}, "span_count": 12, "started_at": stamp},
            {"group": {"session_id": "session-2"}, "span_count": 5, "started_at": stamp},
        ],
        "grouped_by": "session_id",
        "count": 2,
        "total": 37,
        "truncated": True,
    }
    assert groups.calls == [
        _SpanGroupsCall(
            workspace="default",
            by="session_id",
            filter={
                "agent_name": "research-agent",
                "started_at": {"gte": "2026-06-04T12:00:00+00:00"},
            },
            page=1,
            page_size=100,
            sort="-span_count",
        )
    ]


@pytest.mark.asyncio
async def test_count_agent_sessions_pins_evaluation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = _SpanGroups(data=[SpanGroup(group={"session_id": "session-1"}, span_count=3, started_at=_STAMP)], total=7)

    async with _async_platform() as client:
        _patch_intake_groups(monkeypatch, groups)
        backend = RemoteAnalystBackend(client)
        await backend.count_agent_sessions(
            agent="research-agent",
            workspace="default",
            evaluation_id="run-1",
        )

    assert groups.calls == [
        _SpanGroupsCall(
            workspace="default",
            by="session_id",
            filter={"agent_name": "research-agent", "evaluation_id": "run-1"},
            page=1,
            page_size=1,
            sort="-span_count",
        )
    ]


@pytest.mark.asyncio
async def test_list_span_groups_pins_evaluation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = _SpanGroups(
        data=[SpanGroup(group={"session_id": "session-1"}, span_count=3, started_at=_STAMP)],
        total=7,
    )

    async with _async_platform() as client:
        _patch_intake_groups(monkeypatch, groups)
        backend = RemoteAnalystBackend(client)
        await backend.list_span_groups(
            workspace="default",
            filter={"agent_name": "research-agent"},
            group_by="session_id",
            limit=100,
            evaluation_id="run-1",
        )

    assert groups.calls[0].filter == {
        "agent_name": "research-agent",
        "evaluation_id": "run-1",
    }


_DENVER = ZoneInfo("America/Denver")


def test_previous_scheduled_daily_converts_local_hour_to_utc() -> None:
    # During MDT (UTC-6), 02:00 Denver local is 08:00 UTC.
    now = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)

    scheduled = previous_scheduled(
        now,
        frequency=Frequency.DAILY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )

    assert scheduled == datetime(2026, 6, 10, 8, tzinfo=timezone.utc)


def test_previous_scheduled_daily_rolls_back_when_hour_not_reached() -> None:
    # 03:00 UTC on 2026-06-10 is 21:00 Denver on 2026-06-09, before 02:00 local,
    # so the most recent 02:00-local run was the prior day.
    now = datetime(2026, 6, 10, 3, tzinfo=timezone.utc)

    scheduled = previous_scheduled(
        now,
        frequency=Frequency.DAILY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )

    assert scheduled == datetime(2026, 6, 9, 8, tzinfo=timezone.utc)


def test_previous_scheduled_weekly_lands_on_configured_weekday() -> None:
    # 2026-06-10 is a Wednesday; the prior Monday 02:00 Denver is 2026-06-08.
    now = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)

    scheduled = previous_scheduled(
        now,
        frequency=Frequency.WEEKLY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )

    assert scheduled == datetime(2026, 6, 8, 8, tzinfo=timezone.utc)
    assert scheduled.astimezone(_DENVER).weekday() == int(Weekday.MONDAY)


def test_is_due_true_when_no_prior_run() -> None:
    now = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)

    assert is_due(
        now,
        None,
        frequency=Frequency.DAILY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )


def test_is_due_false_when_run_after_last_scheduled() -> None:
    now = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)
    anchor = datetime(2026, 6, 10, 9, tzinfo=timezone.utc)  # after 08:00 UTC slot

    assert not is_due(
        now,
        anchor,
        frequency=Frequency.DAILY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )


def test_is_due_true_when_run_before_last_scheduled() -> None:
    now = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)
    anchor = datetime(2026, 6, 9, 9, tzinfo=timezone.utc)  # prior day's run

    assert is_due(
        now,
        anchor,
        frequency=Frequency.DAILY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )


def test_is_due_treats_naive_anchor_as_utc() -> None:
    now = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)
    anchor = datetime(2026, 6, 10, 9)  # naive, after the 08:00 UTC slot

    assert not is_due(
        now,
        anchor,
        frequency=Frequency.DAILY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )


def test_weekly_not_due_until_configured_weekday() -> None:
    # Sunday 2026-06-07 12:00 UTC; the upcoming Monday slot has not passed, so
    # the most recent scheduled run is the previous Monday (2026-06-01).
    now = datetime(2026, 6, 7, 12, tzinfo=timezone.utc)
    anchor = datetime(2026, 6, 2, tzinfo=timezone.utc)  # after 2026-06-01 slot

    assert not is_due(
        now,
        anchor,
        frequency=Frequency.WEEKLY,
        run_at_hour=2,
        run_on_weekday=int(Weekday.MONDAY),
        tz=_DENVER,
    )


def test_config_accepts_weekday_name() -> None:
    config = AnalystSchedulerConfig.model_validate({"run_on_weekday": "friday"})

    assert config.run_on_weekday is Weekday.FRIDAY


def test_config_rejects_unknown_timezone() -> None:
    with pytest.raises(ValidationError):
        AnalystSchedulerConfig(timezone="Mars/Olympus_Mons")


class _Results(JobResults):
    def __init__(self) -> None:
        self.saved: list[tuple[str, Path]] = []

    def save(
        self,
        name: str,
        local_path: str | Path,
        *,
        ignore_patterns: list[str] | str | None = None,
    ) -> ResultRef:
        del ignore_patterns
        path = Path(local_path)
        self.saved.append((name, path))
        return ResultRef(name=name, artifact_url=f"file://{path}")


class _RecordingAnalysisRunStatuses:
    def __init__(self) -> None:
        self.updates: list[UpdateAnalysisRunStatusRequest] = []

    def update(
        self,
        *,
        workspace: str,
        agent: str,
        status: AnalysisConfigStatus | str | None = None,
        last_successful_run_at: datetime | None = None,
        last_attempted_at: datetime | None = None,
        last_completed_at: datetime | None = None,
        last_submitted_job: str | None = None,
        last_error: str | None = None,
    ) -> AnalysisRunStatus:
        resolved_status = AnalysisConfigStatus(status) if isinstance(status, str) else status
        update = UpdateAnalysisRunStatusRequest(
            status=resolved_status,
            last_successful_run_at=last_successful_run_at,
            last_attempted_at=last_attempted_at,
            last_completed_at=last_completed_at,
            last_submitted_job=last_submitted_job,
            last_error=last_error,
        )
        self.updates.append(update)
        row = AnalysisRunStatus(
            name=agent,
            workspace=workspace,
            agent=agent,
            status=update.status or AnalysisConfigStatus.IDLE,
            last_successful_run_at=update.last_successful_run_at,
            last_attempted_at=update.last_attempted_at,
            last_completed_at=update.last_completed_at,
            last_submitted_job=update.last_submitted_job or "",
            last_error=update.last_error or "",
        )
        row._id = "analysis-run-status-1"
        row._created_at = _STAMP
        row._updated_at = _STAMP
        row._db_version = 1
        return row


def _ctx(tmp_path: Path) -> JobContext:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=_Results(),
        job_id="insights-job-1",
    )


def _analyze_spec(agent: str = "research-agent") -> AnalyzeSpec:
    return AnalyzeSpec(
        agent=agent,
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
    )


def test_analyze_job_records_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    async_client = object()

    async def fake_run_analyst(**kwargs: object) -> str:
        calls.append(kwargs)
        return "analysis report"

    monkeypatch.setattr("nemo_insights_plugin.jobs.analyze.run_analyst", fake_run_analyst)
    monkeypatch.setattr(
        "nemo_insights_plugin.jobs.analyze.get_async_task_sdk",
        lambda plugin: async_client,
    )
    statuses = _RecordingAnalysisRunStatuses()
    with NeMoPlatform(base_url=_BASE_URL) as sdk:
        monkeypatch.setattr(sdk.insights.analysis_run_statuses, "update", statuses.update)
        result = AnalyzeJob().run(
            _analyze_spec().model_dump(mode="json"),
            ctx=_ctx(tmp_path),
            sdk=sdk,
        )

    assert result["status"] == "completed"
    assert result["artifact"] == {
        "name": "analysis-report",
        "artifact_url": f"file://{tmp_path / 'persistent' / 'analysis-report.txt'}",
    }
    updates = statuses.updates
    assert [u.status for u in updates] == [
        AnalysisConfigStatus.RUNNING,
        AnalysisConfigStatus.IDLE,
    ]
    assert updates[-1].last_submitted_job == "insights-job-1"
    assert (tmp_path / "persistent" / "analysis-report.txt").read_text() == "analysis report"
    assert calls[0]["client"] is async_client
    assert calls[0]["model_refs"] == ConfiguredModelRefs(
        default="default/gpt-5",
        fast="default/gpt-5-mini",
    )


@pytest.mark.asyncio
async def test_analyze_job_compile_requests_storage_without_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "must-not-be-forwarded")
    platform_spec = await AnalyzeJob.compile(
        workspace="default",
        spec=_analyze_spec(),
        entity_client=object(),
        job_name="opt-analyze-default-research-agent-20260608204901",
        async_sdk=object(),
    )

    step = next(iter(platform_spec["steps"]))
    assert [env.model_dump(exclude_none=True) for env in step["environment"]] == [
        {
            "name": PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            "value": DEFAULT_JOB_STORAGE_PATH,
        },
    ]


class _RecordingAnalysisJobs:
    def __init__(
        self,
        *,
        jobs: list[AnalysisJob] | None = None,
    ) -> None:
        self.created: list[CreateAnalysisJobRequest] = []
        self.query_params: list[ListAnalysisJobsQueryParams | None] = []
        self._listed_jobs = list(jobs or [])

    async def list_analysis_jobs(
        self,
        *,
        workspace: str | None = None,
        query_params: ListAnalysisJobsQueryParams | None = None,
    ) -> _AsyncItems[AnalysisJob]:
        del workspace
        self.query_params.append(query_params)
        return _AsyncItems(self._listed_jobs)

    async def create_analysis_job(
        self,
        *,
        workspace: str | None = None,
        body: CreateAnalysisJobRequest,
    ) -> _TypedResponse[AnalysisJob]:
        del workspace
        self.created.append(body)
        job = AnalysisJob(
            name=body.name or "analysis-job",
            spec=body.spec,
            custom_fields=body.custom_fields,
        )
        return _TypedResponse(job)


class _RunStatusLookup:
    def __init__(self, run_status: AnalysisRunStatus | None) -> None:
        self._run_status = run_status

    async def get(
        self,
        entity_type: type[AnalysisRunStatus],
        name: str,
        *,
        workspace: str | None = None,
        parent: str | None = None,
    ) -> AnalysisRunStatus:
        del entity_type, name, workspace, parent
        if self._run_status is None:
            raise NemoEntityNotFoundError("missing")
        return self._run_status


@asynccontextmanager
async def _controller(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jobs: list[AnalysisJob] | None = None,
    run_status: AnalysisRunStatus | None = None,
) -> AsyncIterator[tuple[InsightsAnalysisController, _RecordingAnalysisJobs]]:
    controller = InsightsAnalysisController()
    controller._config = InsightsConfig(
        analyst=AnalystSchedulerConfig(
            frequency=Frequency.DAILY,
            run_at_hour=0,
            job_profile="test-profile",
        )
    )
    async with _async_platform() as sdk:
        jobs_client = client_from_platform(sdk, AsyncAnalysisJobsClient)
        recording_jobs = _RecordingAnalysisJobs(jobs=jobs)
        entities = NemoEntitiesClient(client_from_platform(sdk, AsyncEntitiesClient))
        monkeypatch.setattr(jobs_client, "list_analysis_jobs", recording_jobs.list_analysis_jobs)
        monkeypatch.setattr(jobs_client, "create_analysis_job", recording_jobs.create_analysis_job)
        monkeypatch.setattr(entities, "get", _RunStatusLookup(run_status).get)
        controller._sdk = sdk
        controller._jobs = jobs_client
        controller._entities = entities
        yield controller, recording_jobs


def test_generated_job_name_fits_derived_fileset_name_limit() -> None:
    config = AnalysisConfig(
        name="research-agent-with-a-very-long-name",
        workspace="default",
        agent="research-agent-with-a-very-long-name",
    )

    name = _job_name(config, datetime(2026, 6, 8, 20, 31, 22, tzinfo=timezone.utc))

    assert name.startswith("opt-analyze-default-")
    assert len(name) <= 63 - len("job-fileset-")
    assert len(f"job-fileset-{name}") <= 63


@pytest.mark.asyncio
async def test_controller_submits_due_job(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AnalysisConfig(
        name="research-agent",
        workspace="default",
        agent="research-agent",
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
    )

    async with _controller(monkeypatch) as (controller, jobs):
        await controller._reconcile_config(config)

    assert len(jobs.created) == 1
    created = jobs.created[0]
    created_spec = created.spec
    assert created.name is not None
    assert created.name.startswith("opt-analyze-default-research-agent-")
    assert created.custom_fields == {"insights_analysis_agent": "research-agent"}
    assert created_spec.agent == "research-agent"
    assert created_spec.since is None
    assert created_spec.default_model == "default/gpt-5"
    assert created_spec.fast_model == "default/gpt-5-mini"


@pytest.mark.asyncio
async def test_controller_defers_legacy_config_without_persisted_models(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = AnalysisConfig(name="research-agent", workspace="default", agent="research-agent")

    async with _controller(monkeypatch) as (controller, jobs):
        with caplog.at_level("ERROR", logger="nemo_insights_plugin.controller"):
            await controller._reconcile_config(config)

    assert jobs.created == []
    assert "has no model selection" in caplog.text
    assert "nemo insights analysis enable" in caplog.text


@pytest.mark.asyncio
async def test_controller_skips_active_job(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AnalysisConfig(
        name="research-agent",
        workspace="default",
        agent="research-agent",
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
    )

    async with _controller(
        monkeypatch,
        jobs=[
            AnalysisJob(
                name="existing-analysis-job",
                spec=_analyze_spec(),
                status=PlatformJobStatus.ACTIVE,
                custom_fields={"insights_analysis_agent": "research-agent"},
            )
        ],
    ) as (controller, jobs):
        await controller._reconcile_config(config)

    assert jobs.created == []
