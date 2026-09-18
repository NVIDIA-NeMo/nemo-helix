# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery of the single scheduled attempt and active-job overlap checks."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_insights_plugin.config import InsightsConfig
from nemo_insights_plugin.controller import InsightsAnalysisController
from nemo_insights_plugin.entities import AnalysisConfig, AnalysisConfigStatus, AnalysisRunStatus
from nemo_platform_plugin.client.errors import NotFoundError, raise_for_status
from nemo_platform_plugin.entity_client import NemoEntitiesClient
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus

NOW = datetime(2026, 9, 18, 12, tzinfo=timezone.utc)
PREVIOUS = NOW - timedelta(days=1)


def _config() -> AnalysisConfig:
    return AnalysisConfig(name="demo", agent="demo", workspace="default", default_model="default/model")


def _pending() -> AnalysisRunStatus:
    return AnalysisRunStatus(
        name="demo",
        agent="demo",
        workspace="default",
        status=AnalysisConfigStatus.RUNNING,
        last_submitted_job="scheduled-job",
        last_attempted_at=NOW,
        last_successful_run_at=PREVIOUS,
    )


def _controller(job_status: PlatformJobStatus = PlatformJobStatus.COMPLETED):
    controller = InsightsAnalysisController()
    entities = AsyncMock(spec=NemoEntitiesClient)
    entities.update.side_effect = lambda status: status
    jobs = AsyncMock(spec=AsyncJobsClient)
    job = MagicMock(status=job_status, custom_fields={"insights_analysis_agent": "demo"})
    jobs.get_job.return_value = MagicMock(data=lambda: job)
    controller._entities = entities
    controller._jobs = jobs
    controller._sdk = MagicMock()
    controller._config = InsightsConfig()
    return controller, entities, jobs


@pytest.mark.asyncio
async def test_success_reads_only_tracked_job_and_advances_to_attempt_start() -> None:
    controller, entities, jobs = _controller()
    pending = _pending()
    status = await controller._reconcile_run(_config(), pending)
    assert status.status == AnalysisConfigStatus.IDLE
    assert status.last_successful_run_at == NOW
    assert status.last_completed_at is not None
    jobs.get_job.assert_awaited_once_with(workspace="default", name="scheduled-job")
    entities.list.assert_not_awaited()
    assert pending.status == AnalysisConfigStatus.RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status", PlatformJobStatus.non_terminals())
async def test_active_job_leaves_pending_attempt_unchanged(job_status) -> None:
    controller, entities, _ = _controller(job_status)
    pending = _pending()
    assert await controller._reconcile_run(_config(), pending) is pending
    entities.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status", [PlatformJobStatus.ERROR, PlatformJobStatus.CANCELLED])
async def test_failed_or_cancelled_job_preserves_cursor(job_status) -> None:
    controller, _, _ = _controller(job_status)
    status = await controller._reconcile_run(_config(), _pending())
    assert status.status == AnalysisConfigStatus.ERROR
    assert status.last_successful_run_at == PREVIOUS
    assert job_status.value in status.last_error


@pytest.mark.asyncio
async def test_missing_job_records_failure_without_advancing_cursor() -> None:
    controller, _, jobs = _controller()
    response = httpx.Response(404, request=httpx.Request("GET", "https://platform/jobs/scheduled-job"))
    with pytest.raises(NotFoundError) as exc:
        raise_for_status(response)
    jobs.get_job.side_effect = exc.value
    status = await controller._reconcile_run(_config(), _pending())
    assert status.status == AnalysisConfigStatus.ERROR
    assert status.last_successful_run_at == PREVIOUS


@pytest.mark.asyncio
async def test_read_failure_keeps_pending_attempt_for_next_poll() -> None:
    controller, entities, jobs = _controller()
    jobs.get_job.side_effect = RuntimeError("Jobs unavailable")
    with pytest.raises(RuntimeError, match="Jobs unavailable"):
        await controller._reconcile_run(_config(), _pending())
    entities.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_finished_attempt_is_not_read_again() -> None:
    controller, entities, jobs = _controller()
    status = await controller._reconcile_run(_config(), _pending())
    jobs.get_job.reset_mock()
    entities.update.reset_mock()
    assert await controller._reconcile_run(_config(), status) is status
    assert await controller._reconcile_run(_config(), None) is None
    jobs.get_job.assert_not_awaited()
    entities.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_disable_still_records_completion_without_new_submission() -> None:
    controller, entities, _ = _controller()
    controller._has_active_job = AsyncMock(return_value=False)
    entities.get.return_value = _pending()
    controller._submit_analysis_job = AsyncMock()
    await controller._reconcile_config(_config().model_copy(update={"enabled": False}))
    assert entities.update.await_args.args[0].status == AnalysisConfigStatus.IDLE
    controller._submit_analysis_job.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_name_is_persisted_before_submission_even_when_submission_times_out(monkeypatch, existing) -> None:
    controller, entities, _ = _controller()
    events = []

    async def save(status):
        events.append("save")
        assert status.status == AnalysisConfigStatus.RUNNING
        assert status.last_submitted_job
        assert status.last_attempted_at == NOW
        assert status.last_completed_at is None
        return status

    async def submit(**kwargs):
        events.append("submit")
        saved = (entities.update if existing else entities.create).await_args.args[0]
        assert kwargs["name"] == saved.last_submitted_job
        raise TimeoutError("Response lost")

    entities.create.side_effect = save
    entities.update.side_effect = save
    monkeypatch.setattr("nemo_insights_plugin.controller.submit_analysis_run", submit)
    with pytest.raises(TimeoutError):
        await controller._submit_analysis_job(_config(), _pending() if existing else None, NOW)
    assert events == ["save", "submit"]


@pytest.mark.asyncio
async def test_failed_status_write_prevents_submission(monkeypatch) -> None:
    controller, entities, _ = _controller()
    entities.create.side_effect = RuntimeError("Store unavailable")
    submit = AsyncMock()
    monkeypatch.setattr("nemo_insights_plugin.controller.submit_analysis_run", submit)
    with pytest.raises(RuntimeError, match="Store unavailable"):
        await controller._submit_analysis_job(_config(), None, NOW)
    submit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent, expected", [("demo", True), ("another-agent", False)])
async def test_overlap_query_only_reads_active_jobs_and_matches_agent(agent, expected) -> None:
    controller, _, jobs = _controller()

    async def items():
        yield MagicMock(custom_fields={"insights_analysis_agent": agent})

    jobs.list_jobs.return_value = MagicMock(items=items)
    assert await controller._has_active_job(_config()) is expected
    assert '"status"' in jobs.list_jobs.await_args.kwargs["query_params"]["filter"]
    jobs.get_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_overlap_query_failure_defers_submission() -> None:
    controller, _, jobs = _controller()
    jobs.list_jobs.side_effect = RuntimeError("Jobs unavailable")
    assert await controller._has_active_job(_config())
