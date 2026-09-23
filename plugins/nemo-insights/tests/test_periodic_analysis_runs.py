# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery of the single scheduled attempt and active-job overlap checks."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError, raise_for_status
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_insights_plugin.config import InsightsConfig
from nemo_insights_plugin.controller import InsightsAnalysisController
from nemo_insights_plugin.entities import AnalysisConfig, AnalysisConfigStatus, AnalysisRunStatus

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


def _controller(job_status: HelixJobStatus = HelixJobStatus.COMPLETED):
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
@pytest.mark.parametrize("previous_cursor", [PREVIOUS, None])
async def test_success_without_attempt_timestamp_preserves_cursor(previous_cursor: datetime | None) -> None:
    controller, entities, _ = _controller()
    pending = _pending().model_copy(update={"last_attempted_at": None, "last_successful_run_at": previous_cursor})
    status = await controller._reconcile_run(_config(), pending)
    assert status.status == AnalysisConfigStatus.IDLE
    assert status.last_successful_run_at == previous_cursor
    assert status.last_completed_at is not None
    entities.update.assert_awaited_once_with(status)


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status", HelixJobStatus.non_terminals())
async def test_active_job_leaves_pending_attempt_unchanged(job_status) -> None:
    controller, entities, _ = _controller(job_status)
    pending = _pending()
    assert await controller._reconcile_run(_config(), pending) is pending
    entities.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status", [HelixJobStatus.ERROR, HelixJobStatus.CANCELLED])
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
    query = json.loads(jobs.list_jobs.await_args.kwargs["query_params"]["filter"])
    assert set(query["status"]["$in"]) == {status.value for status in HelixJobStatus.non_terminals()}
    assert query["source"] == {"$in": ["nemo-agents-plugin-execute", "insights"]}
    jobs.get_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_overlap_query_failure_defers_submission() -> None:
    controller, _, jobs = _controller()
    jobs.list_jobs.side_effect = RuntimeError("Jobs unavailable")
    assert await controller._has_active_job(_config())


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status", [HelixJobStatus.COMPLETED, HelixJobStatus.ACTIVE])
@pytest.mark.parametrize("agent_tag", ["another-agent", None])
async def test_wrong_or_missing_agent_tag_preserves_cursor(job_status, agent_tag) -> None:
    controller, entities, jobs = _controller(job_status)
    jobs.get_job.return_value.data().custom_fields = (
        {"insights_analysis_agent": agent_tag} if agent_tag is not None else None
    )
    status = await controller._reconcile_run(_config(), _pending())
    assert status.status == AnalysisConfigStatus.ERROR
    assert status.last_successful_run_at == PREVIOUS
    assert status.last_error == "Reconciled job did not belong to this agent"
    entities.update.assert_awaited_once_with(status)


@pytest.mark.asyncio
async def test_same_prefix_agents_submitted_together_reconcile_only_their_own_jobs(monkeypatch) -> None:
    controller, entities, jobs = _controller()
    attempts = {}
    submitted_jobs = {}

    async def save(status):
        attempts[status.agent] = status
        return status

    async def submit(**kwargs):
        agent = kwargs["request"].agent
        name = kwargs["name"]
        assert attempts[agent].last_submitted_job == name
        assert name not in submitted_jobs
        submitted_jobs[name] = MagicMock(
            status=HelixJobStatus.COMPLETED, custom_fields={"insights_analysis_agent": agent}
        )

    async def get_job(*, workspace, name):
        return MagicMock(data=lambda: submitted_jobs[name])

    entities.create.side_effect = save
    jobs.get_job.side_effect = get_job
    monkeypatch.setattr("nemo_insights_plugin.controller.submit_analysis_run", submit)
    configs = [
        _config().model_copy(update={"name": agent, "agent": agent})
        for agent in ("research-agent-shared-v1", "research-agent-shared-v2")
    ]
    for config in configs:
        await controller._submit_analysis_job(config, None, NOW)
    assert len(submitted_jobs) == 2
    for config in configs:
        pending = attempts[config.agent]
        other = next(value for agent, value in attempts.items() if agent != config.agent)
        mismatched = pending.model_copy(update={"last_submitted_job": other.last_submitted_job})
        rejected = await controller._reconcile_run(config, mismatched)
        assert rejected.status == AnalysisConfigStatus.ERROR
        assert rejected.last_successful_run_at is None
        completed = await controller._reconcile_run(config, pending)
        assert completed.status == AnalysisConfigStatus.IDLE
        assert completed.last_successful_run_at == NOW


@pytest.mark.asyncio
async def test_config_listing_includes_enabled_and_disabled_configs_beyond_first_page() -> None:
    controller, entities, _ = _controller()
    first_page = [
        _config().model_copy(update={"name": f"disabled-{i}", "agent": f"disabled-{i}", "enabled": False})
        for i in range(100)
    ]
    enabled = _config().model_copy(update={"workspace": "another-workspace", "enabled": True})
    disabled = _config().model_copy(update={"enabled": False})
    entities.list.side_effect = [
        MagicMock(data=first_page, pagination=MagicMock(total_pages=2)),
        MagicMock(data=[enabled, disabled], pagination=MagicMock(total_pages=2)),
    ]
    configs = await controller.list_objects()
    assert configs == [*first_page, enabled, disabled]
    assert [call.kwargs for call in entities.list.await_args_list] == [
        {"workspace": "-", "page": 1, "page_size": 100},
        {"workspace": "-", "page": 2, "page_size": 100},
    ]
    # A disabled config on page two must still complete its pending bookkeeping.
    controller._has_active_job = AsyncMock(return_value=False)
    controller._submit_analysis_job = AsyncMock()
    entities.get.return_value = _pending()
    await controller.reconcile_one(configs[-1])
    assert entities.update.await_args.args[0].status == AnalysisConfigStatus.IDLE
    controller._submit_analysis_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_config_listing_does_not_return_partial_results_on_page_failure() -> None:
    controller, entities, _ = _controller()
    entities.list.side_effect = [
        MagicMock(data=[_config()], pagination=MagicMock(total_pages=2)),
        RuntimeError("Entities unavailable"),
    ]
    assert await controller.list_objects() == []
    assert entities.list.await_count == 2


@pytest.mark.asyncio
async def test_empty_config_listing_stops_after_first_page() -> None:
    controller, entities, _ = _controller()
    entities.list.return_value = MagicMock(data=[], pagination=MagicMock(total_pages=0))
    assert await controller.list_objects() == []
    entities.list.assert_awaited_once_with(AnalysisConfig, workspace="-", page=1, page_size=100)
