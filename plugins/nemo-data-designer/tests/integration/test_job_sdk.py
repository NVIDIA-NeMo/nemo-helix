# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Data Designer job-resource lifecycle.

These tests exercise the end-user flow of inspecting a Data Designer job's
status/logs and downloading its artifacts — all the way through the in-process
FastAPI app, the Jobs core service, and the local file backend.

A note on ``get_job_status``: ``task_context`` invokes ``CreateJob.run(...)``
directly, which writes results and emits logs but does **not** drive the Jobs
service controller that would normally roll a step's terminal status up to
the job's ``completed`` state. We patch ``get_job_status`` on the resource as
the (single) network seam to bridge that gap. Everything past that seam — the
``_WaitLogCollector`` filtering, ``_status_is_complete`` branching, log-level
routing, the result download tarball extraction — runs unmocked against the
real services.
"""

from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import data_designer.config as dd
import nemo_data_designer_plugin.testing.utils as u
import pandas as pd
import pytest
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from nemo_data_designer_plugin.jobs.spec import DataDesignerJobConfig
from nemo_data_designer_plugin.sdk.errors import DataDesignerClientError, DataDesignerJobError
from nemo_data_designer_plugin.sdk.job_resources import (
    AsyncDataDesignerJobResource,
    DataDesignerJobResource,
)
from nemo_data_designer_plugin.sdk.job_results import DataDesignerJobResults
from nemo_data_designer_plugin.sdk.resources import AsyncDataDesignerResource, DataDesignerResource

pytestmark = pytest.mark.integration

_JOB_NAME = "data-designer-abc123"


def _make_basic_job_config() -> DataDesignerJobConfig:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a"]),
        )
    )
    return DataDesignerJobConfig(num_records=3, config=builder.build())


@pytest.fixture
async def completed_job() -> AsyncGenerator[u.CreateJobTestContext]:
    """Stand up a real job and run it to completion (writes results, emits logs)."""
    job_config = await u.compile_create_job(_make_basic_job_config(), workspace="default")
    async with u.task_context(job_config, _JOB_NAME) as ctx:
        result = ctx.run_task()
        assert result.exit_code == 0, "task did not complete successfully"
        yield ctx


@pytest.fixture
async def pending_job() -> AsyncGenerator[u.CreateJobTestContext]:
    """Stand up a real job without running it (no results populated)."""
    job_config = await u.compile_create_job(_make_basic_job_config(), workspace="default")
    async with u.task_context(job_config, _JOB_NAME) as ctx:
        yield ctx


@pytest.fixture
def completed_job_resource(completed_job: u.CreateJobTestContext) -> DataDesignerJobResource:
    return DataDesignerResource(completed_job.client).get_job_resource(_JOB_NAME, workspace="default")


@pytest.fixture
async def async_completed_job_resource(completed_job: u.CreateJobTestContext) -> AsyncDataDesignerJobResource:
    return await AsyncDataDesignerResource(completed_job.async_client).get_job_resource(_JOB_NAME, workspace="default")


@pytest.fixture
def pending_job_resource(pending_job: u.CreateJobTestContext) -> DataDesignerJobResource:
    return DataDesignerResource(pending_job.client).get_job_resource(_JOB_NAME, workspace="default")


@contextmanager
def _patch_status(resource: DataDesignerJobResource, status: str) -> Generator[None]:
    """Mock the resource's network status call to return a specific platform-job status."""
    with patch.object(resource, "get_job_status", return_value=status):
        yield


@contextmanager
def _patch_async_status(resource: AsyncDataDesignerJobResource, status: str) -> Generator[None]:
    with patch.object(resource, "get_job_status", new=AsyncMock(return_value=status)):
        yield


@contextmanager
def _no_pause() -> Generator[None]:
    """Skip ``time.sleep`` calls inside ``wait_until_done`` so tests stay fast."""
    with (
        patch("nemo_data_designer_plugin.sdk.job_resources._pause"),
        patch("nemo_data_designer_plugin.sdk.job_resources._async_pause"),
    ):
        yield


# ---------------------------------------------------------------------------
# get_job_resource / get_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_resource_returns_job_for_real_job(completed_job_resource: DataDesignerJobResource) -> None:
    assert isinstance(completed_job_resource, DataDesignerJobResource)

    job = completed_job_resource.get_job()
    assert job["name"] == _JOB_NAME


@pytest.mark.asyncio
async def test_get_job_resource_async_returns_job_for_real_job(
    async_completed_job_resource: AsyncDataDesignerJobResource,
) -> None:
    assert isinstance(async_completed_job_resource, AsyncDataDesignerJobResource)

    job = await async_completed_job_resource.get_job()
    assert job["name"] == _JOB_NAME


# ---------------------------------------------------------------------------
# check_if_complete / _status_is_complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_if_complete_returns_true_for_completed_status(
    pending_job_resource: DataDesignerJobResource,
) -> None:
    with _patch_status(pending_job_resource, "completed"):
        assert pending_job_resource.check_if_complete() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("simulated_status", "expected_log_fragment"),
    [
        ("active", "still running"),
        ("created", "still in the queue"),
        ("pending", "still in the queue"),
        ("error", "stopped with status `error`"),
        ("cancelled", "stopped with status `cancelled`"),
        ("frobnicated", "unknown state"),
    ],
)
async def test_check_if_complete_returns_false_with_friendly_message_for_non_completed_statuses(
    simulated_status: str,
    expected_log_fragment: str,
    caplog: pytest.LogCaptureFixture,
    pending_job_resource: DataDesignerJobResource,
) -> None:
    """Non-raising path should log a user-friendly message for every non-completed status."""

    with _patch_status(pending_job_resource, simulated_status), caplog.at_level("WARNING"):
        assert pending_job_resource.check_if_complete(raise_if_not_complete=False) is False

    assert any(expected_log_fragment in record.message for record in caplog.records), (
        f"expected log message containing {expected_log_fragment!r}, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("simulated_status", ["active", "created", "pending", "error", "cancelled", "frobnicated"])
async def test_check_if_complete_raises_when_requested(
    simulated_status: str, pending_job_resource: DataDesignerJobResource
) -> None:
    with _patch_status(pending_job_resource, simulated_status):
        with pytest.raises(DataDesignerJobError):
            pending_job_resource.check_if_complete(raise_if_not_complete=True)


# ---------------------------------------------------------------------------
# wait_until_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_until_done_logs_success_when_status_completes(
    caplog: pytest.LogCaptureFixture, completed_job_resource: DataDesignerJobResource
) -> None:
    with _no_pause(), _patch_status(completed_job_resource, "completed"), caplog.at_level("INFO"):
        completed_job_resource.wait_until_done()

    assert any("completed successfully" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_wait_until_done_async_logs_success_when_status_completes(
    caplog: pytest.LogCaptureFixture, async_completed_job_resource: AsyncDataDesignerJobResource
) -> None:
    with _no_pause(), _patch_async_status(async_completed_job_resource, "completed"), caplog.at_level("INFO"):
        await async_completed_job_resource.wait_until_done()

    assert any("completed successfully" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_wait_until_done_logs_terminal_failure_for_cancelled_status(
    caplog: pytest.LogCaptureFixture,
    pending_job_resource: DataDesignerJobResource,
) -> None:
    with _no_pause(), _patch_status(pending_job_resource, "cancelled"), caplog.at_level("ERROR"):
        pending_job_resource.wait_until_done()

    assert any("Terminating generation job" in record.message for record in caplog.records)
    assert any("cancelled" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# get_logs
#
# Note: ``DataDesignerJobResource.get_logs`` paginates through Job logs returned
# by the Files service's OTLP endpoint. ``task_context`` runs ``CreateJob.run``
# in-process and bypasses the OTLP log-capture pipeline a real container runner
# would populate, so ``get_logs`` always returns ``[]`` here regardless of what
# the task emitted.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# download_artifacts (sync + async) and DataDesignerJobResults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_artifacts_extracts_dataset_and_loads_analysis(
    tmp_path: Path, completed_job_resource: DataDesignerJobResource
) -> None:
    with _patch_status(completed_job_resource, "completed"):
        results = completed_job_resource.download_artifacts(tmp_path)

    assert isinstance(results, DataDesignerJobResults)

    dataset = results.load_dataset()
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) == 3
    assert dataset["foo"].tolist() == ["a", "a", "a"]

    analysis = results.load_analysis()
    assert isinstance(analysis, DatasetProfilerResults)
    assert analysis.num_records == 3


@pytest.mark.asyncio
async def test_download_artifacts_async_extracts_dataset_and_loads_analysis(
    tmp_path: Path, async_completed_job_resource: AsyncDataDesignerJobResource
) -> None:
    with _patch_async_status(async_completed_job_resource, "completed"):
        results = await async_completed_job_resource.download_artifacts(tmp_path)

    assert isinstance(results, DataDesignerJobResults)
    assert results.load_analysis().num_records == 3


@pytest.mark.asyncio
async def test_load_processor_dataset_raises_for_unknown_processor(
    tmp_path: Path, completed_job_resource: DataDesignerJobResource
) -> None:
    with _patch_status(completed_job_resource, "completed"):
        results = completed_job_resource.download_artifacts(tmp_path)

    with pytest.raises(DataDesignerClientError, match="No artifacts found for processor"):
        results.load_processor_dataset("undefined-processor")


# ---------------------------------------------------------------------------
# load_analysis (resource-level) and _check_if_result_available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_analysis_returns_profiler_results_for_completed_status(
    completed_job_resource: DataDesignerJobResource,
) -> None:
    with _patch_status(completed_job_resource, "completed"):
        analysis = completed_job_resource.load_analysis()

    assert isinstance(analysis, DatasetProfilerResults)
    assert analysis.num_records == 3


@pytest.mark.asyncio
async def test_load_analysis_raises_when_status_is_unknown(pending_job_resource: DataDesignerJobResource) -> None:
    with _patch_status(pending_job_resource, "frobnicated"):
        with pytest.raises(DataDesignerJobError, match="frobnicated"):
            pending_job_resource.load_analysis()


@pytest.mark.asyncio
async def test_load_analysis_when_active_uses_completed_result_if_available(
    caplog: pytest.LogCaptureFixture,
    completed_job_resource: DataDesignerJobResource,
) -> None:
    """``_check_if_result_available`` allows fetching completed results from an ``active`` job
    and emits a 'still cooking' info message.
    """
    with _patch_status(completed_job_resource, "active"), caplog.at_level("INFO"):
        analysis = completed_job_resource.load_analysis()

    assert analysis.num_records == 3
    assert any("still cooking" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_load_analysis_when_terminally_incomplete_warns_and_returns_partial(
    caplog: pytest.LogCaptureFixture,
    completed_job_resource: DataDesignerJobResource,
) -> None:
    with _patch_status(completed_job_resource, "error"), caplog.at_level("WARNING"):
        analysis = completed_job_resource.load_analysis()

    assert analysis.num_records == 3
    assert any("error" in record.message and "analysis" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_load_analysis_raises_friendly_error_when_active_but_result_missing(
    pending_job_resource: DataDesignerJobResource,
) -> None:
    """An ``active`` job whose analysis result hasn't been written yet returns a 404 from the
    Jobs service; ``_check_if_result_available`` translates that into a friendly
    ``"'analysis' result is not available."`` message instead of leaking the underlying
    HTTP error.
    """
    with _patch_status(pending_job_resource, "active"):
        with pytest.raises(DataDesignerJobError, match="'analysis' result is not available"):
            pending_job_resource.load_analysis()
