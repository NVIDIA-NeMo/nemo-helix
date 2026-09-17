# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Data Designer job *task* — the in-container worker
that the Jobs service invokes.

These tests exercise ``nemo_data_designer_plugin.jobs.run`` end-to-end via
``task_context``: they run the task in-process, then verify the produced
results by reading them back through the high-level SDK
(``DataDesignerJobResource`` + ``DataDesignerJobResults``), which is the
recommended way for users to consume job artifacts.

A note on patching ``get_job_status``: ``task_context`` invokes the task
directly and does not drive the Jobs service controller that would normally
roll the task's terminal state up to a ``completed`` job status. The SDK's
``download_artifacts`` gates on that status, so we patch it to ``"completed"``
on the resource as the (single) network seam after the task has actually
written its results.
"""

import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import data_designer.config as dd
import nemo_data_designer_plugin.testing.utils as u
import pandas as pd
import pytest
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from nemo_data_designer_plugin.jobs.run import BUFFER_SIZE
from nemo_data_designer_plugin.jobs.spec import DataDesignerJobConfig
from nemo_data_designer_plugin.jobs.task_results import ANALYSIS_RESULT_NAME, ARTIFACTS_RESULT_NAME
from nemo_data_designer_plugin.sdk.job_results import DataDesignerJobResults
from nemo_data_designer_plugin.sdk.resources import DataDesignerResource
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.jobs.client import JobsClient

pytestmark = pytest.mark.integration


def _load_results(ctx: u.CreateJobTestContext, job_name: str, tmp_path: Path) -> DataDesignerJobResults:
    """Read back a completed task's results through the high-level SDK.

    ``task_context`` does not drive the Jobs service controller, so the platform
    job status never rolls up to ``"completed"`` on its own. We patch the
    resource's ``get_job_status`` to bypass that gate, then download artifacts
    the same way an end user would.
    """
    job_resource = DataDesignerResource(ctx.sdk).get_job_resource(job_name, workspace="default")
    with patch.object(job_resource, "get_job_status", return_value="completed"):
        return job_resource.download_artifacts(tmp_path)


def _get_dataset(ctx: u.CreateJobTestContext, job_name: str, tmp_path: Path) -> pd.DataFrame:
    return _load_results(ctx, job_name, tmp_path).load_dataset()


def _get_analysis(ctx: u.CreateJobTestContext, job_name: str, tmp_path: Path) -> DatasetProfilerResults:
    return _load_results(ctx, job_name, tmp_path).load_analysis()


def _list_job_results(ctx: u.CreateJobTestContext, job_name: str):
    return client_from_platform(ctx.sdk, JobsClient).list_job_results(name=job_name).data()


@pytest.fixture
def _failing_result_manager() -> Generator[None]:
    with patch("nemo_platform_plugin.jobs.result_manager.ResultManager", u.FailingResultManager):
        yield


@pytest.mark.asyncio
async def test_task(tmp_path: Path) -> None:
    test_value = "test-value"
    num_records = 42
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=[test_value])
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=num_records, config=builder.build())

    job_config = await u.compile_create_job(dd_job_config, workspace="default")
    job_name = "data-designer-abc123"

    async with u.task_context(job_config, job_name) as ctx:
        result = ctx.run_task()
        assert result.exit_code == 0

        results = _list_job_results(ctx, job_name)
        assert len(results.data) == 2
        result_names = [r.name for r in results.data]
        assert ANALYSIS_RESULT_NAME in result_names
        assert ARTIFACTS_RESULT_NAME in result_names

        dataset = _get_dataset(ctx, job_name, tmp_path)
        expected_partial_data = pd.DataFrame(data={"foo": [test_value] * num_records})
        # ``check_dtype=False``: the SDK loads via ``read_parquet_dataset``, which yields
        # ``string[pyarrow]`` columns; the inline expected DataFrame is plain ``object``.
        pd.testing.assert_frame_equal(dataset, expected_partial_data, check_dtype=False)

        analysis = _get_analysis(ctx, job_name, tmp_path)
        assert analysis.num_records == 42


@pytest.mark.asyncio
async def test_save_partial_dataset_on_failure(_failing_result_manager: None, tmp_path: Path) -> None:
    test_value = "test-value"
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=[test_value])
        )
    )

    requested_num_records = BUFFER_SIZE * (u.FAILING_RESULT_MANAGER_MAX_SUCCESSFUL_CALLS + 1)
    expected_num_records = BUFFER_SIZE * u.FAILING_RESULT_MANAGER_MAX_SUCCESSFUL_CALLS
    expected_partial_data = pd.DataFrame(data={"foo": [test_value] * expected_num_records})

    dd_job_config = DataDesignerJobConfig(num_records=requested_num_records, config=builder.build())
    job_config = await u.compile_create_job(dd_job_config, workspace="default")
    job_name = "data-designer-abc123"

    async with u.task_context(job_config, job_name) as ctx:
        result = ctx.run_task()
        assert result.exit_code == 1

        results = _list_job_results(ctx, job_name)
        assert len(results.data) == 1
        result_names = [r.name for r in results.data]
        assert ANALYSIS_RESULT_NAME not in result_names
        assert ARTIFACTS_RESULT_NAME in result_names

        dataset = _get_dataset(ctx, job_name, tmp_path)
        pd.testing.assert_frame_equal(dataset, expected_partial_data, check_dtype=False)


# TODO: once we restore batch-completion artifact saves, we can drop this test
# and include the log-related assertion in that test instead (immediately above)
@pytest.mark.asyncio
async def test_exiting_with_error() -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=["a", "b"])
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())
    job_config = await u.compile_create_job(dd_job_config)
    job_name = "data-designer-abc123"

    with (
        capture_job_log_messages() as log_messages,
        patch("nemo_data_designer_plugin.jobs.run.create_execution_context", side_effect=RuntimeError("Yuck")),
    ):
        async with u.task_context(job_config, job_name) as ctx:
            result = ctx.run_task()
            assert result.exit_code == 1
            assert any("Yuck" in message for message in log_messages)


@pytest.mark.asyncio
async def test_seed_dataset(tmp_path: Path) -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(dd.HuggingFaceSeedSource(path="path/to/data.parquet"))
    builder.add_column(column_config=dd.ExpressionColumnConfig(name="full_name", expr=u.FULL_NAME_EXPR))
    dd_job_config = DataDesignerJobConfig(num_records=3, config=builder.build())

    job_config = await u.compile_create_job(dd_job_config, workspace="default")
    job_name = "data-designer-abc123"

    with u.mock_hf_seed_reader():
        async with u.task_context(job_config, job_name) as ctx:
            result = ctx.run_task()
            assert result.exit_code == 0
            dataset = _get_dataset(ctx, job_name, tmp_path)
            assert set(dataset["full_name"].values) == u.FULL_NAMES


class _MessageCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_job_log_messages() -> Iterator[list[str]]:
    job_logger = logging.getLogger("nemo_data_designer_plugin.jobs.run")
    previous_level = job_logger.level
    handler = _MessageCaptureHandler()

    job_logger.addHandler(handler)
    job_logger.setLevel(logging.ERROR)
    try:
        yield handler.messages
    finally:
        job_logger.removeHandler(handler)
        job_logger.setLevel(previous_level)
