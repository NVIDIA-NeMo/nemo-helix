# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import logging
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TypeVar, overload

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.utils.visualization import WithRecordSamplerMixin
from data_designer.logging import RandomEmoji
from nemo_data_designer_plugin.sdk.errors import DataDesignerJobError
from nemo_data_designer_plugin.sdk.job_results import DataDesignerJobResults
from nemo_data_designer_plugin.sdk.logging import with_logging
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError, NotFoundError
from nemo_platform_plugin.data_designer.client import AsyncDataDesignerClient, DataDesignerClient
from nemo_platform_plugin.data_designer.types import DataDesignerJobCollection, DataDesignerJobLogsQueryParams
from nemo_platform_plugin.jobs.archive import safe_extract_tar
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from typing_extensions import Self

logger = logging.getLogger(__name__)

CHECK_PROGRESS_LOG_MSG = (
    "To check on your job's progress, use the `get_job_status` method. "
    "If you want to wait until it's complete, use the `wait_until_done` method."
)
WAIT_INTERVAL_SECONDS = 1
MAX_CONSECUTIVE_POLL_ERRORS = 5
ARTIFACTS_RESULT_NAME = "artifacts"
ANALYSIS_RESULT_NAME = "analysis"
TERMINAL_INCOMPLETE_STATUSES = {"cancelled", "cancelling", "error"}

T = TypeVar("T")


def _pause(seconds: float) -> None:
    time.sleep(seconds)


async def _async_pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _get_job_error(exc: NemoHTTPError) -> DataDesignerJobError:
    return DataDesignerJobError(exc.detail, status_code=exc.status_code)


def _safe_extract_tar(tar: tarfile.TarFile, output_path: Path) -> None:
    safe_extract_tar(tar, output_path, error_cls=DataDesignerJobError)


def _extract_artifact_bytes(artifact_bytes: bytes, output_path: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(artifact_bytes), mode="r:*") as tar:
        _safe_extract_tar(tar, output_path)


@dataclass
class _WaitLogCollector:
    """Collects and processes log entries emitted during job polling."""

    seen_logs: list[dict[str, str]]
    error_occurred: bool
    warning_occurred: bool

    @classmethod
    def create(cls) -> Self:
        return cls(seen_logs=[], error_occurred=False, warning_occurred=False)

    def accept_logs(self, current_logs: list[dict[str, str]]) -> None:
        for log in current_logs[len(self.seen_logs) :]:
            self.seen_logs.append(log)
            if not log.get("name", "").startswith("data_designer"):
                continue
            level = log.get("levelname", "").lower()
            if level == "info":
                logger.info(log["message"])
            elif level in {"warning", "warn"}:
                logger.warning(log["message"])
                self.warning_occurred = True
            elif level == "error":
                logger.error(log["message"])
                self.error_occurred = True

    def log_final_status(self) -> None:
        if self.error_occurred:
            logger.error("🛑 Dataset generation completed with errors.")
        elif self.warning_occurred:
            logger.warning("⚠️ Dataset generation completed with warnings.")
        else:
            logger.info(f"{RandomEmoji.success()} Dataset generation completed successfully.")


@with_logging
class DataDesignerJobResource(WithRecordSamplerMixin):
    @overload
    def __init__(
        self,
        *,
        job_name: str,
        client: DataDesignerClient,
        workspace: str | None,
        job_collection: DataDesignerJobCollection = "create",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        job_name: str,
        platform: NeMoPlatform,
        workspace: str | None,
        job_collection: DataDesignerJobCollection = "create",
    ) -> None: ...

    def __init__(
        self,
        *,
        job_name: str,
        client: DataDesignerClient | None = None,
        platform: NeMoPlatform | None = None,
        workspace: str | None,
        job_collection: DataDesignerJobCollection = "create",
    ):
        if client is None:
            if platform is None:
                raise TypeError("DataDesignerJobResource requires either client= or platform=")
            client = client_from_platform(platform, DataDesignerClient)
        elif platform is not None:
            raise TypeError("Pass only one of client= or platform=")

        self._job_name = job_name
        self._client = client
        self._workspace = workspace
        self._job_collection = job_collection
        self._consecutive_poll_errors = 0

    @property
    def name(self) -> str:
        """The unique identifying name of the job.

        Returns:
            The job name.
        """
        return self._job_name

    def get_job(self) -> dict[str, object]:
        """Get the current job.

        Returns:
            The job dict with up-to-date details.
        """
        try:
            job = self._client.get_job(
                workspace=self._workspace, job_collection=self._job_collection, name=self._job_name
            ).data()
        except NemoHTTPError as exc:
            raise _get_job_error(exc) from exc
        return job.model_dump(mode="json")

    def get_job_status(self) -> PlatformJobStatus | None:
        """Get the current status of the job.

        Returns:
            The current job status.
        """
        try:
            status = self._client.get_job_status(
                workspace=self._workspace, job_collection=self._job_collection, name=self._job_name
            ).data()
        except NemoHTTPError as exc:
            raise _get_job_error(exc) from exc
        return status.status

    def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Check if the job is in a completed state.

        Args:
            raise_if_not_complete: If True, raises DataDesignerJobError when job is not complete.
                                   If False, only logs warnings/errors without raising exceptions.

        Returns:
            True if job is completed, False otherwise.

        Raises:
            DataDesignerJobError: If raise_if_not_complete is True and job is not in completed state.
        """
        status = self.get_job_status()
        return _status_is_complete(status, raise_if_not_complete)

    def wait_until_done(self) -> None:
        """Wait for the job to complete and monitor its progress.

        This method blocks execution until the job reaches a terminal state.
        During the wait, it continuously monitors job logs and displays relevant messages to the user.

        The method will:
        - Poll the job status at regular intervals
        - Display log messages from the data designer service
        - Handle warnings and errors appropriately
        - Provide final status summary when complete
        """
        log_collector = _WaitLogCollector.create()
        job_status = self.get_job_status()
        while job_status != "completed":
            _pause(WAIT_INTERVAL_SECONDS)
            current_logs = self._poll_safe(self.get_logs, log_collector.seen_logs)
            log_collector.accept_logs(current_logs)
            if job_status in TERMINAL_INCOMPLETE_STATUSES:
                log_collector.error_occurred = True
                logger.error(f"🛑 Terminating generation job with status `{job_status}`.")
                break
            job_status = self._poll_safe(self.get_job_status, job_status)
        log_collector.log_final_status()

    def get_logs(self) -> list[dict[str, str]]:
        """Page through and fetch all job logs.

        Returns:
            A list of log entries, where each entry is a dictionary containing log information.
        """
        logs = []
        page_cursor = None
        while True:
            query_params: DataDesignerJobLogsQueryParams | None = {"page_cursor": page_cursor} if page_cursor else None
            try:
                page = self._client.get_job_logs(
                    workspace=self._workspace,
                    job_collection=self._job_collection,
                    name=self._job_name,
                    query_params=query_params,
                ).data()
            except NemoHTTPError as exc:
                raise _get_job_error(exc) from exc
            for log in page.data:
                deserialized = _try_parse_log_message(log.message)
                if deserialized is not None:
                    logs.append(deserialized)
            page_cursor = page.next_page
            if page_cursor is None:
                break
        return logs

    def download_artifacts(self, path: Path | str | None = None) -> DataDesignerJobResults:
        """Download the Job's artifacts to the specified path.

        Args:
            path: Save artifacts to this path. If not specified, creates a local directory using the job name.

        Returns:
            An object with methods for inspecting the saved job results.
        """
        self._check_if_result_available(ARTIFACTS_RESULT_NAME)
        output_path = Path(path or self._job_name)
        logger.info(f"🏺 Downloading artifacts from Job {self._job_name!r}")

        try:
            artifact_bytes = self._client.download_job_result(
                workspace=self._workspace,
                job_collection=self._job_collection,
                job=self._job_name,
                name=ARTIFACTS_RESULT_NAME,
            ).read()
        except NemoHTTPError as exc:
            raise _get_job_error(exc) from exc
        _extract_artifact_bytes(artifact_bytes, output_path)

        try:
            analysis_bytes = self._client.download_job_result(
                workspace=self._workspace,
                job_collection=self._job_collection,
                job=self._job_name,
                name=ANALYSIS_RESULT_NAME,
            ).read()
            analysis = DatasetProfilerResults.model_validate_json(analysis_bytes)
        except Exception as e:
            msg = f"Unable to fetch analysis: {e}"
            logger.warning(msg)
            analysis = msg

        artifacts_dir = output_path / "artifacts"
        logger.info(f"✅ Artifacts downloaded to {artifacts_dir}")
        return DataDesignerJobResults(artifacts_dir, analysis)

    def load_analysis(self) -> DatasetProfilerResults:
        """Load the dataset analysis as a DatasetProfilerResults object.

        Returns:
            The analysis results containing dataset statistics and profiling information.

        Raises:
            DataDesignerJobError: If the job is not completed or if there's an error loading the analysis.
        """
        self._check_if_result_available(ANALYSIS_RESULT_NAME)
        try:
            analysis_bytes = self._client.download_job_result(
                workspace=self._workspace,
                job_collection=self._job_collection,
                job=self._job_name,
                name=ANALYSIS_RESULT_NAME,
            ).read()
            return DatasetProfilerResults.model_validate_json(analysis_bytes)
        except Exception as e:
            raise DataDesignerJobError(f"🛑 Error loading analysis: {e}") from e

    def _check_if_result_available(self, result_name: str) -> None:
        status = self.get_job_status()
        if status == "completed":
            return
        if status == "active" or status in TERMINAL_INCOMPLETE_STATUSES:
            try:
                self._client.get_job_result(
                    workspace=self._workspace,
                    job_collection=self._job_collection,
                    job=self._job_name,
                    name=result_name,
                ).data()
                if status == "active":
                    logger.info(
                        f"{RandomEmoji.cooking()} Your dataset is still cooking. "
                        "Fetching completed results for your enjoyment."
                    )
                else:
                    logger.warning(f"Job ended with status {status!r}. Fetching completed {result_name} result.")
            except NotFoundError as e:
                raise DataDesignerJobError(f"{result_name!r} result is not available.") from e
            except NemoHTTPError as e:
                raise DataDesignerJobError(f"🛑 Error loading dataset: {e.detail}", status_code=e.status_code) from e
        else:
            raise DataDesignerJobError(f"Current job status is {status!r}, results are not available.")

    def _poll_safe(self, fn: Callable[[], T], fallback: T) -> T:
        """Wrapper function to add resilience to network calls made while polling.

        This method will call the provided function and, in the happy path,
        reset the consecutive errors counter and return the result.

        If an error occurs, the consecutive errors counter is incremented.
        - If the threshold is not yet met, the fallback value is returned. Typically
          the fallback value is the last cached response from the network call.
        - If the counter has met the threshold, the counter is reset for future use
          and the caught error is raised.
        """
        try:
            response = fn()
            self._consecutive_poll_errors = 0
            return response
        except Exception:
            self._consecutive_poll_errors += 1
            if self._consecutive_poll_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                self._consecutive_poll_errors = 0
                raise
            return fallback


@with_logging
class AsyncDataDesignerJobResource(WithRecordSamplerMixin):
    @overload
    def __init__(
        self,
        *,
        job_name: str,
        client: AsyncDataDesignerClient,
        workspace: str | None,
        job_collection: DataDesignerJobCollection = "create",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        job_name: str,
        platform: AsyncNeMoPlatform,
        workspace: str | None,
        job_collection: DataDesignerJobCollection = "create",
    ) -> None: ...

    def __init__(
        self,
        *,
        job_name: str,
        client: AsyncDataDesignerClient | None = None,
        platform: AsyncNeMoPlatform | None = None,
        workspace: str | None,
        job_collection: DataDesignerJobCollection = "create",
    ):
        if client is None:
            if platform is None:
                raise TypeError("AsyncDataDesignerJobResource requires either client= or platform=")
            client = client_from_platform(platform, AsyncDataDesignerClient)
        elif platform is not None:
            raise TypeError("Pass only one of client= or platform=")

        self._job_name = job_name
        self._client = client
        self._workspace = workspace
        self._job_collection = job_collection
        self._consecutive_poll_errors = 0

    @property
    def name(self) -> str:
        """The unique identifying name of the job.

        Returns:
            The job name.
        """
        return self._job_name

    async def get_job(self) -> dict[str, object]:
        """Get the current job.

        Returns:
            The job dict with up-to-date details.
        """
        try:
            response = await self._client.get_job(
                workspace=self._workspace, job_collection=self._job_collection, name=self._job_name
            )
        except NemoHTTPError as exc:
            raise _get_job_error(exc) from exc
        return response.data().model_dump(mode="json")

    async def get_job_status(self) -> PlatformJobStatus | None:
        """Get the current status of the job.

        Returns:
            The current job status.
        """
        try:
            response = await self._client.get_job_status(
                workspace=self._workspace, job_collection=self._job_collection, name=self._job_name
            )
        except NemoHTTPError as exc:
            raise _get_job_error(exc) from exc
        return response.data().status

    async def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Check if the job is in a completed state.

        Args:
            raise_if_not_complete: If True, raises DataDesignerJobError when job is not complete.
                                   If False, only logs warnings/errors without raising exceptions.

        Returns:
            True if job is completed, False otherwise.

        Raises:
            DataDesignerJobError: If raise_if_not_complete is True and job is not in completed state.
        """
        status = await self.get_job_status()
        return _status_is_complete(status, raise_if_not_complete)

    async def wait_until_done(self) -> None:
        """Wait for the job to complete and monitor its progress.

        This method blocks execution until the job reaches a terminal state.
        During the wait, it continuously monitors job logs and displays relevant messages to the user.

        The method will:
        - Poll the job status at regular intervals
        - Display log messages from the data designer service
        - Handle warnings and errors appropriately
        - Provide final status summary when complete
        """
        log_collector = _WaitLogCollector.create()
        job_status = await self.get_job_status()
        while job_status != "completed":
            await _async_pause(WAIT_INTERVAL_SECONDS)
            current_logs = await self._poll_safe(self.get_logs, log_collector.seen_logs)
            log_collector.accept_logs(current_logs)
            if job_status in TERMINAL_INCOMPLETE_STATUSES:
                log_collector.error_occurred = True
                logger.error(f"🛑 Terminating generation job with status `{job_status}`.")
                break
            job_status = await self._poll_safe(self.get_job_status, job_status)
        log_collector.log_final_status()

    async def get_logs(self) -> list[dict[str, str]]:
        """Page through and fetch all job logs.

        Returns:
            A list of log entries, where each entry is a dictionary containing log information.
        """
        logs = []
        page_cursor = None
        while True:
            query_params: DataDesignerJobLogsQueryParams | None = {"page_cursor": page_cursor} if page_cursor else None
            try:
                response = await self._client.get_job_logs(
                    workspace=self._workspace,
                    job_collection=self._job_collection,
                    name=self._job_name,
                    query_params=query_params,
                )
            except NemoHTTPError as exc:
                raise _get_job_error(exc) from exc
            page = response.data()
            for log in page.data:
                deserialized = _try_parse_log_message(log.message)
                if deserialized is not None:
                    logs.append(deserialized)
            page_cursor = page.next_page
            if page_cursor is None:
                break
        return logs

    async def download_artifacts(self, path: Path | str | None = None) -> DataDesignerJobResults:
        """Download the Job's artifacts to the specified path.

        Args:
            path: Save artifacts to this path. If not specified, creates a local directory using the job name.

        Returns:
            An object with methods for inspecting the saved job results.
        """
        await self._check_if_result_available(ARTIFACTS_RESULT_NAME)
        output_path = Path(path or self._job_name)
        logger.info(f"🏺 Downloading artifacts from Job {self._job_name!r}")

        try:
            artifact_response = await self._client.download_job_result(
                workspace=self._workspace,
                job_collection=self._job_collection,
                job=self._job_name,
                name=ARTIFACTS_RESULT_NAME,
            )
            artifact_bytes = await artifact_response.read()
        except NemoHTTPError as exc:
            raise _get_job_error(exc) from exc
        await asyncio.to_thread(_extract_artifact_bytes, artifact_bytes, output_path)

        try:
            analysis_response = await self._client.download_job_result(
                workspace=self._workspace,
                job_collection=self._job_collection,
                job=self._job_name,
                name=ANALYSIS_RESULT_NAME,
            )
            analysis_bytes = await analysis_response.read()
            analysis = DatasetProfilerResults.model_validate_json(analysis_bytes)
        except Exception as e:
            msg = f"Unable to fetch analysis: {e}"
            logger.warning(msg)
            analysis = msg

        artifacts_dir = output_path / "artifacts"
        logger.info(f"✅ Artifacts downloaded to {artifacts_dir}")
        return DataDesignerJobResults(artifacts_dir, analysis)

    async def load_analysis(self) -> DatasetProfilerResults:
        """Load the dataset analysis as a DatasetProfilerResults object.

        Returns:
            The analysis results containing dataset statistics and profiling information.

        Raises:
            DataDesignerJobError: If the job is not completed or if there's an error loading the analysis.
        """
        await self._check_if_result_available(ANALYSIS_RESULT_NAME)
        try:
            response = await self._client.download_job_result(
                workspace=self._workspace,
                job_collection=self._job_collection,
                job=self._job_name,
                name=ANALYSIS_RESULT_NAME,
            )
            analysis_bytes = await response.read()
            return DatasetProfilerResults.model_validate_json(analysis_bytes)
        except Exception as e:
            raise DataDesignerJobError(f"🛑 Error loading analysis: {e}") from e

    async def _check_if_result_available(self, result_name: str) -> None:
        status = await self.get_job_status()
        if status == "completed":
            return
        if status == "active" or status in TERMINAL_INCOMPLETE_STATUSES:
            try:
                response = await self._client.get_job_result(
                    workspace=self._workspace,
                    job_collection=self._job_collection,
                    job=self._job_name,
                    name=result_name,
                )
                response.data()
                if status == "active":
                    logger.info(
                        f"{RandomEmoji.cooking()} Your dataset is still cooking. "
                        "Fetching completed results for your enjoyment."
                    )
                else:
                    logger.warning(f"Job ended with status {status!r}. Fetching completed {result_name} result.")
            except NotFoundError as e:
                raise DataDesignerJobError(f"{result_name!r} result is not available.") from e
            except NemoHTTPError as e:
                raise DataDesignerJobError(f"🛑 Error loading dataset: {e.detail}", status_code=e.status_code) from e
        else:
            raise DataDesignerJobError(f"Current job status is {status!r}, results are not available.")

    async def _poll_safe(self, fn: Callable[[], Awaitable[T]], fallback: T) -> T:
        """Wrapper function to add resilience to network calls made while polling.

        This method will call the provided function and, in the happy path,
        reset the consecutive errors counter and return the result.

        If an error occurs, the consecutive errors counter is incremented.
        - If the threshold is not yet met, the fallback value is returned. Typically
          the fallback value is the last cached response from the network call.
        - If the counter has met the threshold, the counter is reset for future use
          and the caught error is raised.
        """
        try:
            response = await fn()
            self._consecutive_poll_errors = 0
            return response
        except Exception:
            self._consecutive_poll_errors += 1
            if self._consecutive_poll_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                self._consecutive_poll_errors = 0
                raise
            return fallback


def _try_parse_log_message(raw_message: str) -> dict[str, str] | None:
    """Best-effort extraction of the JSON payload from a platform log entry.

    Job logs come back as ``log["message"]`` strings. The platform's log
    capture sometimes prepends a stream marker like ``"[stderr] "`` before the
    JSON dict our task emits via ``_make_json_formatter``. Slice from the
    first ``{`` so the prefix doesn't break parsing; non-JSON messages
    (heartbeats, raw stderr lines from third-party libraries, etc.) silently
    return ``None`` and the caller drops them.
    """
    json_start = raw_message.find("{")
    if json_start < 0:
        return None
    try:
        deserialized = json.loads(raw_message[json_start:])
    except Exception:
        return None
    if not isinstance(deserialized, dict) or "message" not in deserialized:
        return None
    return deserialized


def _status_is_complete(status: PlatformJobStatus | None, raise_if_not_complete: bool) -> bool:
    if status == "completed":
        return True
    if status == "active":
        msg = f"Your dataset generation job is still running. {CHECK_PROGRESS_LOG_MSG}"
        if raise_if_not_complete:
            raise DataDesignerJobError(f"🛑 {msg}")
        logger.warning(f"⏳ {msg}")
        return False
    if status in TERMINAL_INCOMPLETE_STATUSES:
        msg = f"🛑 Your dataset generation job stopped with status `{status}`."
        if raise_if_not_complete:
            raise DataDesignerJobError(msg)
        logger.error(msg)
        return False
    if status in {"created", "pending"}:
        msg = f"⏹️ Your dataset generation job is still in the queue with status `{status}`. {CHECK_PROGRESS_LOG_MSG}"
        if raise_if_not_complete:
            raise DataDesignerJobError(msg)
        logger.warning(msg)
        return False
    msg = f"Your job is in an unknown state: `{status}`."
    if raise_if_not_complete:
        raise DataDesignerJobError(msg)
    logger.error(msg)
    return False
