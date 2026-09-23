# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit job resource handles for status polling, log streaming, and artifact download."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.jobs.archive import safe_extract_tar
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from typing_extensions import Self

logger = logging.getLogger(__name__)

WAIT_INTERVAL_SECONDS = 1
MAX_CONSECUTIVE_POLL_ERRORS = 5
ARTIFACTS_RESULT_NAME = "artifacts"
TERMINAL_INCOMPLETE_STATUSES = {"cancelled", "cancelling", "error"}

T = TypeVar("T")


def _pause(seconds: float) -> None:
    time.sleep(seconds)


async def _async_pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _job_url(platform: NeMoHelix | AsyncNeMoHelix, workspace: str, job_name: str, path: str = "") -> str:
    base = str(platform.base_url).rstrip("/")
    return f"{base}/apis/auditor/v2/workspaces/{workspace}/jobs/audit/{job_name}{path}"


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
            if not log["name"].startswith("nemo_auditor"):
                continue
            level = log["levelname"].lower()
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
            logger.error("Audit job completed with errors.")
        elif self.warning_occurred:
            logger.warning("Audit job completed with warnings.")
        else:
            logger.info("Audit job completed successfully.")


def _status_is_complete(status: HelixJobStatus | None, raise_if_not_complete: bool) -> bool:
    if status == "completed":
        return True
    if status == "active":
        msg = "The audit job is still running."
        if raise_if_not_complete:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False
    if status in TERMINAL_INCOMPLETE_STATUSES:
        msg = f"The audit job stopped with status {status!r}."
        if raise_if_not_complete:
            raise RuntimeError(msg)
        logger.error(msg)
        return False
    if status in {"created", "pending"}:
        msg = f"The audit job is still in the queue with status {status!r}."
        if raise_if_not_complete:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False
    msg = f"The audit job is in an unknown state: {status!r}."
    if raise_if_not_complete:
        raise RuntimeError(msg)
    logger.error(msg)
    return False


def _try_parse_log_message(raw_message: str) -> dict[str, str] | None:
    """Best-effort extraction of the JSON payload from a platform log entry."""
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


class AuditorJobResource:
    """Sync SDK handle for a submitted audit job."""

    def __init__(self, *, job_name: str, platform: NeMoHelix, workspace: str) -> None:
        self._job_name = job_name
        self._platform = platform
        self._workspace = workspace
        self._consecutive_poll_errors = 0

    @property
    def name(self) -> str:
        """The unique identifying name of the job."""
        return self._job_name

    def get_job(self) -> dict[str, object]:
        """Fetch the current job dict."""
        resp = self._platform._client.get(_job_url(self._platform, self._workspace, self._job_name))
        resp.raise_for_status()
        return resp.json()

    def get_job_status(self) -> HelixJobStatus | None:
        """Fetch the current platform status of the job."""
        resp = self._platform._client.get(_job_url(self._platform, self._workspace, self._job_name, "/status"))
        resp.raise_for_status()
        return resp.json().get("status")

    def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Return whether the job has reached the ``completed`` status.

        Args:
            raise_if_not_complete: When ``True``, raise ``RuntimeError`` for any
                status other than ``completed``.
        """
        return _status_is_complete(self.get_job_status(), raise_if_not_complete)

    def wait_until_done(self) -> None:
        """Block until the job reaches a terminal status, streaming logs along the way."""
        log_collector = _WaitLogCollector.create()
        job_status = self.get_job_status()
        while job_status != "completed":
            _pause(WAIT_INTERVAL_SECONDS)
            current_logs = self._poll_safe(self.get_logs, log_collector.seen_logs)
            log_collector.accept_logs(current_logs)
            if job_status in TERMINAL_INCOMPLETE_STATUSES:
                log_collector.error_occurred = True
                logger.error(f"Audit job terminated with status {job_status!r}.")
                break
            job_status = self._poll_safe(self.get_job_status, job_status)
        log_collector.log_final_status()

    def get_logs(self) -> list[dict[str, str]]:
        """Page through and return all job log entries."""
        logs = []
        page_cursor = None
        while True:
            params = {"page_cursor": page_cursor} if page_cursor else None
            resp = self._platform._client.get(
                _job_url(self._platform, self._workspace, self._job_name, "/logs"),
                params=params,
            )
            resp.raise_for_status()
            response = resp.json()
            for log in response.get("data", []):
                deserialized = _try_parse_log_message(log.get("message", ""))
                if deserialized is not None:
                    logs.append(deserialized)
            page_cursor = response.get("next_page")
            if page_cursor is None:
                break
        return logs

    def download_artifacts(self, path: Path | str | None = None) -> Path:
        """Download and extract the garak report artifacts for this job.

        Args:
            path: Base output directory. Defaults to a directory named after the job
                in the current working directory.

        Returns:
            The directory that contains the extracted artifacts.

        Raises:
            RuntimeError: If the job has not completed.
        """
        status = self.get_job_status()
        if status != "completed":
            raise RuntimeError(
                f"Artifacts are not available: job {self._job_name!r} has status {status!r}. "
                "Wait until the job completes before downloading artifacts."
            )
        output_path = Path(path or self._job_name)
        resp = self._platform._client.get(
            _job_url(self._platform, self._workspace, self._job_name, f"/results/{ARTIFACTS_RESULT_NAME}/download"),
        )
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:*") as tar:
            safe_extract_tar(tar, output_path, error_cls=RuntimeError)
        return output_path

    def _poll_safe(self, fn: Callable[[], T], fallback: T) -> T:
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


class AsyncAuditorJobResource:
    """Async SDK handle for a submitted audit job."""

    def __init__(self, *, job_name: str, platform: AsyncNeMoHelix, workspace: str) -> None:
        self._job_name = job_name
        self._platform = platform
        self._workspace = workspace
        self._consecutive_poll_errors = 0

    @property
    def name(self) -> str:
        """The unique identifying name of the job."""
        return self._job_name

    async def get_job(self) -> dict[str, object]:
        """Fetch the current job dict."""
        resp = await self._platform._client.get(_job_url(self._platform, self._workspace, self._job_name))
        resp.raise_for_status()
        return resp.json()

    async def get_job_status(self) -> HelixJobStatus | None:
        """Fetch the current platform status of the job."""
        resp = await self._platform._client.get(_job_url(self._platform, self._workspace, self._job_name, "/status"))
        resp.raise_for_status()
        return resp.json().get("status")

    async def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Return whether the job has reached the ``completed`` status.

        Args:
            raise_if_not_complete: When ``True``, raise ``RuntimeError`` for any
                status other than ``completed``.
        """
        return _status_is_complete(await self.get_job_status(), raise_if_not_complete)

    async def wait_until_done(self) -> None:
        """Wait until the job reaches a terminal status, streaming logs along the way."""
        log_collector = _WaitLogCollector.create()
        job_status = await self.get_job_status()
        while job_status != "completed":
            await _async_pause(WAIT_INTERVAL_SECONDS)
            current_logs = await self._poll_safe(self.get_logs, log_collector.seen_logs)
            log_collector.accept_logs(current_logs)
            if job_status in TERMINAL_INCOMPLETE_STATUSES:
                log_collector.error_occurred = True
                logger.error(f"Audit job terminated with status {job_status!r}.")
                break
            job_status = await self._poll_safe(self.get_job_status, job_status)
        log_collector.log_final_status()

    async def get_logs(self) -> list[dict[str, str]]:
        """Page through and return all job log entries."""
        logs = []
        page_cursor = None
        while True:
            params = {"page_cursor": page_cursor} if page_cursor else None
            resp = await self._platform._client.get(
                _job_url(self._platform, self._workspace, self._job_name, "/logs"),
                params=params,
            )
            resp.raise_for_status()
            response = resp.json()
            for log in response.get("data", []):
                deserialized = _try_parse_log_message(log.get("message", ""))
                if deserialized is not None:
                    logs.append(deserialized)
            page_cursor = response.get("next_page")
            if page_cursor is None:
                break
        return logs

    async def download_artifacts(self, path: Path | str | None = None) -> Path:
        """Download and extract the garak report artifacts for this job.

        Args:
            path: Base output directory. Defaults to a directory named after the job
                in the current working directory.

        Returns:
            The directory that contains the extracted artifacts.

        Raises:
            RuntimeError: If the job has not completed.
        """
        status = await self.get_job_status()
        if status != "completed":
            raise RuntimeError(
                f"Artifacts are not available: job {self._job_name!r} has status {status!r}. "
                "Wait until the job completes before downloading artifacts."
            )
        output_path = Path(path or self._job_name)
        resp = await self._platform._client.get(
            _job_url(self._platform, self._workspace, self._job_name, f"/results/{ARTIFACTS_RESULT_NAME}/download"),
        )
        resp.raise_for_status()
        await asyncio.to_thread(
            lambda: _extract_tar(resp.content, output_path),
        )
        return output_path

    async def _poll_safe(self, fn: Callable[[], Awaitable[T]], fallback: T) -> T:
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


def _extract_tar(content: bytes, output_path: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
        safe_extract_tar(tar, output_path, error_cls=RuntimeError)
