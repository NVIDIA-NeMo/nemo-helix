# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync/async job resources for the Anonymizer plugin SDK."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

import httpx
from nemo_anonymizer_plugin.sdk import http
from nemo_anonymizer_plugin.sdk.errors import AnonymizerJobError
from nemo_anonymizer_plugin.sdk.job_results import AnonymizerJobResults
from nemo_anonymizer_plugin.sdk.logging import with_logging
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.jobs.archive import safe_extract_tar
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

logger = logging.getLogger(__name__)

CHECK_PROGRESS_LOG_MSG = (
    "To check on your job's progress, use the `get_job_status` method. "
    "If you want to wait until it's complete, use the `wait_until_done` method."
)
WAIT_INTERVAL_SECONDS = 1
MAX_CONSECUTIVE_POLL_ERRORS = 5
ARTIFACTS_RESULT_NAME = "artifacts"
TERMINAL_INCOMPLETE_STATUSES = {"cancelled", "cancelling", "error"}

T = TypeVar("T")


class _JobIdentity(BaseModel):
    id: str = Field(strict=True, min_length=1)


class _JobStatus(_JobIdentity):
    status: HelixJobStatus | None = None


class _JobDetails(_JobIdentity):
    name: str = Field(strict=True, min_length=1)
    status: HelixJobStatus


_JOB_JSON = TypeAdapter(dict[str, JsonValue])


def _job_url(platform: http.HelixClient, workspace: str | None, path: str) -> str:
    return http.url(platform, workspace, f"/jobs/run{path}")


def _job_path(job_id: str, suffix: str = "") -> str:
    return f"/{http.path_segment(job_id)}{suffix}"


def _safe_extract_tar(tar: tarfile.TarFile, output_path: Path) -> None:
    safe_extract_tar(tar, output_path, error_cls=AnonymizerJobError)


def _raise_for_status(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AnonymizerJobError(exc.response.text) from exc


@with_logging
class AnonymizerJobResource:
    def __init__(self, *, platform: NemoClient, workspace: str | None, job_id: str):
        self._job_id = _JobIdentity(id=job_id).id
        self._platform = platform
        self._workspace = workspace
        self._consecutive_poll_errors = 0

    def _validate_job_id(self, job_id: str) -> None:
        if job_id != self._job_id:
            raise AnonymizerJobError("Job response returned a different Job ID")

    def _client(self) -> httpx.Client:
        return self._platform._client

    def get_job(self) -> dict[str, JsonValue]:
        resp = self._client().get(
            _job_url(self._platform, self._workspace, _job_path(self._job_id)),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        job = _JOB_JSON.validate_json(resp.content)
        self._validate_job_id(_JobIdentity.model_validate(job).id)
        return job

    def get_job_status(self) -> HelixJobStatus | None:
        resp = self._client().get(
            _job_url(self._platform, self._workspace, _job_path(self._job_id, "/status")),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        payload = _JobStatus.model_validate_json(resp.content)
        self._validate_job_id(payload.id)
        return payload.status

    def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        return _status_is_complete(self.get_job_status(), raise_if_not_complete)

    def wait_until_done(self) -> None:
        status = self.get_job_status()
        while status not in {"completed", *TERMINAL_INCOMPLETE_STATUSES}:
            time.sleep(WAIT_INTERVAL_SECONDS)
            status = self._poll_safe(self.get_job_status, status)
        if status != "completed":
            logger.error(f"Anonymizer job ended with status `{status}`.")
        else:
            logger.info("Anonymizer job complete.")

    def get_logs(self) -> list[dict[str, str]]:
        logs = []
        page_cursor = None
        while True:
            params = {"page_cursor": page_cursor} if page_cursor else None
            resp = self._client().get(
                _job_url(self._platform, self._workspace, _job_path(self._job_id, "/logs")),
                headers=http.headers(self._platform),
                params=params,
            )
            _raise_for_status(resp)
            response = resp.json()
            for log in response.get("data", []):
                try:
                    deserialized = json.loads(log["message"])
                    if isinstance(deserialized, dict) and "message" in deserialized:
                        logs.append(deserialized)
                except Exception:
                    pass
            page_cursor = response.get("next_page")
            if page_cursor is None:
                break
        return logs

    def download_artifacts(self, path: Path | str | None = None) -> AnonymizerJobResults:
        job = self._check_if_result_available(ARTIFACTS_RESULT_NAME)
        output_path = Path(path or job.name)
        logger.info(f"Downloading artifacts from job {job.name!r}")

        resp = self._client().get(
            _job_url(
                self._platform,
                self._workspace,
                _job_path(self._job_id, f"/results/{http.path_segment(ARTIFACTS_RESULT_NAME)}/download"),
            ),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:*") as tar:
            _safe_extract_tar(tar, output_path)

        artifacts_dir = output_path / "artifacts"
        logger.info(f"Artifacts downloaded to {artifacts_dir}")
        return AnonymizerJobResults(artifacts_dir)

    def _check_if_result_available(self, result_name: str) -> _JobDetails:
        resp = self._client().get(
            _job_url(self._platform, self._workspace, _job_path(self._job_id, "/status")),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        job = _JobDetails.model_validate_json(resp.content)
        self._validate_job_id(job.id)
        status = job.status
        if status == "completed":
            return job
        raise AnonymizerJobError(f"Current job status is {status!r}; results are not available.")

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


@with_logging
class AsyncAnonymizerJobResource:
    def __init__(self, *, platform: AsyncNemoClient, workspace: str | None, job_id: str):
        self._job_id = _JobIdentity(id=job_id).id
        self._platform = platform
        self._workspace = workspace
        self._consecutive_poll_errors = 0

    def _validate_job_id(self, job_id: str) -> None:
        if job_id != self._job_id:
            raise AnonymizerJobError("Job response returned a different Job ID")

    def _client(self) -> httpx.AsyncClient:
        return self._platform._client

    async def get_job(self) -> dict[str, JsonValue]:
        resp = await self._client().get(
            _job_url(self._platform, self._workspace, _job_path(self._job_id)),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        job = _JOB_JSON.validate_json(resp.content)
        self._validate_job_id(_JobIdentity.model_validate(job).id)
        return job

    async def get_job_status(self) -> HelixJobStatus | None:
        resp = await self._client().get(
            _job_url(self._platform, self._workspace, _job_path(self._job_id, "/status")),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        payload = _JobStatus.model_validate_json(resp.content)
        self._validate_job_id(payload.id)
        return payload.status

    async def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        return _status_is_complete(await self.get_job_status(), raise_if_not_complete)

    async def wait_until_done(self) -> None:
        status = await self.get_job_status()
        while status not in {"completed", *TERMINAL_INCOMPLETE_STATUSES}:
            await asyncio.sleep(WAIT_INTERVAL_SECONDS)
            status = await self._poll_safe(self.get_job_status, status)
        if status != "completed":
            logger.error(f"Anonymizer job ended with status `{status}`.")
        else:
            logger.info("Anonymizer job complete.")

    async def download_artifacts(self, path: Path | str | None = None) -> AnonymizerJobResults:
        job = await self._check_if_result_available(ARTIFACTS_RESULT_NAME)
        output_path = Path(path or job.name)

        resp = await self._client().get(
            _job_url(
                self._platform,
                self._workspace,
                _job_path(self._job_id, f"/results/{http.path_segment(ARTIFACTS_RESULT_NAME)}/download"),
            ),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:*") as tar:
            _safe_extract_tar(tar, output_path)

        artifacts_dir = output_path / "artifacts"
        return AnonymizerJobResults(artifacts_dir)

    async def _check_if_result_available(self, result_name: str) -> _JobDetails:
        resp = await self._client().get(
            _job_url(self._platform, self._workspace, _job_path(self._job_id, "/status")),
            headers=http.headers(self._platform),
        )
        _raise_for_status(resp)
        job = _JobDetails.model_validate_json(resp.content)
        self._validate_job_id(job.id)
        status = job.status
        if status == "completed":
            return job
        raise AnonymizerJobError(f"Current job status is {status!r}; results are not available.")

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


def _status_is_complete(status: HelixJobStatus | None, raise_if_not_complete: bool) -> bool:
    if status == "completed":
        return True
    if status in TERMINAL_INCOMPLETE_STATUSES:
        msg = f"Anonymizer job stopped with status `{status}`."
        if raise_if_not_complete:
            raise AnonymizerJobError(msg)
        logger.error(msg)
        return False
    msg = f"Anonymizer job is still running. {CHECK_PROGRESS_LOG_MSG}"
    if raise_if_not_complete:
        raise AnonymizerJobError(msg)
    logger.warning(msg)
    return False
