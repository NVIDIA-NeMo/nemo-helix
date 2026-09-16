# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level SDK accessor for the Anonymizer plugin service."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Generic, TypeVar

import httpx
import pandas as pd
from nemo_anonymizer_plugin.app.task_config import AnonymizerRequest, PreviewRequest
from nemo_anonymizer_plugin.functions.preview import (
    FailedRecordsFrame,
    LogFrame,
    PreviewDatasetFrame,
    PreviewFrame,
    TraceDatasetFrame,
)
from nemo_anonymizer_plugin.sdk import http
from nemo_anonymizer_plugin.sdk.display import DisplayRecordMixin, set_original_text_column
from nemo_anonymizer_plugin.sdk.errors import (
    AnonymizerClientError,
    AnonymizerConfigValidationError,
    AnonymizerPreviewError,
)
from nemo_anonymizer_plugin.sdk.job_resources import AnonymizerJobResource, AsyncAnonymizerJobResource
from nemo_anonymizer_plugin.sdk.logging import with_logging
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.functions.frames import Done, Error, Heartbeat
from nemo_helix_plugin.sdk import NemoPluginSDKResources
from pydantic import BaseModel, TypeAdapter

logger = logging.getLogger(__name__)

PlatformResourceClient = NeMoHelix | AsyncNeMoHelix
PlatformResourceClientT = TypeVar("PlatformResourceClientT", NeMoHelix, AsyncNeMoHelix)

_PREVIEW_FRAME_ADAPTER = TypeAdapter(PreviewFrame)
_KNOWN_PREVIEW_FRAME_KINDS = {
    "log",
    "preview_dataset",
    "trace_dataset",
    "failed_records",
    "heartbeat",
    "done",
    "error",
}


def _decode_preview_frame(line: str) -> PreviewFrame | None:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        return None
    return _parse_preview_payload(payload)


def _parse_preview_payload(payload: Mapping[str, Any]) -> PreviewFrame | None:
    kind = payload.get("kind")
    if kind not in _KNOWN_PREVIEW_FRAME_KINDS:
        logger.debug("Ignoring unknown preview frame kind: %s", kind)
        return None
    return _PREVIEW_FRAME_ADAPTER.validate_python(payload)


def _build_preview_result(collector: "_PreviewFrameCollector") -> "AnonymizerPreviewResult":
    dataset = collector.dataset
    if dataset is None:
        raise AnonymizerPreviewError("No preview dataset received. Check the logs for details.")
    return AnonymizerPreviewResult(
        dataset=dataset,
        trace_dataset=collector.trace_dataset if collector.trace_dataset is not None else pd.DataFrame(),
        failed_records=collector.failed_records,
    )


@dataclass
class AnonymizerPreviewResult(DisplayRecordMixin):
    """User-facing preview result.

    Attributes:
      dataset:        Public dataframe (e.g. with replaced/rewritten text).
      trace_dataset:  Internal trace dataframe (detection + intermediate cols).
      failed_records: Per-record failures, if any.
    """

    dataset: pd.DataFrame
    trace_dataset: pd.DataFrame
    failed_records: list[dict] = field(default_factory=list)
    _display_cycle_index: int = field(default=0, init=False, repr=False)

    def _display_trace_dataframe(self) -> pd.DataFrame:
        return self.trace_dataset


@dataclass
class _PreviewFrameCollector:
    dataset: pd.DataFrame | None = None
    trace_dataset: pd.DataFrame | None = None
    failed_records: list[dict] = field(default_factory=list)
    log_levels_seen: set[str] = field(default_factory=set)

    def __enter__(self) -> "_PreviewFrameCollector":
        logger.info("Starting Anonymizer preview")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if isinstance(exc_val, AnonymizerPreviewError):
            raise exc_val
        if exc_val:
            raise _get_error(exc_val)
        if self.dataset is None:
            raise AnonymizerPreviewError("No preview dataset received. Check the logs for details.")

    def accept(self, frame: BaseModel) -> None:
        preview_frame = _parse_preview_payload(frame.model_dump(mode="json"))
        if preview_frame is None:
            return
        match preview_frame:
            case LogFrame():
                self._accept_log(preview_frame)
            case PreviewDatasetFrame():
                self.dataset = pd.DataFrame(preview_frame.records).convert_dtypes(dtype_backend="pyarrow")
            case TraceDatasetFrame():
                self.trace_dataset = pd.DataFrame(preview_frame.records).convert_dtypes(dtype_backend="pyarrow")
                set_original_text_column(self.trace_dataset, preview_frame.original_text_column)
            case FailedRecordsFrame():
                self.failed_records = list(preview_frame.records)
            case Error():
                raise AnonymizerPreviewError(preview_frame.message)
            case Heartbeat() | Done():
                pass

    def _accept_log(self, frame: LogFrame) -> None:
        self.log_levels_seen.add(frame.level)
        if frame.level == "debug":
            logger.debug(frame.message)
        elif frame.level == "info":
            logger.info(frame.message)
        elif frame.level in {"warning", "warn"}:
            logger.warning(frame.message)
        elif frame.level == "error":
            logger.error(frame.message)


class _BaseAnonymizerResource(Generic[PlatformResourceClientT]):
    def __init__(self, platform: PlatformResourceClientT) -> None:
        self._platform = platform

    def _headers(self) -> dict[str, str]:
        return http.headers(self._platform)

    def _url(self, path: str, workspace: str | None) -> str:
        return http.url(self._platform, workspace, path)

    def _client(self):
        return self._platform._client


@with_logging
class AnonymizerResource(_BaseAnonymizerResource[NeMoHelix]):
    """Sync client for the Anonymizer plugin service."""

    def preview(
        self,
        request: PreviewRequest,
        *,
        workspace: str | None = None,
    ) -> AnonymizerPreviewResult:
        """Run a streaming preview against the Anonymizer service.

        Returns an ``AnonymizerPreviewResult`` once the stream finishes.
        """
        with _PreviewFrameCollector() as collector:
            for frame in self._preview(request=request, workspace=workspace):
                collector.accept(frame)
            return _build_preview_result(collector)

    def _preview(
        self,
        *,
        request: PreviewRequest,
        workspace: str | None,
    ) -> Iterator[PreviewFrame]:
        with self._client().stream(
            "POST",
            self._url("/preview", workspace),
            headers=self._headers(),
            json=request.model_dump(mode="json", exclude_none=True),
        ) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                resp.read()
                raise
            for line in resp.iter_lines():
                if line:
                    frame = _decode_preview_frame(line)
                    if frame is not None:
                        yield frame

    def run(
        self,
        request: AnonymizerRequest,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AnonymizerJobResource:
        """Submit an anonymizer.run job."""
        try:
            resp = self._client().post(
                self._url("/jobs/run", workspace),
                headers=self._headers(),
                json={"spec": request.model_dump(mode="json")},
            )
            resp.raise_for_status()
            job = resp.json()
            logger.info(f"  |-- job name: {job['name']}")
            job_client = AnonymizerJobResource(job_name=job["name"], platform=self._platform, workspace=workspace)
            if wait_until_done:
                job_client.wait_until_done()
            return job_client
        except Exception as e:
            raise _get_error(e) from e

    def get_job_resource(self, job_name: str, workspace: str | None = None) -> AnonymizerJobResource:
        try:
            resp = self._client().get(
                self._url(f"/jobs/run/{http.path_segment(job_name)}", workspace),
                headers=self._headers(),
            )
            resp.raise_for_status()
            return AnonymizerJobResource(job_name=job_name, platform=self._platform, workspace=workspace)
        except Exception as e:
            raise _get_error(e) from e


@with_logging
class AsyncAnonymizerResource(_BaseAnonymizerResource[AsyncNeMoHelix]):
    """Async client for the Anonymizer plugin service."""

    async def preview(
        self,
        request: PreviewRequest,
        *,
        workspace: str | None = None,
    ) -> AnonymizerPreviewResult:
        with _PreviewFrameCollector() as collector:
            async for frame in self._preview(request=request, workspace=workspace):
                collector.accept(frame)
            return _build_preview_result(collector)

    async def _preview(
        self,
        *,
        request: PreviewRequest,
        workspace: str | None,
    ) -> AsyncIterator[PreviewFrame]:
        async with self._client().stream(
            "POST",
            self._url("/preview", workspace),
            headers=self._headers(),
            json=request.model_dump(mode="json", exclude_none=True),
        ) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                await resp.aread()
                raise
            async for line in resp.aiter_lines():
                if line:
                    frame = _decode_preview_frame(line)
                    if frame is not None:
                        yield frame

    async def run(
        self,
        request: AnonymizerRequest,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AsyncAnonymizerJobResource:
        try:
            resp = await self._client().post(
                self._url("/jobs/run", workspace),
                headers=self._headers(),
                json={"spec": request.model_dump(mode="json")},
            )
            resp.raise_for_status()
            job = resp.json()
            job_client = AsyncAnonymizerJobResource(job_name=job["name"], platform=self._platform, workspace=workspace)
            if wait_until_done:
                await job_client.wait_until_done()
            return job_client
        except Exception as e:
            raise _get_error(e) from e

    async def get_job_resource(self, job_name: str, workspace: str | None = None) -> AsyncAnonymizerJobResource:
        try:
            resp = await self._client().get(
                self._url(f"/jobs/run/{http.path_segment(job_name)}", workspace),
                headers=self._headers(),
            )
            resp.raise_for_status()
            return AsyncAnonymizerJobResource(job_name=job_name, platform=self._platform, workspace=workspace)
        except Exception as e:
            raise _get_error(e) from e


def _get_error(e: BaseException) -> AnonymizerClientError:
    if isinstance(e, httpx.HTTPStatusError):
        _read_error_response(e.response)
        detail = _response_text(e.response)
        detail = _response_json_detail(e.response, fallback=detail)
        if e.response.status_code == 422:
            return AnonymizerConfigValidationError(f"Config validation failed!\n{detail}")
        return AnonymizerClientError(f"Something went wrong!\n{detail}")
    return AnonymizerClientError(f"Something went wrong!\n{e}")


def _read_error_response(response: httpx.Response) -> None:
    try:
        response.read()
    except (httpx.HTTPError, httpx.StreamError):
        logger.debug("Failed to read Anonymizer error response body.", exc_info=True)


def _response_text(response: httpx.Response) -> str:
    try:
        return response.text
    except httpx.ResponseNotRead:
        logger.debug("Cannot include Anonymizer error response body because it was not read.", exc_info=True)
        return response.reason_phrase or f"HTTP {response.status_code}"
    except UnicodeDecodeError:
        logger.debug("Failed to decode Anonymizer error response body as text.", exc_info=True)
        return response.reason_phrase or f"HTTP {response.status_code}"


def _response_json_detail(response: httpx.Response, *, fallback: str) -> str:
    try:
        detail_json = response.json()
    except json.JSONDecodeError:
        logger.debug("Anonymizer error response body is not JSON.", exc_info=True)
        return fallback
    except UnicodeDecodeError:
        logger.debug("Failed to decode Anonymizer error response JSON.", exc_info=True)
        return fallback
    except httpx.ResponseNotRead:
        logger.debug("Cannot parse Anonymizer error response JSON because the body was not read.", exc_info=True)
        return fallback

    if not isinstance(detail_json, Mapping):
        logger.debug("Anonymizer error response JSON was %s, expected object.", type(detail_json).__name__)
        return fallback

    detail = detail_json.get("detail")
    if detail is None:
        logger.debug("Anonymizer error response JSON did not include a 'detail' field.")
        return fallback
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, default=str)


anonymizer_sdk_resources = NemoPluginSDKResources(
    sync_resource=AnonymizerResource,
    async_resource=AsyncAnonymizerResource,
)
