# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming preview function for the Anonymizer plugin."""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import anyio
import anyio.from_thread
import anyio.to_thread
from anyio.lowlevel import current_token
from fastapi import HTTPException, status
from nemo_anonymizer_plugin.app.context import (
    AnonymizerContext,
    create_anonymizer_context,
    require_model_configs_for_execution,
)
from nemo_anonymizer_plugin.app.errors import AnonymizerInternalError, AnonymizerInvalidConfigError
from nemo_anonymizer_plugin.app.gliner_detector import (
    caller_supplied_entity_detector,
    ensure_gliner_fileset_async,
    is_gliner_cached,
    prewarm_gliner_cache,
)
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec, PreparedAnonymizerInput
from nemo_anonymizer_plugin.app.model_configs import (
    build_model_configs_yaml,
    validate_selected_models_have_model_configs,
)
from nemo_anonymizer_plugin.app.task_config import PreviewRequest
from nemo_anonymizer_plugin.config import get_config
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.function import NemoFunction
from nemo_helix_plugin.function_context import FunctionContext
from nemo_helix_plugin.functions.frames import Done, Error, FrameModel, Heartbeat
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LogLevel = Literal["debug", "info", "warning", "error"]


class PreviewMessageDeliveryError(Exception): ...


# Wire request payload. Uses the shared ``PreviewRequest`` model directly.
PreviewSpec: TypeAlias = PreviewRequest


class LogFrame(FrameModel):
    kind: Literal["log"] = "log"
    level: LogLevel
    message: str


class ModelDownloadFrame(FrameModel):
    """Signals the first-use GLiNER detector weight download so UIs can show it.

    Emitted with ``status='started'`` before the (one-time, ~1.7G) pull-through
    download and ``status='complete'`` once the weights are cached. Only emitted
    when a real download happens; a warm cache yields no frame.
    """

    kind: Literal["model_download"] = "model_download"
    status: Literal["started", "complete"]
    message: str


class PreviewDatasetFrame(FrameModel):
    """Final user-visible dataframe produced by the preview run."""

    kind: Literal["preview_dataset"] = "preview_dataset"
    records: list[dict[str, Any]]


class TraceDatasetFrame(FrameModel):
    """Internal trace dataframe — useful for debugging detection/replacement."""

    kind: Literal["trace_dataset"] = "trace_dataset"
    records: list[dict[str, Any]]
    original_text_column: str | None = None


class FailedRecordsFrame(FrameModel):
    """Records that failed during the pipeline, with reasons."""

    kind: Literal["failed_records"] = "failed_records"
    records: list[dict[str, Any]]


PreviewFrame: TypeAlias = Annotated[
    LogFrame
    | ModelDownloadFrame
    | PreviewDatasetFrame
    | TraceDatasetFrame
    | FailedRecordsFrame
    | Heartbeat
    | Done
    | Error,
    Field(discriminator="kind"),
]


class PreviewFunction(NemoFunction[PreviewSpec]):
    name: ClassVar[str] = "preview"
    description: ClassVar[str] = "Streaming preview of an Anonymizer config."
    spec_schema: ClassVar[type[BaseModel]] = PreviewSpec
    frame_schema: ClassVar[Any] = PreviewFrame

    async def run(
        self,
        spec: PreviewSpec,
        *,
        ctx: FunctionContext,
        async_sdk: AsyncNemoClient,
    ) -> AsyncIterator[BaseModel]:
        num_records = _validate_and_get_num_records(spec.num_records)
        validate_selected_models_have_model_configs(
            model_configs=spec.model_configs,
            selected_models=spec.selected_models,
        )
        model_configs = require_model_configs_for_execution(spec.model_configs)

        anon_ctx = create_anonymizer_context(async_sdk, ctx.workspace)
        dd_providers = await anon_ctx.make_model_providers(model_configs)
        model_configs_yaml = build_model_configs_yaml(
            model_configs=model_configs,
            selected_models=spec.selected_models,
        )

        use_in_process_detector = not caller_supplied_entity_detector(spec.selected_models)
        if use_in_process_detector:
            await ensure_gliner_fileset_async(async_sdk)
            if not is_gliner_cached():
                download_message = (
                    "Downloading PII detector model (~1.7G, first run only); subsequent runs load from cache."
                )
                yield ModelDownloadFrame(status="started", message=download_message)
                try:
                    await anyio.to_thread.run_sync(
                        prewarm_gliner_cache, str(async_sdk.base_url), abandon_on_cancel=True
                    )
                except Exception:
                    # The download runs after the first frame is sent, so the framework
                    # can no longer turn this into an HTTP error — surface it as an
                    # in-stream Error frame instead of letting the stream die silently.
                    # Log the real exception internally; keep its text (which can carry a
                    # filesystem path or connection detail) out of the client-facing frames.
                    logger.exception("GLiNER PII detector model download failed")
                    failure_message = "Failed to download the PII detector model."
                    yield LogFrame(level="error", message=failure_message)
                    yield Error(message=failure_message, details={"type": "ModelDownloadError"})
                    return
                yield ModelDownloadFrame(status="complete", message="PII detector model ready.")

        async with _prepare_input(anon_ctx, spec.data) as prepared_input:
            send_stream, receive_stream = anyio.create_memory_object_stream[BaseModel]()
            token = current_token()

            def send_from_thread(frame: BaseModel) -> None:
                try:
                    anyio.from_thread.run(send_stream.send, frame, token=token)
                except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                    raise PreviewMessageDeliveryError(
                        "Caught an anyio resource error. Most likely the request was canceled."
                    ) from None

            from nemo_anonymizer_plugin.functions._preview_logs import attach_preview_handler, request_callback_cvar
            from nemo_anonymizer_plugin.functions._preview_worker import _make_preview

            attach_preview_handler()
            callback_token = request_callback_cvar.set(send_from_thread)

            async def _worker() -> None:
                try:
                    await anyio.to_thread.run_sync(
                        functools.partial(_make_preview, use_in_process_detector=use_in_process_detector),
                        send_from_thread,
                        spec,
                        prepared_input.input,
                        model_configs_yaml,
                        dd_providers,
                        num_records,
                        abandon_on_cancel=True,
                    )
                except (AnonymizerInvalidConfigError, AnonymizerInternalError) as exc:
                    try:
                        await send_stream.send(LogFrame(level="error", message=f"An error occurred: {exc}"))
                        await send_stream.send(Error(message=str(exc), details={"type": type(exc).__name__}))
                    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                        pass
                except Exception as exc:
                    try:
                        await send_stream.send(LogFrame(level="error", message=f"An error occurred: {exc}"))
                        await send_stream.send(Error(message=str(exc), details={"type": type(exc).__name__}))
                    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                        pass
                finally:
                    await send_stream.aclose()

            completed_with_error = False
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_worker)
                    async with receive_stream:
                        async for frame in receive_stream:
                            if isinstance(frame, Error):
                                completed_with_error = True
                            yield frame
                    if not completed_with_error:
                        yield Done()
            finally:
                request_callback_cvar.reset(callback_token)


def _validate_and_get_num_records(requested_num_records: int | None) -> int:
    config = get_config()
    num_records = config.preview_num_records.default
    if requested_num_records is not None:
        if requested_num_records > config.preview_num_records.max:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Max num records for preview requests is {config.preview_num_records.max}",
            )
        num_records = requested_num_records
    return num_records


@asynccontextmanager
async def _prepare_input(
    anon_ctx: AnonymizerContext,
    data: AnonymizerInputSpec,
) -> AsyncIterator[PreparedAnonymizerInput]:
    prepared_input = await anon_ctx.prepare_input(data)
    try:
        yield prepared_input
    finally:
        prepared_input.cleanup()
