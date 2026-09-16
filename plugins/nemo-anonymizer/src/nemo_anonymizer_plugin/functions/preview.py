# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming preview function for the Anonymizer plugin."""

from __future__ import annotations

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
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec, PreparedAnonymizerInput
from nemo_anonymizer_plugin.app.model_configs import (
    build_model_configs_yaml,
    validate_selected_models_have_model_configs,
)
from nemo_anonymizer_plugin.app.task_config import PreviewRequest
from nemo_anonymizer_plugin.config import get_config
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.function import NemoFunction
from nemo_helix_plugin.function_context import FunctionContext
from nemo_helix_plugin.functions.frames import Done, Error, FrameModel, Heartbeat
from pydantic import BaseModel, Field

LogLevel = Literal["debug", "info", "warning", "error"]


class PreviewMessageDeliveryError(Exception): ...


# Wire request payload. Uses the shared ``PreviewRequest`` model directly.
PreviewSpec: TypeAlias = PreviewRequest


class LogFrame(FrameModel):
    kind: Literal["log"] = "log"
    level: LogLevel
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
    LogFrame | PreviewDatasetFrame | TraceDatasetFrame | FailedRecordsFrame | Heartbeat | Done | Error,
    Field(discriminator="kind"),
]


class PreviewFunction(NemoFunction[PreviewSpec]):
    name: ClassVar[str] = "preview"
    description: ClassVar[str] = "Streaming preview of an Anonymizer config."
    spec_schema: ClassVar[type[BaseModel]] = PreviewSpec
    frame_schema: ClassVar[Any] = PreviewFrame
    generate_legacy_verbs: ClassVar[bool] = False

    async def run(
        self,
        spec: PreviewSpec,
        *,
        ctx: FunctionContext,
        async_sdk: AsyncNeMoHelix,
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
                        _make_preview,
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
