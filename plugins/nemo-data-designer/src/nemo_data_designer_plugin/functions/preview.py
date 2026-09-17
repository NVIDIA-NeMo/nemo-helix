# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar, cast

import anyio
import anyio.from_thread
import anyio.to_thread
import data_designer.config as dd
from anyio.lowlevel import current_token
from data_designer.config.utils.io_helpers import serialize_data
from data_designer.errors import DataDesignerError
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.context import create_execution_context, create_validation_context
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError, raise_if_errors
from data_designer_nemo.fileset_file_seed_reader import workspace_cvar
from data_designer_nemo.runnable import resolve_runnable_config
from nemo_data_designer_plugin._data_designer import create_data_designer
from nemo_data_designer_plugin.config import get_config
from nemo_data_designer_plugin.functions._preview_logs import forward_data_designer_logs
from nemo_data_designer_plugin.functions._types import (
    AnalysisFrame,
    DatasetFrame,
    DatasetMetadataFrame,
    LogFrame,
    PreviewSpec,
    ProcessorOutputFrame,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.frames import Done, Error
from pydantic import BaseModel


class PreviewMessageDeliveryError(Exception): ...


class PreviewFunction(NemoFunction[PreviewSpec]):
    name: ClassVar[str] = "preview"
    description: ClassVar[str] = "Generate a small preview dataset by streaming NDJSON frames."
    spec_schema: ClassVar[type[PreviewSpec]] = PreviewSpec
    generate_legacy_verbs: ClassVar[bool] = False

    async def run(
        self,
        spec: PreviewSpec,
        *,
        ctx: FunctionContext,
        sdk: NeMoPlatform,
        async_sdk: AsyncNeMoPlatform,
    ) -> AsyncIterator[BaseModel]:
        # Fail fast on request shape (``num_records``) before doing any config-validation work.
        num_records = _validate_and_get_num_records(spec.num_records)

        validation_ctx = create_validation_context(async_sdk, ctx.workspace)
        errors, _, model_providers = await resolve_runnable_config(validation_ctx, spec.config)
        raise_if_errors(errors)
        execution_ctx = create_execution_context(
            sdk,
            ctx.workspace,
            validated_roots=validation_ctx.validated_filesystem_roots,
        )

        workspace_cvar.set(ctx.workspace)

        send_stream, receive_stream = anyio.create_memory_object_stream[BaseModel]()
        token = current_token()

        def send_from_thread(frame: BaseModel) -> None:
            try:
                anyio.from_thread.run(send_stream.send, frame, token=token)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                raise PreviewMessageDeliveryError(
                    "Caught an anyio resource error. Most likely the request was canceled."
                ) from None

        config_builder = dd.DataDesignerConfigBuilder.from_config(spec.config.to_dict())
        with tempfile.TemporaryDirectory() as artifact_storage_tmpdir:
            data_designer = create_data_designer(
                artifact_path=artifact_storage_tmpdir,
                model_providers=model_providers,
                dd_ctx=execution_ctx,
            )

            try:
                await anyio.to_thread.run_sync(data_designer.check_models, config_builder)
            except DataDesignerError as e:
                raise NDDInvalidConfigError(str(e))
            except TimeoutError as e:
                raise NDDInternalError(str(e))

            async def _worker() -> None:
                try:
                    await anyio.to_thread.run_sync(
                        _make_preview_dataset,
                        send_from_thread,
                        data_designer,
                        config_builder,
                        num_records,
                    )
                except Exception as exc:
                    try:
                        await send_stream.send(LogFrame(level="error", message=f"An error occurred: {exc}"))
                        await send_stream.send(Error(message=str(exc), details={"type": type(exc).__name__}))
                    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                        pass
                finally:
                    await send_stream.aclose()

            completed_with_error = False
            async with anyio.create_task_group() as tg:
                tg.start_soon(_worker)
                async with receive_stream:
                    async for frame in receive_stream:
                        if isinstance(frame, Error):
                            completed_with_error = True
                        yield frame
                if not completed_with_error:
                    yield Done()


def _validate_and_get_num_records(requested_num_records: int | None) -> int:
    """Resolve the effective ``num_records``, raising if the user asked for too many."""
    config = get_config()
    num_records = config.preview_num_records.default
    if requested_num_records:
        if requested_num_records > config.preview_num_records.max:
            raise NDDInvalidConfigError(f"Max num records for preview requests is {config.preview_num_records.max}")
        num_records = requested_num_records

    return num_records


def _make_preview_dataset(
    send_frame: Callable[[BaseModel], None],
    data_designer: DataDesigner,
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
) -> None:
    """
    Synchronous function that runs on a worker thread under
    :func:`anyio.to_thread.run_sync`. The Data Designer engine and its
    filesystem readers use sync APIs here; ``send_frame`` is the only bridge
    back to the async request context.
    """
    with forward_data_designer_logs(send_frame):
        preview_results = data_designer.preview(config_builder, num_records=num_records)

    if (dataset_metadata := preview_results.dataset_metadata) is not None:
        send_frame(DatasetMetadataFrame(metadata=dataset_metadata))

    if (dataset := preview_results.dataset) is not None:
        records = cast(list[dict[str, Any]], dataset.to_dict(orient="records"))
        send_frame(DatasetFrame(records=_to_jsonable_records(records)))

    for processor_name, processor_records in (preview_results.processor_artifacts or {}).items():
        send_frame(
            ProcessorOutputFrame(
                processor_name=processor_name,
                records=_to_jsonable_records(processor_records),
            )
        )

    if (analysis := preview_results.analysis) is not None:
        send_frame(AnalysisFrame(analysis=analysis))


def _to_jsonable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads(serialize_data(records)))
