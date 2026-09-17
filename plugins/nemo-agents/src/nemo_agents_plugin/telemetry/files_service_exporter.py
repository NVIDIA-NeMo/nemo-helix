# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Agent Toolkit telemetry exporter that writes traces via the Nemo Files API.

Uses the typed Files client to store serialized
:class:`~nat.data_models.intermediate_step.IntermediateStep` records as JSONL
under the shared fileset **nemo-agent-telemetry**, scoped by workspace in the
API URL and by agent name in object paths.

Configuration (under ``general.telemetry.tracing``)::

    nemo_files_trace:
      _type: nemo_files
      project: my-agent          # required; always appears as the first path segment (slugified)
      # workspace / agent_name are optional; the agents plugin injects them at deploy
      # and invoke time. Override here for local runs.
      batch_size: 32
      # path_prefix optional; default extra segments after project: "telemetry"
      # Optional relative suffix only; slug(project) is always prepended.
      # Use "{agent}/..." inside path_prefix to add an agent subdirectory.

See also: `NeMo Agent Toolkit — custom telemetry exporters
<https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/custom-components/telemetry-exporters.html>`_.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from nat.builder.builder import Builder  # type: ignore[unresolved-import]
from nat.cli.register_workflow import register_telemetry_exporter  # type: ignore[unresolved-import]
from nat.data_models.intermediate_step import IntermediateStep  # type: ignore[unresolved-import]
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig  # type: ignore[unresolved-import]
from nat.observability.exporter.base_exporter import IsolatedAttribute  # type: ignore[unresolved-import]
from nat.observability.exporter.raw_exporter import RawExporter  # type: ignore[unresolved-import]
from nat.observability.processor.intermediate_step_serializer import (  # type: ignore[unresolved-import]
    IntermediateStepSerializer,
)
from nemo_agents_plugin.utils import get_base_url
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from pydantic import Field

logger = logging.getLogger(__name__)

TELEMETRY_FILESET_NAME = "nemo-agent-telemetry"


class NemoFilesTelemetryExporterConfig(TelemetryExporterBaseConfig, name="nemo_files"):
    """Settings for :class:`NemoFilesServiceRawExporter`."""

    workspace: str = Field(
        default="",
        description="Nemo workspace for Files API; injected at deploy time",
    )
    agent_name: str = Field(default="", description="Logical agent name; injected at deploy time.")
    batch_size: int = Field(default=64, ge=1, description="Lines per upload object.")
    max_buffer_size: int = Field(
        default=1024,
        ge=1,
        description="Maximum buffered items. Oldest items are dropped when exceeded.",
    )


class NemoFilesServiceRawExporter(RawExporter[IntermediateStep, str]):
    """Serialize intermediate steps and upload batched JSONL to a workspace fileset.

    The ``_flush_lock`` guards only the in-memory buffer and sequence counter,
    **never** the HTTP upload.  This avoids a deadlock where holding an
    ``asyncio.Lock`` across multiple awaited HTTP round-trips can starve
    or block concurrent export tasks on the same event loop.
    """

    _buffer: IsolatedAttribute[list[str]] = IsolatedAttribute(list)
    _part_seq: IsolatedAttribute[list[int]] = IsolatedAttribute(lambda: [0])
    # Time-prefixed so run dirs are sortable by recency.
    _run_id: IsolatedAttribute[list[str]] = IsolatedAttribute(
        lambda: [f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:8]}"]
    )
    _flush_lock: IsolatedAttribute[asyncio.Lock] = IsolatedAttribute(asyncio.Lock)
    _upload_semaphore: IsolatedAttribute[asyncio.Semaphore] = IsolatedAttribute(lambda: asyncio.Semaphore(2))

    def __init__(
        self,
        *,
        files_client: AsyncFilesClient,
        workspace: str,
        agent_name: str,
        batch_size: int,
        max_buffer_size: int,
        context_state=None,
    ) -> None:
        super().__init__(context_state=context_state)
        self._files_client = files_client
        self._workspace = workspace
        self._agent_name = agent_name
        self._batch_size = batch_size
        self._max_buffer_size = max_buffer_size
        self.add_processor(IntermediateStepSerializer())

    async def export_processed(self, item: str) -> None:
        batch: list[str] | None = None
        seq: int = 0
        async with self._flush_lock:
            self._buffer.append(item)
            if len(self._buffer) >= self._batch_size:
                batch = list(self._buffer)
                self._buffer.clear()
                seq = self._part_seq[0]
                self._part_seq[0] += 1

        if batch is not None:
            await self._upload_batch(batch, seq)

    async def _upload_batch(self, batch: list[str], seq: int) -> None:
        """Upload a batch of serialized spans.  Called **without** the flush lock."""
        async with self._upload_semaphore:
            run = self._run_id[0]
            remote_path = f"{self._agent_name}/runs/{run}/part-{seq:06d}.jsonl"
            body = "\n".join(batch) + "\n"
            logger.debug(
                "flush: uploading %d spans to %s (seq=%d)",
                len(batch),
                remote_path,
                seq,
            )
            try:
                await self._files_client.create_fileset(
                    workspace=self._workspace,
                    body=CreateFilesetRequest(name=TELEMETRY_FILESET_NAME),
                    exist_ok=True,
                )
                await self._files_client.upload_file(
                    workspace=self._workspace,
                    name=TELEMETRY_FILESET_NAME,
                    path=remote_path,
                    content=body.encode("utf-8"),
                )
                logger.debug("flush: upload OK (%s)", remote_path)
            except Exception:
                async with self._flush_lock:
                    self._buffer[:0] = batch
                    overflow = len(self._buffer) - self._max_buffer_size
                    if overflow > 0:
                        del self._buffer[:overflow]
                        logger.warning(
                            "nemo_files telemetry buffer full; dropped %d oldest items",
                            overflow,
                        )
                logger.exception(
                    "nemo_files telemetry upload failed (fileset=%r workspace=%r path=%r)",
                    TELEMETRY_FILESET_NAME,
                    self._workspace,
                    remote_path,
                )

    async def _cleanup(self) -> None:
        batch: list[str] | None = None
        seq: int = 0
        async with self._flush_lock:
            if self._buffer:
                batch = list(self._buffer)
                self._buffer.clear()
                seq = self._part_seq[0]
                self._part_seq[0] += 1

        if batch is not None:
            await self._upload_batch(batch, seq)

        await super()._cleanup()


@register_telemetry_exporter(config_type=NemoFilesTelemetryExporterConfig)
async def nemo_files_telemetry_exporter(config: NemoFilesTelemetryExporterConfig, builder: Builder):
    """Build an exporter that uploads telemetry to the Nemo Files service."""
    del builder  # unused; required by NAT registration signature

    async_sdk = AsyncNeMoPlatform(base_url=get_base_url())
    exporter = NemoFilesServiceRawExporter(
        files_client=client_from_platform(async_sdk, AsyncFilesClient),
        workspace=config.workspace,
        agent_name=config.agent_name,
        batch_size=config.batch_size,
        max_buffer_size=config.max_buffer_size,
    )
    try:
        yield exporter
    finally:
        await async_sdk.close()
