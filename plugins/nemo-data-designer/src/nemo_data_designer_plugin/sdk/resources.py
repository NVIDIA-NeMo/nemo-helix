# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Generic, TypeVar

import data_designer.config as dd
import httpx
import pandas as pd
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.preview_results import PreviewResults
from data_designer.config.utils.info import InterfaceInfo
from data_designer.logging import RandomEmoji
from nemo_data_designer_plugin.functions._types import (
    AnalysisFrame,
    DatasetFrame,
    DatasetMetadataFrame,
    LogFrame,
    PreviewFrame,
    PreviewSpec,
    ProcessorOutputFrame,
)
from nemo_data_designer_plugin.jobs.retrieval_spec import (
    RetrievalGenerateJobConfig,
    RetrievalPrepareJobConfig,
    RetrievalRunJobConfig,
)
from nemo_data_designer_plugin.jobs.spec import DataDesignerJobConfig
from nemo_data_designer_plugin.sdk.errors import (
    DataDesignerClientError,
    DataDesignerConfigValidationError,
    DataDesignerPreviewError,
    extract_http_error_info,
)
from nemo_data_designer_plugin.sdk.job_resources import AsyncDataDesignerJobResource, DataDesignerJobResource
from nemo_data_designer_plugin.sdk.logging import with_logging
from nemo_data_designer_plugin.sdk.validation import (
    ValidationReport,
    validate_config,
    validate_config_sync,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.data_designer.client import AsyncDataDesignerClient, DataDesignerClient
from nemo_platform_plugin.data_designer.types import DataDesignerJobCollection, DataDesignerJobRequest, PreviewRequest
from nemo_platform_plugin.functions.frames import Done, Error, Heartbeat
from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient
from nemo_platform_plugin.models.types import ModelProvider as NMPModelProvider
from nemo_platform_plugin.sdk import NemoPluginSDKResources
from pydantic import BaseModel, TypeAdapter

logger = logging.getLogger(__name__)

PlatformResourceClient = NeMoPlatform | AsyncNeMoPlatform
PlatformResourceClientT = TypeVar("PlatformResourceClientT", NeMoPlatform, AsyncNeMoPlatform)

_PREVIEW_FRAME_ADAPTER = TypeAdapter(PreviewFrame)
_KNOWN_PREVIEW_FRAME_KINDS = {
    "analysis",
    "dataset",
    "dataset_metadata",
    "done",
    "error",
    "heartbeat",
    "log",
    "processor_output",
}


def _decode_preview_frame(line: str) -> PreviewFrame | None:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        logger.debug("Ignoring non-object preview frame: %r", payload)
        return None
    return _parse_preview_payload(payload)


def _parse_preview_frame(frame: BaseModel) -> PreviewFrame | None:
    return _parse_preview_payload(frame.model_dump(mode="json"))


def _parse_preview_payload(payload: Mapping[str, Any]) -> PreviewFrame | None:
    kind = payload.get("kind")
    if kind not in _KNOWN_PREVIEW_FRAME_KINDS:
        logger.debug("Ignoring unknown preview frame kind: %s", kind)
        return None
    return _PREVIEW_FRAME_ADAPTER.validate_python(payload)


@dataclass
class _PreviewFrameCollector:
    """Collect and validate frames from the streaming preview response."""

    dataset: pd.DataFrame | None = None
    dataset_metadata: DatasetMetadata | None = None
    analysis: DatasetProfilerResults | None = None
    processor_artifacts: dict[str, list[dict]] = field(default_factory=dict)
    log_levels_seen: set[str] = field(default_factory=set)

    def __enter__(self):
        logger.info("🚀 Starting preview generation")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if isinstance(exc_val, DataDesignerPreviewError):
            raise exc_val
        if exc_val:
            raise _get_error(exc_val)
        if self.dataset is None:
            raise DataDesignerPreviewError("No dataset generated. Check the logs for details about what failed.")
        if self.dataset_metadata is None:
            raise DataDesignerPreviewError(
                "No dataset metadata received. Check the logs for details about what failed."
            )
        if self.analysis is None:
            raise DataDesignerPreviewError("No analysis received. Check the logs for details about what failed.")
        self._log_end_preview()

    def accept(self, frame: BaseModel) -> None:
        preview_frame = _parse_preview_frame(frame)
        if preview_frame is None:
            return

        match preview_frame:
            case LogFrame():
                self._accept_log(preview_frame)
            case DatasetFrame():
                self._accept_dataset(preview_frame)
            case DatasetMetadataFrame():
                self._accept_dataset_metadata(preview_frame)
            case AnalysisFrame():
                self._accept_analysis(preview_frame)
            case ProcessorOutputFrame():
                self._accept_processor_output(preview_frame)
            case Error():
                raise DataDesignerPreviewError(preview_frame.message)
            case Heartbeat() | Done():
                pass

    def _accept_log(self, frame: LogFrame) -> None:
        level = frame.level

        self.log_levels_seen.add(level)

        if level == "debug":
            logger.debug(frame.message)
        elif level == "info":
            logger.info(frame.message)
        elif level in {"warning", "warn"}:
            logger.warning(frame.message)
        elif level == "error":
            logger.error(frame.message)

    def _accept_dataset(self, frame: DatasetFrame) -> None:
        # The dataset is mission-critical. If we can't load it from the message,
        # or we can but the dataset is empty, raise a preview error.
        try:
            self.dataset = pd.DataFrame(frame.records).convert_dtypes(dtype_backend="pyarrow")
        except Exception as e:
            raise DataDesignerPreviewError(f"🛑 Error generating preview dataset: {e}") from e

        if len(self.dataset) == 0:
            raise DataDesignerPreviewError(
                "🛑 Dataset is empty — all records were dropped due to generation or processing failures. "
                "Check the warnings above for details on which columns failed."
            )

    def _accept_dataset_metadata(self, frame: DatasetMetadataFrame) -> None:
        # Dataset metadata is mission-critical. If we can't load it, raise a preview error.
        try:
            self.dataset_metadata = frame.metadata
        except Exception as e:
            raise DataDesignerPreviewError(f"🛑 Error loading dataset metadata: {e}") from e

    def _accept_analysis(self, frame: AnalysisFrame) -> None:
        # Analysis is mission-critical. If we can't load it, raise a preview error.
        try:
            self.analysis = frame.analysis
        except Exception as e:
            raise DataDesignerPreviewError(f"🛑 Error profiling preview dataset: {e}") from e

    def _accept_processor_output(self, frame: ProcessorOutputFrame) -> None:
        # If a processor artifact fails to deserialize, log it but keep the preview result.
        try:
            self.processor_artifacts[frame.processor_name] = frame.records
        except Exception as e:
            logger.error(f"🛑 Error loading processor output: {e}")
            self.log_levels_seen.add("error")

    def get_processor_artifacts(self) -> dict[str, list[dict]] | None:
        return self.processor_artifacts or None

    def _log_end_preview(self) -> None:
        if "error" in self.log_levels_seen:
            logger.error("🛑 Preview completed with errors.")
        elif "warning" in self.log_levels_seen or "warn" in self.log_levels_seen:
            logger.warning("⚠️ Preview completed with warnings.")
        else:
            logger.info(f"{RandomEmoji.success()} Preview complete!")


class _BaseDataDesignerResource(Generic[PlatformResourceClientT]):
    """Shared platform handle for sync and async plugin SDK resources."""

    def __init__(self, platform: PlatformResourceClientT) -> None:
        self._platform = platform


@with_logging
class DataDesignerResource(_BaseDataDesignerResource[NeMoPlatform]):
    """High-level sync client for the Data Designer plugin service."""

    def __init__(self, platform: NeMoPlatform) -> None:
        super().__init__(platform)
        self._data_designer_client = client_from_platform(platform, DataDesignerClient)
        self._models_client = client_from_platform(platform, ModelsClient)

    def preview(
        self,
        config_builder: dd.DataDesignerConfigBuilder,
        *,
        num_records: int | None = None,
        workspace: str | None = None,
    ) -> PreviewResults:
        """Generate a set of preview records based on your current Data Designer configuration.

        This method is meant for fast iteration on your Data Designer configuration.

        Args:
            config_builder: Data Designer configuration builder.
            num_records: The number of records to generate. Must be less than or equal to the
                service-side configured max number of preview records.
            workspace: The workspace to run the request in. If not supplied, uses the workspace
                of the base NeMoPlatform object.
            timeout: The timeout for the preview call in seconds.

        Returns:
            An object containing the preview dataset and tools for inspecting the results.
        """
        config = config_builder.build()
        request = PreviewSpec(config=config, num_records=num_records)

        with _PreviewFrameCollector() as message_collector:
            for frame in self._preview(request=request, workspace=workspace):
                message_collector.accept(frame)
            return PreviewResults(
                config_builder=config_builder,
                dataset=message_collector.dataset,
                dataset_metadata=message_collector.dataset_metadata,
                analysis=message_collector.analysis,
                processor_artifacts=message_collector.get_processor_artifacts(),
            )

    def _preview(
        self,
        *,
        request: PreviewSpec,
        workspace: str | None,
    ) -> Iterator[PreviewFrame]:
        body = PreviewRequest.model_validate(request.model_dump(mode="json", exclude_none=True))
        with self._data_designer_client.preview(workspace=workspace, body=body).stream() as frames:
            for frame in frames:
                preview_frame = _parse_preview_frame(frame)
                if preview_frame is not None:
                    yield preview_frame

    def create(
        self,
        config_builder: dd.DataDesignerConfigBuilder,
        *,
        num_records: int = 100,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> DataDesignerJobResource:
        """Create a Data Designer generation job.

        Args:
            config_builder: Data Designer configuration builder.
            num_records: The number of records to generate.
            workspace: The workspace in which to run the job. If not supplied, uses
                the workspace of the base NeMoPlatform object.
            wait_until_done: Set to True to poll the job status and block until the
                job reaches a terminal state.

        Returns:
            An object with methods for querying the job's status and results.
        """
        config = config_builder.build()
        request = DataDesignerJobConfig(config=config, num_records=num_records)
        return self._submit_named_job("create", request, workspace=workspace, wait_until_done=wait_until_done)

    def get_job_resource(self, job_name: str, workspace: str | None = None) -> DataDesignerJobResource:
        """Get a high-level resource for an existing data generation job.

        Args:
            job_name: The name of the job.
            workspace: The workspace in which the job ran.

        Returns:
            An object containing methods for querying job status,
            retrieving the generated dataset, and accessing job metadata.

        Raises:
            ValueError: If the job ID provided is empty.
        """
        try:
            self._data_designer_client.get_job(workspace=workspace, name=job_name).data()
        except Exception as e:
            raise _get_error(e) from e
        return DataDesignerJobResource(job_name=job_name, client=self._data_designer_client, workspace=workspace)

    def get_default_model_configs(self) -> list[dd.ModelConfig]:
        """Default model configs are not supported in the NeMo Platform Data Designer service."""

        return []

    def get_default_model_providers(self) -> list[dd.ModelProvider]:
        """Get the model providers available for inference.

        Returns:
            A list of ModelProvider objects available for inference.
        """
        nmp_providers = self._models_client.list_providers(workspace="-")
        return [_nmp_provider_to_ndd_provider(self._models_client, provider) for provider in nmp_providers.items()]

    def get_info(self) -> InterfaceInfo:
        return InterfaceInfo(model_providers=self.get_default_model_providers())

    def validate(
        self,
        config_builder: dd.DataDesignerConfigBuilder,
        *,
        workspace: str | None = None,
    ) -> ValidationReport:
        """Validate a Data Designer config.

        This runs the same client-side checks ``preview`` / ``create`` perform
        internally, but never short-circuits — every detectable problem is
        reported. The remote pass is a client-side simulation and does not
        contact the data-designer service.

        Args:
            config_builder: Data Designer configuration builder.
            workspace: Workspace used to resolve provider references and seed
                sources for the remote pass. Falls back to the platform
                client's default workspace, then to ``"default"``.

        Returns:
            A :class:``ValidationReport``
        """
        resolved_workspace = workspace or self._platform.workspace or "default"
        return validate_config_sync(
            config_builder,
            sdk=self._platform,
            workspace=resolved_workspace,
        )

    def retrieval_generate(
        self,
        spec: RetrievalGenerateJobConfig,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> DataDesignerJobResource:
        """Submit a Stage 0 retrieval SDG job."""
        return self._submit_named_job("retrieval-generate", spec, workspace=workspace, wait_until_done=wait_until_done)

    def retrieval_prepare(
        self,
        spec: RetrievalPrepareJobConfig,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> DataDesignerJobResource:
        """Submit a Stage 1 retrieval prepare job."""
        return self._submit_named_job("retrieval-prepare", spec, workspace=workspace, wait_until_done=wait_until_done)

    def retrieval_run(
        self,
        spec: RetrievalRunJobConfig,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> DataDesignerJobResource:
        """Submit generate then prepare as one multi-step jobs-service workflow."""
        return self._submit_named_job("retrieval-run", spec, workspace=workspace, wait_until_done=wait_until_done)

    def _submit_named_job(
        self,
        job_name: DataDesignerJobCollection,
        spec: BaseModel,
        *,
        workspace: str | None,
        wait_until_done: bool,
    ) -> DataDesignerJobResource:
        try:
            body = DataDesignerJobRequest.model_validate({"spec": spec.model_dump(mode="json")})
            job = self._data_designer_client.create_job(workspace=workspace, job_collection=job_name, body=body).data()
            logger.info(f"  |-- job name: {job.name}")
            job_client = DataDesignerJobResource(
                job_name=job.name,
                client=self._data_designer_client,
                workspace=workspace,
                job_collection=job_name,
            )
            if wait_until_done:
                job_client.wait_until_done()
            return job_client
        except Exception as e:
            raise _get_error(e) from e


@with_logging
class AsyncDataDesignerResource(_BaseDataDesignerResource[AsyncNeMoPlatform]):
    """High-level async client for the Data Designer plugin service."""

    def __init__(self, platform: AsyncNeMoPlatform) -> None:
        super().__init__(platform)
        self._data_designer_client = client_from_platform(platform, AsyncDataDesignerClient)
        self._models_client = client_from_platform(platform, AsyncModelsClient)

    async def preview(
        self,
        config_builder: dd.DataDesignerConfigBuilder,
        *,
        num_records: int | None = None,
        workspace: str | None = None,
    ) -> PreviewResults:
        """Generate a set of preview records based on your current Data Designer configuration.

        This method is meant for fast iteration on your Data Designer configuration.

        Args:
            config_builder: Data Designer configuration builder.
            num_records: The number of records to generate. Must be less than or equal to the
                service-side configured max number of preview records.
            workspace: The workspace to run the request in. If not supplied, uses the workspace
                of the base NeMoPlatform object.
            timeout: The timeout for the preview call in seconds.

        Returns:
            An object containing the preview dataset and tools for inspecting the results.
        """
        config = config_builder.build()
        request = PreviewSpec(config=config, num_records=num_records)

        with _PreviewFrameCollector() as message_collector:
            async for frame in self._preview(request=request, workspace=workspace):
                message_collector.accept(frame)
            return PreviewResults(
                config_builder=config_builder,
                dataset=message_collector.dataset,
                dataset_metadata=message_collector.dataset_metadata,
                analysis=message_collector.analysis,
                processor_artifacts=message_collector.get_processor_artifacts(),
            )

    async def _preview(
        self,
        *,
        request: PreviewSpec,
        workspace: str | None,
    ) -> AsyncIterator[PreviewFrame]:
        body = PreviewRequest.model_validate(request.model_dump(mode="json", exclude_none=True))
        response = await self._data_designer_client.preview(workspace=workspace, body=body)
        async with response.stream() as frames:
            async for frame in frames:
                preview_frame = _parse_preview_frame(frame)
                if preview_frame is not None:
                    yield preview_frame

    async def create(
        self,
        config_builder: dd.DataDesignerConfigBuilder,
        *,
        num_records: int = 100,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AsyncDataDesignerJobResource:
        """Create a Data Designer generation job.

        Args:
            config_builder: Data Designer configuration builder.
            num_records: The number of records to generate.
            workspace: The workspace in which to run the job. If not supplied, uses
                the workspace of the base NeMoPlatform object.
            wait_until_done: Set to True to poll the job status and block until the
                job reaches a terminal state.

        Returns:
            An object with methods for querying the job's status and results.
        """
        config = config_builder.build()
        request = DataDesignerJobConfig(config=config, num_records=num_records)
        return await self._submit_named_job("create", request, workspace=workspace, wait_until_done=wait_until_done)

    async def get_job_resource(self, job_name: str, workspace: str | None = None) -> AsyncDataDesignerJobResource:
        """Get a high-level resource for an existing data generation job.

        Args:
            job_name: The name of the job.
            workspace: The workspace in which the job ran.

        Returns:
            An object containing methods for querying job status,
            retrieving the generated dataset, and accessing job metadata.

        Raises:
            ValueError: If the job ID provided is empty.
        """
        try:
            response = await self._data_designer_client.get_job(workspace=workspace, name=job_name)
            response.data()
        except Exception as e:
            raise _get_error(e) from e
        return AsyncDataDesignerJobResource(job_name=job_name, client=self._data_designer_client, workspace=workspace)

    async def get_default_model_configs(self) -> list[dd.ModelConfig]:
        """Default model configs are not supported in the NeMo Platform Data Designer service."""

        return []

    async def get_default_model_providers(self) -> list[dd.ModelProvider]:
        """Get the model providers available for inference.

        Returns:
            A list of ModelProvider objects available for inference.
        """
        nmp_providers = await self._models_client.list_providers(workspace="-")
        return [
            _nmp_provider_to_ndd_provider(self._models_client, provider) async for provider in nmp_providers.items()
        ]

    async def get_info(self) -> InterfaceInfo:
        return InterfaceInfo(model_providers=await self.get_default_model_providers())

    async def validate(
        self,
        config_builder: dd.DataDesignerConfigBuilder,
        *,
        workspace: str | None = None,
    ) -> ValidationReport:
        """Async equivalent of :meth:`DataDesignerResource.validate`."""
        resolved_workspace = workspace or self._platform.workspace or "default"
        return await validate_config(
            config_builder,
            async_sdk=self._platform,
            workspace=resolved_workspace,
        )

    async def retrieval_generate(
        self,
        spec: RetrievalGenerateJobConfig,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AsyncDataDesignerJobResource:
        return await self._submit_named_job(
            "retrieval-generate", spec, workspace=workspace, wait_until_done=wait_until_done
        )

    async def retrieval_prepare(
        self,
        spec: RetrievalPrepareJobConfig,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AsyncDataDesignerJobResource:
        return await self._submit_named_job(
            "retrieval-prepare", spec, workspace=workspace, wait_until_done=wait_until_done
        )

    async def retrieval_run(
        self,
        spec: RetrievalRunJobConfig,
        *,
        workspace: str | None = None,
        wait_until_done: bool = False,
    ) -> AsyncDataDesignerJobResource:
        return await self._submit_named_job("retrieval-run", spec, workspace=workspace, wait_until_done=wait_until_done)

    async def _submit_named_job(
        self,
        job_name: DataDesignerJobCollection,
        spec: BaseModel,
        *,
        workspace: str | None,
        wait_until_done: bool,
    ) -> AsyncDataDesignerJobResource:
        try:
            body = DataDesignerJobRequest.model_validate({"spec": spec.model_dump(mode="json")})
            response = await self._data_designer_client.create_job(
                workspace=workspace, job_collection=job_name, body=body
            )
            job = response.data()
            logger.info(f"  |-- job name: {job.name}")
            job_client = AsyncDataDesignerJobResource(
                job_name=job.name,
                client=self._data_designer_client,
                workspace=workspace,
                job_collection=job_name,
            )
            if wait_until_done:
                await job_client.wait_until_done()
            return job_client
        except Exception as e:
            raise _get_error(e) from e


def _get_error(e: BaseException) -> DataDesignerClientError:
    if isinstance(e, NemoHTTPError):
        status_code = e.status_code
        detail = e.detail
        if status_code == 422:
            return DataDesignerConfigValidationError(f"‼️ Config validation failed!\n{detail}", status_code=status_code)
        return DataDesignerClientError(f"‼️ Something went wrong!\n{detail}", status_code=status_code)
    if isinstance(e, httpx.HTTPStatusError):
        status_code, detail = extract_http_error_info(e)
        if status_code == 422:
            return DataDesignerConfigValidationError(f"‼️ Config validation failed!\n{detail}", status_code=status_code)
        return DataDesignerClientError(f"‼️ Something went wrong!\n{detail}", status_code=status_code)
    return DataDesignerClientError(f"‼️ Something went wrong!\n{e}")


def _nmp_provider_to_ndd_provider(
    models: ModelsClient | AsyncModelsClient,
    nmp_provider: NMPModelProvider,
) -> dd.ModelProvider:
    return dd.ModelProvider(
        name=f"{nmp_provider.workspace}/{nmp_provider.name}",
        endpoint=models.get_provider_route_openai_url(nmp_provider),
    )


data_designer_sdk_resources = NemoPluginSDKResources(
    sync_resource=DataDesignerResource,
    async_resource=AsyncDataDesignerResource,
)
