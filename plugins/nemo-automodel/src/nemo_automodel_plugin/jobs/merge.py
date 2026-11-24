# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge a LoRA adapter into its base model and register the result as a new model.

Four steps, as in training: download the base model and the adapter, merge on a GPU, upload the merged
checkpoint as a fileset, register it as a model entity.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nemo_automodel_plugin.schema import AutomodelSchema, OutputRequest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    GPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    ResourcesSpec,
)
from nemo_helix_plugin.jobs.docker import validate_gpu_available_for_docker
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.refs import parse_entity_ref
from nhx.automodel.app.constants import DEFAULT_MODEL_PATH
from nhx.automodel.app.jobs.compiler import (
    _build_file_upload_config,
    _cpu_tasks_executor,
    _get_base_environment,
    _get_cpu_resources,
    _require_fileset_for_download,
)
from nhx.automodel.app.jobs.training.compiler import (
    _resolve_checkpoint_head_type,
    _resolve_custom_implementation_override,
)
from nhx.automodel.config import config
from nhx.automodel.images import (
    AUTOMODEL_PYTHON_ENTRYPOINT,
    FILE_IO_TASK_COMMAND,
    MODEL_ENTITY_TASK_COMMAND,
    get_training_image,
)
from nhx.automodel.tasks.merge_adapter import ADAPTER_PATH, MergeAdapterStepConfig
from nhx.customization_common.contributor.jobs import require_container_runtime
from nhx.customization_common.schemas.checkpoint import traits_from_model_spec
from nhx.customization_common.schemas.file_io import DownloadItem, FileIOTaskConfig, FileSetRef, model_upload_lifecycle
from nhx.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
from nhx.customization_common.service.platform_client import (
    AsyncCustomizationHelixClients,
    async_customization_platform_clients_from_platform,
    fetch_model_entity,
    validate_output_name_not_in_flight,
)
from nhx.customization_common.tasks.file_io_metadata import build_output_fileset_metadata_from_model_entity
from pydantic import BaseModel, Field


class AutomodelMergeSpec(AutomodelSchema):
    model: str = Field(description="Base model entity the adapter was trained on ('name' or 'workspace/name').")
    adapter: str = Field(description="LoRA adapter entity to merge ('name' or 'workspace/name').")
    output: OutputRequest = Field(description="The merged model entity; its fileset has the same name.")
    num_gpus: int = Field(default=1, ge=1, description="GPUs for the merge; a model too big for them spills to CPU.")


async def compile_merge(
    workspace: str, spec: AutomodelMergeSpec, platform: AsyncCustomizationHelixClients, profile: str
) -> HelixJobSpec:
    base = await fetch_model_entity(spec.model, workspace, platform)
    base_ref = f"{base.workspace}/{base.name}"
    if _resolve_checkpoint_head_type(base) in ("embedding", "cross_encoder"):
        raise HelixJobCompilationError(
            f"'{base_ref}' is not a causal or vision-language model; train it with merge=true."
        )
    ref = parse_entity_ref(spec.adapter, workspace)
    try:
        adapter = (await platform.models.get_adapter(name=ref.name, workspace=ref.workspace)).data()
    except NotFoundError as exc:
        raise HelixJobCompilationError(f"Adapter '{ref.workspace}/{ref.name}' not found.") from exc
    if adapter.model not in (None, base_ref):
        raise HelixJobCompilationError(f"Adapter '{spec.adapter}' was trained on '{adapter.model}', not '{base_ref}'.")
    if adapter.lora_config is None:
        raise HelixJobCompilationError(f"Adapter '{spec.adapter}' is not a LoRA adapter: it records no lora_config.")
    # The downloads and the guard share these refs. A bare fileset name means the job's workspace to the
    # file_io task, whatever workspace its entity is in, and the upload writes to output.name there.
    base_fileset = FileSetRef.model_validate(_require_fileset_for_download(base.fileset, base_ref))
    adapter_fileset = FileSetRef.model_validate(adapter.fileset)
    inputs = {base_ref, f"{ref.workspace}/{ref.name}"}
    inputs |= {f"{fs.workspace or workspace}/{fs.name}" for fs in (base_fileset, adapter_fileset)}
    if f"{workspace}/{spec.output.name}" in inputs:
        raise HelixJobCompilationError(
            f"output.name '{spec.output.name}' would overwrite the base model, the adapter, or one of their filesets."
        )
    await validate_output_name_not_in_flight(spec.output.name, workspace, platform)

    cpu, env = _get_cpu_resources(), _get_base_environment()

    def cpu_step(name: str, command: list[str], task: BaseModel, **extra: Any) -> HelixJobStep:
        executor = _cpu_tasks_executor(command, cpu, profile)
        return HelixJobStep(name=name, executor=executor, environment=env, config=task.model_dump(mode="json"), **extra)

    downloads = [
        DownloadItem(src=base_fileset, dest=DEFAULT_MODEL_PATH),
        DownloadItem(src=adapter_fileset, dest=ADAPTER_PATH),
    ]
    merge = MergeAdapterStepConfig(
        trust_remote_code=bool(base.trust_remote_code),
        has_vision_tower=traits_from_model_spec(base.spec).has_vision_tower,
        override_custom_impl=_resolve_custom_implementation_override(base),
        chat_template=base.spec.chat_template if base.spec else None,
    )
    lora = adapter.lora_config
    entity = ModelEntityTaskConfig(
        name=spec.output.name,
        workspace=workspace,
        fileset=FileSetRef(workspace=None, name=spec.output.name),
        base_model=base_ref,
        model_entity=base_ref,
        peft=PEFTConfig(type="lora_merged", rank=lora.rank, alpha=lora.alpha or lora.rank),
        trust_remote_code=bool(base.trust_remote_code),
    )
    upload = _build_file_upload_config(spec.output.name, build_output_fileset_metadata_from_model_entity(base))
    gpu = GPUExecutionProviderSpec(
        provider="gpu",
        profile=profile,
        container=ContainerSpec(
            image=config.training_automodel_image or get_training_image(),
            entrypoint=AUTOMODEL_PYTHON_ENTRYPOINT,
            command=["-m", "nhx.automodel.tasks.merge_adapter"],
        ),
        resources=ResourcesSpec(num_gpus=spec.num_gpus, shm_size="8Gi"),
    )
    return HelixJobSpec(
        steps=[
            cpu_step("model-and-adapter-download", FILE_IO_TASK_COMMAND, FileIOTaskConfig(download=downloads)),
            # No lifecycle: the merge reports no progress, so a staleness timeout would kill a healthy one.
            HelixJobStep(name="merge", executor=gpu, environment=env, config=merge.model_dump(mode="json")),
            cpu_step("model-upload", FILE_IO_TASK_COMMAND, upload, lifecycle=model_upload_lifecycle()),
            cpu_step("model-entity-creation", MODEL_ENTITY_TASK_COMMAND, entity),
        ]
    )


class AutomodelMergeJob(NemoJob):
    name: ClassVar[str] = "automodel.merge"
    description: ClassVar[str] = "Merge a LoRA adapter into its base model and register the result as a new model."
    job_collection_path: ClassVar[str | None] = "/automodel-merge/jobs"
    spec_schema: ClassVar[type[BaseModel] | None] = AutomodelMergeSpec
    dependencies: ClassVar[list[str]] = ["entities", "auth", "jobs", "secrets", "files", "models"]

    @classmethod
    async def compile(
        cls,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNemoClient,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        await asyncio.to_thread(require_container_runtime, "Automodel merge")  # the probe is sync
        job = await compile_merge(
            workspace,
            AutomodelMergeSpec.model_validate(spec.model_dump()),
            async_customization_platform_clients_from_platform(async_sdk),
            profile or config.default_training_execution_profile,
        )
        validate_gpu_available_for_docker(job)
        return job
