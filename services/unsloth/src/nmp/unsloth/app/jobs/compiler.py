# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job compiler — transforms ``UnslothJobOutput`` into a 4-step ``PlatformJobSpec``.

Invoked from :meth:`UnslothJob.compile` via :mod:`nmp.unsloth.compile`.
The four steps mirror automodel:

1. file_io download   — pull model fileset + dataset fileset to the PVC
2. training            — GPU step running ``train_sft``
3. file_io upload      — push the saved checkpoint to a new fileset
4. model_entity        — create the output ``ModelEntity`` referencing it
"""

from __future__ import annotations

import logging

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    PlatformJobSpec,
    PlatformJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.models.types import ModelEntity
from nmp.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nmp.customization_common.schemas.file_io import (
    DownloadItem,
    FileIOTaskConfig,
    FileSetRef,
    UploadItem,
)
from nmp.customization_common.schemas.model_entity import (
    DeploymentParameters as ModelEntityDeploymentParameters,
)
from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
from nmp.customization_common.service.platform_client import fetch_model_entity
from nmp.customization_common.tasks.file_io_metadata import build_output_fileset_metadata_from_model_entity
from nmp.unsloth.app.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_MODEL_PATH,
    DEFAULT_VALIDATION_DATASET_PATH,
)
from nmp.unsloth.app.jobs.training.compiler import compile_training_step
from nmp.unsloth.config import config
from nmp.unsloth.entities.values import FinetuningType
from nmp.unsloth.images import (
    FILE_IO_TASK_COMMAND,
    MODEL_ENTITY_TASK_COMMAND,
    UNSLOTH_PYTHON_ENTRYPOINT,
    get_tasks_image,
)
from nmp.unsloth.schemas import UnslothJobOutput

logger = logging.getLogger(__name__)


def _get_cpu_resources() -> ResourcesSpec:
    return ResourcesSpec(
        limits=ResourcesLimitsSpec(
            cpu=config.default_job_resource_cpu_limit,
            memory=config.default_job_resource_memory_limit,
        ),
        requests=ResourcesRequestsSpec(
            cpu=config.default_job_resource_cpu_request,
            memory=config.default_job_resource_memory_request,
        ),
    )


def _get_base_environment() -> list[EnvironmentVariable]:
    return [
        EnvironmentVariable(
            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            value=DEFAULT_JOB_STORAGE_PATH,
        ),
    ]


def _resolve_finetuning_type(spec: UnslothJobOutput) -> FinetuningType:
    """Map the plugin's flat ``finetuning_type`` + ``save_method`` onto the enum."""
    if spec.training.finetuning_type == "lora":
        if spec.output.save_method in {"merged_16bit", "merged_4bit"}:
            return FinetuningType.LORA_MERGED
        return FinetuningType.LORA
    return FinetuningType.ALL_WEIGHTS


def _build_peft_config(spec: UnslothJobOutput) -> PEFTConfig | None:
    if spec.training.finetuning_type != "lora":
        return None
    assert spec.training.lora is not None  # validated by UnslothJobInput
    return PEFTConfig(
        type=_resolve_finetuning_type(spec),
        rank=spec.training.lora.rank,
        alpha=spec.training.lora.alpha,
    )


def _same_fileset_ref(a: str, b: str, *, workspace: str) -> bool:
    """Return True when two platform fileset refs denote the same fileset."""
    ra = FileSetRef.model_validate(a)
    rb = FileSetRef.model_validate(b)
    return (ra.workspace or workspace) == (rb.workspace or workspace) and ra.name == rb.name


def _resolve_validation_dataset_path(
    job_spec: UnslothJobOutput,
    workspace: str,
) -> str | None:
    """Map ``spec.dataset.validation_path`` to the local PVC path the download step uses."""
    if not job_spec.dataset.validation_path:
        return None
    if _same_fileset_ref(
        job_spec.dataset.path,
        job_spec.dataset.validation_path,
        workspace=workspace,
    ):
        return DEFAULT_DATASET_PATH
    return DEFAULT_VALIDATION_DATASET_PATH


def _require_fileset(name: str | None, *, label: str) -> str:
    if not name or not str(name).strip():
        raise PlatformJobCompilationError(
            f"{label} has no fileset attached. Attach a platform FileSet "
            "(workspace/name) with model weights before running training.",
        )
    return str(name)


def _build_file_download_config(
    job_spec: UnslothJobOutput,
    me: ModelEntity,
    *,
    workspace: str,
) -> FileIOTaskConfig:
    """Compile the download step: model fileset + dataset fileset."""
    model_fileset = _require_fileset(
        me.fileset,
        label=f"Model '{me.workspace}/{me.name}'",
    )
    downloads = [
        DownloadItem(
            src=FileSetRef.model_validate(model_fileset),
            dest=DEFAULT_MODEL_PATH,
        ),
        DownloadItem(
            src=FileSetRef.model_validate(job_spec.dataset.path),
            dest=DEFAULT_DATASET_PATH,
        ),
    ]
    if job_spec.dataset.validation_path and not _same_fileset_ref(
        job_spec.dataset.path,
        job_spec.dataset.validation_path,
        workspace=workspace,
    ):
        downloads.append(
            DownloadItem(
                src=FileSetRef.model_validate(job_spec.dataset.validation_path),
                dest=DEFAULT_VALIDATION_DATASET_PATH,
            ),
        )
    return FileIOTaskConfig(download=downloads)


def _build_file_upload_config(job_spec: UnslothJobOutput, me: ModelEntity) -> FileIOTaskConfig:
    """Compile the upload step.

    ``workspace=None`` tells the file_io task to use the job's workspace
    from its :class:`NMPJobContext`.
    """
    return FileIOTaskConfig(
        upload=[
            UploadItem(
                src=DEFAULT_OUTPUT_MODEL_PATH,
                dest=FileSetRef(workspace=None, name=job_spec.output.fileset),
                metadata=build_output_fileset_metadata_from_model_entity(me),
            ),
        ],
    )


def _build_model_entity_config(
    workspace: str,
    job_spec: UnslothJobOutput,
    *,
    trust_remote_code: bool,
) -> ModelEntityTaskConfig:
    # Forward the user-supplied deployment_config from the job spec.
    # String refs are passed through as-is; inline DeploymentParams are
    # converted from the user-facing shape to the task-side shape via
    # model_validate(model_dump()).
    deployment_config: str | ModelEntityDeploymentParameters | None = None
    if isinstance(job_spec.deployment_config, str):
        deployment_config = job_spec.deployment_config
    elif job_spec.deployment_config is not None:
        deployment_config = ModelEntityDeploymentParameters.model_validate(job_spec.deployment_config.model_dump())

    return ModelEntityTaskConfig(
        name=job_spec.output.name,
        workspace=workspace,
        description=job_spec.output.description or "Customized model from unsloth job",
        fileset=FileSetRef(workspace=None, name=job_spec.output.fileset),
        model_entity=job_spec.model.name,
        base_model=job_spec.model.name,
        peft=_build_peft_config(job_spec),
        trust_remote_code=trust_remote_code,
        deployment_config=deployment_config,
    )


async def platform_job_config_compiler(
    workspace: str,
    job_spec: UnslothJobOutput,
    sdk: AsyncNeMoPlatform,
    *,
    job_name: str | None = None,
    profile: str | None = None,
) -> PlatformJobSpec:
    """Compile a canonical unsloth job spec into a 4-step ``PlatformJobSpec``."""
    del job_name  # reserved for future scheduling decisions (e.g. naming jobs)

    logger.info(f"Compiling Unsloth job to PlatformJobSpec: {job_spec.model_dump_json(indent=2)}")

    me = await fetch_model_entity(job_spec.model.name, workspace, sdk)

    cpu_resources = _get_cpu_resources()
    base_env = _get_base_environment()
    task_profile = profile or config.default_training_execution_profile

    validation_dataset_path = _resolve_validation_dataset_path(job_spec, workspace=workspace)
    download_config = _build_file_download_config(job_spec, me, workspace=workspace)
    upload_config = _build_file_upload_config(job_spec, me)
    model_entity_config = _build_model_entity_config(
        workspace,
        job_spec,
        trust_remote_code=me.trust_remote_code or False,
    )

    steps: list[PlatformJobStep] = [
        PlatformJobStep(
            name="model-and-dataset-download",
            executor=CPUExecutionProviderSpec(
                provider="cpu",
                profile=task_profile,
                container=ContainerSpec(
                    image=get_tasks_image(),
                    entrypoint=UNSLOTH_PYTHON_ENTRYPOINT,
                    command=FILE_IO_TASK_COMMAND,
                ),
                resources=cpu_resources,
            ),
            environment=base_env,
            config=download_config.model_dump(mode="json"),
        ),
        compile_training_step(
            job_spec,
            base_env,
            validation_dataset_path=validation_dataset_path,
            profile=task_profile,
        ),
        PlatformJobStep(
            name="model-upload",
            executor=CPUExecutionProviderSpec(
                provider="cpu",
                profile=task_profile,
                container=ContainerSpec(
                    image=get_tasks_image(),
                    entrypoint=UNSLOTH_PYTHON_ENTRYPOINT,
                    command=FILE_IO_TASK_COMMAND,
                ),
                resources=cpu_resources,
            ),
            environment=base_env,
            config=upload_config.model_dump(mode="json"),
        ),
        PlatformJobStep(
            name="model-entity-creation",
            executor=CPUExecutionProviderSpec(
                provider="cpu",
                profile=task_profile,
                container=ContainerSpec(
                    image=get_tasks_image(),
                    entrypoint=UNSLOTH_PYTHON_ENTRYPOINT,
                    command=MODEL_ENTITY_TASK_COMMAND,
                ),
                resources=cpu_resources,
            ),
            environment=base_env,
            config=model_entity_config.model_dump(mode="json"),
        ),
    ]

    return PlatformJobSpec(steps=steps)
