# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job compiler - transforms CustomizationJobOutput into PlatformJobSpec."""

import logging

from nemo_platform_plugin.client.errors import NotFoundError
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
from nemo_platform_plugin.models.types import ModelDeploymentConfig, ModelEntity
from nmp.automodel.api.v2.jobs.schemas import (
    CustomizationJobOutput,
    DeploymentParams,
    DistillationTraining,
    LoRAParams,
    ValidationError,
)
from nmp.automodel.app.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_MODEL_PATH,
    DEFAULT_TEACHER_MODEL_PATH,
)
from nmp.automodel.app.jobs.training.compiler import (
    _extract_model_name,
    _resolve_training_recipe,
    compile_training_step,
)
from nmp.automodel.config import config
from nmp.automodel.entities.values import FinetuningType
from nmp.automodel.images import (
    AUTOMODEL_PYTHON_ENTRYPOINT,
    FILE_IO_TASK_COMMAND,
    MODEL_ENTITY_TASK_COMMAND,
    get_tasks_image,
)
from nmp.common.auth import AuthClient, auth_client_context
from nmp.common.entities.utils import parse_entity_ref
from nmp.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nmp.common.jobs.exceptions import PlatformJobCompilationError
from nmp.customization_common.schemas.file_io import (
    DownloadItem,
    FileIOTaskConfig,
    FileSetRef,
    UploadItem,
)
from nmp.customization_common.schemas.model_entity import (
    DeploymentParameters as ModelEntityDeploymentParameters,
)
from nmp.customization_common.schemas.model_entity import (
    ModelEntityTaskConfig,
)
from nmp.customization_common.schemas.model_entity import (
    PEFTConfig as ModelEntityPEFTConfig,
)
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients, fetch_model_entity
from nmp.customization_common.tasks.file_io_metadata import build_output_fileset_metadata_from_model_entity

logger = logging.getLogger(__name__)


def _get_cpu_resources() -> ResourcesSpec:
    """Get default CPU resources for download/upload tasks."""
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


def _cpu_tasks_executor(command: list[str], cpu_resources: ResourcesSpec, profile: str) -> CPUExecutionProviderSpec:
    """CPU container steps for fileset I/O and model-entity create.

    Local Docker maps ``cpu/default`` to host subprocess (see ``local.yaml``),
    which fails because the host has no ``/opt/venv/bin/python``. Use the same
    profile as training (default ``gpu``) so those steps stay Docker-backed.
    """
    return CPUExecutionProviderSpec(
        provider="cpu",
        profile=profile,
        container=ContainerSpec(
            image=get_tasks_image(),
            entrypoint=AUTOMODEL_PYTHON_ENTRYPOINT,
            command=command,
        ),
        resources=cpu_resources,
    )


def _get_base_environment() -> list[EnvironmentVariable]:
    """Get base environment variables for all tasks."""
    return [
        EnvironmentVariable(
            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            value=DEFAULT_JOB_STORAGE_PATH,
        ),
    ]


def _extract_model_uri(me: ModelEntity) -> str | None:
    """Extract model_uri from the model entity.

    Args:
        me: The model entity.

    Returns:
        The fileset string if available, None otherwise.
    """
    return me.fileset if me.fileset else None


def _require_fileset_for_download(fileset_name: str | None, entity_label: str) -> str:
    """Require a platform fileset reference for checkpoint download."""
    if not fileset_name or not str(fileset_name).strip():
        raise PlatformJobCompilationError(
            f"{entity_label} has no fileset. "
            "Attach a platform FileSet (workspace/name) with model weights before running training.",
        )
    return str(fileset_name)


def _append_download_if_present(
    downloads: list[DownloadItem],
    fileset_name: str | None,
    dest: str,
    field_name: str,
) -> None:
    """Append a download item if a FileSet ref is present."""
    if not fileset_name:
        return
    fileset = FileSetRef.model_validate(fileset_name)
    downloads.append(DownloadItem(src=fileset, dest=dest))
    logger.info(f"Detected {field_name} FileSet reference: {fileset}")


def _build_file_download_config(
    job_spec: CustomizationJobOutput,
    me: ModelEntity,
    teacher_me: ModelEntity | None = None,
) -> FileIOTaskConfig:
    """Build the configuration for the file_io task.

    Extracts FileSet references from model_uri and dataset fields.
    Fileset refs use workspace/name or name (optional legacy fileset:// prefix is stripped).

    Args:
        job_spec: The customization job output specification.
        me: The model entity being trained.
        teacher_me: Optional teacher model entity for knowledge distillation jobs.

    Returns:
        FileIOTaskConfig with download items for any fileset refs found.

    """
    downloads: list[DownloadItem] = []

    model_fileset = _require_fileset_for_download(
        _extract_model_uri(me),
        entity_label=f"Model '{me.workspace}/{me.name}'",
    )
    _append_download_if_present(
        downloads,
        fileset_name=model_fileset,
        dest=DEFAULT_MODEL_PATH,
        field_name="model",
    )
    _append_download_if_present(
        downloads,
        fileset_name=job_spec.dataset,
        dest=DEFAULT_DATASET_PATH,
        field_name="dataset",
    )

    if teacher_me is not None:
        teacher_fileset = _require_fileset_for_download(
            _extract_model_uri(teacher_me),
            entity_label=f"Teacher model '{teacher_me.workspace}/{teacher_me.name}'",
        )
        _append_download_if_present(
            downloads,
            fileset_name=teacher_fileset,
            dest=DEFAULT_TEACHER_MODEL_PATH,
            field_name="teacher_model",
        )

    return FileIOTaskConfig(download=downloads)


def _build_file_upload_config(
    output_fileset_name: str,
    fileset_metadata: dict | None = None,
) -> FileIOTaskConfig:
    """Build the configuration for the file_io upload task with a generated fileset name.

    The fileset name is generated at compile time and will be combined with
    the job's workspace at runtime to form the full FileSet reference.

    Args:
        output_fileset_name: The generated name for the output FileSet.
        fileset_metadata: Optional metadata to set on the output fileset (e.g., tool_calling
            config propagated from the source model entity).

    Returns:
        FileIOTaskConfig with upload items configured to use the generated name.
    """
    return FileIOTaskConfig(
        upload=[
            UploadItem(
                src=DEFAULT_OUTPUT_MODEL_PATH,
                # workspace is None because at this layer, we don't know the job's workspace.
                dest=FileSetRef(workspace=None, name=output_fileset_name),
                metadata=fileset_metadata,
            )
        ],
    )


def _build_model_entity_config(
    workspace: str, job_spec: CustomizationJobOutput, trust_remote_code: bool = False
) -> ModelEntityTaskConfig:
    """Build the configuration for the model_entity task.

    Args:
        workspace: The workspace for this job.
        job_spec: The customization job input specification.
        trust_remote_code: Whether to trust remote code for the checkpoint.

    Returns:
        ModelEntityTaskConfig with model entity creation settings.
    """
    base_model = _extract_model_name(job_spec)

    assert job_spec.output is not None, "output must be set by input-to-output transformer"
    training = job_spec.training

    peft_config: ModelEntityPEFTConfig | None = None
    if isinstance(training.peft, LoRAParams):
        peft_config = ModelEntityPEFTConfig(
            type=training.finetuning_type,
            alpha=training.peft.alpha,
            rank=training.peft.rank,
        )

    # Only forward the user-supplied deployment_config from the job spec.
    # tool_call_config from the *source* model entity's spec is propagated
    # separately via fileset metadata (see build_output_fileset_metadata_from_model_entity),
    # so we intentionally do not merge it here.
    deployment_config: str | ModelEntityDeploymentParameters | None = None
    if isinstance(job_spec.deployment_config, str):
        deployment_config = job_spec.deployment_config
    elif job_spec.deployment_config is not None:
        deployment_config = ModelEntityDeploymentParameters.model_validate(job_spec.deployment_config.model_dump())

    return ModelEntityTaskConfig(
        name=job_spec.output.name,
        workspace=workspace,
        description="Customized model from job",
        fileset=FileSetRef(
            workspace=None,
            name=job_spec.output.fileset,
        ),
        base_model=base_model,
        model_entity=job_spec.model,
        peft=peft_config,
        trust_remote_code=trust_remote_code,
        deployment_config=deployment_config,
    )


async def _resolve_deployment_config_ref(
    config_ref: str,
    workspace: str,
    platform: AsyncCustomizationPlatformClients,
) -> ModelDeploymentConfig:
    """Resolve a ``name`` or ``workspace/name`` string to a ModelDeploymentConfig."""
    ref = parse_entity_ref(config_ref, default_workspace=workspace)
    try:
        response = await platform.models.get_deployment_config(name=ref.name, workspace=ref.workspace)
        return response.data()
    except NotFoundError as e:
        raise PlatformJobCompilationError(
            f"deployment_config references '{config_ref}' which does not exist in workspace '{ref.workspace}'."
        ) from e
    except Exception as e:
        raise PlatformJobCompilationError(f"Failed to resolve deployment_config '{config_ref}': {e}") from e


async def _validate_deployment_config(
    workspace: str,
    transformed_spec: CustomizationJobOutput,
    platform: AsyncCustomizationPlatformClients,
    auth_client: AuthClient,
) -> None:
    """Validate deployment_config consistency before training starts.

    Catches contradictory or impossible configurations early so the user
    gets a clear error instead of a silent failure after expensive training.
    """
    dc = transformed_spec.deployment_config
    if dc is None:
        return

    # Inline deployment params: check permission-gated fields.
    if isinstance(dc, DeploymentParams):
        tcc = dc.tool_call_config
        if tcc and tcc.tool_call_plugin:
            if not await auth_client.has_permissions(workspace, ["models.tool-call-plugin.set"]):
                raise PlatformJobCompilationError(
                    "Insufficient permissions to set tool_call_plugin. "
                    "Requires the models.tool-call-plugin.set permission."
                )
        return

    # String reference to an existing deployment config: validate consistency.
    if not isinstance(dc, str):
        return

    ft_type = transformed_spec.training.finetuning_type
    is_lora = ft_type == FinetuningType.LORA
    produces_new_model = ft_type in (FinetuningType.ALL_WEIGHTS, FinetuningType.LORA_MERGED)
    resolved_config = await _resolve_deployment_config_ref(dc, workspace, platform)

    # LoRA job referencing a config that has lora_enabled=False
    if is_lora and resolved_config.model_spec.lora_enabled is False:
        raise PlatformJobCompilationError(
            f"deployment_config references '{dc}' which has lora_enabled=false, "
            "but this is a LoRA training job. The deployment would not load LoRA adapters. "
            "Use a deployment config with lora_enabled=true, or provide inline deployment parameters."
        )

    # SFT or lora_merged referencing a string config
    if produces_new_model:
        output_name = transformed_spec.output.name
        try:
            response = await platform.models.get_model(name=output_name, workspace=workspace)
            existing_me = response.data()
        except NotFoundError:
            # Output model entity doesn't exist yet, so a string
            # ref is inherently invalid -- it was created for a different model.
            raise PlatformJobCompilationError(
                f"deployment_config cannot be a string reference ('{dc}') for {ft_type.value} training "
                "that creates a new model entity. The referenced config was created for a different model. "
                "Use inline deployment parameters (e.g., DeploymentParams(gpu=1, lora_enabled=True)) instead."
            )

        # Output model entity already exists (retraining to create a new FileSet).
        # Verify the config actually targets this model entity.
        model_spec = resolved_config.model_spec
        config_targets_model = (resolved_config.model_entity_id == f"{existing_me.workspace}/{existing_me.name}") or (
            model_spec.model_name == existing_me.name and model_spec.model_namespace == existing_me.workspace
        )
        if not config_targets_model:
            raise PlatformJobCompilationError(
                f"deployment_config references '{dc}' which targets a different model entity "
                f"than the output model '{existing_me.workspace}/{existing_me.name}'. "
                "The deployment config must target the same model entity being retrained, "
                "or use inline deployment parameters instead."
            )


async def platform_job_config_compiler(
    workspace: str,
    job_spec: CustomizationJobOutput,
    platform: AsyncCustomizationPlatformClients,
) -> PlatformJobSpec:
    """Compile canonical job spec into a four-step PlatformJobSpec."""
    transformed_spec = job_spec
    logger.info("Compiling Automodel job to PlatformJobSpec: %s", transformed_spec.model_dump_json(indent=2))

    try:
        transformed_spec.validate_for_training()
    except ValidationError as e:
        raise PlatformJobCompilationError(str(e)) from e

    # output is a required field in CustomizationJobOutput
    cpu_resources = _get_cpu_resources()
    base_env = _get_base_environment()
    task_profile = transformed_spec.training.execution_profile or config.default_training_execution_profile

    # Fetch the primary model entity
    me = await fetch_model_entity(transformed_spec.model, workspace, platform)

    # For distillation jobs, also fetch the teacher model entity
    teacher_me: ModelEntity | None = None
    if isinstance(transformed_spec.training, DistillationTraining):
        try:
            teacher_me = await fetch_model_entity(transformed_spec.training.teacher_model, workspace, platform)
        except ValueError as e:
            raise PlatformJobCompilationError(
                f"Teacher model '{transformed_spec.training.teacher_model}' not found. "
                "Verify the teacher model entity exists."
            ) from e
        except PermissionError as e:
            raise PlatformJobCompilationError(
                f"Access denied to teacher model '{transformed_spec.training.teacher_model}'."
            ) from e

    if transformed_spec.deployment_config is not None:
        auth_client = auth_client_context.get()
        if auth_client is None:
            raise PlatformJobCompilationError(
                "No auth context available; cannot validate deployment config permissions.",
            )
        await _validate_deployment_config(workspace, transformed_spec, platform, auth_client)

    file_io_download_config = _build_file_download_config(transformed_spec, me, teacher_me)
    training_recipe = _resolve_training_recipe(me, transformed_spec.training.recipe)

    # The embedding NIM requires ONNX format, which cannot represent standalone LoRA adapters.
    # LoRA with merge=True (lora_merged) is allowed because it produces a full-weight model after training.
    if training_recipe.value in ("bi_encoder", "cross_encoder") and (
        transformed_spec.training.finetuning_type == FinetuningType.LORA
    ):
        raise PlatformJobCompilationError(
            "NeMo Platform does not support unmerged LoRA for embedding or cross-encoder models. "
            "Embedding NIM requires ONNX (no standalone adapters); ranking NIM expects a full-weight checkpoint. "
            "Use peft with merge=True (lora_merged) or omit peft for all_weights training."
        )

    # Extract chat_template and tool_call_config from the source model entity's spec
    # (populated from fileset metadata by the model-spec-runner background task).
    # These are propagated to:
    #   1. The training step config (chat_template takes highest priority in template resolution)
    #   2. The output fileset metadata (so the model-spec-runner sets them on the output model)
    fileset_metadata = build_output_fileset_metadata_from_model_entity(me)
    file_io_upload_config = _build_file_upload_config(transformed_spec.output.fileset, fileset_metadata)

    # Build model_entity config for creating the model entity
    trust_remote_code = me.trust_remote_code or False
    model_entity_config = _build_model_entity_config(workspace, transformed_spec, trust_remote_code)

    steps = [
        # Step 1: Download model and dataset files from Files service
        PlatformJobStep(
            name="model-and-dataset-download",
            executor=_cpu_tasks_executor(FILE_IO_TASK_COMMAND, cpu_resources, task_profile),
            environment=base_env,
            config=file_io_download_config.model_dump(mode="json"),
        ),
        # Step 2: Training job
        compile_training_step(
            transformed_spec,
            base_env,
            me,
            teacher_me=teacher_me,
        ),
        # Step 3: Upload customized model
        PlatformJobStep(
            name="model-upload",
            executor=_cpu_tasks_executor(FILE_IO_TASK_COMMAND, cpu_resources, task_profile),
            environment=base_env,
            config=file_io_upload_config.model_dump(mode="json"),
        ),
        # Step 4: Create model entity
        PlatformJobStep(
            name="model-entity-creation",
            executor=_cpu_tasks_executor(MODEL_ENTITY_TASK_COMMAND, cpu_resources, task_profile),
            environment=base_env,
            config=model_entity_config.model_dump(mode="json"),
        ),
    ]

    return PlatformJobSpec(steps=steps)
