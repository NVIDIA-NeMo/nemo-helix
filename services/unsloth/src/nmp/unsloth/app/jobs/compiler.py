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

from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.deployment import (
    LORA_ENABLED_REQUIRED_MESSAGE,
    DeploymentParams,
    is_unbound_deployment_config,
)
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
from nemo_platform_plugin.models.types import ModelDeploymentConfig, ModelEntity
from nmp.common.auth import auth_client_context
from nmp.common.entities.utils import parse_entity_ref
from nmp.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nmp.customization_common.schemas.file_io import (
    DownloadItem,
    FileIOTaskConfig,
    FileSetRef,
    UploadItem,
)
from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
from nmp.customization_common.service.platform_client import (
    AsyncCustomizationPlatformClients,
    fetch_model_entity,
)
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
    return ModelEntityTaskConfig(
        name=job_spec.output.name,
        workspace=workspace,
        description=job_spec.output.description or "Customized model from unsloth job",
        fileset=FileSetRef(workspace=None, name=job_spec.output.fileset),
        model_entity=job_spec.model.name,
        base_model=job_spec.model.name,
        peft=_build_peft_config(job_spec),
        trust_remote_code=trust_remote_code,
        deployment_config=job_spec.deployment_config,
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


async def _require_tool_call_plugin_permission(workspace: str) -> None:
    """Gate ``tool_call_plugin``, the one deployment field that needs a permission check.

    Auth is resolved here rather than up front: every other deployment_config
    shape validates without it, so demanding an auth context for all of them
    would fail compilation for jobs that never consult it.
    """
    auth_client = auth_client_context.get()
    if auth_client is None:
        raise PlatformJobCompilationError(
            "No auth context available; cannot validate the tool_call_plugin permission.",
        )
    if not await auth_client.has_permissions(workspace, ["models.tool-call-plugin.set"]):
        raise PlatformJobCompilationError(
            "Insufficient permissions to set tool_call_plugin. Requires the models.tool-call-plugin.set permission."
        )


def _config_targets_model(config: ModelDeploymentConfig, workspace: str, name: str) -> bool:
    """Whether ``config`` can serve the model entity ``workspace/name``.

    ``model_entity_id`` is the canonical link; older configs only carry the
    name/namespace pair on ``model_spec``, so both are accepted. A config that
    names no model at all serves any model: the model_entity task binds it to the
    trained one at deploy time.
    """
    if is_unbound_deployment_config(config):
        return True
    model_spec = config.model_spec
    return (config.model_entity_id == f"{workspace}/{name}") or (
        model_spec.model_name == name and model_spec.model_namespace == workspace
    )


async def _validate_deployment_config(
    workspace: str,
    job_spec: UnslothJobOutput,
    platform: AsyncCustomizationPlatformClients,
) -> None:
    """Validate deployment_config consistency before training starts.

    Without this, a referenced config naming an unrelated model compiles happily and
    the model_entity task deploys *that* model when the run finishes -- a silently
    wrong deployment discovered only after the GPU hours are spent.

    The finetuning type is re-derived from the spec rather than read off a property:
    the compiler is entered with an ``UnslothJobOutput``, and the submit-time
    predicate (``trains_standalone_lora_adapter``) lives on ``UnslothJobInput``, so
    any path not going through the plugin's input schema would bypass it.
    """
    dc = job_spec.deployment_config
    if dc is None:
        return

    # A merged save folds the adapter into the base weights, so only the unmerged
    # case trains a standalone adapter served from the base model's deployment. The
    # other two register a model entity of their own, which is the opposite branch.
    ft_type = _resolve_finetuning_type(job_spec)
    is_lora_adapter = ft_type == FinetuningType.LORA

    # Inline deployment params: check permission-gated fields.
    if isinstance(dc, DeploymentParams):
        # UnslothJobInput rejects this at submit, but the compiler is entered with an
        # UnslothJobOutput, which carries no such validator -- re-assert it here.
        if is_lora_adapter and not dc.lora_enabled:
            raise PlatformJobCompilationError(LORA_ENABLED_REQUIRED_MESSAGE)
        tcc = dc.tool_call_config
        if tcc and tcc.tool_call_plugin:
            await _require_tool_call_plugin_permission(workspace)
        return

    resolved_config = await _resolve_deployment_config_ref(dc, workspace, platform)

    # A LoRA adapter cannot be served by a base deployment that does not load adapters.
    if is_lora_adapter and resolved_config.model_spec.lora_enabled is False:
        raise PlatformJobCompilationError(
            f"deployment_config references '{dc}' which has lora_enabled=false, "
            "but this is a LoRA training job. The deployment would not load LoRA adapters. "
            "Use a deployment config with lora_enabled=true, or provide inline deployment parameters."
        )

    if is_lora_adapter:
        # The adapter is served from its base model's deployment, so a referenced
        # config is only usable if it deploys that base model.
        base = parse_entity_ref(job_spec.model.name, workspace)
        if not _config_targets_model(resolved_config, base.workspace, base.name):
            raise PlatformJobCompilationError(
                f"deployment_config references '{dc}' which targets a different model entity than the base model "
                f"'{base.workspace}/{base.name}'. A LoRA adapter is served from its base model's deployment, "
                "so the config must target that base model, or use inline deployment parameters instead."
            )
        return

    # Full-weight and merged training register their own model entity, so the config must
    # be able to serve it. That entity usually does not exist yet, and two shapes are
    # legitimate ahead of it -- a config created up front that points *forward* at the
    # output model, and an unbound config that names no model at all and is bound to the
    # trained model at deploy time. So validate the target rather than the entity's
    # existence; the same comparison covers a retrain, where it already exists.
    output_name = job_spec.output.name
    if not _config_targets_model(resolved_config, workspace, output_name):
        raise PlatformJobCompilationError(
            f"deployment_config references '{dc}' which targets a different model entity "
            f"than the {ft_type.value} training output '{workspace}/{output_name}'. The deployment "
            "config must target the model this run produces, name no model at all, or use inline "
            "deployment parameters instead."
        )


async def _validate_adapter_base_model(
    workspace: str,
    job_spec: UnslothJobOutput,
    platform: AsyncCustomizationPlatformClients,
) -> None:
    """Reject a LoRA job whose output name is an adapter of a different base model."""
    # Adapter names are unique per workspace, so retraining an existing adapter name under a
    # different base model conflicts on create and cannot be updated. Without this the job
    # trains to completion first and only fails in the model-entity step.
    if _resolve_finetuning_type(job_spec) != FinetuningType.LORA:
        return

    output_name = job_spec.output.name
    try:
        existing = (await platform.models.get_adapter(name=output_name, workspace=workspace)).data()
    except NotFoundError:
        return

    base = parse_entity_ref(job_spec.model.name, workspace)
    expected = f"{base.workspace}/{base.name}"
    if existing.model is not None and existing.model != expected:
        raise PlatformJobCompilationError(
            f"Adapter '{workspace}/{output_name}' already exists on base model '{existing.model}', "
            f"but this job trains against '{expected}'. Adapter names are unique per workspace, so "
            "the existing adapter cannot be re-parented. Choose a different output.name, or train "
            f"against '{existing.model}'."
        )


async def platform_job_config_compiler(
    workspace: str,
    job_spec: UnslothJobOutput,
    platform: AsyncCustomizationPlatformClients,
    *,
    job_name: str | None = None,
    profile: str | None = None,
) -> PlatformJobSpec:
    """Compile a canonical unsloth job spec into a 4-step ``PlatformJobSpec``."""
    del job_name  # reserved for future scheduling decisions (e.g. naming jobs)

    logger.info(f"Compiling Unsloth job to PlatformJobSpec: {job_spec.model_dump_json(indent=2)}")

    me = await fetch_model_entity(job_spec.model.name, workspace, platform)

    await _validate_adapter_base_model(workspace, job_spec, platform)
    await _validate_deployment_config(workspace, job_spec, platform)

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
