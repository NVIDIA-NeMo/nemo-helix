# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job compiler — transforms ``RlJobOutput`` into a 4-step ``PlatformJobSpec``.

Steps mirror unsloth/automodel:

1. file_io download    — pull model fileset + dataset (+ environment for GRPO) to the PVC
2. training            — Ray DPO/GRPO step (single-node GPU or multi-node distributed)
3. file_io upload      — push the trained checkpoint to a new fileset
4. model_entity        — create the output ``ModelEntity`` referencing it

The training step's executor is selected by ``parallelism.num_nodes``:
``num_nodes == 1`` means a single-node ``GPUExecutionProviderSpec``;
``num_nodes > 1`` means a multi-node ``DistributedGPUExecutionProviderSpec``.
Multi-node additionally requires a shared filesystem for Ray's
cross-node ENDED/barrier coordination, enforced here with a fail-fast.
"""

from __future__ import annotations

import logging

from nemo_platform_plugin.integrations import IntegrationsSpec
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    DistributedGPUExecutionProviderSpec,
    EnvironmentVariable,
    GPUExecutionProviderSpec,
    PlatformJobSpec,
    PlatformJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.models.types import ModelEntity
from nmp.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nmp.customization_common.integrations import (
    collect_integration_secret_envs,
    warn_incomplete_integrations,
)
from nmp.customization_common.schemas.file_io import (
    DownloadItem,
    FileIOTaskConfig,
    FileSetRef,
    UploadItem,
)
from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients, fetch_model_entity
from nmp.customization_common.tasks.file_io_metadata import build_output_fileset_metadata_from_model_entity
from nmp.rl.app.constants import (
    BASE_LOG_DIR_ENVVAR,
    DEFAULT_DATASET_PATH,
    DEFAULT_ENVIRONMENT_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_MODEL_PATH,
    NMP_BROKER_HOST_ENVVAR,
    NMP_BROKER_PORT_ENVVAR,
    NMP_JOB_STORAGE_PVC_ENVVAR,
    NMP_VLLM_HOST_ENVVAR,
    NMP_VLLM_PORT_ENVVAR,
    SANDBOX_DATASET_PATH,
    SANDBOX_ENVIRONMENT_PATH,
)
from nmp.rl.app.jobs.training.schemas import (
    DPOConfig,
    GRPOConfig,
    LoRAConfig,
    MLflowConfig,
    ModelConfig,
    TrainingBackend,
    TrainingStepConfig,
    WandBConfig,
)
from nmp.rl.config import config, platform_config
from nmp.rl.entities.values import FinetuningType, TrainingType
from nmp.rl.images import (
    FILE_IO_TASK_COMMAND,
    MODEL_ENTITY_TASK_COMMAND,
    RL_PYTHON_ENTRYPOINT,
    get_tasks_image,
    get_training_image,
)
from nmp.rl.schemas import DPOTraining, GRPOTraining, RlJobOutput

logger = logging.getLogger(__name__)


GPU_TRAINING_SHM_GIB_PER_GPU = 8


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


def _gpu_training_resources(*, num_nodes: int = 1, num_gpus: int) -> ResourcesSpec:
    return ResourcesSpec(
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        shm_size=f"{GPU_TRAINING_SHM_GIB_PER_GPU * max(1, num_gpus)}Gi",
    )


def _base_environment() -> list[EnvironmentVariable]:
    return [EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]


def _require_fileset(name: str | None, *, label: str) -> str:
    if not name or not str(name).strip():
        raise PlatformJobCompilationError(
            f"{label} has no fileset attached. Attach a platform FileSet (workspace/name) before training.",
        )
    return str(name)


def _build_download_config(job_spec: RlJobOutput, me: ModelEntity, *, workspace: str) -> FileIOTaskConfig:
    model_fileset = _require_fileset(me.fileset, label=f"Model '{me.workspace}/{me.name}'")
    dataset_ref = FileSetRef.model_validate(job_spec.dataset)
    if dataset_ref.workspace is None:
        dataset_ref = FileSetRef(workspace=workspace, name=dataset_ref.name)

    downloads = [
        DownloadItem(src=FileSetRef.model_validate(model_fileset), dest=DEFAULT_MODEL_PATH),
        DownloadItem(src=dataset_ref, dest=DEFAULT_DATASET_PATH),
    ]

    if job_spec.training_type == TrainingType.GRPO:
        env_ref = FileSetRef.model_validate(_require_fileset(job_spec.environment, label="Environment"))
        if env_ref.workspace is None:
            env_ref = FileSetRef(workspace=workspace, name=env_ref.name)
        downloads.append(DownloadItem(src=env_ref, dest=DEFAULT_ENVIRONMENT_PATH))

    return FileIOTaskConfig(download=downloads)


def _build_upload_config(job_spec: RlJobOutput, me) -> FileIOTaskConfig:
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
    workspace: str, job_spec: RlJobOutput, *, trust_remote_code: bool
) -> ModelEntityTaskConfig:
    method = job_spec.training_type.value.upper()
    peft: PEFTConfig | None = None
    if isinstance(job_spec.training, GRPOTraining) and job_spec.training.finetuning_type == "lora":
        lora = job_spec.training.lora
        assert lora is not None  # validated on GRPOTraining
        peft = PEFTConfig(type=FinetuningType.LORA, rank=lora.rank, alpha=lora.alpha)
        description = f"{method}-trained LoRA adapter from nmp-rl job ({job_spec.model})"
    else:
        description = f"{method}-trained model from nmp-rl job ({job_spec.model})"
    return ModelEntityTaskConfig(
        name=job_spec.output.name,
        workspace=workspace,
        description=description,
        fileset=FileSetRef(workspace=None, name=job_spec.output.fileset),
        model_entity=job_spec.model,
        base_model=job_spec.model,
        peft=peft,
        trust_remote_code=trust_remote_code,
        deployment_config=None,
    )


def _build_integrations_config(integrations: IntegrationsSpec | None) -> TrainingStepConfig.IntegrationsConfig:
    """Map the public ``IntegrationsSpec`` onto the training step's ``IntegrationsConfig``.

    Without this the step config carries the empty default and the driver's
    ``build_wandb_config`` / ``build_mlflow_config`` (and the backend's MLFLOW_URI
    setup) all see ``None`` — silently disabling W&B/MLflow even when the job
    requested them. Field names line up except MLflow's run name (public ``name``
    maps to the step's ``run_name``). Secrets (``api_key_secret``) are NOT copied
    here; ``collect_integration_secret_envs`` injects them as env vars in
    :func:`_build_training_step`.
    """
    if integrations is None:
        return TrainingStepConfig.IntegrationsConfig()

    wandb_cfg = None
    if integrations.wandb is not None:
        w = integrations.wandb
        wandb_cfg = WandBConfig(
            project=w.project,
            name=w.name,
            entity=w.entity,
            tags=w.tags,
            notes=w.notes,
            base_url=w.base_url,
        )

    mlflow_cfg = None
    if integrations.mlflow is not None:
        m = integrations.mlflow
        mlflow_cfg = MLflowConfig(
            experiment_name=m.experiment_name,
            run_name=m.name,
            tags=m.tags,
            description=m.description,
            tracking_uri=m.tracking_uri,
        )

    return TrainingStepConfig.IntegrationsConfig(wandb=wandb_cfg, mlflow=mlflow_cfg)


def _build_dpo_training_step_config(job_spec: RlJobOutput, *, trust_remote_code: bool) -> TrainingStepConfig:
    """Map a DPO spec onto the backend-agnostic step config."""
    t = job_spec.training
    if not isinstance(t, DPOTraining):
        raise PlatformJobCompilationError(f"Expected a DPO training spec, got {type(t).__name__}.")
    p = t.parallelism
    return TrainingStepConfig(
        backend=TrainingBackend.NEMO_RL,
        model=ModelConfig(
            path=DEFAULT_MODEL_PATH,
            name=job_spec.model,
            max_seq_length=t.max_seq_length,
            trust_remote_code=trust_remote_code,
        ),
        dataset=TrainingStepConfig.DatasetConfig(path=DEFAULT_DATASET_PATH),
        training=TrainingStepConfig.TrainingConfig(
            training_type=TrainingType.DPO,
            finetuning_type=FinetuningType.ALL_WEIGHTS,
            dpo=DPOConfig(
                ref_policy_kl_penalty=t.ref_policy_kl_penalty,
                preference_average_log_probs=t.preference_average_log_probs,
                sft_average_log_probs=t.sft_average_log_probs,
                preference_loss_weight=t.preference_loss_weight,
                sft_loss_weight=t.sft_loss_weight,
                max_grad_norm=t.max_grad_norm,
            ),
        ),
        schedule=TrainingStepConfig.ScheduleConfig(
            epochs=t.epochs,
            max_steps=t.max_steps,
            val_check_interval=t.val_check_interval,
            # DPO validation is a logprob pass with no generation, so dpo_config always
            # runs it at step 0. The knob is GRPO-only, where a baseline costs a rollout pass.
            val_at_end=t.val_at_end,
            keep_top_k=t.keep_top_k,
            progress_reporting=t.progress_reporting,
        ),
        batch=TrainingStepConfig.BatchConfig(global_batch_size=t.batch_size, micro_batch_size=t.micro_batch_size),
        optimizer=TrainingStepConfig.OptimizerConfig(
            optimizer_type=t.optimizer_type,
            learning_rate=t.learning_rate,
            min_learning_rate=t.min_learning_rate,
            weight_decay=t.weight_decay,
            beta1=t.adam_beta1,
            beta2=t.adam_beta2,
            eps=t.adam_eps,
            warmup_steps=t.warmup_steps,
        ),
        parallelism=TrainingStepConfig.ParallelismConfig(
            num_nodes=p.num_nodes,
            num_gpus_per_node=p.num_gpus_per_node,
            tensor_parallel_size=p.tensor_parallel_size,
            pipeline_parallel_size=p.pipeline_parallel_size,
            context_parallel_size=p.context_parallel_size,
            sequence_parallel=p.sequence_parallel,
            activation_checkpointing=t.activation_checkpointing,
        ),
        integrations=_build_integrations_config(job_spec.integrations),
        output_model=job_spec.output.name,
        seed=t.seed if t.seed is not None else 42,
    )


def _build_grpo_training_step_config(job_spec: RlJobOutput, *, trust_remote_code: bool) -> TrainingStepConfig:
    """Map a GRPO spec onto the backend-agnostic step config."""
    t = job_spec.training
    if not isinstance(t, GRPOTraining):
        raise PlatformJobCompilationError(f"Expected a GRPO training spec, got {type(t).__name__}.")
    if t.policy_backend is None:
        # GRPOTraining resolves it from finetuning_type, so only a hand-built spec gets here.
        raise PlatformJobCompilationError("policy_backend is unset; submit through GRPOTraining.")
    p = t.parallelism
    sandboxed = config.sandboxed_gym_default
    if sandboxed and not platform_config.sandbox_cluster_capable:
        raise PlatformJobCompilationError(
            "GRPO jobs with custom environment filesets require sandboxed Gym (platform default). "
            "OpenSandbox is not yet available on this cluster (sandbox_cluster_capable=false). "
            "Set sandboxClusterCapable=true (platform.sandbox_cluster_capable) once OpenSandbox "
            "is installed, or NMP_RL_SANDBOXED_GYM_DEFAULT=false for trusted dev smoke tests only."
        )
    # The Gym host mounts the job-storage claim itself to read the environment and
    # dataset, and the training container only ever learns the storage *path*. Without
    # the claim name the sandbox config is unsatisfiable, so fail here rather than
    # after the model download and vLLM startup.
    if sandboxed and not config.job_storage_pvc_claim:
        raise PlatformJobCompilationError(
            "Sandboxed GRPO requires the job-storage PVC claim name so the Gym sandbox can "
            "mount the downloaded environment and dataset. Set NMP_RL_JOB_STORAGE_PVC_CLAIM "
            "to the same claim the Jobs controller mounts for job storage."
        )

    return TrainingStepConfig(
        backend=TrainingBackend.NEMO_RL,
        model=ModelConfig(
            path=DEFAULT_MODEL_PATH,
            name=job_spec.model,
            max_seq_length=t.max_seq_length,
            trust_remote_code=trust_remote_code,
            v4_compatible=t.v4_compatible,
        ),
        dataset=TrainingStepConfig.DatasetConfig(path=DEFAULT_DATASET_PATH),
        gym=TrainingStepConfig.GymConfig(
            environment_path=DEFAULT_ENVIRONMENT_PATH,
            sandbox_environment_path=SANDBOX_ENVIRONMENT_PATH if sandboxed else None,
            sandbox_dataset_path=SANDBOX_DATASET_PATH if sandboxed else None,
            sandboxed=sandboxed,
            gym_runtime_image=config.gym_runtime_image or get_training_image(),
            allow_internet=config.sandbox_allow_internet,
            public_dns_allow=config.sandbox_public_dns_allow,
            sandbox_server_protocol=platform_config.sandbox_server_protocol,
            sandbox_resources=config.sandbox_resources,
            sandbox_ttl_s=config.sandbox_ttl_s,
            sandbox_rollout_chunk_size=config.sandbox_rollout_chunk_size,
            sandbox_rollout_max_in_flight=config.sandbox_rollout_max_in_flight,
        ),
        training=TrainingStepConfig.TrainingConfig(
            training_type=TrainingType.GRPO,
            finetuning_type=FinetuningType.LORA if t.finetuning_type == "lora" else FinetuningType.ALL_WEIGHTS,
            grpo=GRPOConfig(
                num_generations_per_prompt=t.num_generations_per_prompt,
                num_prompts_per_step=t.num_prompts_per_step,
                temperature=t.temperature,
                max_new_tokens=t.max_new_tokens,
                top_k=t.top_k,
                normalize_rewards=t.normalize_rewards,
                use_leave_one_out_baseline=t.use_leave_one_out_baseline,
                advantage_clip_low=t.advantage_clip_low,
                advantage_clip_high=t.advantage_clip_high,
                overlong_filtering=t.overlong_filtering,
                max_rollout_turns=t.max_rollout_turns,
                ref_policy_kl_penalty=t.ref_policy_kl_penalty,
                ratio_clip_min=t.ratio_clip_min,
                ratio_clip_max=t.ratio_clip_max,
                ratio_clip_c=t.ratio_clip_c,
                use_on_policy_kl_approximation=t.use_on_policy_kl_approximation,
                use_importance_sampling_correction=t.use_importance_sampling_correction,
                max_grad_norm=t.max_grad_norm,
                truncated_importance_sampling_type=t.truncated_importance_sampling_type,
                truncated_importance_sampling_ratio=t.truncated_importance_sampling_ratio,
                truncated_importance_sampling_ratio_min=t.truncated_importance_sampling_ratio_min,
                use_dynamic_sampling=t.use_dynamic_sampling,
                dynamic_sampling_max_gen_batches=t.dynamic_sampling_max_gen_batches,
                batch_multiplier=t.batch_multiplier,
                # exclude_none: RewardShapingConfig reads each penalty field independently, so a
                # null would land as an explicit "no penalty" rather than leaving its default.
                reward_shaping=(t.reward_shaping.model_dump(exclude_none=True) if t.reward_shaping else None),
                reward_scaling=(t.reward_scaling.model_dump() if t.reward_scaling else None),
                batching_strategy=t.batching_strategy,
                train_mb_tokens=t.train_mb_tokens,
                sequence_length_round=t.sequence_length_round,
                automodel_kwargs=t.automodel_kwargs,
                router_aux_loss_coef=t.router_aux_loss_coef,
                hf_config_overrides=t.hf_config_overrides,
                vllm_tensor_parallel_size=t.vllm_tensor_parallel_size,
                vllm_gpu_memory_utilization=t.vllm_gpu_memory_utilization,
            ),
            lora=(
                LoRAConfig(
                    rank=t.lora.rank,
                    alpha=t.lora.alpha,
                    dropout=t.lora.dropout,
                    target_modules=t.lora.target_modules,
                    exclude_modules=t.lora.exclude_modules,
                    use_triton=t.lora.use_triton,
                )
                if t.finetuning_type == "lora" and t.lora is not None
                else None
            ),
        ),
        schedule=TrainingStepConfig.ScheduleConfig(
            epochs=t.epochs,
            max_steps=t.max_steps,
            val_check_interval=t.val_check_interval,
            val_at_start=t.val_at_start,
            val_at_end=t.val_at_end,
            keep_top_k=t.keep_top_k,
            progress_reporting=t.progress_reporting,
        ),
        batch=TrainingStepConfig.BatchConfig(global_batch_size=t.batch_size, micro_batch_size=t.micro_batch_size),
        optimizer=TrainingStepConfig.OptimizerConfig(
            optimizer_type=t.optimizer_type,
            learning_rate=t.learning_rate,
            min_learning_rate=t.min_learning_rate,
            weight_decay=t.weight_decay,
            beta1=t.adam_beta1,
            beta2=t.adam_beta2,
            eps=t.adam_eps,
            warmup_steps=t.warmup_steps,
        ),
        parallelism=TrainingStepConfig.ParallelismConfig(
            num_nodes=p.num_nodes,
            num_gpus_per_node=p.num_gpus_per_node,
            tensor_parallel_size=p.tensor_parallel_size,
            pipeline_parallel_size=p.pipeline_parallel_size,
            context_parallel_size=p.context_parallel_size,
            expert_parallel_size=p.expert_parallel_size,
            sequence_parallel=p.sequence_parallel,
            activation_checkpointing=t.activation_checkpointing,
            policy_backend=t.policy_backend,
        ),
        integrations=_build_integrations_config(job_spec.integrations),
        output_model=job_spec.output.name,
        seed=t.seed if t.seed is not None else 42,
    )


def _build_training_step_config(job_spec: RlJobOutput, *, trust_remote_code: bool) -> TrainingStepConfig:
    if job_spec.training_type == TrainingType.GRPO:
        return _build_grpo_training_step_config(job_spec, trust_remote_code=trust_remote_code)
    return _build_dpo_training_step_config(job_spec, trust_remote_code=trust_remote_code)


def _build_training_step(
    job_spec: RlJobOutput,
    base_env: list[EnvironmentVariable],
    *,
    trust_remote_code: bool,
    profile: str | None,
) -> PlatformJobStep:
    """Build the Ray DPO training step, selecting the executor by ``num_nodes``.

    Multi-node (``num_nodes > 1``) requires shared storage for Ray's cross-node
    ENDED/barrier coordination — fail fast when it is not configured.
    """
    p = job_spec.training.parallelism
    num_nodes = p.num_nodes
    num_gpus_per_node = p.num_gpus_per_node

    step_config = _build_training_step_config(job_spec, trust_remote_code=trust_remote_code)
    step_name = "grpo-training" if job_spec.training_type == TrainingType.GRPO else "dpo-training"

    container = ContainerSpec(
        image=get_training_image(),
        entrypoint=RL_PYTHON_ENTRYPOINT,
        command=["-m", "nmp.rl.tasks.training"],
    )

    warn_incomplete_integrations(job_spec.integrations)
    environment = [*base_env, *collect_integration_secret_envs(job_spec.integrations)]

    if job_spec.training_type == TrainingType.GRPO:
        for name, value in (
            (NMP_VLLM_HOST_ENVVAR, "127.0.0.1"),
            (NMP_VLLM_PORT_ENVVAR, "8000"),
            (NMP_BROKER_HOST_ENVVAR, "127.0.0.1"),
            (NMP_BROKER_PORT_ENVVAR, "51234"),
        ):
            environment.append(EnvironmentVariable(name=name, value=value))
        if config.job_storage_pvc_claim:
            environment.append(EnvironmentVariable(name=NMP_JOB_STORAGE_PVC_ENVVAR, value=config.job_storage_pvc_claim))

    executor: GPUExecutionProviderSpec | DistributedGPUExecutionProviderSpec
    if num_nodes > 1:
        shared_dir = config.multinode_shared_storage_path
        if not shared_dir:
            raise PlatformJobCompilationError(
                f"Multi-node NeMo-RL training (num_nodes={num_nodes}) requires a shared filesystem for Ray's "
                "cross-node coordination. Set NMP_RL_MULTINODE_SHARED_STORAGE_PATH to a path mounted on every "
                "node (e.g. an NFS mount) before submitting a multi-node job.",
            )
        # Ray's bootstrap writes the ENDED marker + barriers under BASE_LOG_DIR.
        environment = [*environment, EnvironmentVariable(name=BASE_LOG_DIR_ENVVAR, value=shared_dir)]
        resolved_profile = profile or config.default_distributed_execution_profile
        executor = DistributedGPUExecutionProviderSpec(
            provider="gpu_distributed",
            container=container,
            resources=_gpu_training_resources(num_nodes=num_nodes, num_gpus=num_gpus_per_node),
            profile=resolved_profile if resolved_profile is not None else "default",
        )
    else:
        resolved_profile = profile or config.default_training_execution_profile
        executor = GPUExecutionProviderSpec(
            provider="gpu",
            container=container,
            resources=_gpu_training_resources(num_gpus=num_gpus_per_node),
            profile=resolved_profile if resolved_profile is not None else "default",
        )

    return PlatformJobStep(
        name=step_name,
        executor=executor,
        environment=environment,
        config=step_config.model_dump(mode="json"),
    )


async def platform_job_config_compiler(
    workspace: str,
    job_spec: RlJobOutput,
    platform: AsyncCustomizationPlatformClients,
    *,
    job_name: str | None = None,
    profile: str | None = None,
) -> PlatformJobSpec:
    """Compile a canonical NeMo-RL job spec into a 4-step ``PlatformJobSpec``."""
    del job_name  # reserved for future scheduling decisions

    # Log only non-sensitive, high-level context. The full spec embeds
    # `integrations` (W&B / MLflow tokens and tracking URIs), so it must not be
    # dumped at INFO.
    p = job_spec.training.parallelism
    method = job_spec.training_type.value.upper()
    logger.info(
        "Compiling NeMo-RL %s job to PlatformJobSpec: model=%s, dataset=%s, output=%s, "
        "num_nodes=%d, num_gpus_per_node=%d",
        method,
        job_spec.model,
        job_spec.dataset,
        job_spec.output.name,
        p.num_nodes,
        p.num_gpus_per_node,
    )

    job_spec.validate_for_training()

    me = await fetch_model_entity(job_spec.model, workspace, platform)
    trust_remote_code = me.trust_remote_code or False

    cpu_resources = _get_cpu_resources()
    base_env = _base_environment()

    def _cpu_task_step(
        name: str,
        command: list[str],
        task_config: FileIOTaskConfig | ModelEntityTaskConfig,
    ) -> PlatformJobStep:
        return PlatformJobStep(
            name=name,
            executor=CPUExecutionProviderSpec(
                provider="cpu",
                container=ContainerSpec(
                    image=get_tasks_image(),
                    entrypoint=RL_PYTHON_ENTRYPOINT,
                    command=command,
                ),
                resources=cpu_resources,
            ),
            environment=base_env,
            config=task_config.model_dump(mode="json"),
        )

    steps: list[PlatformJobStep] = [
        _cpu_task_step(
            "model-dataset-environment-download"
            if job_spec.training_type == TrainingType.GRPO
            else "model-and-dataset-download",
            FILE_IO_TASK_COMMAND,
            _build_download_config(job_spec, me, workspace=workspace),
        ),
        _build_training_step(job_spec, base_env, trust_remote_code=trust_remote_code, profile=profile),
        _cpu_task_step("model-upload", FILE_IO_TASK_COMMAND, _build_upload_config(job_spec, me)),
        _cpu_task_step(
            "model-entity-creation",
            MODEL_ENTITY_TASK_COMMAND,
            _build_model_entity_config(workspace, job_spec, trust_remote_code=trust_remote_code),
        ),
    ]

    return PlatformJobSpec(steps=steps)
