# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training step compiler."""

import logging

from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    DistributedGPUExecutionProviderSpec,
    EnvironmentVariable,
    GPUExecutionProviderSpec,
    PlatformJobStep,
    ResourcesSpec,
    StepLifecycle,
)
from nemo_platform_plugin.models.types import ModelEntity
from nmp.automodel.api.v2.jobs.schemas import (
    AnyTraining,
    CustomizationJobOutput,
    DistillationTraining,
    LoRAParams,
)
from nmp.automodel.app.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEACHER_MODEL_PATH,
    V4_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from nmp.automodel.app.jobs.training.schemas import (
    DistillationConfig,
    LoRAConfig,
    ModelConfig,
    RetrievalConfig,
    TrainingRecipe,
    TrainingStepConfig,
)
from nmp.automodel.config import config
from nmp.automodel.entities.values import Precision, TrainingType
from nmp.automodel.images import AUTOMODEL_PYTHON_ENTRYPOINT, get_training_image
from nmp.common.model_utils import is_embedding_model
from nmp.customization_common.integrations import (
    collect_integration_secret_envs,
    warn_incomplete_integrations,
)

logger = logging.getLogger(__name__)


def _resolve_is_embedding_model(me: ModelEntity) -> bool:
    """Resolve embedding flag while preserving compatibility with legacy specs."""
    if me.spec is None:
        return is_embedding_model(me.name)

    # Do not rely on `me.spec is not None` alone:
    # older persisted ModelSpec payloads may not include `is_embedding_model`.
    # Pydantic fills missing fields with the default (False), which would
    # incorrectly classify legacy embedding models as LLMs.
    model_fields_set = getattr(me.spec, "model_fields_set", getattr(me.spec, "__fields_set__", set()))
    if "is_embedding_model" not in model_fields_set:
        return is_embedding_model(me.name)

    return me.spec.is_embedding_model or False


def _resolve_checkpoint_head_type(me: ModelEntity) -> str:
    """Resolve the persisted checkpoint head while supporting legacy model specs."""
    if me.spec is None:
        return "embedding" if is_embedding_model(me.name) else "unknown"

    fields_set = getattr(me.spec, "model_fields_set", getattr(me.spec, "__fields_set__", set()))
    if "head_type" in fields_set:
        return me.spec.head_type or "unknown"
    return "embedding" if _resolve_is_embedding_model(me) else "unknown"


def _resolve_training_recipe(me: ModelEntity, requested: str) -> TrainingRecipe:
    """Resolve an explicit/automatic job recipe independently of backbone architecture."""
    if requested != "auto":
        return TrainingRecipe(requested)

    head_type = _resolve_checkpoint_head_type(me)
    if head_type == "embedding":
        return TrainingRecipe.BI_ENCODER
    if head_type == "cross_encoder":
        return TrainingRecipe.CROSS_ENCODER
    return TrainingRecipe.SFT


def _resolve_v4_compatible(me: ModelEntity) -> bool:
    """Check if the model requires transformers-v4-compatible checkpoint output."""
    if me.spec is None:
        return False
    checkpoint_model_name = getattr(me.spec, "checkpoint_model_name", None)
    is_v4_compatible = checkpoint_model_name in V4_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    logger.info(f"Checkpoint model name {checkpoint_model_name} is v4 compatible: {is_v4_compatible}")
    return is_v4_compatible


def _resolve_custom_implementation_override(me: ModelEntity) -> bool:
    if me.spec is None:
        return False

    checkpoint_model_name = getattr(me.spec, "checkpoint_model_name", None)
    if checkpoint_model_name == "NemotronHForCausalLM" and getattr(me.spec, "moe_config", None) is None:
        # V2 Model is being used, v3 uses MoE - However V2 gets recognized as V3 and fails
        return True

    if (
        checkpoint_model_name == "MistralForCausalLM"
        and getattr(me.spec, "family", None) == "mistral"
        and getattr(me.spec, "is_chat", False)
    ):
        # Mistral 7b v0.3 Instruct has the custom tokenizer implementation fail with:
        """2026-03-02 18:35:51 | INFO | root | Using model config to instantiate tokenizer
        2026-03-02 18:35:53 | INFO | nemo_automodel._transformers.auto_tokenizer | Using custom tokenizer MistralCommonBackend for model type 'mistral'
        2026-03-02 18:35:53 | WARNING | nemo_automodel._transformers.tokenization.tokenization_mistral_common | Multiple tokenizer files found in directory: /var/run/scratch/job/model. Using tokenizer.model.v3.
        Instantiation failed for `ColumnMappedTextInstructionDataset`
        Accepted signature : (path_or_dataset_id: Union[str, List[str]], column_mapping: Dict[str, str], tokenizer, *, split: Optional[str] = 'train', name: Optional[str] = None, answer_only_loss_mask: bool = True, seq_length: Optional[int] = None, padding: Union[str, bool] = 'do_not_pad', truncation: Union[str, bool] = 'do_not_truncate', limit_dataset_samples: Optional[int] = None, use_hf_chat_template: bool = False) -> None
        Positional args    : ()
        Keyword args       : {   'answer_only_loss_mask': True,
            'column_mapping': {'answer': 'completion', 'question': 'prompt'},
            'padding': 'do_not_pad',
            'path_or_dataset_id': '/run/scratch/job/training/dataset/train.jsonl',
            'seq_length': 1024,
            'split': 'train',
            'tokenizer': '******',
            'truncation': 'longest_first'}
        Exception          : piece id is out of range.
        """
        return True

    return False


def compile_training_step(
    job_spec: CustomizationJobOutput,
    base_env: list[EnvironmentVariable],
    me: ModelEntity,
    teacher_me: ModelEntity | None = None,
) -> PlatformJobStep:
    """Compile job input to a PlatformJobStep for training.

    Args:
        job_spec: The customization job output specification.
        base_env: Base environment variables for the job step.
        me: The model entity being trained.
        teacher_me: Optional teacher model entity for knowledge distillation jobs.

    """
    job_spec.validate_for_training()
    if TrainingType(job_spec.training.type) == TrainingType.DPO:
        raise ValueError("DPO training is not supported by nmp-automodel")
    trust_remote_code = me.trust_remote_code or False
    chat_template = me.spec.chat_template if me.spec else None
    checkpoint_head_type = _resolve_checkpoint_head_type(me)
    override_custom_impl = _resolve_custom_implementation_override(me)
    v4_compatible = _resolve_v4_compatible(me)
    training = job_spec.training
    training_recipe = _resolve_training_recipe(me, training.recipe)
    training = training.with_resolved_recipe(training_recipe.value)
    if isinstance(training, DistillationTraining) and training_recipe != TrainingRecipe.SFT:
        raise ValueError("Knowledge distillation only supports the sft recipe.")
    p = training.parallelism
    num_gpus_per_node = p.num_gpus_per_node

    training_config = TrainingStepConfig(
        model=_translate_model_config(
            job_spec,
            DEFAULT_MODEL_PATH,
            trust_remote_code=trust_remote_code,
            checkpoint_head_type=checkpoint_head_type,
            chat_template=chat_template,
            override_custom_impl=override_custom_impl,
            v4_compatible=v4_compatible,
        ),
        dataset=TrainingStepConfig.DatasetConfig(
            path=DEFAULT_DATASET_PATH,
        ),
        training=_translate_training_config(training, me, teacher_me=teacher_me, recipe=training_recipe),
        schedule=TrainingStepConfig.ScheduleConfig(
            epochs=training.epochs,
            max_steps=training.max_steps,
            val_check_interval=training.val_check_interval,
            progress_reporting=training.progress_reporting,
        ),
        batch=TrainingStepConfig.BatchConfig(
            global_batch_size=training.batch_size,
            micro_batch_size=training.micro_batch_size,
            sequence_packing=training.sequence_packing,
            sequence_packing_max_samples=training.sequence_packing_max_samples,
        ),
        optimizer=TrainingStepConfig.OptimizerConfig(
            optimizer_name=training.optimizer,
            lr_decay_style=training.lr_decay_style,
            learning_rate=training.learning_rate,
            min_learning_rate=training.min_learning_rate,
            eps=training.adam_eps,
            weight_decay=training.weight_decay,
            beta1=training.adam_beta1,
            beta2=training.adam_beta2,
            warmup_steps=training.warmup_steps,
        ),
        parallelism=TrainingStepConfig.ParallelismConfig(
            num_nodes=p.num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            tensor_parallel_size=p.tensor_parallel_size,
            pipeline_parallel_size=p.pipeline_parallel_size,
            context_parallel_size=p.context_parallel_size,
            expert_parallel_size=p.expert_parallel_size,
            sequence_parallel=p.sequence_parallel,
        ),
        integrations=job_spec.integrations,
        output_model=job_spec.output.name,
        retrieval=_translate_retrieval_config(training),
    )

    container = ContainerSpec(
        image=_get_training_image(),
        entrypoint=AUTOMODEL_PYTHON_ENTRYPOINT,
        command=["-m", "nmp.automodel.tasks.training"],
    )

    profile = (
        training.execution_profile
        if training.execution_profile is not None
        else config.default_training_execution_profile
    )

    if p.num_nodes > 1:
        logger.debug(f"Using distributed GPU executor: num_nodes={p.num_nodes}, num_gpus_per_node={num_gpus_per_node}")
        executor = DistributedGPUExecutionProviderSpec(
            provider="gpu_distributed",
            profile=profile,
            container=container,
            resources=ResourcesSpec(
                num_gpus=num_gpus_per_node,
                num_nodes=p.num_nodes,
            ),
        )
    else:
        logger.debug(f"Using single-node GPU executor: num_gpus={num_gpus_per_node}")
        executor = GPUExecutionProviderSpec(
            provider="gpu",
            profile=profile,
            container=container,
            resources=ResourcesSpec(
                num_gpus=num_gpus_per_node,
            ),
        )

    warn_incomplete_integrations(job_spec.integrations)
    secret_envs = collect_integration_secret_envs(job_spec.integrations)

    return PlatformJobStep(
        name="training",
        executor=executor,
        environment=[*base_env, *secret_envs, EnvironmentVariable(name="HF_DATASETS_OFFLINE", value="1")],
        config=training_config.model_dump(mode="json"),
        lifecycle=StepLifecycle(staleness_timeout_seconds=config.training_staleness_timeout_seconds),
    )


def _translate_model_config(
    job_spec: CustomizationJobOutput,
    path: str,
    trust_remote_code: bool = False,
    checkpoint_head_type: str = "unknown",
    chat_template: str | None = None,
    override_custom_impl: bool = False,
    v4_compatible: bool = False,
) -> ModelConfig:
    """Translate job spec to internal ModelConfig."""
    training = job_spec.training
    return ModelConfig(
        path=path,
        name=_extract_model_name(job_spec),
        max_seq_length=training.max_seq_length,
        precision=training.precision,
        attn_implementation=training.attn_implementation,
        trust_remote_code=trust_remote_code,
        checkpoint_head_type=checkpoint_head_type,
        is_embedding_model=checkpoint_head_type == "embedding",
        chat_template=chat_template,
        override_custom_impl=override_custom_impl,
        v4_compatible=v4_compatible,
    )


def _translate_training_config(
    training: AnyTraining,
    me: ModelEntity,
    teacher_me: ModelEntity | None = None,
    recipe: TrainingRecipe = TrainingRecipe.SFT,
) -> TrainingStepConfig.TrainingConfig:
    """Translate API training method to internal TrainingConfig.

    Args:
        training: The API training configuration.
        me: The primary model entity.
        teacher_me: Teacher model entity, populated for distillation jobs.
    """
    training_type = TrainingType(training.type)
    lora = _translate_lora_config(training.peft, me) if isinstance(training.peft, LoRAParams) else None

    kd = None
    if isinstance(training, DistillationTraining):
        teacher_trust_remote_code = (teacher_me.trust_remote_code or False) if teacher_me else False
        kd = DistillationConfig(
            teacher_model=ModelConfig(
                path=DEFAULT_TEACHER_MODEL_PATH,
                name=training.teacher_model,
                precision=Precision(training.teacher_precision),
                trust_remote_code=teacher_trust_remote_code,
            ),
            ratio=training.distillation_ratio,
            temperature=training.distillation_temperature,
        )

    return TrainingStepConfig.TrainingConfig(
        training_type=training_type,
        recipe=recipe,
        finetuning_type=training.finetuning_type,
        lora=lora,
        kd=kd,
    )


def _translate_retrieval_config(training: AnyTraining) -> RetrievalConfig | None:
    raw = getattr(training, "retrieval", None)
    if raw is None:
        return None
    if isinstance(raw, RetrievalConfig):
        return raw
    return RetrievalConfig.model_validate(raw.model_dump(mode="python"))


def _translate_lora_config(api_lora: LoRAParams, me: ModelEntity) -> LoRAConfig:
    """Translate API LoRAConfig to internal LoRAConfig."""
    lora = LoRAConfig(
        rank=api_lora.rank,
        alpha=api_lora.alpha,
        dropout=api_lora.dropout,
        target_modules=api_lora.target_modules,
        exclude_modules=api_lora.exclude_modules,
        use_triton=api_lora.use_triton,
    )

    if not lora.target_modules:
        if me.spec and me.spec.checkpoint_model_name == "NemotronHForCausalLM":
            # Need to remove out_proj from the list of target modules
            modules = set()
            if me.spec.linear_layers:
                for ll in me.spec.linear_layers:
                    m = ll.name.split(".")[-1]
                    if m.endswith("proj"):
                        modules.add(f"*.{m}")
                modules.discard("*.out_proj")

            # In cases when model_spec has linear_layers as null, we need to set the target_modules to default
            # If target_modules is empty we get this error during training:
            # Expected match_all_linear to be true or target_modules/exclude_modules to be non-empty
            lora.target_modules = list(modules) if modules else ["*proj"]
        else:
            lora.target_modules = ["*proj"]
    return lora


def _extract_model_name(job_spec: CustomizationJobOutput) -> str | None:
    """Extract the canonical model name from the model field for template lookup.

    The model name follows the pattern "workspace/name" (e.g., "meta/llama-3.1-8b-instruct")
    which matches the keys in DEFAULT_CHAT_TEMPLATES.
    """
    model = job_spec.model

    if "/" in model:
        logger.debug(f"Extracted model name from URN: {model}")
        return model

    return None


def _get_training_image() -> str:
    """Training container image for the Automodel task."""
    return config.training_automodel_image or get_training_image()
