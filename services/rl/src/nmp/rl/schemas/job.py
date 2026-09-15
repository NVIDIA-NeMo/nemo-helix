# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical NeMo-RL job / training schemas — consumed by the compiler and drivers."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self, Union

from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.customization_common.schema import NamespacedModel
from nmp.customization_common.schemas.values import OutputNameType
from nmp.customization_common.training.reporting import ProgressReportingConfig
from nmp.rl.app.jobs.training.schemas import BatchingStrategy, OptimizerType, PolicyBackend
from nmp.rl.entities.values import TrainingType
from pydantic import ConfigDict, Discriminator, Field, model_validator


class RlSchema(NamespacedModel):
    """Backend base: every RL-owned model emits an ``Rl``-prefixed OpenAPI schema
    name (``DPOTraining`` -> ``RlDPOTraining``), so it can't collide with another
    backend's same-named model in the merged ``/apis/customization`` spec.
    """

    __schema_namespace__ = "Rl"


class ParallelismParams(RlSchema):
    """Distributed training parallelism configuration."""

    num_gpus_per_node: int = Field(default=1, gt=0, description="Number of GPUs per node.")
    num_nodes: int = Field(default=1, gt=0, description="Number of nodes (>1 → multi-node Ray cluster).")
    tensor_parallel_size: int = Field(default=1, gt=0, description="Tensor parallel size.")
    pipeline_parallel_size: int = Field(default=1, gt=0, description="Pipeline parallel size.")
    context_parallel_size: int = Field(default=1, gt=0, description="Context parallel size.")
    expert_parallel_size: int = Field(
        default=1,
        gt=0,
        description="Expert parallel size for MoE models. GRPO only, and implemented only by "
        "policy_backend='automodel'.",
    )
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism.")


class RewardShapingParams(RlSchema):
    """DAPO-style reward shaping (NeMo-RL ``grpo.reward_shaping``).

    Both penalties act on responses the generation cap cut off, which score zero on a
    verifier that needs a complete answer. ``overlong_*`` ramps a penalty in over the last
    ``overlong_buffer_length`` tokens before ``max_response_length``; ``stop_properly_penalty_coef``
    scales the reward of anything truncated outright.
    """

    overlong_buffer_length: int | None = Field(
        default=None,
        gt=0,
        description="Tokens before max_response_length over which the penalty ramps to full.",
    )
    overlong_buffer_penalty: float | None = Field(
        default=None,
        ge=0.0,
        description="Penalty applied at the end of the buffer.",
    )
    max_response_length: int | None = Field(
        default=None,
        gt=0,
        description="Length beyond which a response is penalised. Usually the generation cap.",
    )
    stop_properly_penalty_coef: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Scale factor on the reward of a truncated response. 0 zeroes it, 1 is no penalty.",
    )

    @model_validator(mode="after")
    def _overlong_penalty_is_fully_specified(self) -> Self:
        """The three overlong fields are only meaningful together.

        NeMo-RL reads them independently and would silently skip the penalty if one were
        missing, which looks identical to shaping being off.
        """
        overlong = (
            self.overlong_buffer_length,
            self.overlong_buffer_penalty,
            self.max_response_length,
        )
        if any(v is not None for v in overlong) and not all(v is not None for v in overlong):
            raise ValueError(
                "reward_shaping needs overlong_buffer_length, overlong_buffer_penalty and "
                "max_response_length together, or none of them."
            )
        if all(v is None for v in overlong) and self.stop_properly_penalty_coef is None:
            raise ValueError(
                "reward_shaping is enabled but specifies no penalty. Set the overlong_* trio, "
                "stop_properly_penalty_coef, or omit reward_shaping entirely."
            )
        # apply_reward_shaping takes the stop_properly branch first and returns from it, so the
        # overlong parameters are read only when that coefficient is unset. NeMo-RL logs which
        # ones it ignored and carries on, which is a quiet way to run without the shaping the
        # job asked for.
        if self.stop_properly_penalty_coef is not None and any(v is not None for v in overlong):
            raise ValueError(
                "reward_shaping accepts either stop_properly_penalty_coef or the overlong_* "
                "parameters, not both: NeMo-RL applies the stop-properly penalty and ignores "
                "the overlong ones."
            )
        return self


class RewardScalingParams(RlSchema):
    """Linear reward rescaling (NeMo-RL ``grpo.reward_scaling``).

    Each reward is clamped to ``[source_min, source_max]`` and mapped onto
    ``[target_min, target_max]``. The DAPO recipes use it to move a binary verifier's
    ``[0, 1]`` onto ``[-1, 1]``, so a wrong answer carries negative reward rather than
    merely less positive reward.
    """

    source_min: float = Field(default=0.0, description="Low end of the incoming reward range.")
    source_max: float = Field(default=1.0, description="High end of the incoming reward range.")
    target_min: float = Field(default=0.0, description="Low end of the rescaled range.")
    target_max: float = Field(default=1.0, description="High end of the rescaled range.")

    @model_validator(mode="after")
    def _ranges_are_non_degenerate(self) -> Self:
        # A zero-width source collapses every reward onto one value, which zeroes the
        # advantage and stalls the run without raising anywhere downstream.
        if self.source_min >= self.source_max:
            raise ValueError("reward_scaling.source_min must be below source_max.")
        if self.target_min >= self.target_max:
            raise ValueError("reward_scaling.target_min must be below target_max.")
        return self


class LoRAParams(RlSchema):
    """LoRA hyperparameters for GRPO (DTensor ``lora_cfg``). No merge-at-export."""

    rank: int = Field(default=16, gt=0, description="LoRA rank (r); maps to NeMo-RL lora_cfg.dim.")
    alpha: int = Field(default=32, gt=0, description="LoRA scaling factor (effective lr multiplier = alpha/rank).")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="LoRA dropout probability.")
    target_modules: list[str] | None = Field(
        default=None,
        description="Module names to adapt. Empty/None with match_all_linear applies to all linear layers.",
    )
    exclude_modules: list[str] | None = Field(
        default=None,
        description="Module name patterns to exclude from LoRA.",
    )
    use_triton: bool | None = Field(
        default=None,
        description="Use Automodel's Triton LoRA kernels: faster, but incompatible with "
        "tensor_parallel_size > 1 (they take raw tensors, and TP makes the adapter weights "
        "DTensors). Unset lets the compiler pick -- true at TP 1, false above. Setting it true "
        "with TP > 1 is rejected, not silently downgraded.",
    )


class _TrainingBase(RlSchema):
    """Common training configuration shared by all RL methods."""

    model_config = ConfigDict(protected_namespaces=())

    # --- Optimizer ---
    optimizer_type: OptimizerType | None = Field(
        default=None,
        description="Optimizer + LR-scheduler combination (AdamW/Adam × cosine-annealing/flat-LR). "
        "Defaults to AdamW with cosine annealing.",
    )
    learning_rate: float = Field(default=1e-4, gt=0.0, description="Peak learning rate.")
    min_learning_rate: float | None = Field(default=None, ge=0.0, description="Minimum LR for cosine decay.")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay coefficient.")
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0, description="Adam beta1.")
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0, description="Adam beta2.")
    adam_eps: float = Field(default=1e-5, gt=0.0, description="Adam epsilon (numerical stability term).")
    warmup_steps: int = Field(default=0, ge=0, description="Linear warmup steps.")

    # --- Schedule ---
    epochs: int = Field(default=1, gt=0, description="Number of passes through the dataset.")
    max_steps: int | None = Field(default=None, gt=0, description="Max training steps (overrides epochs if set).")
    val_check_interval: float | None = Field(
        default=None,
        description="Validation interval. Float <= 1.0 is fraction of epoch; > 1.0 is step count.",
    )
    val_at_end: bool = Field(
        default=True,
        description="Run a final validation pass after the last training step. Keep enabled so the "
        "final checkpoint carries validation metrics and best-checkpoint selection works; "
        "set False only to skip the extra eval. GRPO ignores this when the dataset ships no "
        "validation.jsonl.",
    )
    progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)

    # --- Checkpointing ---
    keep_top_k: int = Field(
        default=1, gt=0, description="Number of best checkpoints to retain (ranked by validation loss)."
    )

    # --- Batch ---
    batch_size: int = Field(default=32, gt=0, description="Global batch size across all GPUs.")
    micro_batch_size: int = Field(default=1, gt=0, description="Per-GPU micro batch size.")
    activation_checkpointing: bool = Field(
        default=False,
        description="Recompute activations during the backward pass to reduce memory at the cost of compute. "
        "Enable to fit larger models or longer sequences.",
    )

    # --- Model ---
    max_seq_length: int = Field(default=2048, gt=0, description="Maximum token sequence length for training.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility.")

    # --- Infrastructure ---
    parallelism: ParallelismParams = Field(default_factory=ParallelismParams)
    execution_profile: str | None = Field(
        default=None,
        min_length=1,
        description="Execution profile for the GPU training step (operator-configured). "
        "Falls back to the service default when omitted.",
    )


class DPOTraining(_TrainingBase):
    """Direct Preference Optimization (full-weight only — PEFT unsupported)."""

    # No default: ``TrainingMethod`` is a discriminated union, so the tag has to be
    # present in the submitted JSON. Defaulting it here would make the generated
    # OpenAPI schema advertise the field as optional while the server rejects it.
    type: Literal["dpo"]
    ref_policy_kl_penalty: float = Field(
        default=0.05, ge=0.0, description="KL penalty coefficient (beta in the DPO paper)."
    )
    preference_average_log_probs: bool = Field(
        default=False, description="Average log probabilities for preference loss calculation."
    )
    sft_average_log_probs: bool = Field(
        default=False, description="Average log probabilities for SFT regularization loss."
    )
    preference_loss_weight: float = Field(default=1.0, ge=0.0, description="Weight for the preference (DPO) loss term.")
    sft_loss_weight: float = Field(
        default=0.0, ge=0.0, description="Weight for SFT regularization loss (0 = disabled)."
    )
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Maximum gradient norm for clipping.")


class GRPOTraining(_TrainingBase):
    """Group Relative Policy Optimization with NeMo Gym environments.

    ``finetuning_type`` is ``all_weights`` (default) or ``lora`` only.
    ``lora_merged`` is not supported for GRPO on the platform DTensor path.
    """

    # No default: ``TrainingMethod`` is a discriminated union, so the tag has to be
    # present in the submitted JSON (same as DPOTraining.type).
    type: Literal["grpo"]
    finetuning_type: Literal["all_weights", "lora"] = Field(
        default="all_weights",
        description="Full-weight GRPO or LoRA adapter training. lora_merged is not supported.",
    )
    lora: LoRAParams | None = Field(
        default=None,
        description="LoRA hyperparameters. Defaults applied when finetuning_type is lora.",
    )
    policy_backend: PolicyBackend | None = Field(
        default=None,
        description="NeMo-RL policy worker that trains the model. `lora` requires `automodel` "
        "(NeMo Automodel + Transformer Engine, Hopper or newer; the only backend with expert "
        "parallelism and automodel_kwargs). `all_weights` defaults to `dtensor` (stock HuggingFace "
        "on PyTorch FSDP2, also the pre-Hopper option) and may be set to `automodel` for a "
        "consolidated HuggingFace export. Omit it to get the default for the finetuning type.",
    )
    v4_compatible: bool = Field(
        default=True,
        description="Keep the base checkpoint's transformers-v4 config.json on the trained "
        "export. The platform's vLLM cannot read the v5 config Automodel otherwise writes. "
        "Set false to export that v5 config instead.",
    )
    val_at_start: bool = Field(
        default=False,
        description="Run a validation pass before the first training step. Enable it to measure "
        "uplift: the baseline and the trained result then come from one job, on the same data with "
        "the same generation settings, instead of a separate baseline run that has to be kept in "
        "sync. Off by default because a GRPO baseline costs a full rollout pass. Ignored when the "
        "dataset ships no validation.jsonl. DPO always validates at step 0 and has no such knob, "
        "since its validation needs no generation.",
    )
    num_generations_per_prompt: int = Field(
        default=8, gt=0, description="Group size: rollouts sampled per prompt, used for relative advantages."
    )
    num_prompts_per_step: int | None = Field(
        default=None,
        gt=0,
        description="Prompts sampled per training step. Derived from batch_size / "
        "num_generations_per_prompt when omitted; the product of the two must be a "
        "multiple of batch_size.",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Sampling temperature for rollout generation. Must be greater than 0: "
        "GRPO's advantage is the spread of rewards inside a prompt group, so greedy sampling "
        "makes every rollout in a group identical, the spread zero, and the whole run a no-op. "
        "Applies to validation rollouts too, which are generated with the same settings.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Cap on tokens generated per rollout turn. Defaults to max_seq_length, "
        "which lets a rollout run until the context is exhausted; vLLM clamps it to whatever "
        "the prompt leaves. Cannot exceed max_seq_length. NOTE: NeMo Gym's verifiers_agent "
        "does not yet honour this -- it reads max_tokens from its own environment config and "
        "drops the per-row value, so the effective cap is "
        "min(agent max_tokens, max_seq_length - prompt_len). Until that is fixed, bound "
        "response length through max_seq_length or the environment's own max_tokens.",
    )
    top_k: int | None = Field(
        default=None,
        gt=0,
        description="Restrict rollout sampling to the k most likely tokens at each step. Omit to "
        "sample from the full distribution, which is what standard GRPO does. Narrowing this has "
        "the same risk as lowering temperature: the rollouts in a prompt group become more alike, "
        "and GRPO learns from how much they differ.",
    )
    normalize_rewards: bool = Field(
        default=True, description="Divide each group's advantages by their standard deviation."
    )
    use_leave_one_out_baseline: bool = Field(
        default=True,
        description="Compare each rollout against the mean of the *other* rollouts in its group "
        "rather than against the group mean including itself. Excluding a rollout from its own "
        "baseline removes the bias that comparing it to itself introduces.",
    )
    advantage_clip_low: float | None = Field(
        default=None,
        description="Lower bound applied to advantages after normalization. Omit for no bound. "
        "Bounding both ends limits how much one unusual rollout can move the policy.",
    )
    advantage_clip_high: float | None = Field(
        default=None,
        description="Upper bound applied to advantages after normalization. Omit for no bound.",
    )
    overlong_filtering: bool = Field(
        default=False,
        description="Zero the loss contribution of rollouts truncated by the generation limit. "
        "Enable when a low max_new_tokens truncates many rollouts, so the policy is not penalised "
        "for responses it was cut off from finishing.",
    )
    max_rollout_turns: int = Field(
        default=1, gt=0, description="Maximum agent turns per rollout. Single-turn environments use 1."
    )
    ref_policy_kl_penalty: float = Field(
        default=0.0, ge=0.0, description="KL penalty coefficient against the reference policy."
    )
    ratio_clip_min: float = Field(default=0.2, ge=0.0, description="Lower PPO-style importance ratio clip bound.")
    ratio_clip_max: float = Field(default=0.28, ge=0.0, description="Upper PPO-style importance ratio clip bound.")
    ratio_clip_c: float | None = Field(
        default=None,
        gt=1.0,
        description="Dual-clip bound. Puts a floor under the loss for tokens whose advantage is "
        "negative, so a single badly-scored rollout cannot dominate the update. Must be greater "
        "than 1; 3 is the usual choice. Omit to leave dual clipping off.",
    )
    use_on_policy_kl_approximation: bool = Field(
        default=True,
        description="Estimate the KL term against the policy that generated the rollouts rather "
        "than the one currently training. The two differ once a rollout batch is reused across "
        "several optimizer steps.",
    )
    use_importance_sampling_correction: bool = Field(
        default=True,
        description="Reweight each token by how much more or less likely the training policy is "
        "to produce it than the rollout policy was. Corrects for the drift that builds up when a "
        "rollout batch is reused, at the cost of higher variance.",
    )
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Maximum gradient norm for clipping.")

    # --- Truncated importance sampling (DAPO) ---
    truncated_importance_sampling_type: Literal["tis", "icepop", "seq-mask-tis"] | None = Field(
        default=None,
        description="Bound the rollout-vs-training importance weights, so a policy that has drifted "
        "from the one that generated the batch cannot dominate the update. Watch "
        "`token_mult_prob_error`: a value climbing past ~1.05 is the drift this corrects. "
        "`tis` clamps weights into the range; `icepop` zeroes out-of-range tokens "
        "(reference bounds 0.5-5); `seq-mask-tis` zeroes whole sequences by their geometric-mean "
        "ratio (reference bounds 0.999-1.002). None leaves it off, which is NeMo-RL's default.",
    )
    truncated_importance_sampling_ratio: float | None = Field(
        default=None,
        gt=0.0,
        description="Upper bound on the importance weight. Required whenever "
        "`truncated_importance_sampling_type` is set.",
    )
    truncated_importance_sampling_ratio_min: float | None = Field(
        default=None,
        ge=0.0,
        description="Lower bound on the importance weight. Required for `icepop` and `seq-mask-tis`; "
        "optional for `tis`, which floors at 0 when unset.",
    )

    # --- Dynamic sampling (DAPO) ---
    use_dynamic_sampling: bool = Field(
        default=False,
        description="Discard prompt groups whose rewards have zero standard deviation, and keep "
        "generating until the step has a full batch of groups that do not. Those groups contribute "
        "exactly no gradient, so on a dataset the model finds too hard (or too easy) most of a step "
        "is otherwise wasted -- watch `baseline_reward/pct_mixed` for how much. Costs roughly "
        "`1 / pct_mixed` times the generation per step.",
    )
    dynamic_sampling_max_gen_batches: int = Field(
        default=10,
        gt=0,
        description="How many generation batches one step may consume trying to fill itself before "
        "the run fails. Only read when `use_dynamic_sampling` is true.",
    )
    batch_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description="Over-generate each step by this factor so dynamic sampling has candidates to "
        "filter. Set it near `1 / pct_mixed`. Rejected above 1.0 unless `use_dynamic_sampling` is "
        "true, which is what NeMo-RL asserts at startup.",
    )

    # --- Reward shaping (DAPO) ---
    reward_shaping: RewardShapingParams | None = Field(
        default=None,
        description="Penalise over-long and improperly-terminated responses before the loss sees "
        "the reward. None leaves shaping off.",
    )
    reward_scaling: RewardScalingParams | None = Field(
        default=None,
        description="Linearly rescale each reward before advantages are computed. None leaves "
        "scaling off. The DAPO recipes set target_min to -1.0, which turns a binary verifier's "
        "0 into a negative reward instead of a merely smaller positive one.",
    )

    # --- Micro-batch construction ---
    batching_strategy: BatchingStrategy = Field(
        default=BatchingStrategy.DYNAMIC,
        description="How rollouts are grouped into training micro-batches. `dynamic` fills each "
        "micro-batch to a token budget so short rollouts share a batch instead of each paying for a "
        "full-length pad. `sequence_packing` concatenates rollouts into packed sequences; it is "
        "rejected for context_parallel_size > 1. `static` is one rollout per slot.",
    )
    train_mb_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Token budget per training micro-batch, read by `dynamic` and `sequence_packing`. "
        "Defaults to max_seq_length * micro_batch_size, the peak `static` already provisions for. "
        "Lower it if you OOM.",
    )
    sequence_length_round: int = Field(
        default=64,
        gt=0,
        description="Round bucketed micro-batch sequence lengths up to a multiple of this. Only read by `dynamic`.",
    )

    # --- Per-architecture backend settings ---
    automodel_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Passed to policy.dtensor_cfg.automodel_kwargs; requires policy_backend='automodel'. "
        "{'force_hf': true} loads the stock HuggingFace modules; a {'backend': {...}} block picks "
        "the Transformer-Engine / DeepEP MoE implementation. Unset means Automodel auto-detects.",
    )
    router_aux_loss_coef: float | None = Field(
        default=None,
        ge=0.0,
        description="MoE router auxiliary-loss coefficient, applied as a top-level HuggingFace config "
        "override. Use 0.0 to drop the load-balancing term during RL. Unset keeps the model's own "
        "value. Models that nest their config (Qwen3.5 reads it under `text_config`) need "
        "`hf_config_overrides` instead — a top-level key they do not read is absorbed silently.",
    )
    hf_config_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Passed to NeMo-RL's `policy.hf_config_overrides` verbatim, which forwards it to "
        "the training model as HuggingFace config kwargs and to vLLM as `hf_overrides`. Nested keys "
        "are preserved, so this reaches models that namespace their config, e.g. "
        '`{"text_config": {"router_aux_loss_coef": 0.0}}` for Qwen3.5.',
    )
    vllm_tensor_parallel_size: int | None = Field(
        default=None,
        gt=0,
        description="Tensor parallel size for the vLLM rollout engine, independent of the policy's. "
        "Defaults to min(parallelism.tensor_parallel_size, parallelism.num_gpus_per_node).",
    )
    vllm_gpu_memory_utilization: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of each GPU vLLM reserves for weights plus KV cache. Raise toward 0.7 "
        "for large models; the rest is left for the colocated policy.",
    )

    @model_validator(mode="after")
    def _lora_defaults(self) -> Self:
        if self.finetuning_type == "lora" and self.lora is None:
            self.lora = LoRAParams()
        if self.finetuning_type == "all_weights" and self.lora is not None:
            raise ValueError("lora must be omitted when finetuning_type is all_weights")
        return self

    @model_validator(mode="after")
    def _triton_lora_kernels_require_no_tensor_parallelism(self) -> Self:
        """Triton LoRA kernels cannot take the DTensors that TP produces.

        ``TritonLinearLoRA.forward`` passes the adapter weights straight to a Triton kernel,
        and ``lora_kernel.py`` never calls ``.to_local()``. NeMo-RL turns the pairing into a
        bare assert in ``automodel/setup.py``, which fires inside a Ray worker only after the
        model loads. Unset means "compiler picks", so only an explicit ``true`` is rejected.
        """
        tp = self.parallelism.tensor_parallel_size
        if self.lora is not None and self.lora.use_triton is True and tp > 1:
            raise ValueError(
                f"lora.use_triton=true is incompatible with parallelism.tensor_parallel_size={tp}: "
                "Automodel's Triton LoRA kernels do not accept the DTensor-sharded adapter weights "
                "tensor parallelism produces. Leave lora.use_triton unset to let the compiler pick, "
                "set it false, or drop to tensor_parallel_size=1."
            )
        return self

    @model_validator(mode="after")
    def _policy_backend_defaults_to_the_one_that_supports_the_request(self) -> Self:
        """Follow ``finetuning_type`` when no backend was named; an explicit value is left alone."""
        if self.policy_backend is None:
            self.policy_backend = PolicyBackend.AUTOMODEL if self.finetuning_type == "lora" else PolicyBackend.DTENSOR
        return self

    @model_validator(mode="after")
    def _policy_backend_supports_requested_features(self) -> Self:
        """Reject pairings the backend does not support, before the job reaches a GPU.

        ``dtensor`` asserts ``lora_cfg.enabled is False`` and ignores expert parallelism
        and ``automodel_kwargs``. ``automodel`` trains LoRA and full weights.
        """
        if self.policy_backend is not PolicyBackend.DTENSOR:
            return self
        conflicts = []
        if self.finetuning_type == "lora":
            conflicts.append("finetuning_type='lora'")
        if self.parallelism.expert_parallel_size > 1:
            conflicts.append(f"parallelism.expert_parallel_size={self.parallelism.expert_parallel_size}")
        if self.automodel_kwargs:
            conflicts.append("automodel_kwargs")
        if conflicts:
            raise ValueError(f"{', '.join(conflicts)} require policy_backend='automodel'.")
        return self

    @model_validator(mode="after")
    def _hf_config_overrides_do_not_collide(self) -> Self:
        # router_aux_loss_coef writes the same top-level key, and HuggingFace absorbs an
        # unknown kwarg without complaining -- so a silent loser here means the MoE aux
        # loss stays on and only shows up as degraded accuracy many steps in.
        if self.router_aux_loss_coef is not None and "router_aux_loss_coef" in (self.hf_config_overrides or {}):
            raise ValueError(
                "router_aux_loss_coef is set both directly and inside hf_config_overrides. "
                "Keep one: use hf_config_overrides when the model nests it (e.g. Qwen3.5 reads "
                "it under text_config)."
            )
        return self

    @model_validator(mode="after")
    def _truncated_importance_sampling_is_usable(self) -> Self:
        """Mirror the asserts ClippedPGLossFn makes at startup.

        Each of these otherwise fails inside the driver after the model is loaded and the
        first rollout batch has been generated, which is an expensive way to learn about a
        typo in the job spec.
        """
        tis_type = self.truncated_importance_sampling_type
        ratio, ratio_min = (
            self.truncated_importance_sampling_ratio,
            self.truncated_importance_sampling_ratio_min,
        )
        if tis_type is None:
            if ratio is not None or ratio_min is not None:
                raise ValueError(
                    "truncated_importance_sampling_ratio/_min are ignored unless "
                    "truncated_importance_sampling_type is set."
                )
            return self
        if not self.use_importance_sampling_correction:
            raise ValueError("truncated importance sampling requires use_importance_sampling_correction=true.")
        if ratio is None:
            raise ValueError(
                f"truncated_importance_sampling_ratio is required when "
                f"truncated_importance_sampling_type is {tis_type!r}."
            )
        if tis_type in ("icepop", "seq-mask-tis") and ratio_min is None:
            raise ValueError(
                f"truncated_importance_sampling_ratio_min is required when "
                f"truncated_importance_sampling_type is {tis_type!r}."
            )
        if ratio_min is not None and ratio_min > ratio:
            raise ValueError(
                "truncated_importance_sampling_ratio_min must not exceed truncated_importance_sampling_ratio."
            )
        return self

    @model_validator(mode="after")
    def _batch_multiplier_requires_dynamic_sampling(self) -> Self:
        # NeMo-RL asserts exactly this at startup; without dynamic sampling the extra
        # prompts are generated and then trained on, silently changing the batch size.
        if self.batch_multiplier != 1.0 and not self.use_dynamic_sampling:
            raise ValueError("batch_multiplier may only be set when use_dynamic_sampling is true.")
        return self

    @model_validator(mode="after")
    def _advantage_clip_range_is_usable(self) -> Self:
        # Either bound alone is fine; together they have to leave room between them.
        # Reversed or equal bounds clamp every advantage to one value, which zeroes
        # the gradient and makes the run do nothing for the reason least visible in
        # the logs -- reward stays flat and no error is raised anywhere.
        low, high = self.advantage_clip_low, self.advantage_clip_high
        if low is not None and high is not None and low >= high:
            raise ValueError(
                f"advantage_clip_low ({low}) must be less than advantage_clip_high ({high}); "
                f"an empty or inverted range clips every advantage to the same value and the "
                f"policy stops receiving a gradient"
            )
        return self

    @model_validator(mode="after")
    def _generation_length_fits_context(self) -> Self:
        # max_seq_length is the whole context, prompt included. Asking for more generated
        # tokens than that is always unsatisfiable, and vLLM would silently clamp it
        # rather than say so.
        if self.max_new_tokens is not None and self.max_new_tokens > self.max_seq_length:
            raise ValueError(
                f"max_new_tokens ({self.max_new_tokens}) cannot exceed max_seq_length "
                f"({self.max_seq_length}); max_seq_length is the total prompt + generation budget"
            )
        return self


TrainingMethod = Annotated[Union[DPOTraining, GRPOTraining], Discriminator("type")]


class _OutputBase(RlSchema):
    name: str = Field(max_length=255, examples=["my-dpo-llama"])


class OutputRequest(_OutputBase):
    """Output artifact configuration provided by the user."""


class OutputResponse(_OutputBase):
    """Resolved output artifact details."""

    type: OutputNameType = Field(default=OutputNameType.MODEL)
    fileset: str = Field(max_length=255)


class RlJobOutput(RlSchema):
    """Canonical NeMo-RL job spec (output of the plugin transform)."""

    model_config = ConfigDict(protected_namespaces=())

    name: str | None = Field(default=None)
    model: str = Field(description="Model entity reference ('name' or 'workspace/name').")
    dataset: str = Field(description="Dataset fileset reference ('name' or 'workspace/name').")
    environment: str | None = Field(default=None)
    training: TrainingMethod = Field(description="Training method and hyperparameters.")
    integrations: IntegrationsSpec | None = Field(default=None)
    output: OutputResponse = Field(description="Output artifact created by this job.")

    @property
    def training_type(self) -> TrainingType:
        return TrainingType(self.training.type)

    def validate_for_training(self) -> None:
        """Validate parallelism/batch consistency before compiling."""
        training = self.training
        p = training.parallelism
        total_gpus = p.num_gpus_per_node * p.num_nodes
        # NeMo-RL derives dp_size = world_size // (tp * cp * ep), so ep belongs in this product.
        model_parallel_size = (
            p.tensor_parallel_size * p.pipeline_parallel_size * p.context_parallel_size * p.expert_parallel_size
        )
        if total_gpus % model_parallel_size != 0:
            raise ValueError(
                f"Total GPUs ({total_gpus}) must be divisible by tensor_parallel_size "
                f"({p.tensor_parallel_size}) * pipeline_parallel_size ({p.pipeline_parallel_size}) * "
                f"context_parallel_size ({p.context_parallel_size}) * expert_parallel_size "
                f"({p.expert_parallel_size}) = {model_parallel_size}"
            )
        derived_dp = total_gpus // model_parallel_size
        gb, mb = training.batch_size, training.micro_batch_size
        divisor = mb * derived_dp
        if gb % divisor != 0:
            raise ValueError(
                f"batch_size ({gb}) must be divisible by micro_batch_size ({mb}) * "
                f"data_parallel_size ({derived_dp}) = {divisor}."
            )
        if isinstance(training, GRPOTraining):
            if not self.environment:
                raise ValueError("GRPO jobs require an environment fileset reference.")
            # The rollout engine meshes over the same GPUs under its own rule: vllm_generation.py
            # asserts world_size % tp == 0, and one engine does not span nodes.
            vllm_tp = training.vllm_tensor_parallel_size
            if vllm_tp is not None:
                if vllm_tp > p.num_gpus_per_node:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({vllm_tp}) cannot exceed num_gpus_per_node "
                        f"({p.num_gpus_per_node}); a single vLLM engine does not shard across nodes."
                    )
                if total_gpus % vllm_tp != 0:
                    raise ValueError(
                        f"Total GPUs ({total_gpus}) must be divisible by vllm_tensor_parallel_size ({vllm_tp})."
                    )
            # The rollout batch is num_prompts_per_step * num_generations_per_prompt, and
            # NeMo-RL shards it by the train batch. When num_prompts_per_step is derived,
            # the floor division can leave the two out of step (e.g. 32 // 5 = 6 -> 30 vs
            # 32), which only surfaces as an assert at the first optimizer step.
            gen = training.num_generations_per_prompt
            prompts = training.num_prompts_per_step or max(gb // max(gen, 1), 1)
            rollout = prompts * gen
            if rollout % gb != 0:
                raise ValueError(
                    f"num_prompts_per_step ({prompts}) * num_generations_per_prompt ({gen}) "
                    f"= {rollout} must be a multiple of batch_size ({gb}). Choose a "
                    f"num_generations_per_prompt that divides batch_size, or set "
                    f"num_prompts_per_step explicitly."
                )

    @model_validator(mode="after")
    def _output_type_matches_training(self) -> Self:
        if self.training_type == TrainingType.DPO and self.output.type != OutputNameType.MODEL:
            raise ValueError("DPO produces a full-weight model; output.type must be 'model'.")
        if self.training_type == TrainingType.GRPO:
            assert isinstance(self.training, GRPOTraining)
            if self.training.finetuning_type == "lora":
                if self.output.type != OutputNameType.ADAPTER:
                    raise ValueError("GRPO LoRA produces an adapter; output.type must be 'adapter'.")
            elif self.output.type != OutputNameType.MODEL:
                raise ValueError("GRPO all_weights produces a full-weight model; output.type must be 'model'.")
        return self
