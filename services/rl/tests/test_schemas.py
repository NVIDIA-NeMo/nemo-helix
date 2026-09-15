# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public NeMo-RL schema tests: field defaults and the cross-field validators."""

from __future__ import annotations

from typing import Any

import pytest
from nmp.customization_common.schemas.values import OutputNameType
from nmp.rl.app.jobs.training.schemas import BatchingStrategy, OptimizerType, PolicyBackend
from nmp.rl.schemas import DPOTraining, GRPOTraining, OutputResponse, ParallelismParams, RlJobOutput


def _make_output(name: str = "out", out_type: OutputNameType = OutputNameType.MODEL) -> OutputResponse:
    return OutputResponse(name=name, type=out_type, fileset=f"{name}-fs")


def _make_job_output(training: DPOTraining, out_type: OutputNameType = OutputNameType.MODEL) -> RlJobOutput:
    return RlJobOutput(
        model="default/base",
        dataset="default/prefs",
        training=training,
        output=_make_output(out_type=out_type),
    )


def test_dpo_training_defaults_preserve_prior_behavior() -> None:
    """The newly exposed knobs default to the values the compiler used to hardcode."""
    t = DPOTraining(type="dpo")
    assert t.type == "dpo"
    # Newly exposed configurability.
    assert t.optimizer_type is None  # → AdamW + cosine annealing
    assert t.adam_eps == 1e-5
    assert t.activation_checkpointing is False
    assert t.keep_top_k == 1
    # val_at_end defaults True so the final checkpoint carries validation metrics
    # and best-checkpoint selection works (otherwise NeMo-RL falls back to latest).
    assert t.val_at_end is True
    # Existing DPO hyperparameters.
    assert t.ref_policy_kl_penalty == 0.05
    assert t.sft_loss_weight == 0.0


def test_dpo_training_accepts_overrides() -> None:
    t = DPOTraining(
        type="dpo",
        optimizer_type=OptimizerType.ADAM_WITH_FLAT_LR,
        adam_eps=1e-8,
        activation_checkpointing=True,
        keep_top_k=3,
        val_at_end=True,
    )
    assert t.optimizer_type is OptimizerType.ADAM_WITH_FLAT_LR
    assert t.adam_eps == 1e-8
    assert t.activation_checkpointing is True
    assert t.keep_top_k == 3
    assert t.val_at_end is True


@pytest.mark.parametrize("bad", [0.0, -1e-5])
def test_adam_eps_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError):
        DPOTraining(type="dpo", adam_eps=bad)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", -1e-4),
        ("min_learning_rate", -1e-4),
        ("weight_decay", -0.01),
        ("adam_beta1", -0.1),
        ("adam_beta1", 1.0),
        ("adam_beta2", -0.1),
        ("adam_beta2", 1.0),
    ],
)
def test_optimizer_bounds(field: str, bad: float) -> None:
    kwargs: dict[str, Any] = {field: bad}
    with pytest.raises(ValueError):
        DPOTraining(type="dpo", **kwargs)


def test_keep_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DPOTraining(type="dpo", keep_top_k=0)


def test_validate_for_training_accepts_consistent_single_gpu() -> None:
    # 1 GPU, no model parallelism, gb divisible by micro*dp → no error.
    job = _make_job_output(DPOTraining(type="dpo", batch_size=32, micro_batch_size=1))
    job.validate_for_training()


def test_validate_for_training_rejects_indivisible_model_parallel() -> None:
    # total_gpus=1 but tensor_parallel_size=2 → 1 % 2 != 0.
    job = _make_job_output(
        DPOTraining(type="dpo", parallelism=ParallelismParams(num_gpus_per_node=1, tensor_parallel_size=2)),
    )
    with pytest.raises(ValueError, match="must be divisible by tensor_parallel_size"):
        job.validate_for_training()


def test_validate_for_training_rejects_indivisible_batch() -> None:
    # total_gpus=2, mp=1 → data_parallel=2; batch_size=3 not divisible by micro(1)*dp(2).
    job = _make_job_output(
        DPOTraining(
            type="dpo",
            parallelism=ParallelismParams(num_gpus_per_node=2),
            batch_size=3,
            micro_batch_size=1,
        ),
    )
    with pytest.raises(ValueError, match="batch_size"):
        job.validate_for_training()


def test_dpo_output_must_be_full_weight_model() -> None:
    # DPO is full-weight; an adapter output is rejected at construction time.
    with pytest.raises(ValueError, match="full-weight model"):
        _make_job_output(DPOTraining(type="dpo"), out_type=OutputNameType.ADAPTER)


def test_grpo_requires_environment() -> None:
    job = RlJobOutput(
        model="default/base",
        dataset="default/gym-data",
        training=GRPOTraining(type="grpo"),
        output=_make_output(),
    )
    with pytest.raises(ValueError, match="environment fileset"):
        job.validate_for_training()


def test_grpo_training_discriminator() -> None:
    t = GRPOTraining(type="grpo")
    assert t.type == "grpo"
    assert t.num_generations_per_prompt == 8
    assert t.finetuning_type == "all_weights"
    assert t.lora is None


def test_grpo_lora_defaults_params() -> None:
    t = GRPOTraining(type="grpo", finetuning_type="lora")
    assert t.lora is not None
    assert t.lora.rank == 16
    assert t.lora.alpha == 32


def test_grpo_advanced_knobs_default_to_current_behaviour() -> None:
    """Turning hardcoded values into settings must not move an unchanged job.

    The first three were fixed in the config compiler before they were fields; the
    rest are NeMo-RL's own defaults, which the compiler was already passing through.
    """
    t = GRPOTraining(type="grpo")

    assert t.normalize_rewards is True
    assert t.use_leave_one_out_baseline is True
    assert t.use_on_policy_kl_approximation is True
    assert t.use_importance_sampling_correction is True
    assert t.ratio_clip_c is None
    assert t.advantage_clip_low is None
    assert t.advantage_clip_high is None
    assert t.top_k is None


def test_grpo_dual_clip_must_exceed_one() -> None:
    """The loss asserts this at startup; rejecting it here fails the request instead."""
    with pytest.raises(ValueError, match="ratio_clip_c"):
        GRPOTraining(type="grpo", ratio_clip_c=1.0)


def test_grpo_rejects_an_inverted_advantage_clip_range() -> None:
    """Bounds that cross clip every advantage to one value and kill the gradient."""
    with pytest.raises(ValueError, match="advantage_clip_low"):
        GRPOTraining(type="grpo", advantage_clip_low=5.0, advantage_clip_high=-5.0)


def test_grpo_rejects_an_empty_advantage_clip_range() -> None:
    """Equal bounds are the same failure, and just as quiet."""
    with pytest.raises(ValueError, match="advantage_clip_low"):
        GRPOTraining(type="grpo", advantage_clip_low=1.0, advantage_clip_high=1.0)


def test_grpo_accepts_a_single_advantage_clip_bound() -> None:
    """Bounding one side is a real request; the pair check only applies to both."""
    assert GRPOTraining(type="grpo", advantage_clip_low=-5.0).advantage_clip_high is None
    assert GRPOTraining(type="grpo", advantage_clip_high=5.0).advantage_clip_low is None


def test_grpo_lora_rejects_params_with_all_weights() -> None:
    from nmp.rl.schemas import LoRAParams

    with pytest.raises(ValueError, match="lora must be omitted"):
        GRPOTraining(type="grpo", finetuning_type="all_weights", lora=LoRAParams(rank=8))


def test_router_aux_loss_coef_collides_with_hf_config_overrides() -> None:
    """Both write the same top-level key, and HuggingFace absorbs an unknown kwarg silently,
    so a losing setting shows up only as degraded accuracy many steps in."""
    with pytest.raises(ValueError, match="set both directly and inside hf_config_overrides"):
        GRPOTraining(
            type="grpo",
            router_aux_loss_coef=0.0,
            hf_config_overrides={"router_aux_loss_coef": 0.0},
        )


def test_router_aux_loss_coef_may_coexist_with_a_nested_override() -> None:
    """Only the top-level key collides. Nesting it under text_config is the Qwen3.5 case."""
    t = GRPOTraining(
        type="grpo",
        router_aux_loss_coef=0.0,
        hf_config_overrides={"text_config": {"router_aux_loss_coef": 0.0}},
    )
    assert t.router_aux_loss_coef == 0.0


def test_batching_defaults() -> None:
    t = GRPOTraining(type="grpo")
    assert t.batching_strategy is BatchingStrategy.DYNAMIC
    assert t.train_mb_tokens is None  # derived as max_seq_length * micro_batch_size
    assert t.sequence_length_round == 64


def test_use_triton_defaults_to_unset() -> None:
    """Unset is what lets the compiler resolve it from TP without overriding a caller."""
    from nmp.rl.schemas import LoRAParams

    assert LoRAParams().use_triton is None
    t = GRPOTraining(type="grpo", finetuning_type="lora")
    assert t.lora is not None and t.lora.use_triton is None


def test_triton_lora_rejected_with_tensor_parallelism() -> None:
    """The Triton kernels take raw tensors and TP makes them DTensors; NeMo-RL turns the
    pairing into a bare assert that fires only after the model loads."""
    from nmp.rl.schemas import LoRAParams

    with pytest.raises(ValueError, match="use_triton=true is incompatible"):
        GRPOTraining(
            type="grpo",
            finetuning_type="lora",
            lora=LoRAParams(rank=16, use_triton=True),
            parallelism=ParallelismParams(num_gpus_per_node=2, tensor_parallel_size=2),
        )


def test_triton_lora_accepted_without_tensor_parallelism() -> None:
    from nmp.rl.schemas import LoRAParams

    t = GRPOTraining(type="grpo", finetuning_type="lora", lora=LoRAParams(rank=16, use_triton=True))
    assert t.lora is not None and t.lora.use_triton is True


def test_triton_lora_explicitly_disabled_is_allowed_with_tensor_parallelism() -> None:
    """False is the value TP needs, so asking for it must not trip the same check."""
    from nmp.rl.schemas import LoRAParams

    t = GRPOTraining(
        type="grpo",
        finetuning_type="lora",
        lora=LoRAParams(rank=16, use_triton=False),
        parallelism=ParallelismParams(num_gpus_per_node=2, tensor_parallel_size=2),
    )
    assert t.lora is not None and t.lora.use_triton is False


def test_policy_backend_defaults_to_the_backend_that_supports_the_request() -> None:
    """An unset value follows finetuning_type; one default for both would reject half the configs."""
    assert GRPOTraining(type="grpo").policy_backend is PolicyBackend.DTENSOR
    assert GRPOTraining(type="grpo", finetuning_type="lora").policy_backend is PolicyBackend.AUTOMODEL


def test_an_explicit_backend_is_never_overridden_by_the_default() -> None:
    """The default only fills a gap; a stated value still stands or is rejected outright."""
    assert GRPOTraining(type="grpo", policy_backend=PolicyBackend.DTENSOR).policy_backend is PolicyBackend.DTENSOR


def test_policy_backend_has_no_megatron_member_yet() -> None:
    """The image can already run MegatronPolicyWorker, so the enum is the only thing stopping
    a request from reaching a backend whose megatron_cfg the compiler still emits inert."""
    assert {b.value for b in PolicyBackend} == {"dtensor", "automodel"}


def test_automodel_backend_accepts_full_weight() -> None:
    """Automodel writes a consolidated HuggingFace tree the publisher can copy."""
    t = GRPOTraining(type="grpo", finetuning_type="all_weights", policy_backend=PolicyBackend.AUTOMODEL)
    assert t.policy_backend is PolicyBackend.AUTOMODEL
    assert t.finetuning_type == "all_weights"


def test_dtensor_backend_rejects_lora() -> None:
    """V1 asserts ``lora_cfg.enabled is False`` in the Ray worker; fail before the GPU."""
    with pytest.raises(ValueError, match="require policy_backend='automodel'"):
        GRPOTraining(type="grpo", finetuning_type="lora", policy_backend=PolicyBackend.DTENSOR)


def test_dtensor_backend_rejects_expert_parallelism() -> None:
    with pytest.raises(ValueError, match="expert_parallel_size=8"):
        GRPOTraining(
            type="grpo",
            policy_backend=PolicyBackend.DTENSOR,
            parallelism=ParallelismParams(num_gpus_per_node=8, expert_parallel_size=8),
        )


def test_dtensor_backend_rejects_automodel_kwargs() -> None:
    with pytest.raises(ValueError, match="automodel_kwargs"):
        GRPOTraining(
            type="grpo",
            policy_backend=PolicyBackend.DTENSOR,
            automodel_kwargs={"force_hf": True},
        )


def test_dtensor_backend_reports_every_conflict_at_once() -> None:
    """One submission, one error listing all of it -- not three round-trips."""
    with pytest.raises(ValueError) as excinfo:
        GRPOTraining(
            type="grpo",
            finetuning_type="lora",
            policy_backend=PolicyBackend.DTENSOR,
            parallelism=ParallelismParams(num_gpus_per_node=8, expert_parallel_size=8),
            automodel_kwargs={"force_hf": True},
        )
    message = str(excinfo.value)
    assert "finetuning_type='lora'" in message
    assert "expert_parallel_size=8" in message
    assert "automodel_kwargs" in message


def test_dtensor_backend_accepts_plain_full_weight() -> None:
    """Full-weight with no v2-only feature is exactly what `dtensor` is for."""
    t = GRPOTraining(type="grpo", policy_backend=PolicyBackend.DTENSOR)
    assert t.policy_backend is PolicyBackend.DTENSOR


def test_grpo_lora_rejects_lora_merged() -> None:
    # `type` is required, so omitting it raises before finetuning_type is ever
    # looked at -- this would pass even if lora_merged were accepted. Match on the
    # field name so the assertion is about lora_merged and nothing else.
    with pytest.raises(ValueError, match="finetuning_type"):
        GRPOTraining.model_validate({"type": "grpo", "finetuning_type": "lora_merged"})


def test_grpo_lora_requires_adapter_output() -> None:
    with pytest.raises(ValueError, match="adapter"):
        RlJobOutput(
            model="default/base",
            dataset="default/gym-data",
            environment="default/env",
            training=GRPOTraining(type="grpo", finetuning_type="lora"),
            output=_make_output(out_type=OutputNameType.MODEL),
        )


def test_grpo_lora_accepts_adapter_output() -> None:
    job = RlJobOutput(
        model="default/base",
        dataset="default/gym-data",
        environment="default/env",
        training=GRPOTraining(type="grpo", finetuning_type="lora", lora=None),
        output=_make_output(out_type=OutputNameType.ADAPTER),
    )
    assert job.output.type is OutputNameType.ADAPTER
    job.validate_for_training()


def test_grpo_sampling_defaults_preserve_previous_behaviour() -> None:
    """These were hardcoded in the compiler; the defaults must reproduce them exactly."""
    t = GRPOTraining(type="grpo")
    assert t.temperature == 1.0
    # None means "use the whole context", which is what max_seq_length gave before.
    assert t.max_new_tokens is None


def test_grpo_rejects_greedy_temperature() -> None:
    """Temperature 0 makes every rollout in a group identical.

    GRPO's advantage is the spread of rewards within a prompt group. With no spread the
    advantage is zero for every sample and the run trains on nothing -- silently, since
    rewards still look plausible. Cheaper to reject at submit time than to discover after
    150 steps of flat loss.
    """
    with pytest.raises(ValueError):
        GRPOTraining(type="grpo", temperature=0.0)


def test_grpo_rejects_max_new_tokens_larger_than_context() -> None:
    with pytest.raises(ValueError, match="cannot exceed max_seq_length"):
        GRPOTraining(type="grpo", max_seq_length=2048, max_new_tokens=4096)


def test_grpo_accepts_max_new_tokens_up_to_the_context() -> None:
    t = GRPOTraining(type="grpo", max_seq_length=2048, max_new_tokens=2048)
    assert t.max_new_tokens == 2048


def _grpo_job(training: GRPOTraining, out_type: OutputNameType = OutputNameType.MODEL) -> RlJobOutput:
    return RlJobOutput(
        model="default/base",
        dataset="default/gym-data",
        environment="default/env",
        training=training,
        output=_make_output(out_type=out_type),
    )


def test_expert_parallel_size_joins_the_model_parallel_divisor() -> None:
    """EP draws from the same world as TP/CP: NeMo-RL derives dp = world // (tp * cp * ep)."""
    job = _grpo_job(
        GRPOTraining(
            type="grpo",
            # EP is automodel-only, and automodel is LoRA-only, so an EP run is a LoRA run.
            finetuning_type="lora",
            parallelism=ParallelismParams(num_gpus_per_node=8, expert_parallel_size=3),
        ),
        OutputNameType.ADAPTER,
    )
    with pytest.raises(ValueError, match="expert_parallel_size"):
        job.validate_for_training()


def test_expert_parallel_size_accepted_when_it_divides_the_world() -> None:
    job = _grpo_job(
        GRPOTraining(
            type="grpo",
            finetuning_type="lora",
            parallelism=ParallelismParams(num_gpus_per_node=8, expert_parallel_size=8),
            batch_size=8,
            num_generations_per_prompt=8,
        ),
        OutputNameType.ADAPTER,
    )
    job.validate_for_training()


def test_expert_parallel_size_defaults_to_one() -> None:
    assert ParallelismParams().expert_parallel_size == 1


def test_vllm_tensor_parallel_size_cannot_span_nodes() -> None:
    """One vLLM engine lives inside a node; TP above the node size is unsatisfiable."""
    job = _grpo_job(
        GRPOTraining(
            type="grpo",
            parallelism=ParallelismParams(num_nodes=2, num_gpus_per_node=4),
            vllm_tensor_parallel_size=8,
            batch_size=8,
            num_generations_per_prompt=8,
        ),
    )
    with pytest.raises(ValueError, match="cannot exceed num_gpus_per_node"):
        job.validate_for_training()


def test_vllm_tensor_parallel_size_must_divide_total_gpus() -> None:
    job = _grpo_job(
        GRPOTraining(
            type="grpo",
            parallelism=ParallelismParams(num_gpus_per_node=6),
            vllm_tensor_parallel_size=4,
            batch_size=6,
            num_generations_per_prompt=6,
        ),
    )
    with pytest.raises(ValueError, match="vllm_tensor_parallel_size"):
        job.validate_for_training()


def test_backend_settings_default_to_unset() -> None:
    """Unset means the compiler picks, so a plain job compiles without any of these."""
    t = GRPOTraining(type="grpo")
    assert t.vllm_tensor_parallel_size is None  # falls back to min(training tp, gpus per node)
    assert t.vllm_gpu_memory_utilization == 0.5
    assert t.automodel_kwargs is None
    assert t.router_aux_loss_coef is None


def test_router_aux_loss_coef_accepts_zero() -> None:
    """0.0 is the value an MoE RL run actually wants, so it must survive validation."""
    assert GRPOTraining(type="grpo", router_aux_loss_coef=0.0).router_aux_loss_coef == 0.0


def test_no_validation_generations_knob_is_exposed() -> None:
    """NeMo-RL's validate() runs one rollout per validation row and has no per-prompt fan-out.

    ``grpo.GRPOConfig`` is ``extra="allow"``, so a ``num_val_generations_per_prompt`` key would
    be accepted and read by nothing. mean@k comes from repeating rows in validation.jsonl.
    """
    assert "num_val_generations_per_prompt" not in GRPOTraining.model_fields
