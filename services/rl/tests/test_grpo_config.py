# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO config compilation (sandbox paths + egress)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from nmp.customization_common.service.context import NMPJobContext
from nmp.rl.app.jobs.training.schemas import (
    BatchingStrategy,
    GRPOConfig,
    LoRAConfig,
    ModelConfig,
    PolicyBackend,
    TrainingBackend,
    TrainingStepConfig,
)
from nmp.rl.entities.values import FinetuningType, TrainingType
from nmp.rl.tasks.training.backends.nemo_rl.grpo_config import compile_grpo_config
from nmp.rl.tasks.training.backends.nemo_rl.sandbox_config import (
    DEFAULT_ROLLOUT_CHUNK_SIZE,
    DEFAULT_ROLLOUT_MAX_IN_FLIGHT,
)


@pytest.fixture
def job_ctx(tmp_path: Path) -> NMPJobContext:
    return NMPJobContext(
        workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step="grpo-training",
        task="training",
        jobs_url=None,
        files_url=None,
        storage_path=tmp_path,
        config_path=tmp_path / "config.yaml",
    )


def _write_gym_dataset(root: Path, filename: str = "training.jsonl") -> None:
    row = {
        "task_idx": 0,
        "vf_env_id": "ascii-tree",
        "responses_create_params": {"input": [{"role": "user", "content": "hello"}]},
        "agent_ref": {"name": "verifiers_agent"},
        "answer": "42",
        "example_id": "ex-0",
        "info": {},
    }
    (root / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")


def _make_grpo_step(
    tmp_path: Path,
    dataset_pvc: Path,
    *,
    finetuning_type: FinetuningType = FinetuningType.ALL_WEIGHTS,
    lora: LoRAConfig | None = None,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    activation_checkpointing: bool = False,
    grpo: GRPOConfig | None = None,
    policy_backend: PolicyBackend = PolicyBackend.AUTOMODEL,
    v4_compatible: bool = True,
) -> TrainingStepConfig:
    env_root = tmp_path / "environment"
    env_root.mkdir(exist_ok=True)
    (env_root / "nemo-environment.yaml").write_text(
        "format: adapter-wheels-v1\nadapter:\n  agent: verifiers_agent\n"
        "config_paths:\n  - configs/verifiers_agent.yaml\n",
        encoding="utf-8",
    )
    return TrainingStepConfig(
        backend=TrainingBackend.NEMO_RL,
        model=ModelConfig(
            path=str(tmp_path / "model"),
            max_seq_length=512,
            v4_compatible=v4_compatible,
        ),
        dataset=TrainingStepConfig.DatasetConfig(path=str(dataset_pvc)),
        gym=TrainingStepConfig.GymConfig(
            environment_path=str(env_root),
            sandbox_environment_path="/job/environment",
            sandbox_dataset_path="/job/dataset",
            sandboxed=True,
            gym_runtime_image="nvcr.io/nvidia/nmp-rl-training:test",
        ),
        training=TrainingStepConfig.TrainingConfig(
            training_type=TrainingType.GRPO,
            finetuning_type=finetuning_type,
            grpo=grpo or GRPOConfig(num_generations_per_prompt=4),
            lora=lora,
        ),
        schedule=TrainingStepConfig.ScheduleConfig(epochs=1),
        batch=TrainingStepConfig.BatchConfig(global_batch_size=8, micro_batch_size=1),
        optimizer=TrainingStepConfig.OptimizerConfig(),
        parallelism=TrainingStepConfig.ParallelismConfig(
            num_gpus_per_node=1,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            activation_checkpointing=activation_checkpointing,
            policy_backend=policy_backend,
        ),
        output_model="out",
        workspace_path=str(tmp_path / "workspace"),
    )


def _prepared_step(
    tmp_path: Path,
    *,
    finetuning_type: FinetuningType = FinetuningType.ALL_WEIGHTS,
    lora: LoRAConfig | None = None,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    activation_checkpointing: bool = False,
    grpo: GRPOConfig | None = None,
    policy_backend: PolicyBackend = PolicyBackend.AUTOMODEL,
    v4_compatible: bool = True,
) -> tuple[TrainingStepConfig, Path]:
    dataset_pvc = tmp_path / "dataset"
    dataset_pvc.mkdir(exist_ok=True)
    _write_gym_dataset(dataset_pvc)
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return (
        _make_grpo_step(
            tmp_path,
            dataset_pvc,
            finetuning_type=finetuning_type,
            lora=lora,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            activation_checkpointing=activation_checkpointing,
            grpo=grpo,
            policy_backend=policy_backend,
            v4_compatible=v4_compatible,
        ),
        dataset_pvc,
    )


def test_compile_grpo_config_sandboxed_paths(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, dataset_pvc = _prepared_step(tmp_path)
    cfg = compile_grpo_config(step, job_ctx)

    nemo_gym = cfg["env"]["nemo_gym"]
    assert cfg["env"]["should_use_nemo_gym"] is True
    assert nemo_gym["sandboxed"] is True
    assert nemo_gym["environment_path"] == "/job/environment"
    assert nemo_gym["environment_offline"] is True
    # The master reads the dataset itself, so the dataloader path stays on job storage
    # even though the sandbox sees the same file at /job/dataset.
    assert cfg["data"]["train"]["data_path"] == str(dataset_pvc / "training.jsonl")
    assert nemo_gym["sandbox"]["network_policy"]["egress_allow"]
    # The Gym actor builds the sandbox's own env inside the sandbox pod, where paths differ
    # from the training pod's. Emitting them here would conflict, and nothing reads them.
    assert "bootstrap_env" not in nemo_gym
    assert "host_work_path" not in nemo_gym
    # config_paths must reach Gym in sandboxed mode too, or no servers start — and they must
    # be anchored to the SANDBOX mount. The manifest stores them relative to the package root;
    # Gym resolves relative paths against its CWD (the runtime image's WORKDIR), so a relative
    # entry sends it looking at /opt/nemo-rl/configs/... and it dies with FileNotFoundError.
    assert nemo_gym["config_paths"] == ["/job/environment/configs/verifiers_agent.yaml"]
    # job_id is a sandbox pod label; without it every job shares one default.
    assert nemo_gym["job_id"] == "job-123"
    # Full-weight omits lora_cfg entirely rather than emitting enabled=False.
    assert "lora_cfg" not in cfg["policy"]["dtensor_cfg"]

    sandbox = nemo_gym["sandbox"]
    assert sandbox["environment_pvc_claim"] == "nmp-job-storage"
    assert sandbox["workspace_pvc_claim"] == "nmp-job-storage"
    assert sandbox["dataset_pvc_claim"] == "nmp-job-storage"
    assert sandbox["environment_sub_path"] == "jobs/default/job-123/environment"
    assert sandbox["dataset_sub_path"] == "jobs/default/job-123/dataset"


def test_compile_grpo_config_colocated_anchors_config_paths(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mode A anchors config_paths to the job-storage package, not the sandbox mount.

    Same underlying hazard as the sandboxed case: the manifest stores paths relative to the
    package root, and Gym resolves them against its CWD. Colocated Gym has a different root
    from the sandbox, so the two modes must not share one answer.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, dataset_pvc = _prepared_step(tmp_path)
    assert step.gym is not None
    step.gym.sandboxed = False
    # Unlike the sandboxed path, mode A runs prepare_dataset, which needs at least two rows
    # to carve out a validation split.
    train_jsonl = dataset_pvc / "training.jsonl"
    train_jsonl.write_text(train_jsonl.read_text(encoding="utf-8") * 2, encoding="utf-8")

    nemo_gym = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]

    assert nemo_gym["config_paths"] == [f"{step.gym.environment_path}/configs/verifiers_agent.yaml"]
    assert Path(nemo_gym["config_paths"][0]).is_absolute()


def test_compile_grpo_config_sandboxed_requires_pvc_claim(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NMP_JOB_STORAGE_PVC_CLAIM", raising=False)
    step, _ = _prepared_step(tmp_path)
    with pytest.raises(ValueError, match="NMP_JOB_STORAGE_PVC_CLAIM"):
        compile_grpo_config(step, job_ctx)


def test_compile_grpo_config_disables_validation_without_val_split(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Gym dataset only has to ship training.jsonl; NeMo-RL asserts on a missing
    val dataset whenever val_period / val_at_start / val_at_end is set."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    cfg = compile_grpo_config(step, job_ctx)

    assert cfg["data"]["validation"] is None
    assert cfg["grpo"]["val_period"] == 0
    assert cfg["grpo"]["val_at_end"] is False
    assert cfg["grpo"]["val_at_start"] is False
    assert cfg["checkpointing"]["metric_name"] is None
    assert cfg["checkpointing"]["save_period"] > 0


def test_compile_grpo_config_emits_val_at_start_when_requested(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """val_at_start gives the uplift baseline: step-0 validation on the same data."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, dataset_pvc = _prepared_step(tmp_path)
    _write_gym_dataset(dataset_pvc, filename="validation.jsonl")
    step.schedule.val_at_start = True

    cfg = compile_grpo_config(step, job_ctx)

    assert cfg["grpo"]["val_at_start"] is True
    assert cfg["grpo"]["val_period"] > 0


def test_compile_grpo_config_val_at_start_defaults_off(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GRPO baseline pass costs a full rollout, so it must be opt-in."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, dataset_pvc = _prepared_step(tmp_path)
    _write_gym_dataset(dataset_pvc, filename="validation.jsonl")

    cfg = compile_grpo_config(step, job_ctx)

    assert cfg["grpo"]["val_at_start"] is False


def test_compile_grpo_config_ignores_val_at_start_without_val_split(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NeMo-RL asserts on a missing val dataloader whenever val_at_start is set."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    step.schedule.val_at_start = True

    cfg = compile_grpo_config(step, job_ctx)

    assert cfg["grpo"]["val_at_start"] is False
    assert cfg["grpo"]["val_period"] == 0


def test_compiled_config_selects_only_prefetched_actors(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the config -> Ray actor -> venv coupling.

    NeMo-RL picks the policy actor from the compiled config, and
    ray_actor_environment_registry maps each actor to a py_executable (a `uv run
    --extra <X>` venv). Only the extras prefetched in
    docker/rl/Dockerfile.nmp-rl-base exist in the image; anything else is built on
    the node at job startup, which on a deny-egress training cluster fails outright.

    Each assertion below corresponds to a prefetch filter in that Dockerfile. If you
    flip one, add the matching actor to the prefetch filter list *and* its
    verification loop first.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    cfg = compile_grpo_config(step, job_ctx)

    # policy_backend defaults to `automodel` -> DTensorPolicyWorkerV2 -> the
    # `automodel` venv, matching upstream NeMo-RL's reference grpo config.
    assert cfg["policy"]["dtensor_cfg"]["_v2"] is True
    # megatron_cfg would select MegatronPolicyWorker -> `mcore`.
    assert cfg["policy"]["megatron_cfg"]["enabled"] is False
    # vLLM is the only prefetched generation backend (`vllm`); sglang/trtllm are not.
    assert cfg["policy"]["generation"]["backend"] == "vllm"


def test_compile_grpo_config_enables_lora(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=32, alpha=64, use_triton=True),
    )
    cfg = compile_grpo_config(step, job_ctx)
    lora_cfg = cfg["policy"]["dtensor_cfg"]["lora_cfg"]
    assert lora_cfg["enabled"] is True
    assert lora_cfg["dim"] == 32
    assert lora_cfg["alpha"] == 64
    assert lora_cfg["use_triton"] is True
    assert cfg["policy"]["dtensor_cfg"]["_v2"] is True


def test_compiled_config_has_master_config_required_fields(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fields NeMo-RL's ``MasterConfig`` requires but gives no default for.

    ``MasterConfig`` is a pydantic model, so a missing key is a hard
    ValidationError -- and it is raised by the *driver*, after the Ray cluster is
    already up. On a real job that is minutes of GPU time and four pods in before
    anything complains, and none of it is visible to the compiler's own tests.

    ``make_sequence_length_divisible_by`` was set by dpo_config but not here, and
    the three generation fields are GRPO-only (DPO runs no vLLM), so nothing
    exercised them. Values mirror upstream's reference config
    ``examples/configs/grpo_math_1B.yaml``.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    policy = compile_grpo_config(step, job_ctx)["policy"]

    # None is meaningful, not absent: generation/__init__.py fills stop_token_ids
    # with [tokenizer.eos_token_id] when it is None.
    for field in ("top_k", "stop_token_ids", "stop_strings"):
        assert field in policy["generation"], f"policy.generation.{field} missing"
        assert policy["generation"][field] is None

    assert policy["make_sequence_length_divisible_by"] == step.parallelism.tensor_parallel_size


def test_tokenizer_omits_chat_template_when_none(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with no chat template must omit the key, not emit ``None``.

    TokenizerConfig declares ``chat_template: NotRequired[str]``, so absent is
    valid and ``None`` is not -- MasterConfig rejects it with "Input should be a
    valid string". ``resolve_chat_template`` returns None whenever the model ships
    no template and the user gave none, which is every model without a built-in
    one. Qwen3 has one, so a single-model GPU run never sees this.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    # The fixture model dir has no tokenizer, so resolution falls through to None.
    tokenizer = compile_grpo_config(step, job_ctx)["policy"]["tokenizer"]

    assert "chat_template" not in tokenizer
    # chat_template_kwargs is NotRequired[dict | None], so None is accepted there.
    assert tokenizer["chat_template_kwargs"] is None
    assert tokenizer["name"] == step.model.path


def test_tokenizer_keeps_chat_template_when_present(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    monkeypatch.setattr(
        "nmp.rl.tasks.training.backends.nemo_rl.grpo_config.resolve_chat_template",
        lambda **_: "{{ bos_token }}",
    )
    step, _ = _prepared_step(tmp_path)
    tokenizer = compile_grpo_config(step, job_ctx)["policy"]["tokenizer"]

    assert tokenizer["chat_template"] == "{{ bos_token }}"


def test_compiled_vllm_cfg_has_required_typeddict_fields(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-NotRequired members of NeMo-RL's ``VllmSpecificArgs``.

    ``vllm_cfg`` is a TypedDict that MasterConfig does not validate the keys of, so
    a missing one is not a config error — it is a KeyError raised wherever vLLM
    first reads it, well after the Ray cluster is up.

    ``skip_tokenizer_init`` is deliberately NOT asserted: it is required by the
    TypedDict, but generation/__init__.py fills it in when absent based on
    stop_strings and expose_http_server, and hardcoding it would override that
    VLM-aware logic.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    vllm_cfg = compile_grpo_config(step, job_ctx)["policy"]["generation"]["vllm_cfg"]

    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "async_engine",
        "kv_cache_dtype",
    ):
        assert field in vllm_cfg, f"vllm_cfg.{field} missing"

    # Literal["auto", "fp8", "fp8_e4m3"]; fp8 additionally requires precision=fp8.
    assert vllm_cfg["kv_cache_dtype"] == "auto"
    assert "skip_tokenizer_init" not in vllm_cfg


@pytest.mark.parametrize(
    ("finetuning_type", "lora", "expected"),
    [
        (FinetuningType.ALL_WEIGHTS, None, False),
        (FinetuningType.LORA, LoRAConfig(rank=8), True),
        # finetuning_type is what enables LoRA; an omitted lora block just takes defaults.
        (FinetuningType.LORA, None, True),
    ],
)
def test_lora_cfg_follows_finetuning_type_under_automodel(
    tmp_path: Path,
    job_ctx: NMPJobContext,
    monkeypatch: pytest.MonkeyPatch,
    finetuning_type: FinetuningType,
    lora: LoRAConfig | None,
    expected: bool,
) -> None:
    """``finetuning_type`` drives ``lora_cfg``, and nothing else does. ``_v2`` stays true
    across all three cases: the backend is the caller's choice, not a function of LoRA.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, finetuning_type=finetuning_type, lora=lora)
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    # Read exactly the way NeMo-RL does, so "key omitted" and "enabled: False" are
    # equivalent here and the test does not depend on which one we emit.
    assert dtensor_cfg.get("lora_cfg", {}).get("enabled", False) is expected
    assert dtensor_cfg["_v2"] is True


@pytest.mark.parametrize(
    ("tensor_parallel_size", "expected"),
    [(1, True), (2, False)],
)
def test_unset_triton_is_resolved_from_tensor_parallel_size(
    tmp_path: Path,
    job_ctx: NMPJobContext,
    monkeypatch: pytest.MonkeyPatch,
    tensor_parallel_size: int,
    expected: bool,
) -> None:
    """Unset means the compiler picks the only value that works: NeMo-RL asserts on
    Triton + TP > 1, so choosing wrong here is a crash, not a slowdown."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=16),
        tensor_parallel_size=tensor_parallel_size,
    )
    # Batch divisibility for tp=2 on 1 gpu would fail validate_for_training — this is the
    # compiler path only, so TP is just a knob on lora_cfg here.
    cfg = compile_grpo_config(step, job_ctx)
    assert cfg["policy"]["dtensor_cfg"]["lora_cfg"]["use_triton"] is expected


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_triton_is_passed_through_verbatim(
    tmp_path: Path,
    job_ctx: NMPJobContext,
    monkeypatch: pytest.MonkeyPatch,
    requested: bool,
) -> None:
    """An explicit choice survives compilation. The TP > 1 conflict is rejected up in
    ``GRPOTraining``, so the compiler never has to second-guess the caller."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=16, use_triton=requested),
    )
    cfg = compile_grpo_config(step, job_ctx)
    assert cfg["policy"]["dtensor_cfg"]["lora_cfg"]["use_triton"] is requested


def test_sandbox_egress_comes_from_the_compiled_step_not_service_config(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """allow_internet must follow gym.allow_internet on the step config.

    RlConfig is not readable from the training pod, so the compiler resolves the operator
    setting and passes it through. Reading it here instead would silently use the default.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None
    step.gym.allow_internet = False

    nemo_gym = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]
    sandbox = nemo_gym["sandbox"]
    assert sandbox["allow_internet"] is False
    assert nemo_gym["environment_offline"] is True

    step.gym.allow_internet = True
    nemo_gym = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]
    sandbox = nemo_gym["sandbox"]
    assert sandbox["allow_internet"] is True
    assert nemo_gym["environment_offline"] is False


def test_public_dns_allow_reaches_the_sandbox_network_policy(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator-configured suffixes must land where NeMo-RL reads them.

    NeMo-RL takes them from ``sandbox.network_policy.public_dns_allow`` (nemo_gym_actor), not
    from ``sandbox`` directly, and only consults them when ``allow_internet`` is set. Its
    built-in list is ``*.com``/``*.org``, so an index on any other TLD needs this to resolve.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["network_policy"]["public_dns_allow"] == []

    step.gym.allow_internet = True
    step.gym.public_dns_allow = ["hub.primeintellect.ai"]
    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["network_policy"]["public_dns_allow"] == ["hub.primeintellect.ai"]


def test_compiled_config_survives_the_yaml_round_trip(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compiled dict must round-trip through the writer/reader pair the job uses.

    ``TrainingRunner._compile_library_config`` writes it with ``yaml.dump`` and the driver
    reads it back with ``OmegaConf.load``, whose loader is SafeLoader-based. ``yaml.dump``
    happily emits ``!!python/tuple`` for a tuple, which that loader then refuses -- so a
    tuple anywhere in the config kills the job at driver start with an opaque
    ``ConstructorError`` pointing at a line number, minutes after compile succeeded.
    ``yaml.safe_load`` stands in for OmegaConf's loader here (omegaconf is a NeMo-RL image
    dependency, not a platform one) and rejects the same tags. Asserting on the round trip
    rather than on any one field catches the next non-plain value here, not on a cluster.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None
    step.gym.allow_internet = True
    step.gym.public_dns_allow = ["hub.primeintellect.ai"]

    cfg = compile_grpo_config(step, job_ctx)
    config_path = tmp_path / "nemo_rl_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(cfg, handle, default_flow_style=False)

    text = config_path.read_text(encoding="utf-8")
    assert "!!python/" not in text
    loaded = yaml.safe_load(text)
    assert loaded["env"]["nemo_gym"]["sandbox"]["network_policy"]["public_dns_allow"] == ["hub.primeintellect.ai"]


def test_sandbox_server_protocol_reaches_the_host_provider(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheme must be declarable, because NeMo-RL cannot know it.

    NeMo-RL defaults the health/rollout proxy URLs to https. Against a plain-HTTP
    in-cluster OpenSandbox server every poll then stalls in the TLS handshake and the job
    dies after ready_timeout_s with a bare `<urlopen error timed out>`. The same key also
    feeds the SDK connection used by create_host, so create and health cannot disagree.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None

    # Unset leaves NeMo-RL's default in place rather than asserting one here.
    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["host_provider_options"] == {}

    step.gym.sandbox_server_protocol = "http"
    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["host_provider_options"] == {"connection": {"protocol": "http"}}


def test_generation_sampling_comes_from_the_grpo_hyperparameters(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temperature has to reach policy.generation, which is what stamps every Gym row.

    ``_prepare_nemo_gym_rows`` copies temperature/top_p/max_output_tokens from that block
    onto each row before it is POSTed, so this is the single point that decides how both
    colocated and sandboxed rollouts sample.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, grpo=GRPOConfig(num_generations_per_prompt=4, temperature=0.7, top_k=20))

    generation = compile_grpo_config(step, job_ctx)["policy"]["generation"]
    assert generation["temperature"] == 0.7
    assert generation["top_k"] == 20
    # Neutral value, not a knob: the job schema exposes no top_p.
    assert generation["top_p"] == 1.0


def test_generation_samples_the_full_distribution_by_default(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset top_k means no truncation, which is what standard GRPO does."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    assert compile_grpo_config(step, job_ctx)["policy"]["generation"]["top_k"] is None


def test_generation_temperature_defaults_to_one(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The previous hardcoded value stays the default, so existing jobs do not shift."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    assert compile_grpo_config(step, job_ctx)["policy"]["generation"]["temperature"] == 1.0


def test_normalize_rewards_reaches_the_block_the_estimator_reads(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The advantage estimator reads `grpo.adv_estimator`, not `grpo`.

    NeMo-RL's YAML recipes wire the two together with OmegaConf interpolation, which
    this config never goes through -- it is assembled in Python. Writing only the
    top-level field left AdvEstimatorConfig on its own default of True, so asking for
    unnormalized advantages did nothing and said nothing.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, normalize_rewards=False, use_leave_one_out_baseline=False),
    )

    grpo_cfg = compile_grpo_config(step, job_ctx)["grpo"]
    assert grpo_cfg["adv_estimator"]["name"] == "grpo"
    assert grpo_cfg["adv_estimator"]["normalize_rewards"] is False
    assert grpo_cfg["adv_estimator"]["use_leave_one_out_baseline"] is False
    # Written in both places so the config matches a resolved NeMo-RL recipe, which
    # carries the top-level pair as the interpolation source.
    assert grpo_cfg["normalize_rewards"] is False
    assert grpo_cfg["use_leave_one_out_baseline"] is False


def test_advantage_estimation_defaults_are_unchanged(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both were effectively on before they were knobs; a default job must not shift."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    estimator = compile_grpo_config(step, job_ctx)["grpo"]["adv_estimator"]
    assert estimator["normalize_rewards"] is True
    assert estimator["use_leave_one_out_baseline"] is True


def test_advantage_clip_bounds_reach_the_grpo_block(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Applied after normalization, so they live on `grpo` rather than on the loss."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, advantage_clip_low=-5.0, advantage_clip_high=5.0),
    )

    grpo_cfg = compile_grpo_config(step, job_ctx)["grpo"]
    assert grpo_cfg["advantage_clip_low"] == -5.0
    assert grpo_cfg["advantage_clip_high"] == 5.0


def test_advantages_are_unbounded_by_default(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None on both sides is NeMo-RL's default and standard GRPO."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    grpo_cfg = compile_grpo_config(step, job_ctx)["grpo"]
    assert grpo_cfg["advantage_clip_low"] is None
    assert grpo_cfg["advantage_clip_high"] is None


def test_loss_clipping_and_correction_knobs_reach_the_loss(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three belong to ClippedPGLossConfig, so they compile into `loss_fn`."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(
            num_generations_per_prompt=4,
            ratio_clip_c=3.0,
            use_on_policy_kl_approximation=False,
            use_importance_sampling_correction=False,
        ),
    )

    loss_fn = compile_grpo_config(step, job_ctx)["loss_fn"]
    assert loss_fn["ratio_clip_c"] == 3.0
    assert loss_fn["use_on_policy_kl_approximation"] is False
    assert loss_fn["use_importance_sampling_correction"] is False


def test_loss_knob_defaults_are_unchanged(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These two were hardcoded on before they were knobs; the default keeps them on.

    NeMo-RL's own defaults are False. The platform has been running with them True, so
    the default here follows the platform rather than the library: turning them into
    settings must not silently change the loss every existing job computes.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    loss_fn = compile_grpo_config(step, job_ctx)["loss_fn"]
    assert loss_fn["use_on_policy_kl_approximation"] is True
    assert loss_fn["use_importance_sampling_correction"] is True
    # Dual clipping stays off unless asked for, matching NeMo-RL.
    assert loss_fn["ratio_clip_c"] is None


def test_max_new_tokens_is_independent_of_max_seq_length(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation length must be settable without shrinking the context.

    max_seq_length sizes the context in three places (total sequence, vLLM max_model_len,
    and previously the generation cap). Bounding response length by lowering it would also
    shrink the prompt budget, so the two have to be separate fields.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, grpo=GRPOConfig(num_generations_per_prompt=4, max_new_tokens=128))

    policy = compile_grpo_config(step, job_ctx)["policy"]
    assert policy["generation"]["max_new_tokens"] == 128
    # Context is untouched: max_seq_length still sizes both of these.
    assert policy["max_total_sequence_length"] == 512
    assert policy["generation"]["vllm_cfg"]["max_model_len"] == 512


def test_max_new_tokens_defaults_to_the_full_context(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset keeps NeMo-RL's own recipe convention: generate until the context runs out."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    policy = compile_grpo_config(step, job_ctx)["policy"]
    assert policy["generation"]["max_new_tokens"] == policy["max_total_sequence_length"]


def test_sandbox_resources_reach_the_sandbox_when_the_operator_sets_them(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox runs a Gym server per config entry plus its own Ray instance.

    On the OpenSandbox default that is enough to be OOMKilled mid-rollout, which surfaces
    to the training pod as a proxy 502 rather than as a memory error, so the operator needs
    a way to size the pod.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None
    step.gym.sandbox_resources = {"cpu": "2", "memory": "8Gi"}

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["resources"] == {"cpu": "2", "memory": "8Gi"}


def test_sandbox_resources_unset_leaves_the_provider_default(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset must not be emitted at all rather than this compiler asserting a size.

    The dump is exclude_none, so an unset value leaves the key off entirely and NeMo-RL's
    own default applies.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert "resources" not in sandbox


def test_sandbox_ttl_reaches_the_sandbox_when_the_operator_sets_it(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ttl_s reaps leaked sandboxes, so it has to outlast the longest accepted run.

    NeMo-RL defaults it to 4h. A GRPO job that runs longer loses its sandbox mid-rollout
    and dies with a proxy 502, which reads as a transport fault rather than an expiry.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None
    step.gym.sandbox_ttl_s = 86_400

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["ttl_s"] == 86_400


def test_sandbox_ttl_unset_keeps_the_nemo_rl_default(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["ttl_s"] == 14_400


def test_lora_exclude_modules_turns_off_match_all_linear(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Automodel's ModuleMatcher rejects match_all_linear alongside exclude_modules, and it
    raises inside the policy worker -- after Ray, vLLM and the Gym sandbox are all up.

    Exclude-only is how NemotronH has to be configured: its Mamba mixer gives LoRA no
    gradient on out_proj under the CUDA-kernel path.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=128, alpha=512, exclude_modules=["*out_proj*"], use_triton=False),
    )
    lora_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]["lora_cfg"]

    assert lora_cfg["exclude_modules"] == ["*out_proj*"]
    assert lora_cfg["target_modules"] == []
    assert lora_cfg["match_all_linear"] is False


def test_lora_target_modules_turn_off_match_all_linear(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=16, target_modules=["*_proj"]),
    )
    lora_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]["lora_cfg"]

    assert lora_cfg["target_modules"] == ["*_proj"]
    assert lora_cfg["match_all_linear"] is False


def test_lora_without_module_lists_matches_all_linear(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No lists at all is the one case that should still adapt every linear layer."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=16),
    )
    lora_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]["lora_cfg"]

    assert lora_cfg["match_all_linear"] is True
    assert lora_cfg["target_modules"] == []
    assert lora_cfg["exclude_modules"] == []


def test_policy_backend_dtensor_omits_v2(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dtensor`` must reach V1 -- the `fsdp` venv, stock HF modules, no Transformer
    Engine. NeMo-RL reads ``.get("_v2", False)``, so absent and false are equivalent."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, policy_backend=PolicyBackend.DTENSOR)
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    assert "_v2" not in dtensor_cfg
    assert dtensor_cfg["enabled"] is True


def test_automodel_all_weights_requests_consolidated_safetensors(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    checkpointing = compile_grpo_config(step, job_ctx)["checkpointing"]

    assert checkpointing["model_save_format"] == "safetensors"
    assert checkpointing["save_consolidated"] is True


def test_v4_compatible_defaults_on_for_consolidated_export(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the consolidated export carries a transformers v5 config.json, which the
    vLLM the platform serves the published model with cannot read."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    checkpointing = compile_grpo_config(step, job_ctx)["checkpointing"]

    assert checkpointing["v4_compatible"] is True


def test_v4_compatible_can_be_turned_off(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, v4_compatible=False)
    checkpointing = compile_grpo_config(step, job_ctx)["checkpointing"]

    assert checkpointing["v4_compatible"] is False


def _write_base_model_config(tmp_path: Path, transformers_version: str | None) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    config: dict[str, object] = {"architectures": ["Qwen3ForCausalLM"]}
    if transformers_version is not None:
        config["transformers_version"] = transformers_version
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_v4_compatible_warns_when_the_base_checkpoint_is_v5(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    _write_base_model_config(tmp_path, "5.1.0")
    step, _ = _prepared_step(tmp_path)
    with caplog.at_level("WARNING"):
        compile_grpo_config(step, job_ctx)

    assert any("base checkpoint is transformers v5" in record.message for record in caplog.records)


def test_v4_compatible_does_not_warn_for_a_v4_checkpoint(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    _write_base_model_config(tmp_path, "4.51.0")
    step, _ = _prepared_step(tmp_path)
    with caplog.at_level("WARNING"):
        compile_grpo_config(step, job_ctx)

    assert not any("base checkpoint is transformers v" in record.message for record in caplog.records)


def test_v4_compatible_off_does_not_warn_for_a_v5_checkpoint(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    _write_base_model_config(tmp_path, "5.1.0")
    step, _ = _prepared_step(tmp_path, v4_compatible=False)
    with caplog.at_level("WARNING"):
        compile_grpo_config(step, job_ctx)

    assert not any("base checkpoint is transformers v" in record.message for record in caplog.records)


def test_dtensor_v1_omits_model_save_format(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, policy_backend=PolicyBackend.DTENSOR)
    checkpointing = compile_grpo_config(step, job_ctx)["checkpointing"]

    assert "model_save_format" not in checkpointing
    assert "save_consolidated" not in checkpointing


def test_automodel_lora_still_requests_consolidated_export(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        finetuning_type=FinetuningType.LORA,
        lora=LoRAConfig(rank=8, alpha=16),
    )
    checkpointing = compile_grpo_config(step, job_ctx)["checkpointing"]

    assert checkpointing["save_consolidated"] is True
    assert "model_save_format" not in checkpointing


def test_expert_parallel_size_reaches_dtensor(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``automodel/setup.py`` is the sole reader of ``dtensor_cfg.expert_parallel_size``.
    Without ``_v2`` the V1 worker ignores it and shards nothing, so a MoE run OOMs with no
    sign why -- hence ``GRPOTraining`` rejects the pairing rather than letting it compile.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, expert_parallel_size=8)
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    assert dtensor_cfg["expert_parallel_size"] == 8
    assert dtensor_cfg["_v2"] is True


def test_activation_checkpointing_on_moe_ignores_the_router(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recomputing the router is what makes AC crash on an MoE.

    The bf16 top-k is nondeterministic, so the recomputed forward can route a token to a
    different expert than the saved one did; the permutation buffers then differ by a row
    and backward raises CheckpointError out of a TE kernel. Automodel only saves the router
    output when this is set.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, expert_parallel_size=8, activation_checkpointing=True)
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    assert dtensor_cfg["activation_checkpointing"] is True
    assert dtensor_cfg["moe_parallelizer"] == {"ignore_router_for_ac": True}


@pytest.mark.parametrize(
    ("expert_parallel_size", "activation_checkpointing"),
    [(8, False), (1, True), (1, False)],
)
def test_moe_parallelizer_omitted_without_both_moe_and_checkpointing(
    tmp_path: Path,
    job_ctx: NMPJobContext,
    monkeypatch: pytest.MonkeyPatch,
    expert_parallel_size: int,
    activation_checkpointing: bool,
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        expert_parallel_size=expert_parallel_size,
        activation_checkpointing=activation_checkpointing,
    )
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    assert "moe_parallelizer" not in dtensor_cfg


def test_expert_parallel_size_omitted_when_unused(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    assert "expert_parallel_size" not in dtensor_cfg


def test_automodel_kwargs_reach_dtensor(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force_hf is what makes NemotronH loadable at all on the DTensor path."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, automodel_kwargs={"force_hf": True}),
    )
    dtensor_cfg = compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]

    assert dtensor_cfg["automodel_kwargs"] == {"force_hf": True}
    assert dtensor_cfg["_v2"] is True


def test_automodel_kwargs_omitted_when_unset(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    assert "automodel_kwargs" not in compile_grpo_config(step, job_ctx)["policy"]["dtensor_cfg"]


def test_router_aux_loss_coef_becomes_an_hf_config_override(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.0 is the value an MoE RL run wants, so the field is checked against None, not
    truthiness -- an ``if coef:`` guard would drop it."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, router_aux_loss_coef=0.0),
    )

    assert compile_grpo_config(step, job_ctx)["policy"]["hf_config_overrides"] == {"router_aux_loss_coef": 0.0}


def test_hf_config_overrides_omitted_when_router_coef_unset(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    assert "hf_config_overrides" not in compile_grpo_config(step, job_ctx)["policy"]


def test_vllm_tensor_parallel_size_is_independent_of_training_tp(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 30B model needs several GPUs to hold inference weights while NeMo-RL's recipe for it
    keeps DTensor at tp=1, since tensor parallelism over hybrid Mamba layers is untested.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        tensor_parallel_size=1,
        grpo=GRPOConfig(num_generations_per_prompt=4, vllm_tensor_parallel_size=4),
    )
    policy = compile_grpo_config(step, job_ctx)["policy"]

    assert policy["generation"]["vllm_cfg"]["tensor_parallel_size"] == 4
    assert policy["dtensor_cfg"]["tensor_parallel_size"] == 1


def test_vllm_tensor_parallel_size_falls_back_to_the_coupled_default(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset falls back to min(training tp, gpus per node)."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, tensor_parallel_size=2)

    vllm_cfg = compile_grpo_config(step, job_ctx)["policy"]["generation"]["vllm_cfg"]
    # _prepared_step pins num_gpus_per_node=1, so the min clamps to it.
    assert vllm_cfg["tensor_parallel_size"] == 1


def test_vllm_gpu_memory_utilization_is_configurable(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, vllm_gpu_memory_utilization=0.7),
    )
    default_step, _ = _prepared_step(tmp_path)

    assert compile_grpo_config(step, job_ctx)["policy"]["generation"]["vllm_cfg"]["gpu_memory_utilization"] == 0.7
    assert (
        compile_grpo_config(default_step, job_ctx)["policy"]["generation"]["vllm_cfg"]["gpu_memory_utilization"] == 0.5
    )


def test_compiled_config_carries_the_progress_reporting_extras(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compiled grpo block is the only channel to the driver's logger.

    ``grpo_driver`` builds NemoRLLogger via ``for_schedule``, reading these three with
    getattr. They are extras NeMo-RL never looks at, so nothing downstream fails loudly if
    the compiler stops emitting them -- progress reporting just silently reverts to
    defaults, and the job still succeeds.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    grpo = compile_grpo_config(step, job_ctx)["grpo"]

    assert grpo["steps_per_epoch"] >= 1
    assert "progress_time_series_metrics" in grpo
    assert "progress_min_report_interval_seconds" in grpo


def test_rollout_chunk_size_defaults_to_the_upstream_value(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset carries NeMo-RL's own default, so an undeclared knob changes no behaviour.

    Same shape as ttl_s: the mirror holds a non-null default, so the key is always emitted
    and only its value tracks whether the operator declared one.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None
    assert step.gym.sandbox_rollout_chunk_size is None

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["rollout_chunk_size"] == DEFAULT_ROLLOUT_CHUNK_SIZE


def test_operator_can_pin_the_rollout_chunk_size(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long generations can make one chunk outlive the sandbox proxy's per-request cap.

    That cap is fixed by the OpenSandbox server build, not exposed in its config, so the
    workable chunk size is deployment-specific and has to be declarable.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None

    step.gym.sandbox_rollout_chunk_size = 2
    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["rollout_chunk_size"] == 2


def test_rollout_max_in_flight_defaults_to_the_upstream_value(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset leaves NeMo-RL's default, so declaring the knob is never required."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None
    assert step.gym.sandbox_rollout_max_in_flight is None

    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["rollout_max_in_flight"] == DEFAULT_ROLLOUT_MAX_IN_FLIGHT


def test_operator_can_pin_the_rollout_max_in_flight(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lowering chunk_size without raising this throttles the step.

    In-flight rollouts are chunk_size × max_in_flight, so a deployment that shrinks
    chunks to stay under the proxy cap has to raise concurrency if it wants the same
    throughput it had before chunking.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)
    assert step.gym is not None

    step.gym.sandbox_rollout_max_in_flight = 64
    sandbox = compile_grpo_config(step, job_ctx)["env"]["nemo_gym"]["sandbox"]
    assert sandbox["rollout_max_in_flight"] == 64


def test_dapo_components_default_off(tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unstated job must compile to the same NeMo-RL config as before these knobs existed."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path)

    cfg = compile_grpo_config(step, job_ctx)
    assert cfg["grpo"]["use_dynamic_sampling"] is False
    assert cfg["grpo"]["batch_multiplier"] == 1.0
    assert cfg["grpo"]["reward_shaping"] == {"enabled": False}
    # ClippedPGLossFn gates the whole TIS block on the type being non-null.
    assert cfg["loss_fn"]["truncated_importance_sampling_type"] is None


def test_truncated_importance_sampling_reaches_the_loss_fn(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TIS bounds the rollout-vs-training importance weights, which is what stops a run
    whose ``token_mult_prob_error`` is climbing from training against a policy that no
    longer generated its data. It only takes effect through ``loss_fn``.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(
            num_generations_per_prompt=4,
            truncated_importance_sampling_type="icepop",
            truncated_importance_sampling_ratio=5.0,
            truncated_importance_sampling_ratio_min=0.5,
        ),
    )

    loss_fn = compile_grpo_config(step, job_ctx)["loss_fn"]
    assert loss_fn["truncated_importance_sampling_type"] == "icepop"
    assert loss_fn["truncated_importance_sampling_ratio"] == 5.0
    assert loss_fn["truncated_importance_sampling_ratio_min"] == 0.5
    # seq-mask-tis asserts against this, and NeMo-RL reads it from the same block.
    assert loss_fn["sequence_level_importance_ratios"] is False
    assert loss_fn["use_importance_sampling_correction"] is True


def test_dynamic_sampling_and_reward_shaping_reach_grpo(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both live on the grpo block, and reward_shaping is gated on its own ``enabled``."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(
            num_generations_per_prompt=4,
            use_dynamic_sampling=True,
            batch_multiplier=4.0,
            dynamic_sampling_max_gen_batches=6,
            reward_shaping={"stop_properly_penalty_coef": 0.0},
        ),
    )

    grpo = compile_grpo_config(step, job_ctx)["grpo"]
    assert grpo["use_dynamic_sampling"] is True
    assert grpo["batch_multiplier"] == 4.0
    assert grpo["dynamic_sampling_max_gen_batches"] == 6
    assert grpo["reward_shaping"] == {"enabled": True, "stop_properly_penalty_coef": 0.0}


def test_reward_scaling_reaches_grpo(tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """The DAPO recipes map a binary verifier's [0,1] onto [-1,1] before advantages.

    Unlike shaping, every bound has a non-null default, so the block is always emitted and
    only ``enabled`` decides whether NeMo-RL applies it.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, grpo=GRPOConfig(num_generations_per_prompt=4))
    assert compile_grpo_config(step, job_ctx)["grpo"]["reward_scaling"] == {"enabled": False}

    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(
            num_generations_per_prompt=4,
            reward_scaling={"source_min": 0.0, "source_max": 1.0, "target_min": -1.0, "target_max": 1.0},
        ),
    )
    assert compile_grpo_config(step, job_ctx)["grpo"]["reward_scaling"] == {
        "enabled": True,
        "source_min": 0.0,
        "source_max": 1.0,
        "target_min": -1.0,
        "target_max": 1.0,
    }


def test_batching_defaults_to_dynamic_with_derived_budget(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default is dynamic, with a budget derived as max_seq_length * micro_batch_size."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, grpo=GRPOConfig(num_generations_per_prompt=4))
    policy = compile_grpo_config(step, job_ctx)["policy"]

    # max_seq_length=512, micro_batch_size=1 from the fixture.
    assert policy["dynamic_batching"] == {
        "enabled": True,
        "train_mb_tokens": 512,
        "logprob_mb_tokens": 512,
        "sequence_length_round": 64,
    }
    assert policy["sequence_packing"] == {"enabled": False}


def test_batching_static_disables_both(tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, batching_strategy=BatchingStrategy.STATIC),
    )
    policy = compile_grpo_config(step, job_ctx)["policy"]
    assert policy["dynamic_batching"] == {"enabled": False}
    assert policy["sequence_packing"] == {"enabled": False}


def test_batching_sequence_packing_carries_algorithm(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(
            num_generations_per_prompt=4,
            batching_strategy=BatchingStrategy.SEQUENCE_PACKING,
            train_mb_tokens=2048,
        ),
    )
    policy = compile_grpo_config(step, job_ctx)["policy"]
    assert policy["dynamic_batching"] == {"enabled": False}
    # No sequence_length_round: NeMo-RL only ever reads it off dynamic_batching, so
    # emitting it here would be a dead key that reads as if packing honoured it.
    assert policy["sequence_packing"] == {
        "enabled": True,
        "train_mb_tokens": 2048,
        "logprob_mb_tokens": 2048,
        "algorithm": "modified_first_fit_decreasing",
    }


@pytest.mark.parametrize("strategy", list(BatchingStrategy))
def test_batching_modes_are_never_both_enabled(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch, strategy: BatchingStrategy
) -> None:
    """NeMo-RL asserts the pair is mutually exclusive; violating it fails only after the
    model download and vLLM startup, so pin the invariant here."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, grpo=GRPOConfig(num_generations_per_prompt=4, batching_strategy=strategy))
    policy = compile_grpo_config(step, job_ctx)["policy"]
    assert not (policy["dynamic_batching"]["enabled"] and policy["sequence_packing"]["enabled"])


def test_batching_rejects_budget_below_max_seq_length(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget under max_seq_length leaves the longest rollout unable to fit anywhere."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, train_mb_tokens=256),  # fixture max_seq_length=512
    )
    with pytest.raises(ValueError, match="below max_seq_length"):
        compile_grpo_config(step, job_ctx)


def test_sequence_packing_rejected_under_context_parallel(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DTensorPolicyWorker rejects packing under CP; catch it at compile time instead."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, batching_strategy=BatchingStrategy.SEQUENCE_PACKING),
    )
    step.parallelism.context_parallel_size = 2
    with pytest.raises(ValueError, match="not supported with context_parallel_size"):
        compile_grpo_config(step, job_ctx)

    # dynamic has no such restriction on the GRPO + DTensor path.
    assert step.training.grpo is not None
    step.training.grpo.batching_strategy = BatchingStrategy.DYNAMIC
    assert compile_grpo_config(step, job_ctx)["policy"]["dynamic_batching"]["enabled"] is True


def test_hf_config_overrides_passthrough_preserves_nesting(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested keys must survive: Qwen3.5 reads router_aux_loss_coef under text_config."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    overrides = {"text_config": {"router_aux_loss_coef": 0.0}}
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(num_generations_per_prompt=4, hf_config_overrides=overrides),
    )
    cfg = compile_grpo_config(step, job_ctx)
    assert cfg["policy"]["hf_config_overrides"] == {"text_config": {"router_aux_loss_coef": 0.0}}
    # Copied, not aliased: the caller's dict must not be mutated by compilation.
    assert overrides == {"text_config": {"router_aux_loss_coef": 0.0}}


def test_hf_config_overrides_absent_leaves_key_off(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(tmp_path, grpo=GRPOConfig(num_generations_per_prompt=4))
    assert "hf_config_overrides" not in compile_grpo_config(step, job_ctx)["policy"]


def test_router_aux_loss_coef_layers_onto_passthrough(
    tmp_path: Path, job_ctx: NMPJobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scalar shortcut still writes top-level, alongside unrelated nested overrides."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    step, _ = _prepared_step(
        tmp_path,
        grpo=GRPOConfig(
            num_generations_per_prompt=4,
            router_aux_loss_coef=0.0,
            hf_config_overrides={"text_config": {"rope_theta": 1000.0}},
        ),
    )
    assert compile_grpo_config(step, job_ctx)["policy"]["hf_config_overrides"] == {
        "text_config": {"rope_theta": 1000.0},
        "router_aux_loss_coef": 0.0,
    }
