# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nmp-rl-training: does the config we compile actually satisfy NeMo-RL's schemas?

Built as part of the docker-bake.hcl bake group (smoke-test stage) and run on a CPU
runner -- no GPU hardware required.

WHY THIS LIVES IN THE IMAGE RATHER THAN THE SERVICE TEST SUITE
--------------------------------------------------------------
``compile_grpo_config`` and NeMo-RL's schemas are only ever in the same interpreter
here. ``services/rl/tests`` cannot import MasterConfig (it needs torch), and NeMo-RL
cannot see the compiler. So the two halves of the contract were never checked
against each other, and two config-shape bugs reached a live GPU run:

  MasterConfig: policy.generation.{top_k,stop_token_ids,stop_strings} and
                policy.make_sequence_length_divisible_by were missing
  vllm_cfg:     kv_cache_dtype was missing -> KeyError deep inside setup()

Both surfaced only after Ray was up, minutes into a job, and each cost an image
build plus a redeploy to find. Running the check inside the image also removes the
skew that makes an external validator untrustworthy: an older image with pydantic
<2.13 accepts configs the deployed one rejects, because 2.13 enforces required keys
on nested TypedDicts and 2.12 does not. Here the validator and the code being
validated are the same artifact by construction.

WHAT IT DOES NOT COVER
----------------------
Config *shape*, not behaviour. Sandbox creation, vLLM startup, rollouts and reward
parsing all still need a cluster.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke_nmp_rl_training


def _write_fixture(root: Path) -> Path:
    """Materialize the on-disk inputs compile_grpo_config reads.

    It counts JSONL rows and parses the environment manifest, so this cannot be a
    pure in-memory config.
    """
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    row = {
        "task_idx": 0,
        "vf_env_id": "smoke-env",
        "responses_create_params": {"input": [{"role": "user", "content": "hello"}]},
        "agent_ref": {"name": "verifiers_agent"},
        "answer": "42",
        "example_id": "ex-0",
        "info": {},
    }
    (dataset / "training.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    env_root = root / "environment"
    env_root.mkdir(parents=True, exist_ok=True)
    (env_root / "nemo-environment.yaml").write_text(
        "format: adapter-wheels-v1\nadapter:\n  agent: verifiers_agent\nconfig_paths:\n  - configs/verifiers_agent.yaml\n",
        encoding="utf-8",
    )
    (root / "workspace").mkdir(parents=True, exist_ok=True)
    return dataset


def _compile(root: Path, *, lora: bool, moe: bool = False) -> dict:
    """Compile a sandboxed GRPO config the way the training task does.

    ``moe`` mirrors NeMo-RL's NemotronH 30B-A3B recipes, whose two shapes differ:
    ``grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`` uses ``force_hf`` plus an out_proj
    exclusion and no expert parallelism, while ``grpo-nanov3-30BA3B-2n8g-fsdp2.yaml``
    uses expert parallelism plus a Transformer-Engine / DeepEP backend block.
    """
    from nmp.customization_common.service.context import NMPJobContext
    from nmp.rl.app.jobs.training.schemas import (
        GRPOConfig,
        LoRAConfig,
        ModelConfig,
        TrainingBackend,
        TrainingStepConfig,
    )
    from nmp.rl.entities.values import FinetuningType, TrainingType
    from nmp.rl.tasks.training.backends.nemo_rl.grpo_config import compile_grpo_config

    moe_backend = {
        "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
        "attn": "te",
        "linear": "te",
        "rms_norm": "torch_fp32",
        "experts": "torch_mm",
        "enable_deepep": True,
        "rope_fusion": False,
        "enable_hf_state_dict_adapter": True,
    }
    grpo_kwargs: dict = {"num_generations_per_prompt": 4}
    if moe:
        grpo_kwargs |= {
            "automodel_kwargs": {"force_hf": True} if lora else {"backend": moe_backend},
            "router_aux_loss_coef": 0.0,
            "vllm_tensor_parallel_size": 4,
            "vllm_gpu_memory_utilization": 0.7,
        }
    if lora:
        lora_cfg = (
            LoRAConfig(rank=128, alpha=512, exclude_modules=["*out_proj*"], use_triton=False)
            if moe
            else LoRAConfig(rank=8)
        )
    else:
        lora_cfg = None

    dataset = _write_fixture(root)
    step = TrainingStepConfig(
        backend=TrainingBackend.NEMO_RL,
        model=ModelConfig(path=str(root / "model"), max_seq_length=512),
        dataset=TrainingStepConfig.DatasetConfig(path=str(dataset)),
        gym=TrainingStepConfig.GymConfig(
            environment_path=str(root / "environment"),
            sandbox_environment_path="/job/environment",
            sandbox_dataset_path="/job/dataset",
            sandboxed=True,
            gym_runtime_image="nvcr.io/nvidia/nmp-rl-training:test",
        ),
        training=TrainingStepConfig.TrainingConfig(
            training_type=TrainingType.GRPO,
            finetuning_type=FinetuningType.LORA if lora else FinetuningType.ALL_WEIGHTS,
            grpo=GRPOConfig(**grpo_kwargs),
            lora=lora_cfg,
        ),
        schedule=TrainingStepConfig.ScheduleConfig(epochs=1),
        batch=TrainingStepConfig.BatchConfig(global_batch_size=8, micro_batch_size=1),
        optimizer=TrainingStepConfig.OptimizerConfig(),
        parallelism=TrainingStepConfig.ParallelismConfig(
            num_gpus_per_node=8 if moe else 1,
            tensor_parallel_size=1,
            expert_parallel_size=8 if moe and not lora else 1,
        ),
        output_model="out",
        workspace_path=str(root / "workspace"),
    )
    ctx = NMPJobContext(
        workspace="default",
        job_id="smoke-job",
        attempt_id="attempt-1",
        step="grpo-training",
        task="training",
        jobs_url=None,
        files_url=None,
        storage_path=root,
        config_path=root / "config.yaml",
    )
    return compile_grpo_config(step, ctx)


@pytest.mark.parametrize("lora", [False, True], ids=["full_weight", "lora"])
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_compiled_grpo_config_satisfies_master_config(tmp_path, monkeypatch, lora, moe):
    """The driver builds MasterConfig from this dict; a missing field is fatal there."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    from nemo_rl.algorithms.grpo import MasterConfig

    MasterConfig(**_compile(tmp_path, lora=lora, moe=moe))


@pytest.mark.parametrize("lora", [False, True], ids=["full_weight", "lora"])
def test_moe_knobs_land_on_the_keys_nemo_rl_reads(tmp_path, monkeypatch, lora):
    """Each of these keys is optional in NeMo-RL, so a wrong name or nesting level passes
    every schema check and does nothing. ``expert_parallel_size`` is read only by
    ``nemo_rl/models/automodel/setup.py``, ``automodel_kwargs`` only by the v2 worker, and
    ``hf_config_overrides`` sits on ``policy``, not on ``dtensor_cfg``.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    policy = _compile(tmp_path, lora=lora, moe=True)["policy"]
    dtensor_cfg = policy["dtensor_cfg"]

    assert dtensor_cfg["_v2"] is True
    assert dtensor_cfg["automodel_kwargs"]
    assert policy["hf_config_overrides"] == {"router_aux_loss_coef": 0.0}
    # Generation shards for capacity; training does not shard tensors at all here.
    assert policy["generation"]["vllm_cfg"]["tensor_parallel_size"] == 4
    assert policy["generation"]["vllm_cfg"]["gpu_memory_utilization"] == 0.7
    assert dtensor_cfg["tensor_parallel_size"] == 1

    if lora:
        assert dtensor_cfg["lora_cfg"]["exclude_modules"] == ["*out_proj*"]
        assert dtensor_cfg["lora_cfg"]["match_all_linear"] is False
        assert "expert_parallel_size" not in dtensor_cfg
    else:
        assert dtensor_cfg["expert_parallel_size"] == 8


def test_dtensor_cfg_keys_are_known_to_nemo_rl(tmp_path, monkeypatch):
    """DTensorConfig is a TypedDict, so an invented key is accepted and ignored, never
    rejected."""
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    from nemo_rl.models.policy import DTensorConfig

    known = set(DTensorConfig.__required_keys__) | set(DTensorConfig.__optional_keys__)
    for lora in (False, True):
        emitted = set(_compile(tmp_path, lora=lora, moe=True)["policy"]["dtensor_cfg"])
        assert not emitted - known, f"unknown dtensor_cfg keys: {sorted(emitted - known)}"


@pytest.mark.parametrize("lora", [False, True], ids=["full_weight", "lora"])
def test_compiled_vllm_cfg_has_required_typeddict_keys(tmp_path, monkeypatch, lora):
    """vllm_cfg members MasterConfig does not validate.

    It is a TypedDict, so pydantic accepts a partial one and the miss becomes a
    KeyError wherever vLLM first reads it -- e.g. grpo.py's kv_cache_dtype check,
    inside setup(), long after config load.

    ``skip_tokenizer_init`` is exempt: generation/__init__.py sets it when absent,
    choosing from stop_strings and expose_http_server, and hardcoding it would
    override logic that exists for VLMs.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    from nemo_rl.models.generation.vllm.config import VllmSpecificArgs

    vllm_cfg = _compile(tmp_path, lora=lora)["policy"]["generation"]["vllm_cfg"]
    missing = (set(VllmSpecificArgs.__required_keys__) - {"skip_tokenizer_init"}) - set(vllm_cfg)

    assert not missing, f"vllm_cfg missing required keys: {sorted(missing)}"


def test_lora_selects_the_v2_dtensor_worker(tmp_path, monkeypatch):
    """LoRA is implemented only in DTensorPolicyWorkerV2.

    lm_policy.py asserts "LoRA is not supported for DTensorPolicyWorker V1" when it
    sees an enabled lora_cfg without _v2, so the two must be emitted together.
    """
    monkeypatch.setenv("NMP_JOB_STORAGE_PVC_CLAIM", "nmp-job-storage")
    dtensor_cfg = _compile(tmp_path, lora=True)["policy"]["dtensor_cfg"]

    assert dtensor_cfg["lora_cfg"]["enabled"] is True
    assert dtensor_cfg["_v2"] is True
