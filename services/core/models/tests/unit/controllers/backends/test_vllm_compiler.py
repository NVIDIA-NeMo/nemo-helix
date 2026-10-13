# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the backend-agnostic vLLM compiler."""

from types import SimpleNamespace

from nhx.core.models.controllers.backends import vllm_compiler
from nhx.core.models.controllers.backends.common import DeploymentConfigView


def _view(**kwargs) -> DeploymentConfigView:
    return DeploymentConfigView(**kwargs)


def _model_entity(
    *,
    spec=None,
    trust_remote_code=False,
    base_model=None,
    finetuning_type=None,
) -> SimpleNamespace:
    """A lightweight stand-in for a ModelEntity (avoids MagicMock(spec=) collision)."""
    return SimpleNamespace(
        spec=spec,
        trust_remote_code=trust_remote_code,
        base_model=base_model,
        finetuning_type=finetuning_type,
    )


def _spec(**kwargs) -> SimpleNamespace:
    base = {"hidden_size": 4096, "num_attention_heads": 32, "num_kv_heads": 8, "context_size": 8192}
    base.update(kwargs)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


def test_resolve_vllm_image_falls_back_to_defaults():
    """When the config sets no image, the configured defaults are used verbatim."""
    view = _view(gpu=1)
    name, tag = vllm_compiler.resolve_vllm_image(view, "default-image", "default-tag")
    assert name == "default-image"
    assert tag == "default-tag"


def test_resolve_vllm_image_override():
    """An image set on the config takes precedence over the defaults."""
    view = _view(gpu=1, image_name="my-mirror/vllm", image_tag="custom")
    name, tag = vllm_compiler.resolve_vllm_image(view, "default-image", "default-tag")
    assert name == "my-mirror/vllm"
    assert tag == "custom"


# ---------------------------------------------------------------------------
# Serve args
# ---------------------------------------------------------------------------


def test_compile_vllm_args_basic():
    view = _view(gpu=1, model_namespace="default", model_name="qwen")
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    assert args[0] == "/model-store"
    assert "--served-model-name" in args
    assert args[args.index("--served-model-name") + 1] == "default/qwen"


def test_compile_vllm_args_uses_hf_alternate_for_customized_embedding():
    model_entity = _model_entity(
        spec=_spec(head_type="embedding"),
        base_model="default/base-embed",
        finetuning_type="all_weights",
    )
    args = vllm_compiler.compile_vllm_args(_view(gpu=1, model_name="embed"), model_entity)
    assert args[0] == vllm_compiler.VLLM_HF_ALTERNATE_PATH


def test_compile_vllm_args_uses_hf_alternate_for_legacy_embedding_alias():
    model_entity = _model_entity(
        spec=_spec(head_type="unknown", is_embedding_model=True),
        base_model="default/base-embed",
        finetuning_type="all_weights",
    )
    args = vllm_compiler.compile_vllm_args(_view(gpu=1, model_name="embed"), model_entity)
    assert args[0] == vllm_compiler.VLLM_HF_ALTERNATE_PATH


def test_compile_vllm_args_cross_encoder_does_not_use_hf_alternate():
    model_entity = _model_entity(
        spec=_spec(head_type="cross_encoder", is_embedding_model=True),
        base_model="default/base-rerank",
        finetuning_type="all_weights",
    )
    args = vllm_compiler.compile_vllm_args(_view(gpu=1, model_name="rerank"), model_entity)
    assert args[0] == vllm_compiler.MODEL_STORE_PATH


def test_compile_vllm_args_uses_root_for_base_embedding():
    model_entity = _model_entity(spec=_spec(head_type="embedding", is_embedding_model=True))
    args = vllm_compiler.compile_vllm_args(_view(gpu=1, model_name="embed"), model_entity)
    assert args[0] == vllm_compiler.MODEL_STORE_PATH


def test_compile_vllm_args_served_name_without_namespace():
    view = _view(gpu=1, model_name="qwen")
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    assert args[args.index("--served-model-name") + 1] == "qwen"


def test_compile_vllm_args_appends_additional_args_verbatim():
    view = _view(gpu=1, model_name="qwen", additional_args=["--max-model-len", "8192"])
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    assert args[-2:] == ["--max-model-len", "8192"]


def test_compile_vllm_args_user_served_name_not_overridden():
    view = _view(gpu=1, model_name="qwen", additional_args=["--served-model-name", "custom"])
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    # Compiler must not inject its own served-model-name when the user set one.
    assert args.count("--served-model-name") == 1
    assert args[args.index("--served-model-name") + 1] == "custom"


def test_compile_vllm_args_enable_lora():
    view = _view(gpu=1, model_name="qwen", lora_enabled=True)
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    assert "--enable-lora" in args


def test_compile_vllm_args_chat_template():
    view = _view(gpu=1, model_name="qwen", chat_template="{{ messages }}")
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    assert args[args.index("--chat-template") + 1] == "{{ messages }}"


def test_compile_vllm_args_trust_remote_code():
    # trust-remote-code is a vllm serve CLI flag, not a (non-existent) env var.
    model_entity = _model_entity(trust_remote_code=True)
    view = _view(gpu=1, model_name="qwen")
    args = vllm_compiler.compile_vllm_args(view, model_entity=model_entity)
    assert "--trust-remote-code" in args


def test_compile_vllm_args_no_trust_remote_code_by_default():
    model_entity = _model_entity(trust_remote_code=False)
    view = _view(gpu=1, model_name="qwen")
    args = vllm_compiler.compile_vllm_args(view, model_entity=model_entity)
    assert "--trust-remote-code" not in args
    # Also absent when there is no model entity at all.
    assert "--trust-remote-code" not in vllm_compiler.compile_vllm_args(view, model_entity=None)


def test_compile_vllm_args_user_trust_remote_code_not_duplicated():
    model_entity = _model_entity(trust_remote_code=True)
    view = _view(gpu=1, model_name="qwen", additional_args=["--trust-remote-code"])
    args = vllm_compiler.compile_vllm_args(view, model_entity=model_entity)
    assert args.count("--trust-remote-code") == 1


def test_compile_vllm_args_cpu_only_no_tp():
    view = _view(gpu=0, model_name="qwen")
    args = vllm_compiler.compile_vllm_args(view, model_entity=None)
    assert "--tensor-parallel-size" not in args


def test_compile_vllm_args_user_tp_not_overridden():
    model_entity = _model_entity(spec=_spec())
    view = _view(gpu=4, model_name="qwen", additional_args=["--tensor-parallel-size", "2"])
    args = vllm_compiler.compile_vllm_args(view, model_entity=model_entity)
    assert args.count("--tensor-parallel-size") == 1
    assert args[args.index("--tensor-parallel-size") + 1] == "2"


# ---------------------------------------------------------------------------
# Tensor parallelism heuristic
# ---------------------------------------------------------------------------


def test_compute_tp_single_gpu():
    assert vllm_compiler.compute_tensor_parallel_size(1, None) == 1


def test_compute_tp_cpu_only():
    assert vllm_compiler.compute_tensor_parallel_size(0, None) == 1


def test_compute_tp_missing_spec_defaults_to_one():
    model_entity = _model_entity(spec=None)
    assert vllm_compiler.compute_tensor_parallel_size(4, model_entity) == 1


def test_compute_tp_picks_largest_valid_divisor():
    # hidden_size/heads/kv_heads all divisible by 1,2,4 -> picks 4 (== gpu count).
    model_entity = _model_entity(spec=_spec(hidden_size=4096, num_attention_heads=32, num_kv_heads=8))
    assert vllm_compiler.compute_tensor_parallel_size(4, model_entity) == 4


def test_compute_tp_respects_kv_head_constraint():
    # num_kv_heads=2 only divides by 1 and 2 (not 4) -> TP capped at 2.
    model_entity = _model_entity(spec=_spec(hidden_size=4096, num_attention_heads=32, num_kv_heads=2))
    assert vllm_compiler.compute_tensor_parallel_size(4, model_entity) == 2


def test_compute_tp_no_valid_divisor_defaults_to_one():
    # Odd dims not divisible by any divisor of 4 except 1.
    model_entity = _model_entity(spec=_spec(hidden_size=4097, num_attention_heads=33, num_kv_heads=7))
    assert vllm_compiler.compute_tensor_parallel_size(4, model_entity) == 1


def test_compute_tp_missing_kv_heads_falls_back_to_heads():
    model_entity = _model_entity(spec=_spec(hidden_size=4096, num_attention_heads=8, num_kv_heads=None))
    # Falls back to num_heads=8 for the kv constraint; 4 divides 8 -> TP=4.
    assert vllm_compiler.compute_tensor_parallel_size(4, model_entity) == 4


# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------


def test_compile_vllm_env_lora():
    view = _view(gpu=1, model_name="qwen", lora_enabled=True)
    env = vllm_compiler.compile_vllm_env_vars(view)
    assert env["VLLM_PLUGINS"] == "lora_filesystem_resolver"
    assert env["VLLM_LORA_RESOLVER_CACHE_DIR"] == vllm_compiler.VLLM_LORA_CACHE_DIR
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"


def test_compile_vllm_env_no_lora():
    view = _view(gpu=1, model_name="qwen", lora_enabled=False)
    env = vllm_compiler.compile_vllm_env_vars(view)
    assert "VLLM_PLUGINS" not in env


def test_compile_vllm_env_additional_envs_merged():
    view = _view(gpu=1, model_name="qwen", additional_envs={"FOO": "bar", "NUM": 3})
    env = vllm_compiler.compile_vllm_env_vars(view)
    assert env["FOO"] == "bar"
    assert env["NUM"] == "3"
