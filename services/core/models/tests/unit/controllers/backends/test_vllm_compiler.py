# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the backend-agnostic vLLM compiler."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from nemo_helix_plugin.models.types import ModelEntity, ModelSpec, ToolCallConfig
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


@pytest.mark.parametrize("auto_tool_choice", [True, False, None])
def test_compile_vllm_args_model_tool_call_config(auto_tool_choice: bool | None):
    model = _tool_model(ToolCallConfig(tool_call_parser="llama3_json", auto_tool_choice=auto_tool_choice))
    args = vllm_compiler.compile_vllm_args(_view(), model)
    assert args[args.index("--tool-call-parser") + 1] == "llama3_json"
    assert ("--enable-auto-tool-choice" in args) is (auto_tool_choice is True)


def _tool_model(config: ToolCallConfig | None) -> ModelEntity:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ModelEntity(
        id="base-id",
        name="base",
        workspace="default",
        created_at=now,
        updated_at=now,
        spec=ModelSpec(
            checkpoint_model_name="llama-base",
            family="llama",
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=32,
            num_kv_heads=8,
            ffn_hidden_size=8192,
            vocab_size=128256,
            tied_embeddings=True,
            gated_mlp=True,
            base_num_parameters=1_000_000_000,
            precision="bfloat16",
            tool_call_config=config,
        ),
    )


@pytest.mark.parametrize(
    "deployment_config,expected_parser,expected_auto",
    [
        (ToolCallConfig(tool_call_parser="hermes"), "hermes", True),
        (ToolCallConfig(auto_tool_choice=False), "llama3_json", False),
        (ToolCallConfig(auto_tool_choice=True), "llama3_json", True),
        (ToolCallConfig(tool_call_parser="hermes", auto_tool_choice=False), "hermes", False),
        (ToolCallConfig(), "llama3_json", True),
    ],
)
def test_compile_vllm_args_deployment_tool_config_overrides_per_field(
    deployment_config: ToolCallConfig,
    expected_parser: str,
    expected_auto: bool,
):
    model = _tool_model(ToolCallConfig(tool_call_parser="llama3_json", auto_tool_choice=True))
    args = vllm_compiler.compile_vllm_args(_view(tool_call_config=deployment_config), model)
    assert args[args.index("--tool-call-parser") + 1] == expected_parser
    assert ("--enable-auto-tool-choice" in args) is expected_auto


def test_compile_vllm_args_deployment_enables_auto_tool_choice():
    model = _tool_model(ToolCallConfig(auto_tool_choice=False))
    config = ToolCallConfig(tool_call_parser="hermes", auto_tool_choice=True)
    args = vllm_compiler.compile_vllm_args(_view(tool_call_config=config), model)
    assert args == ["/model-store", "--tool-call-parser", "hermes", "--enable-auto-tool-choice"]


def test_compile_vllm_args_deployment_tool_config_without_model():
    config = ToolCallConfig(tool_call_parser="hermes", auto_tool_choice=True)
    args = vllm_compiler.compile_vllm_args(_view(tool_call_config=config), None)
    assert args == ["/model-store", "--tool-call-parser", "hermes", "--enable-auto-tool-choice"]


@pytest.mark.parametrize("parser_args", [["--tool-call-parser", "custom"], ["--tool-call-parser=custom"]])
@pytest.mark.parametrize("auto_flag", ["--enable-auto-tool-choice", "--no-enable-auto-tool-choice"])
@pytest.mark.parametrize("deployment_auto", [True, False])
def test_compile_vllm_args_raw_tool_flags_override_config(
    parser_args: list[str],
    auto_flag: str,
    deployment_auto: bool,
):
    model = _tool_model(ToolCallConfig(tool_call_parser="llama3_json", auto_tool_choice=True))
    raw_args = [*parser_args, auto_flag]
    deployment_config = ToolCallConfig(tool_call_parser="hermes", auto_tool_choice=deployment_auto)
    args = vllm_compiler.compile_vllm_args(_view(tool_call_config=deployment_config, additional_args=raw_args), model)
    assert args == ["/model-store", *raw_args]


@pytest.mark.parametrize("from_deployment", [False, True])
def test_auto_tool_choice_without_parser_warns_and_is_not_inferred(from_deployment: bool, caplog):
    config = ToolCallConfig(auto_tool_choice=True)
    model = None if from_deployment else _tool_model(config)
    view = _view(tool_call_config=config if from_deployment else None)
    assert vllm_compiler.compile_vllm_args(view, model) == ["/model-store"]
    assert "auto tool choice" in caplog.text and "parser" in caplog.text


@pytest.mark.parametrize("parser_args", [["--tool-call-parser", "hermes"], ["--tool-call-parser=hermes"]])
def test_raw_parser_satisfies_inferred_auto_tool_choice(parser_args: list[str]):
    model = _tool_model(ToolCallConfig(auto_tool_choice=True))
    args = vllm_compiler.compile_vllm_args(_view(additional_args=parser_args), model)
    assert args == ["/model-store", "--enable-auto-tool-choice", *parser_args]


def test_raw_auto_choice_without_parser_is_not_silently_removed():
    raw = ["--enable-auto-tool-choice"]
    assert vllm_compiler.compile_vllm_args(_view(additional_args=raw), None) == ["/model-store", *raw]


def test_custom_plugin_without_mounted_path_skips_inferred_parser_and_auto(caplog):
    model = _tool_model(
        ToolCallConfig(
            tool_call_parser="custom",
            tool_call_plugin="default/parser-plugin",
            auto_tool_choice=True,
        )
    )
    assert vllm_compiler.compile_vllm_args(_view(), model) == ["/model-store"]
    assert "plugin" in caplog.text and "mounted" in caplog.text


def test_custom_plugin_uses_backend_supplied_mount_path():
    model = _tool_model(
        ToolCallConfig(
            tool_call_parser="custom",
            tool_call_plugin="default/parser-plugin",
            auto_tool_choice=True,
        )
    )
    args = vllm_compiler.compile_vllm_args(_view(), model, tool_call_plugin_path="/scratch/plugin/plugin.py")
    assert args == [
        "/model-store",
        "--tool-parser-plugin",
        "/scratch/plugin/plugin.py",
        "--tool-call-parser",
        "custom",
        "--enable-auto-tool-choice",
    ]
    assert "default/parser-plugin" not in args


@pytest.mark.parametrize(
    "plugin_args", [["--tool-parser-plugin", "/user/parser.py"], ["--tool-parser-plugin=/user/parser.py"]]
)
@pytest.mark.parametrize("mounted_path", [None, "/scratch/plugin/plugin.py"])
def test_raw_plugin_path_overrides_backend_mount(plugin_args: list[str], mounted_path: str | None):
    model = _tool_model(
        ToolCallConfig(
            tool_call_parser="custom",
            tool_call_plugin="default/parser-plugin",
            auto_tool_choice=True,
        )
    )
    args = vllm_compiler.compile_vllm_args(
        _view(additional_args=plugin_args),
        model,
        tool_call_plugin_path=mounted_path,
    )
    assert args == ["/model-store", "--tool-call-parser", "custom", "--enable-auto-tool-choice", *plugin_args]


def test_compile_vllm_args_no_tool_config_unchanged():
    assert vllm_compiler.compile_vllm_args(_view(), _tool_model(None)) == ["/model-store"]
    assert vllm_compiler.compile_vllm_args(_view(tool_call_config=ToolCallConfig()), None) == ["/model-store"]


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


@pytest.mark.parametrize("raw", [["--served_model_name", "custom"], ["--served_model_name=custom"]])
def test_compile_vllm_args_underscore_spelling_counts_as_user_flag(raw: list[str]):
    # vLLM accepts underscores in flag names, so the compiler must not add a second one.
    args = vllm_compiler.compile_vllm_args(_view(gpu=1, model_name="qwen", additional_args=raw), None)
    assert args == ["/model-store", *raw]


# ---------------------------------------------------------------------------
# Nemotron 3.5 Super VL defaults
# ---------------------------------------------------------------------------


def _super_vl(
    *,
    base_model=None,
    finetuning_type=None,
    tool_call_config=None,
    architecture="NemotronH_Omni_Reasoning_V3",
    mtp_layers=1,
) -> SimpleNamespace:
    """The spec the model spec task records: language-model dims plus the top-level architecture."""
    spec = _spec(
        num_kv_heads=2,
        architectures=[architecture],
        tool_call_config=tool_call_config,
        num_nextn_predict_layers=mtp_layers,
    )
    return _model_entity(spec=spec, trust_remote_code=True, base_model=base_model, finetuning_type=finetuning_type)


_TP4 = ["/model-store", "--tensor-parallel-size", "4"]
_QWEN3_CODER = ["--tool-call-parser", "qwen3_coder", "--enable-auto-tool-choice"]
_SUPER_VL_DEFAULTS = [
    "--max-model-len",
    "114688",
    "--reasoning-parser",
    "nemotron_v3",
    "--mamba-ssm-cache-dtype",
    "float32",
    "--limit-mm-per-prompt",
    '{"image": 1, "video": 1}',
]
_MTP = ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}']


@pytest.mark.parametrize(
    ("view_kwargs", "model", "expected"),
    [
        pytest.param(
            {},
            _super_vl(architecture=architecture),
            [*_TP4, *_QWEN3_CODER, "--trust-remote-code", *_SUPER_VL_DEFAULTS, *_MTP],
            id=f"base-{architecture}",
        )
        for architecture in sorted(vllm_compiler.NEMOTRON_SUPER_VL_ARCHITECTURES)
    ]
    + [
        # Adapters are rank 64 and served without MTP.
        pytest.param(
            {"lora_enabled": True},
            _super_vl(),
            [*_TP4, *_QWEN3_CODER, "--enable-lora", "--trust-remote-code", *_SUPER_VL_DEFAULTS]
            + ["--max-lora-rank", "64", "--max-loras", "2"],
            id="lora",
        ),
        # A fine-tuned checkpoint may lack the MTP layer; the model API accepts finetuning_type
        # without base_model.
        pytest.param(
            {},
            _super_vl(base_model="default/super-vl-base"),
            [*_TP4, *_QWEN3_CODER, "--trust-remote-code", *_SUPER_VL_DEFAULTS],
            id="fine-tuned-base-model",
        ),
        pytest.param(
            {},
            _super_vl(finetuning_type="all_weights"),
            [*_TP4, *_QWEN3_CODER, "--trust-remote-code", *_SUPER_VL_DEFAULTS],
            id="fine-tuned-finetuning-type-only",
        ),
        # Nemotron 3.5 Bolt shares the architecture but has no MTP layer; a model registered
        # before the spec recorded the layer count is treated the same way.
        *(
            pytest.param(
                {},
                _super_vl(mtp_layers=mtp_layers),
                [*_TP4, *_QWEN3_CODER, "--trust-remote-code", *_SUPER_VL_DEFAULTS],
                id=f"no-mtp-layer-{mtp_layers}",
            )
            for mtp_layers in (0, None)
        ),
        pytest.param(
            {},
            _super_vl(tool_call_config=ToolCallConfig(tool_call_parser="hermes", auto_tool_choice=False)),
            [*_TP4, "--tool-call-parser", "hermes", "--trust-remote-code", *_SUPER_VL_DEFAULTS, *_MTP],
            id="configured-tool-parser-wins",
        ),
        # Same language-model dims, but a text checkpoint: no multimodal defaults.
        pytest.param(
            {},
            _model_entity(spec=_spec(num_kv_heads=2, architectures=["NemotronHForCausalLM"])),
            _TP4,
            id="text-nemotron-h",
        ),
    ],
)
def test_super_vl_defaults(view_kwargs: dict, model: SimpleNamespace, expected: list[str]):
    # TP 4 on 4 GPUs: vLLM copies Super VL's 2 KV heads across ranks.
    assert vllm_compiler.compile_vllm_args(_view(gpu=4, **view_kwargs), model) == expected


@pytest.mark.parametrize(
    ("raw", "replaced"),
    [
        (
            ["--max-model-len", "32768", "--speculative-config", '{"method": "ngram"}'],
            {"--max-model-len", "--speculative-config"},
        ),
        (
            ["--max_model_len=32768", "--speculative_config.method", "ngram"],
            {"--max-model-len", "--speculative-config"},
        ),
        # Opt-in flags are off by default and keep every platform default alongside them.
        (["--enable-expert-parallel"], set()),
        (["--skip-mm-profiling"], set()),
        (["--mm-encoder-attn-backend", "TORCH_SDPA"], set()),
    ],
)
def test_super_vl_additional_args(raw: list[str], replaced: set[str]):
    kept = vllm_compiler.compile_vllm_args(_view(gpu=4), _super_vl())
    for flag in replaced:
        index = kept.index(flag)
        del kept[index : index + 2]
    assert vllm_compiler.compile_vllm_args(_view(gpu=4, additional_args=raw), _super_vl()) == [*kept, *raw]


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


@pytest.mark.parametrize(
    ("gpu", "hidden_size", "num_heads", "num_kv_heads", "expected"),
    [
        # vLLM copies 2 KV heads across 4 or 8 ranks; Nemotron 3.5 Super VL needs TP 4+ to fit.
        (4, 4096, 32, 2, 4),
        (8, 4096, 32, 2, 8),
        # 3 KV heads neither split across 2 or 4 ranks nor copy evenly onto them.
        (4, 3072, 24, 3, 1),
    ],
)
def test_compute_tp_kv_head_rule(gpu: int, hidden_size: int, num_heads: int, num_kv_heads: int, expected: int):
    spec = _spec(hidden_size=hidden_size, num_attention_heads=num_heads, num_kv_heads=num_kv_heads)
    assert vllm_compiler.compute_tensor_parallel_size(gpu, _model_entity(spec=spec)) == expected


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
