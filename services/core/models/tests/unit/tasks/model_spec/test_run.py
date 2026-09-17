# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.metadata import FilesetMetadata
from nemo_platform_plugin.files.storage_config import LocalStorageConfig
from nemo_platform_plugin.files.types import FilesetOutput, FilesetPurpose
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    ModelEntity,
    UpdateModelEntityRequest,
)
from nemo_platform_plugin.models.types import (
    ModelSpec as PluginModelSpec,
)
from nmp.core.models.schemas import (
    LinearLayerSpec,
    MambaConfig,
    ModelSpec,
    MoEConfig,
    SlidingWindowConfig,
    ToolCallConfig,
)
from nmp.core.models.tasks.model_spec.run import ModelSpecRunner
from nmp.core.models.tasks.model_spec.schemas import ModelSpecTaskConfig, NMPJobContext


@dataclass(frozen=True)
class _Response:
    body: Any

    def data(self) -> Any:
        return self.body


def _core_model_spec() -> ModelSpec:
    return ModelSpec(
        context_size=32_768,
        num_virtual_tokens=16,
        is_chat=True,
        checkpoint_model_name="Qwen/Qwen3-0.6B",
        family="qwen3",
        num_layers=28,
        hidden_size=1024,
        num_attention_heads=16,
        num_kv_heads=8,
        ffn_hidden_size=3072,
        vocab_size=151_936,
        tied_embeddings=True,
        gated_mlp=True,
        base_num_parameters=600_000_000,
        precision="bfloat16",
        moe_config=MoEConfig(
            num_experts=8,
            num_experts_per_tok=2,
            num_expert_layers=12,
            expert_ffn_size=1536,
            num_shared_experts=1,
        ),
        mamba_config=MambaConfig(
            is_hybrid=True,
            num_mamba_layers=4,
            num_attention_layers=24,
            num_mlp_layers=28,
            state_size=128,
            conv_kernel=4,
        ),
        sliding_window_config=SlidingWindowConfig(window_size=4096),
        linear_layers=[
            LinearLayerSpec(name="model.layers.0.self_attn.q_proj", in_features=1024, out_features=1024),
            LinearLayerSpec(name="model.layers.0.mlp.gate_proj", in_features=1024, out_features=3072),
        ],
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
        tool_call_config=ToolCallConfig(
            tool_call_parser="llama3_json",
            tool_call_plugin="default/qwen-tools",
            auto_tool_choice=True,
        ),
    )


def _model_entity(name: str) -> ModelEntity:
    now = datetime(2026, 1, 1)
    return ModelEntity(
        id="model-123",
        name=name,
        workspace="default",
        created_at=now,
        updated_at=now,
        fileset="fileset://default/qwen3-fileset",
        trust_remote_code=False,
    )


def _fileset(tmp_path: Path) -> FilesetOutput:
    return FilesetOutput(
        id="fileset-123",
        name="qwen3-fileset",
        workspace="default",
        description="Qwen3 checkpoint",
        purpose=FilesetPurpose.MODEL,
        storage=LocalStorageConfig(path=str(tmp_path / "checkpoint")),
        metadata=FilesetMetadata(),
        custom_fields={},
        project="default",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def test_analyze_checkpoint_updates_model_with_plugin_model_spec(tmp_path: Path) -> None:
    files_sdk = MagicMock()
    files_sdk.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(path="config.json"),
            SimpleNamespace(path="model.safetensors"),
        ]
    )
    files_sdk.download.side_effect = lambda **kwargs: kwargs["local_path"].mkdir(parents=True, exist_ok=True)
    sdk = cast(NeMoPlatform, SimpleNamespace(files=files_sdk))

    model_name = "qwen3-0-6b-automodel"
    model_entity = _model_entity(model_name)
    fileset = _fileset(tmp_path)

    models_client = MagicMock()
    models_client.get_model.return_value = _Response(model_entity)

    def update_model(**kwargs: Any) -> _Response:
        body = cast(UpdateModelEntityRequest, kwargs["body"])
        return _Response(model_entity.model_copy(update={"spec": body.spec}))

    models_client.update_model.side_effect = update_model

    files_client = MagicMock()
    files_client.get_fileset.return_value = _Response(fileset)

    def client_factory(_sdk: NeMoPlatform, client_cls: type[Any]) -> Any:
        if client_cls is ModelsClient:
            return models_client
        if client_cls is FilesClient:
            return files_client
        raise AssertionError(f"Unexpected client class: {client_cls}")

    inferred_spec = _core_model_spec()
    parallelism_api = types.ModuleType("nmp.core.models.parallelism.api")
    infer_model_cfg_from_hf = MagicMock(return_value=inferred_spec)
    find_minimum_gpus_from_metadata = MagicMock(side_effect=[(4, {}), (2, {})])
    setattr(parallelism_api, "infer_model_cfg_from_hf", infer_model_cfg_from_hf)
    setattr(parallelism_api, "find_minimum_gpus_from_metadata", find_minimum_gpus_from_metadata)
    parallelism_pkg = types.ModuleType("nmp.core.models.parallelism")
    parallelism_pkg.__path__ = []

    job_ctx = NMPJobContext(
        workspace="default",
        job_id="job-123",
        attempt_id="attempt-0",
        step="model-spec",
        task="model-spec",
        jobs_url=None,
        files_url=None,
        models_url=None,
        storage_path=tmp_path,
        config_path=None,
    )

    with (
        patch.dict(
            sys.modules,
            {
                "nmp.core.models.parallelism": parallelism_pkg,
                "nmp.core.models.parallelism.api": parallelism_api,
            },
        ),
        patch("nmp.core.models.tasks.model_spec.run.client_from_platform", side_effect=client_factory),
    ):
        runner = ModelSpecRunner(sdk=sdk, job_ctx=job_ctx)
        result = runner.analyze_checkpoint(ModelSpecTaskConfig(workspace="default", name=model_name))

    models_client.update_model.assert_called_once()
    update_kwargs = models_client.update_model.call_args.kwargs
    assert update_kwargs["name"] == model_name
    assert update_kwargs["workspace"] == "default"

    body = update_kwargs["body"]
    assert isinstance(body, UpdateModelEntityRequest)
    assert isinstance(body.spec, PluginModelSpec)
    assert body.spec.model_dump() == inferred_spec.model_dump()
    assert result.spec == body.spec

    assert body.spec.minimum_gpus_all_weights == 4
    assert body.spec.minimum_gpus_lora == 2
    assert body.spec.moe_config is not None
    assert body.spec.moe_config.num_experts == 8
    assert body.spec.mamba_config is not None
    assert body.spec.mamba_config.num_mamba_layers == 4
    assert body.spec.sliding_window_config is not None
    assert body.spec.sliding_window_config.window_size == 4096
    assert body.spec.linear_layers is not None
    assert body.spec.linear_layers[0].name == "model.layers.0.self_attn.q_proj"
    assert body.spec.tool_call_config is not None
    assert body.spec.tool_call_config.tool_call_parser == "llama3_json"

    infer_model_cfg_from_hf.assert_called_once()
    assert infer_model_cfg_from_hf.call_args.args == (str(tmp_path / "model"),)
    assert type(infer_model_cfg_from_hf.call_args.args[0]) is str
    assert infer_model_cfg_from_hf.call_args.kwargs == {
        "is_trusted": False,
        "file_listing": ["config.json", "model.safetensors"],
    }

    files_sdk.download.assert_called_once_with(
        remote_path=["config.json"],
        local_path=tmp_path / "model",
        fileset="qwen3-fileset",
        workspace="default",
    )


def _spec_with(**overrides: Any) -> ModelSpec:
    return _core_model_spec().model_copy(update=overrides)


def test_merge_existing_spec_preserves_a_determined_reasoning_toggle() -> None:
    entity = _model_entity("qwen3")
    entity.spec = PluginModelSpec.model_validate(_spec_with(chat_template_controls_reasoning=True).model_dump())
    fresh = _spec_with(chat_template_controls_reasoning=None)

    ModelSpecRunner._merge_existing_spec(entity, fresh)

    assert fresh.chat_template_controls_reasoning is True


def test_merge_existing_spec_preserves_a_determined_false() -> None:
    entity = _model_entity("qwen3")
    entity.spec = PluginModelSpec.model_validate(_spec_with(chat_template_controls_reasoning=False).model_dump())
    fresh = _spec_with(chat_template_controls_reasoning=None)

    ModelSpecRunner._merge_existing_spec(entity, fresh)

    assert fresh.chat_template_controls_reasoning is False


def test_merge_existing_spec_does_not_override_a_fresh_analysis() -> None:
    entity = _model_entity("qwen3")
    entity.spec = PluginModelSpec.model_validate(_spec_with(chat_template_controls_reasoning=False).model_dump())
    fresh = _spec_with(chat_template_controls_reasoning=True)

    ModelSpecRunner._merge_existing_spec(entity, fresh)

    assert fresh.chat_template_controls_reasoning is True


OVERRIDABLE_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
    "{%- if enable_thinking %}<think>{%- endif %}"
)


def test_rederive_uses_a_fileset_supplied_template_over_the_tokenizer_answer() -> None:
    # Tokenizer said the checkpoint toggles; the fileset then overrode the served
    # template with one that ignores the kwarg.
    spec = _spec_with(chat_template_controls_reasoning=True, chat_template="hello {{ messages[0]['content'] }}")

    ModelSpecRunner._rederive_reasoning_control(spec)

    assert spec.chat_template_controls_reasoning is False


def test_rederive_promotes_a_toggling_override() -> None:
    spec = _spec_with(chat_template_controls_reasoning=False, chat_template=OVERRIDABLE_TEMPLATE)

    ModelSpecRunner._rederive_reasoning_control(spec)

    assert spec.chat_template_controls_reasoning is True


def test_rederive_leaves_the_tokenizer_answer_when_no_override_arrived() -> None:
    spec = _spec_with(chat_template_controls_reasoning=True, chat_template=None)

    ModelSpecRunner._rederive_reasoning_control(spec)

    assert spec.chat_template_controls_reasoning is True
