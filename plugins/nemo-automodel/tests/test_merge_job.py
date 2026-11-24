# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA merge job: the real compiler against a fake platform API, and the SDK's routing."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_automodel_plugin.jobs.merge import AutomodelMergeSpec, compile_merge
from nemo_automodel_plugin.sdk.resources import AutomodelCustomization
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.models.types import Adapter, Lora, ModelEntity, ModelSpec
from nhx.automodel.app.constants import DEFAULT_MODEL_PATH
from nhx.automodel.tasks.merge_adapter import ADAPTER_PATH, MergeAdapterStepConfig
from nhx.customization_common.sdk.client import make_customization_sdk_context
from nhx.customization_common.service.platform_client import AsyncCustomizationHelixClients

NOW = datetime(2026, 1, 1)
SPEC_FIELDS: dict[str, Any] = dict(
    checkpoint_model_name="SomeForCausalLM", family="gemma", num_layers=2, hidden_size=8, num_attention_heads=2,
    num_kv_heads=2, ffn_hidden_size=16, vocab_size=32, tied_embeddings=False, gated_mlp=True,
    base_num_parameters=1000, precision="bfloat16",
)  # fmt: skip


def _model(**overrides: Any) -> ModelEntity:
    fields: dict[str, Any] = {"id": "m", "workspace": "default", "name": "bolt", "fileset": "default/bolt-weights"}
    return ModelEntity(created_at=NOW, updated_at=NOW, **{**fields, **overrides})


def _adapter(**overrides: Any) -> Adapter:
    fields: dict[str, Any] = {
        "name": "cord-lora", "workspace": "default", "fileset": "default/cord-lora", "finetuning_type": "lora",
        "lora_config": Lora(rank=16, alpha=32), "model": "default/bolt",
    }  # fmt: skip
    return Adapter(**{**fields, **overrides})


def _platform(base: ModelEntity, adapter: Adapter | None) -> AsyncCustomizationHelixClients:
    def found(entity: Any) -> MagicMock:
        return MagicMock(data=MagicMock(return_value=entity))

    async def get_adapter(name: str, workspace: str) -> MagicMock:
        if adapter is None or (workspace, name) != (adapter.workspace, adapter.name):
            raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://test")))
        return found(adapter)

    models = AsyncMock()
    models.get_model.return_value = found(base)
    models.get_adapter.side_effect = get_adapter
    page = MagicMock()
    page.items.return_value.__aiter__.return_value = []
    return AsyncCustomizationHelixClients(
        files=AsyncMock(), models=models, jobs=MagicMock(list_jobs=AsyncMock(return_value=page))
    )


async def _compile(base: ModelEntity | None = None, adapter: Adapter | None = None, **spec: Any):
    body = {"model": "bolt", "adapter": "cord-lora", "output": {"name": "bolt-cord"}, **spec}
    platform = _platform(base or _model(), adapter or _adapter())
    return await compile_merge("default", AutomodelMergeSpec.model_validate(body), platform, "gpu")


@pytest.mark.asyncio
async def test_compiles_download_merge_upload_and_register_steps() -> None:
    download, merge, upload, entity = (await _compile(num_gpus=2)).steps

    assert [item["dest"] for item in download.config["download"]] == [DEFAULT_MODEL_PATH, ADAPTER_PATH]
    assert [item["src"]["name"] for item in download.config["download"]] == ["bolt-weights", "cord-lora"]
    assert merge.executor.resources.num_gpus == 2
    assert merge.lifecycle.staleness_timeout_seconds == 0  # it reports no progress
    assert upload.config["upload"][0]["dest"]["name"] == "bolt-cord"
    assert entity.config["base_model"] == "default/bolt"
    assert entity.config["peft"] == {"type": "lora_merged", "rank": 16, "alpha": 32}


@pytest.mark.asyncio
async def test_a_vision_checkpoint_is_merged_with_the_image_text_class_and_its_chat_template() -> None:
    base = _model(trust_remote_code=True, spec=ModelSpec(**SPEC_FIELDS, has_vision_tower=True, chat_template="{{ m }}"))

    config = MergeAdapterStepConfig.model_validate((await _compile(base=base)).steps[1].config)

    assert (config.has_vision_tower, config.trust_remote_code, config.chat_template) == (True, True, "{{ m }}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"adapter": _adapter(model="default/other")}, "trained on 'default/other'"),
        ({"adapter": _adapter(name="other")}, "not found"),
        ({"adapter": _adapter(lora_config=None)}, "lora_config"),
        ({"output": {"name": "bolt"}}, "would overwrite"),
        ({"output": {"name": "cord-lora"}}, "would overwrite"),
        ({"base": _model(spec=ModelSpec(**SPEC_FIELDS, head_type="embedding"))}, "not a causal"),
        ({"base": _model(fileset=None)}, "has no fileset"),
    ],
)
async def test_rejects_a_merge_that_cannot_work(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(HelixJobCompilationError, match=message):
        await _compile(**kwargs)


def test_merge_jobs_are_listed_from_their_own_collection() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        page = {"page": 1, "page_size": 10, "current_page_size": 0, "total_pages": 0, "total_results": 0}
        return httpx.Response(200, json={"data": [], "pagination": page})

    client = NemoClient(
        base_url="http://x", workspace="default", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    namespace = AutomodelCustomization(make_customization_sdk_context(client))

    namespace.merge_jobs.list(workspace="ws")
    namespace.jobs.list(workspace="ws")

    assert paths == [
        "/apis/customization/v2/workspaces/ws/automodel-merge/jobs",
        "/apis/customization/v2/workspaces/ws/automodel/jobs",
    ]
