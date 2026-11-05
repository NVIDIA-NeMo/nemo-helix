# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType
from typing import Any

import pytest
from nemo_agents_plugin.fabric.gateway_models import PLATFORM_IGW_API_KEY_ENV, gateway_models, is_gateway_url

IGW = "http://platform/apis/inference-gateway/v2/workspaces/ws"


def _config(**models: dict[str, Any]) -> dict[str, Any]:
    return {"models": models}


def test_shared_and_harness_models_on_the_gateway_are_found_in_config_order() -> None:
    config = {
        "models": {
            "default": {"model": "a", "base_url": f"{IGW}/openai/-/v1"},
            "direct": {"model": "b", "base_url": "https://provider.test/v1"},
        },
        "harnesses": {"codex": {"model": {"model": "c", "base_url": f"{IGW}/openai/-/v1", "api_key_env": "CODEX_KEY"}}},
    }

    found = gateway_models(config)

    assert [(model.virtual_model, model.api_key_env) for model in found] == [
        ("ws/a", PLATFORM_IGW_API_KEY_ENV),
        ("ws/c", "CODEX_KEY"),
    ]
    assert found[1].config is config["harnesses"]["codex"]["model"]


@pytest.mark.parametrize(
    ("model", "virtual_model"),
    [
        pytest.param({"model": "vm", "base_url": f"{IGW}/openai/-/v1"}, "ws/vm", id="openai-route-body-model"),
        pytest.param({"model": "other/vm", "base_url": f"{IGW}/openai/-/v1"}, "other/vm", id="qualified-body-model"),
        pytest.param({"model": "ignored", "base_url": f"{IGW}/model/routed/-/v1"}, "ws/routed", id="model-route"),
        pytest.param({"model": "x", "base_url": f"{IGW}/provider/stub/-/v1"}, None, id="provider-route"),
    ],
)
def test_the_virtual_model_comes_from_the_route(model: dict[str, Any], virtual_model: str | None) -> None:
    [found] = gateway_models(_config(default=model))

    assert found.virtual_model == virtual_model


@pytest.mark.parametrize(
    "model",
    [
        pytest.param({"model": "vm", "settings": {"base_url": f"{IGW}/openai/-/v1"}}, id="legacy-settings"),
        pytest.param(
            {"model": "vm", "base_url": "", "settings": {"base_url": f"{IGW}/openai/-/v1"}}, id="empty-base-url"
        ),
    ],
)
def test_a_gateway_url_is_found_where_the_runtime_reads_it(model: dict[str, Any]) -> None:
    [found] = gateway_models(_config(default=model))

    assert found.virtual_model == "ws/vm"


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("https://helix.test/nemo/apis/inference-gateway/v2/workspaces/ws/openai/-/v1", id="path-prefix"),
    ],
)
def test_a_gateway_route_behind_a_path_prefix_is_not_a_gateway_url(base_url: str) -> None:
    """Prefixed ingress is unsupported: the job proxy would forward the prefix on top of ``NHX_BASE_URL``'s."""
    assert not is_gateway_url(base_url)
    assert gateway_models(_config(default={"model": "vm", "base_url": base_url})) == []


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_config(default={"model": "x", "base_url": "https://provider.test/v1"}), id="direct"),
        pytest.param(
            _config(default={"model": "x", "base_url": "https://evil.test/v1?to=/apis/inference-gateway/"}),
            id="marker-in-query",
        ),
        pytest.param({"models": "not-a-mapping", "harnesses": {"h": "not-a-mapping"}}, id="malformed"),
    ],
)
def test_models_off_the_gateway_are_not_found(config: dict[str, Any]) -> None:
    assert gateway_models(config) == []


def test_read_only_mappings_are_read_too() -> None:
    model = MappingProxyType({"model": "vm", "base_url": f"{IGW}/openai/-/v1"})

    [found] = gateway_models(MappingProxyType({"models": MappingProxyType({"default": model})}))

    assert found.virtual_model == "ws/vm"
    assert found.config is model
