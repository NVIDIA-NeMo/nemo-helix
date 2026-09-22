# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import httpx
import pytest
from nemo_optimization.preflight import preflight_validate_llm_models
from nemo_platform_plugin.client.client import NemoClient

_VIRTUAL_MODELS_PATH = re.compile(
    r"^/apis/inference-gateway/v2/workspaces/(?P<workspace>[^/]+)/virtual-models/(?P<name>[^/]+)$"
)


class _RecordingVirtualModels:
    """In-memory VirtualModels API: records each ``GET`` and answers 404 for ``missing`` names."""

    def __init__(self, *, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.calls: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        match = _VIRTUAL_MODELS_PATH.match(request.url.path)
        assert request.method == "GET" and match is not None, f"unexpected request {request.method} {request.url}"
        name, workspace = match["name"], match["workspace"]
        self.calls.append({"name": name, "workspace": workspace})
        if name in self.missing:
            return httpx.Response(404, json={"detail": f"VirtualModel {name!r} not found"})
        return httpx.Response(200, json={"name": name, "workspace": workspace})


def _StubSDK(virtual_models: _RecordingVirtualModels) -> NemoClient:
    return NemoClient(
        base_url="http://test",
        http_client=httpx.Client(transport=httpx.MockTransport(virtual_models), base_url="http://test"),
    )


def test_preflight_noop_without_sdk() -> None:
    preflight_validate_llm_models(
        {"models": {"default": {"provider": "nvidia", "model": "x"}}}, workspace="ws", sdk=None
    )


def test_preflight_validates_fabric_models_without_base_url() -> None:
    vms = _RecordingVirtualModels()
    sdk = _StubSDK(vms)
    preflight_validate_llm_models(
        {
            "models": {
                "default": {"provider": "nvidia", "model": "demo-model"},
                "judge": {"provider": "openai", "model": "demo-model"},
            }
        },
        workspace="ws",
        sdk=sdk,
    )
    assert vms.calls == [{"name": "demo-model", "workspace": "ws"}]


def test_preflight_skips_fabric_models_with_external_base_url() -> None:
    vms = _RecordingVirtualModels()
    sdk = _StubSDK(vms)
    preflight_validate_llm_models(
        {
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/meta/llama-3.1-8b-instruct",
                    "base_url": "https://inference-api.nvidia.com/v1",
                }
            }
        },
        workspace="ws",
        sdk=sdk,
    )
    assert vms.calls == []


def test_preflight_still_validates_legacy_llms() -> None:
    vms = _RecordingVirtualModels(missing={"missing-model"})
    sdk = _StubSDK(vms)
    with pytest.raises(ValueError, match="models.default.model|llms.agent.model_name|missing-model"):
        preflight_validate_llm_models(
            {"llms": {"agent": {"_type": "openai", "model_name": "missing-model"}}},
            workspace="ws",
            sdk=sdk,
        )


def test_preflight_merges_agent_config_models() -> None:
    vms = _RecordingVirtualModels()
    sdk = _StubSDK(vms)
    preflight_validate_llm_models(
        {"optimizer": {"numeric": {"enabled": True}}},
        workspace="ws",
        sdk=sdk,
        agent_config={"models": {"default": {"provider": "openai", "model": "agent-model"}}},
    )
    assert vms.calls == [{"name": "agent-model", "workspace": "ws"}]


def test_preflight_reports_missing_fabric_model() -> None:
    vms = _RecordingVirtualModels(missing={"gone"})
    sdk = _StubSDK(vms)
    with pytest.raises(ValueError, match=r"gone.*optimize_config\.models\.default\.model"):
        preflight_validate_llm_models(
            {"models": {"default": {"provider": "nvidia", "model": "gone"}}},
            workspace="ws",
            sdk=sdk,
        )
