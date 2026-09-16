# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_helix import NotFoundError
from nemo_optimization.preflight import preflight_validate_llm_models


class _StubResponse:
    status_code = 404
    headers: dict[str, str] = {}
    request = None


def _not_found(message: str) -> NotFoundError:
    return NotFoundError(message=message, response=_StubResponse(), body={"detail": message})  # type: ignore[arg-type]


class _RecordingVirtualModels:
    def __init__(self, *, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.calls: list[dict[str, str]] = []

    def retrieve(self, *, name: str, workspace: str) -> object:
        self.calls.append({"name": name, "workspace": workspace})
        if name in self.missing:
            raise _not_found(f"VirtualModel {name!r} not found")
        return object()


class _StubSDK:
    def __init__(self, virtual_models: _RecordingVirtualModels) -> None:
        self.inference = type("Inference", (), {"virtual_models": virtual_models})()


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
        sdk=sdk,  # type: ignore[arg-type]
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
        sdk=sdk,  # type: ignore[arg-type]
    )
    assert vms.calls == []


def test_preflight_still_validates_legacy_llms() -> None:
    vms = _RecordingVirtualModels(missing={"missing-model"})
    sdk = _StubSDK(vms)
    with pytest.raises(ValueError, match="models.default.model|llms.agent.model_name|missing-model"):
        preflight_validate_llm_models(
            {"llms": {"agent": {"_type": "openai", "model_name": "missing-model"}}},
            workspace="ws",
            sdk=sdk,  # type: ignore[arg-type]
        )


def test_preflight_merges_agent_config_models() -> None:
    vms = _RecordingVirtualModels()
    sdk = _StubSDK(vms)
    preflight_validate_llm_models(
        {"optimizer": {"numeric": {"enabled": True}}},
        workspace="ws",
        sdk=sdk,  # type: ignore[arg-type]
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
            sdk=sdk,  # type: ignore[arg-type]
        )
