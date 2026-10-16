# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deployment params shared by every customization backend."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.deployment import (
    DeploymentParams,
    ToolCallParams,
    reject_lora_without_lora_enabled,
)
from pydantic import ValidationError


class TestDeploymentParams:
    def test_defaults(self) -> None:
        params = DeploymentParams()

        assert params.engine == "nim"
        assert params.gpu == 1
        assert params.lora_enabled is True
        assert params.tool_call_config is None

    @pytest.mark.parametrize("gpu", [0, -1])
    def test_rejects_non_positive_gpu(self, gpu: int) -> None:
        """The task-side contract requires gpu > 0; catch it at the submit boundary."""
        with pytest.raises(ValidationError, match="greater than 0"):
            DeploymentParams(gpu=gpu)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            DeploymentParams.model_validate({"gpu": 1, "junk": True})

    def test_accepts_vllm_engine(self) -> None:
        assert DeploymentParams(engine="vllm").engine == "vllm"

    @pytest.mark.parametrize("engine", ["generic", "trtllm"])
    def test_rejects_engines_other_than_vllm_and_nim(self, engine: str) -> None:
        """generic needs an explicit image and has no auto-deploy coverage, so it is not offered."""
        with pytest.raises(ValidationError):
            DeploymentParams.model_validate({"engine": engine})

    def test_tool_call_config_requires_nim(self) -> None:
        """The vLLM compiler ignores tool_call_config, so it is rejected rather than dropped."""
        tool_call_config = ToolCallParams(tool_call_parser="hermes")
        with pytest.raises(ValidationError, match="only supported with engine='nim'"):
            DeploymentParams(engine="vllm", tool_call_config=tool_call_config)

        assert DeploymentParams(tool_call_config=tool_call_config).tool_call_config == tool_call_config

    def test_tool_call_plugin_must_be_a_workspace_qualified_ref(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallParams(tool_call_plugin="no-workspace")

        assert ToolCallParams(tool_call_plugin="default/my-plugin").tool_call_plugin == "default/my-plugin"


class TestRejectLoraWithoutLoraEnabled:
    def test_raises_for_a_lora_job_with_lora_enabled_false(self) -> None:
        with pytest.raises(ValueError, match="lora_enabled must be true"):
            reject_lora_without_lora_enabled(DeploymentParams(lora_enabled=False), trains_lora_adapter=True)

    @pytest.mark.parametrize("engine", ["vllm", "nim"])
    def test_raises_for_either_engine(self, engine: str) -> None:
        with pytest.raises(ValueError, match="lora_enabled must be true"):
            reject_lora_without_lora_enabled(
                DeploymentParams.model_validate({"engine": engine, "lora_enabled": False}), trains_lora_adapter=True
            )

    def test_allows_a_lora_job_with_lora_enabled_true(self) -> None:
        reject_lora_without_lora_enabled(DeploymentParams(lora_enabled=True), trains_lora_adapter=True)

    def test_allows_a_full_weight_job_with_lora_enabled_false(self) -> None:
        reject_lora_without_lora_enabled(DeploymentParams(lora_enabled=False), trains_lora_adapter=False)

    def test_ignores_none(self) -> None:
        reject_lora_without_lora_enabled(None, trains_lora_adapter=True)

    def test_ignores_string_refs(self) -> None:
        """Resolving a ref needs a platform client, so the compiler checks those."""
        reject_lora_without_lora_enabled("workspace/some-config", trains_lora_adapter=True)
