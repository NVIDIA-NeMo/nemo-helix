# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nhx_evals_sdk.execution.config."""

from __future__ import annotations

import pytest
from nhx_evals_sdk.enums import AgentFormat
from nhx_evals_sdk.execution.config import resolve_params
from nhx_evals_sdk.values import (
    Agent,
    GenericAgent,
    Model,
    Retrieval,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)


class TestResolveParams:
    """Coverage for run parameter validation and defaulting."""

    @pytest.mark.parametrize(
        ("target", "expected_type"),
        [
            pytest.param(Model(url="http://example.test/v1", name="test-model"), RunConfigOnlineModel, id="model"),
            pytest.param(
                GenericAgent(
                    url="http://agent.test",
                    name="test-agent",
                    format=AgentFormat.GENERIC,
                    body={"query": "{{ prompt }}"},
                    response_path="$.answer",
                ),
                RunConfigOnline,
                id="agent",
            ),
        ],
    )
    def test_rejects_missing_target_specific_params(self, target: Model | Agent, expected_type: type[object]) -> None:
        """Targeted evaluation should require callers to choose the matching params type."""
        del expected_type

        with pytest.raises(TypeError):
            resolve_params(target=target)

    def test_defaults_offline_params(self) -> None:
        """Offline evaluation may omit params and use default RunConfig."""
        assert resolve_params() == RunConfig()

    def test_accepts_model_online_params(self) -> None:
        """Model targets require RunConfigOnlineModel."""
        target = Model(url="http://example.test/v1", name="test-model")
        params = RunConfigOnlineModel(ignore_request_failure=True)

        assert resolve_params(params=params, target=target) is params

    def test_converts_generic_online_params_for_model_target(self) -> None:
        """Model targets accept generic online params from JSON specs."""
        target = Model(url="http://example.test/v1", name="test-model")
        params = RunConfigOnline(parallelism=3, ignore_request_failure=True)

        resolved = resolve_params(params=params, target=target)

        assert isinstance(resolved, RunConfigOnlineModel)
        assert resolved.parallelism == 3
        assert resolved.ignore_request_failure is True

    def test_accepts_agent_online_params(self) -> None:
        """Agent targets require RunConfigOnline-compatible params."""
        target = GenericAgent(
            url="http://agent.test",
            name="test-agent",
            format=AgentFormat.GENERIC,
            body={"query": "{{ prompt }}"},
            response_path="$.answer",
        )
        params = RunConfigOnline(ignore_request_failure=True)

        assert resolve_params(params=params, target=target) is params

    def test_rejects_model_online_params_for_agent(self) -> None:
        """Agent targets should not accept model-only online params."""
        target = GenericAgent(
            url="http://agent.test",
            name="test-agent",
            format=AgentFormat.GENERIC,
            body={"query": "{{ prompt }}"},
            response_path="$.answer",
        )

        with pytest.raises(TypeError, match="agent target requires RunConfigOnline"):
            resolve_params(params=RunConfigOnlineModel(), target=target)

    def test_rejects_online_params_without_target(self) -> None:
        """Offline evaluation should use offline params rather than silently accepting online settings."""
        with pytest.raises(TypeError, match="offline evaluation requires RunConfig"):
            resolve_params(params=RunConfigOnline(ignore_request_failure=True))

    def test_defaults_retrieval_params_to_online_model(self) -> None:
        """Retrieval targets default to RunConfigOnlineModel when params are omitted."""
        target = Retrieval(embeddings=Model(url="http://example.test/v1", name="embed"))

        assert isinstance(resolve_params(target=target), RunConfigOnlineModel)
