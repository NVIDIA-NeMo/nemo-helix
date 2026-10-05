# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nemo_agents_plugin.refs`.

Covers the agent-specific ref types (:class:`AgentRef` and the
:data:`AgentTarget` union it forms with
:class:`~nemo_helix_plugin.refs.EndpointURL`) and the shape-based
:func:`classify_agent_target` dispatch the ``EvaluateAgentJob`` uses to
decide between a NeMo Helix agent reference and a literal endpoint URL.

The generic ref machinery (``StrRef``, ``LocalDir``, ``FilesetRef``,
``classify_output_target``) is tested in
``packages/nemo_helix_plugin/tests/test_refs.py`` so this suite stays focused
on what the agents plugin layers on top.
"""

from __future__ import annotations

import pytest
from nemo_agents_plugin.refs import (
    AgentRef,
    AgentTarget,
    EndpointURL,
    classify_agent_target,
)
from pydantic import BaseModel


class TestAgentRefSubclass:
    """``AgentRef`` is a thin :class:`StrRef` subclass with an opt-in metavar."""

    def test_agent_ref_is_a_str(self) -> None:
        assert isinstance(AgentRef("calculator"), str)

    def test_subclass_identity_preserved(self) -> None:
        """``isinstance`` checks let resolvers branch when the call site went explicit."""
        assert isinstance(AgentRef("x"), AgentRef)
        assert not isinstance(AgentRef("x"), EndpointURL)

    def test_equality_with_plain_str(self) -> None:
        assert AgentRef("calculator") == "calculator"
        assert "calculator" == AgentRef("calculator")

    def test_advertises_cli_metavar(self) -> None:
        assert AgentRef.__cli_metavar__ == "AGENT_REF"


class TestClassifyAgentTarget:
    """Shape-based dispatch — ``"://"`` is the canonical URL marker."""

    @pytest.mark.parametrize(
        "value",
        [
            "http://localhost:8080",
            "https://api.example.com/v1/agent",
            "http://127.0.0.1:8080/apis/agents/v2/workspaces/default/agents/calc/-",
        ],
    )
    def test_urls_classify_as_endpoint(self, value: str) -> None:
        assert classify_agent_target(value) is EndpointURL

    @pytest.mark.parametrize(
        "value",
        [
            "calculator",
            "default/calculator",
            "agents/default/calculator",
            "calculator-v2",
        ],
    )
    def test_names_classify_as_agent_ref(self, value: str) -> None:
        assert classify_agent_target(value) is AgentRef


class TestPydanticIntegration:
    """The :class:`StrRef` hook lets Pydantic accept any string for an ``AgentTarget`` field.

    The disambiguation is intentionally *not* validated at the model
    layer — both arms of :data:`AgentTarget` accept any string value.
    Resolvers (here, the ``EvaluateAgentJob``) inspect the shape at run
    time when they have the runtime context to give an actionable error.
    """

    def test_union_field_accepts_url(self) -> None:
        class _M(BaseModel):
            target: AgentTarget | None = None

        # ``ty`` doesn't see through the StrRef Pydantic core schema that
        # accepts a bare ``str`` and wraps it in the subclass; the runtime
        # assertion below is the actual behaviour we care about.
        assert _M(target="http://localhost:8080").target == "http://localhost:8080"  # ty: ignore[invalid-argument-type]

    def test_union_field_accepts_bare_name(self) -> None:
        class _M(BaseModel):
            target: AgentTarget | None = None

        assert _M(target="calculator").target == "calculator"  # ty: ignore[invalid-argument-type]

    def test_union_field_accepts_none(self) -> None:
        class _M(BaseModel):
            target: AgentTarget | None = None

        assert _M().target is None
