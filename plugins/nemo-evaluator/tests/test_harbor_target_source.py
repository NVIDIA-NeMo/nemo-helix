# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``HarborRunnerTarget.source``: one agent per target, selected by name or import path."""

from __future__ import annotations

import pytest
from nemo_evaluator.jobs.agent_spec import (
    HarborBuiltinAgentSource,
    HarborImportedAgentSource,
    HarborRunnerTarget,
    target_agent_identity,
)
from pydantic import ValidationError


def test_a_harbor_source_is_exactly_one_agent() -> None:
    """The wire type cannot name two agents at once, or none; the schema, not a validator, says so."""
    assert HarborRunnerTarget().source == HarborBuiltinAgentSource(name="oracle")
    for source in (
        {},
        {"name": "oracle", "import_path": "x:Y"},
        {"model_name": "m"},
        {"name": "oracle", "agent": "calc"},
    ):
        with pytest.raises(ValidationError):
            HarborRunnerTarget.model_validate({"source": source})
    with pytest.raises(ValidationError, match="agent_name"):
        HarborRunnerTarget.model_validate({"source": {"name": "oracle"}, "agent_name": "codex"})


def test_the_harbor_source_implies_what_harbor_is_handed() -> None:
    """The runtime translation reads three agent settings; each source shape fills them its own way."""
    builtin = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model_name="m"))
    imported = HarborRunnerTarget(source=HarborImportedAgentSource(import_path="x:Y", model_name="m"))
    assert (builtin.agent_name, builtin.agent_import_path, builtin.agent_model_name) == ("codex", None, "m")
    assert (imported.agent_name, imported.agent_import_path, imported.agent_model_name) == (None, "x:Y", "m")
    assert target_agent_identity(builtin) == ("codex", "m")
    assert target_agent_identity(imported) == ("x:Y", "m")


def test_the_legacy_flat_harbor_agent_fields_are_lifted_into_source_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One release of tolerance, with the precedence the flat fields had: an import path over a built-in name."""
    cases = [
        ({"agent_name": "codex", "agent_model_name": "m"}, HarborBuiltinAgentSource(name="codex", model_name="m")),
        ({"agent_name": "codex", "agent_import_path": "x:Y"}, HarborImportedAgentSource(import_path="x:Y")),
        ({"agent_model_name": "m"}, HarborBuiltinAgentSource(name="oracle", model_name="m")),
        ({"agent_name": None}, HarborBuiltinAgentSource(name="oracle")),
    ]
    for legacy, source in cases:
        with caplog.at_level("WARNING", logger="nemo_evaluator.jobs.agent_spec"):
            target = HarborRunnerTarget.model_validate({"kind": "harbor", "reward_key": "r", **legacy})
        assert target.source == source, legacy
        assert target.reward_key == "r"
    assert caplog.text.count("deprecated") == len(cases)
    assert "source" in HarborRunnerTarget.model_fields and "agent_name" not in HarborRunnerTarget.model_fields
