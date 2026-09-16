# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_agent_optimization_plugin.schemas.optimize import (
    AgentOptimizeSpec,
    OptimizeSpec,
    OptimizeSubmitSpec,
)
from pydantic import ValidationError


def _valid() -> dict:
    return {
        "agent": "my-ws/my-agent",
        "optimize_config_fileset": "my-ws/bundle",
        "optimize_config": "configs/optimize.yaml",
        "output_agent": "my-agent-opt",
    }


def test_agent_optimize_spec_accepts_a_minimal_valid_payload() -> None:
    spec = AgentOptimizeSpec.model_validate(_valid())
    assert spec.agent == "my-ws/my-agent"
    assert spec.workspace == "default"


@pytest.mark.parametrize("missing", ["agent", "optimize_config_fileset", "optimize_config", "output_agent"])
def test_every_normalized_field_is_required(missing: str) -> None:
    payload = _valid()
    del payload[missing]
    with pytest.raises(ValidationError):
        AgentOptimizeSpec.model_validate(payload)


@pytest.mark.parametrize("bad", ["/abs/optimize.yaml", "../escape.yaml", "~/optimize.yaml", "D:opt.yml"])
def test_config_must_stay_inside_the_bundle(bad: str) -> None:
    payload = _valid() | {"optimize_config": bad}
    with pytest.raises(ValidationError, match="relative to the fileset root"):
        AgentOptimizeSpec.model_validate(payload)


def test_config_fileset_must_be_an_entity_ref() -> None:
    payload = _valid() | {"optimize_config_fileset": "not/a/valid/ref"}
    with pytest.raises(ValidationError, match="must be 'name' or 'workspace/name'"):
        AgentOptimizeSpec.model_validate(payload)


def test_optimize_spec_adds_strategy() -> None:
    spec = OptimizeSpec.model_validate(_valid() | {"strategy": "nat"})
    assert spec.strategy == "nat"


def test_submit_spec_has_no_workspace_field() -> None:
    assert "workspace" not in OptimizeSubmitSpec.model_fields
