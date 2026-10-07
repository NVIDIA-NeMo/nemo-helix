# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GymRunnerTarget.source``: the Gym agent as one field -- component, config, instance."""

from __future__ import annotations

import pytest
from nemo_evaluator.jobs.agent_spec import GymAgentSource, GymPlacement, GymRunnerTarget, target_agent_identity
from nemo_evaluator.jobs.runner_targets import runner_to_target
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from pydantic import ValidationError

_CONFIG = "responses_api_agents/simple_agent/configs/simple_agent.yaml"


def test_the_source_is_required_and_names_one_component() -> None:
    with pytest.raises(ValidationError):
        GymRunnerTarget.model_validate({"resources_server": "mcqa"})
    with pytest.raises(ValidationError):
        GymRunnerTarget.model_validate({"source": {"config": _CONFIG}, "resources_server": "mcqa"})
    with pytest.raises(ValidationError, match="agent_ref_name"):
        GymRunnerTarget.model_validate(
            {
                "source": {"component": "simple_agent", "config": _CONFIG},
                "resources_server": "mcqa",
                "agent_ref_name": "x",
            }
        )


def test_the_source_implies_what_gym_is_handed() -> None:
    """The runtime translation and the sandbox plan read three settings; the source fills them."""
    target = GymRunnerTarget(
        source=GymAgentSource(component="langgraph_agent", config=_CONFIG, instance="rewoo_agent"),
        resources_server="mcqa",
    )
    assert (target.agent, target.agent_config, target.agent_ref_name) == ("langgraph_agent", _CONFIG, "rewoo_agent")
    assert target_agent_identity(target) == ("langgraph_agent", None)
    with pytest.raises(ValidationError, match="source.config"):
        GymRunnerTarget(source=GymAgentSource(component="simple_agent"), resources_server="mcqa")


def test_the_legacy_flat_gym_agent_fields_are_lifted_into_source_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One release of tolerance for specs and persisted jobs written before `source` existed."""
    legacy = {
        "kind": "gym",
        "agent": "simple_agent",
        "agent_config": _CONFIG,
        "agent_ref_name": "x",
        "resources_server": "mcqa",
    }
    with caplog.at_level("WARNING", logger="nemo_evaluator.jobs.agent_spec"):
        target = GymRunnerTarget.model_validate(legacy)
    assert target.source == GymAgentSource(component="simple_agent", config=_CONFIG, instance="x")
    assert "deprecated" in caplog.text
    assert "agent" not in GymRunnerTarget.model_fields and "source" in GymRunnerTarget.model_fields


def test_a_live_runner_and_its_placement_become_one_source(tmp_path) -> None:
    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(agent="langgraph_agent", agent_config=_CONFIG, resources_server="mcqa")
    )
    target = runner_to_target(runner, GymPlacement(agent_ref_name="rewoo_agent"))
    assert isinstance(target, GymRunnerTarget)
    assert target.source == GymAgentSource(component="langgraph_agent", config=_CONFIG, instance="rewoo_agent")
