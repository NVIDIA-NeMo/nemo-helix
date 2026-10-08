# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GymRunnerTarget.source``: the Gym agent as one field -- component, config, instance."""

from __future__ import annotations

import pytest
from nemo_evals.api.schemas import AgentRef
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_spec import (
    REGISTERED_AGENT_GYM_COMPONENT,
    AgentEvalInputSpec,
    AgentEvalSpec,
    GymAgentSource,
    GymPlacement,
    GymRunnerTarget,
    RegisteredAgentSource,
    registered_agent_config,
    registered_agent_files,
    registered_agent_source,
    target_agent_identity,
)
from nemo_evals.jobs.runner_targets import runner_to_target
from nhx_evals_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
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
    with caplog.at_level("WARNING", logger="nemo_evals.jobs.agent_spec"):
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


_RESOLVED_TASK = {
    "id": "t",
    "spec": {"kind": "evaluator", "intent": "x", "inputs": {}, "metrics": []},
}


def test_a_registered_agent_is_a_gym_source_and_excludes_the_gym_agent_keys() -> None:
    """The union admits a registered `agent` next to a Gym `component`, never both, and no `model_name`."""
    for source in (
        {"component": "simple_agent", "agent": "calc"},
        {"agent": "calc", "config": _CONFIG},
        {"agent": "calc", "model_name": "m"},
    ):
        with pytest.raises(ValidationError):
            GymRunnerTarget.model_validate({"source": source, "resources_server": "mcqa"})
    target = GymRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="dev/calculator-agent")), resources_server="mcqa"
    )
    assert target.agent_config is not None  # no `source.config` needed: the staging step generates it


def test_a_registered_gym_source_implies_the_platform_component_and_a_generated_instance() -> None:
    target = GymRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="dev/Calc-Agent.v2")), resources_server="mcqa"
    )
    assert target.agent == REGISTERED_AGENT_GYM_COMPONENT
    assert target.agent_ref_name == "registered_calc_agent_v2"
    assert (
        target.agent_config
        == f"responses_api_agents/{REGISTERED_AGENT_GYM_COMPONENT}/configs/registered_calc_agent_v2.yaml"
    )
    assert target_agent_identity(target) == ("Calc-Agent.v2", None)
    assert registered_agent_source(target) is target.source


def test_resolved_config_is_derived_never_submitted_and_required_to_run() -> None:
    with pytest.raises(ValidationError, match="a Gym agent needs none"):
        GymRunnerTarget(
            source=GymAgentSource(component="a", config=_CONFIG), resources_server="mcqa", resolved_config={}
        )
    tasks = [{"id": "t", "intent": "x", "inputs": {}}]
    submitted = {
        "kind": "gym",
        "source": {"agent": "calc"},
        "resources_server": "mcqa",
        "resolved_config": {"harness": {}},
    }
    with pytest.raises(ValidationError, match="not the submitter"):
        AgentEvalInputSpec.model_validate({"tasks": tasks, "target": submitted})
    with pytest.raises(ValidationError, match="must be resolved before run"):
        AgentEvalSpec.model_validate(
            {
                "tasks": [_RESOLVED_TASK],
                "target": {"kind": "gym", "source": {"agent": "calc"}, "resources_server": "mcqa"},
            }
        )
    resolved = GymRunnerTarget(
        source=RegisteredAgentSource(
            agent=AgentRef(root="dev/calc"), files=FilesetRef(root="dev/agent-files-0123abcd4567")
        ),
        resources_server="mcqa",
        resolved_config={"harness": {"adapter_id": "x"}, "skills": {"paths": ["skills/a"]}},
    )
    AgentEvalSpec.model_validate({"tasks": [_RESOLVED_TASK], "target": resolved.model_dump(mode="json")})
    assert registered_agent_config(resolved) == resolved.resolved_config
    assert registered_agent_files(resolved) == FilesetRef(root="dev/agent-files-0123abcd4567")
