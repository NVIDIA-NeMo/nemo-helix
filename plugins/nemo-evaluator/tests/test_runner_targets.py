# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describing a live runner as the target spec that reproduces it as a job."""

from __future__ import annotations

import pytest
from nemo_evaluator.api.fields import TasksetRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, GymPlacement, GymRunnerTarget
from nemo_evaluator.jobs.runner_targets import UnsubmittableRunnerError, runner_to_target
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from nemo_evaluator_sdk.values import SecretRef
from pydantic import ValidationError

#: Target fields a runtime config cannot supply, so a round-trip cannot check them here: ``kind``
#: discriminates the target union, and the other two come from the ``GymPlacement``.
WIRE_ONLY_TARGET_FIELDS = {"kind", "environment", "agent_ref_name"}


def _configured() -> GymRuntimeConfig:
    """A config with every field set away from its default.

    Defaults would let a dropped field pass by coincidence — the target would carry the same value
    the runner had, for the wrong reason.
    """
    return GymRuntimeConfig(
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="gdpval",
        model_type="openai_model",
        bind_resources_server=False,
        hydra_params={"simple_agent": {"responses_api_agents": {"x": 1}}},
        env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/shared/cache"},
        env_secrets={"NVIDIA_API_KEY": SecretRef("evals/nvidia-api-key")},
        num_repeats=3,
        concurrency=7,
        startup_timeout_s=1800.0,
        collection_timeout_s=3600.0,
        shutdown_grace_s=45.0,
        reward_key="score",
    )


def test_a_gym_runner_describes_itself_as_a_submittable_target() -> None:
    # The point of the conversion: an evaluation someone got working locally becomes a job spec
    # without retyping its configuration — which is where the subtle divergences come from.
    config = _configured()

    target = runner_to_target(GymAgentTaskRunner(config=config))

    assert isinstance(target, GymRunnerTarget)
    assert target.kind == "gym"
    # Every runtime field arrives, unchanged. What is excluded is the set with no runtime
    # counterpart, listed in one place so a new wire-only field is a deliberate addition here
    # rather than a puzzling failure.
    assert target.model_dump(exclude=WIRE_ONLY_TARGET_FIELDS) == config.model_dump()


def test_every_field_actually_travels_rather_than_defaulting() -> None:
    # Guards the failure this conversion exists to prevent: a field silently dropped, so the
    # submitted job runs something different from what was tested. Asserting values differ from
    # their defaults is what makes the round-trip above meaningful.
    config = _configured()
    defaults = {
        name: field.get_default(call_default_factory=True) for name, field in GymRuntimeConfig.model_fields.items()
    }
    carried = runner_to_target(GymAgentTaskRunner(config=config)).model_dump(exclude=WIRE_ONLY_TARGET_FIELDS)

    indistinguishable = [name for name, value in carried.items() if value == defaults.get(name)]
    assert not indistinguishable, (
        f"these fields match their defaults, so carrying them is untested: {indistinguishable}"
    )


def test_the_config_read_back_is_the_one_the_runner_holds() -> None:
    # The conversion reads `runner.config`, not `runner_info()`, because the latter redacts
    # credential-shaped values — rebuilding from it would submit `<redacted>` as a real setting.
    config = GymRuntimeConfig(
        agent="a",
        agent_config="c",
        resources_server="r",
        env_vars={"OPENAI_API_KEY": "sk-real-value", "HTTPS_PROXY": "http://proxy:8080"},
    )
    runner = GymAgentTaskRunner(config=config)

    target = runner_to_target(runner)

    assert runner.runner_info().config["env_vars"]["OPENAI_API_KEY"] == "<redacted>"
    assert isinstance(target, GymRunnerTarget)
    assert target.env_vars["OPENAI_API_KEY"] == "sk-real-value"


def test_configuration_with_no_json_form_is_refused_rather_than_failing_at_submit() -> None:
    """`hydra_params` and `env_vars` are typed loosely enough to hold anything.

    A callable survives `GymRuntimeConfig` construction and only fails inside
    `model_dump(mode="json")` when the spec is posted — a `PydanticSerializationError` raised from
    the transport, naming neither the runner nor the field. This module promises to refuse what
    cannot travel, so the refusal has to happen here.
    """
    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="a",
            agent_config="c",
            resources_server="r",
            hydra_params={"callback": lambda value: value},
        )
    )

    with pytest.raises(UnsubmittableRunnerError) as excinfo:
        runner_to_target(runner)

    message = str(excinfo.value)
    assert "JSON" in message, "the message should say what is wrong with the value"
    assert "hydra_params" in message, "and name the fields that are free-form"


def test_an_unsupported_runner_is_refused_by_name() -> None:
    # Only Gym has a target today. A runner with no wire form must say so rather than produce a
    # spec that silently runs something else.
    class _Unsupported:
        def runner_info(self):  # pragma: no cover - never reached
            raise NotImplementedError

        async def run_tasks(self, tasks, config=None):  # pragma: no cover - never reached
            return []

    with pytest.raises(UnsubmittableRunnerError) as excinfo:
        runner_to_target(_Unsupported())

    message = str(excinfo.value)
    assert "_Unsupported" in message, "the message must name the runner that could not be converted"
    # Points at the alternative rather than dead-ending.
    assert "AgentEvaluator()" in message


def test_a_custom_environment_run_with_a_secret_is_submittable_from_a_runner() -> None:
    """A FileSet environment and a model credential, arriving from the runner and the placement.

    A sandboxed deployment refuses credential-shaped plaintext in ``env_vars``, so this combination
    is what a real custom-environment evaluation needs, and both halves have to reach one target.
    """
    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="gdpval",
            env_secrets={"NVIDIA_API_KEY": SecretRef("evals/nvidia-api-key")},
        )
    )
    placement = GymPlacement(
        environment=FilesetRef(root="evals/gdpval-env"),
        agent_ref_name="gdpval_simple_agent",
    )

    target = runner_to_target(runner, placement)

    assert isinstance(target, GymRunnerTarget)
    assert target.environment is not None and target.environment.root == "evals/gdpval-env"
    assert target.env_secrets["NVIDIA_API_KEY"].root == "evals/nvidia-api-key"
    assert target.agent_ref_name == "gdpval_simple_agent"
    assert target.agent_config == "responses_api_agents/simple_agent/configs/simple_agent.yaml"
    AgentEvalInputSpec(tasks=TasksetRef("gdpval"), target=target)


def test_a_runner_submitted_without_a_placement_keeps_its_own_agent_config() -> None:
    target = runner_to_target(GymAgentTaskRunner(config=_configured()))

    assert isinstance(target, GymRunnerTarget)
    assert target.agent_config == "responses_api_agents/simple_agent/configs/simple_agent.yaml"
    assert target.environment is None
    assert target.agent_ref_name is None


def test_a_placement_for_a_runner_that_cannot_be_placed_is_refused() -> None:
    # Ignoring a placement aimed at the wrong runner would submit a job without the environment.
    class _Unsupported:
        def runner_info(self):  # pragma: no cover - never reached
            raise NotImplementedError

        async def run_tasks(self, tasks, config=None):  # pragma: no cover - never reached
            return []

    with pytest.raises(UnsubmittableRunnerError, match="GymPlacement"):
        runner_to_target(_Unsupported(), GymPlacement(environment=FilesetRef(root="ws/env")))


def test_a_hand_written_target_cannot_name_a_variable_both_ways_either() -> None:
    # Same rule as GymRuntimeConfig, because a hand-written spec never passes through one.
    with pytest.raises(ValidationError, match="GYM_MODEL_KEY"):
        GymRunnerTarget(
            agent="simple_agent",
            agent_config="c",
            resources_server="mcqa",
            env_vars={"GYM_MODEL_KEY": "plaintext"},
            env_secrets={"GYM_MODEL_KEY": SecretRef("evals/nvidia-api-key")},
        )
