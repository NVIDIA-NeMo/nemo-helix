# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting a taskset evaluation with a live agent runner, through ``Evaluator.submit``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_evaluator.api.fields import TasksetRef
from nemo_evaluator.jobs.agent_spec import GymPlacement
from nemo_evaluator.sdk.resources import Evaluator
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_helix_plugin.evaluator.client import EvaluatorClient


def _evaluator() -> tuple[Evaluator, MagicMock]:
    """An Evaluator with its executor stubbed — these tests are about dispatch, not transport."""
    evaluator = Evaluator.__new__(Evaluator)
    executor = MagicMock()
    evaluator._executor = executor
    return evaluator, executor


def _runner() -> GymAgentTaskRunner:
    return GymAgentTaskRunner(
        config=GymRuntimeConfig(agent="simple_agent", agent_config="c.yaml", resources_server="mcqa", num_repeats=3)
    )


@pytest.mark.parametrize("empty", [False, True])
def test_saved_trials_reach_job_spec_without_runner_conversion(monkeypatch, empty):
    evaluator = Evaluator(client=EvaluatorClient(base_url="http://test", workspace="default"))
    create_job = MagicMock()
    monkeypatch.setattr(evaluator._executor, "create_agent_eval", create_job)
    convert = MagicMock(side_effect=AssertionError("Offline submission must not convert a runner"))
    monkeypatch.setattr("nemo_evaluator.sdk._executor.runner_to_target", convert)
    trials = (
        []
        if empty
        else [
            AgentEvalTrial(
                id="trial-1",
                task_id="task-1",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="Done"),
            )
        ]
    )
    tasks = TasksetRef("default/suite")
    job = evaluator.submit(tasks=tasks, trials=trials)
    assert job is create_job.return_value
    spec = create_job.call_args.kwargs["spec"]
    assert spec.tasks == tasks
    assert spec.trials == trials
    assert spec.target is None
    assert spec.model_dump(mode="json")["trials"] == [trial.model_dump(mode="json") for trial in trials]
    convert.assert_not_called()


@pytest.mark.parametrize("case", ["both", "neither", "placement", "no_tasks", "row_option"])
def test_saved_trial_submission_rejects_invalid_combinations(case):
    evaluator, executor = _evaluator()
    options = {"tasks": TasksetRef("suite"), "trials": []}
    if case == "both":
        options["target"] = _runner()
    elif case == "neither":
        del options["trials"]
    elif case == "placement":
        options["placement"] = GymPlacement()
    elif case == "no_tasks":
        del options["tasks"]
    else:
        options["config"] = MagicMock()
    with pytest.raises(TypeError):
        evaluator.submit(**options)  # ty: ignore[no-matching-overload] -- exercise invalid public call shapes
    executor.submit_agent_eval.assert_not_called()
    executor.submit.assert_not_called()


@pytest.mark.parametrize("directory", ["omitted", "none", "path"])
def test_harbor_submission_converts_target_before_creating_job(monkeypatch, tmp_path, directory):
    from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

    evaluator = Evaluator(client=EvaluatorClient(base_url="http://test", workspace="default"))
    create_job = MagicMock()
    monkeypatch.setattr(evaluator._executor, "create_agent_eval", create_job)
    jobs_dir = tmp_path / "harbor-jobs"
    options = {} if directory == "omitted" else {"jobs_dir": None if directory == "none" else jobs_dir}
    runner = HarborAgentTaskRunner(
        config=HarborRuntimeConfig(
            **options, agent_name="oracle", reward_key="score", n_attempts=2, n_concurrent_trials=3
        )
    )
    tasks = TasksetRef("default/suite")
    job = evaluator.submit(tasks=tasks, target=runner)
    assert job is create_job.return_value
    create_job.assert_called_once()
    kwargs = create_job.call_args.kwargs
    assert kwargs["workspace"] == "default"
    assert kwargs["spec"].tasks == tasks
    assert kwargs["spec"].target == HarborRunnerTarget(
        agent_name="oracle", reward_key="score", n_attempts=2, n_concurrent_trials=3
    )
    assert "jobs_dir" not in kwargs["spec"].target.model_dump()
    if directory == "path":
        assert not jobs_dir.exists()


def test_a_taskset_and_a_live_runner_route_to_the_agent_eval_path() -> None:
    # The shape this change exists for: what you ran locally with AgentEvaluator() becomes a job,
    # with the runner carried through rather than described again by hand.
    evaluator, executor = _evaluator()
    runner = _runner()

    evaluator.submit(tasks=TasksetRef("my-taskset"), target=runner)

    kwargs = executor.submit_agent_eval.call_args.kwargs
    assert kwargs["target"] is runner, "the live runner must reach the executor, not a copy of it"
    assert kwargs["tasks"].root == "my-taskset"
    # ...and the row path is not also taken.
    executor.submit.assert_not_called()


def test_the_row_path_is_untouched_by_the_new_overload() -> None:
    # The overload adds a shape; it must not change the existing one. A regression here would break
    # every caller that predates taskset evaluation.
    evaluator, executor = _evaluator()

    # An explicit packager: a MagicMock metric is not a built-in, and the real resolver refuses to
    # guess a bundling policy for one. That rejection is the row path working, not a test problem.
    evaluator.submit(metric=MagicMock(), dataset=MagicMock(), metric_bundle_packager=MagicMock())

    executor.submit.assert_called_once()
    executor.submit_agent_eval.assert_not_called()


def test_a_runner_passed_to_the_row_path_is_refused_rather_than_sent_as_an_endpoint() -> None:
    # The likely mistake once both shapes exist: reaching for a runner while still describing the
    # work as rows. Without this, the runner would travel the row path and be described as a model
    # endpoint, failing somewhere far from the call that got it wrong.
    evaluator, executor = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        # Rejected statically too — the runtime guard is for callers without a type checker.
        evaluator.submit(metric=MagicMock(), dataset=MagicMock(), target=_runner())  # ty: ignore[invalid-argument-type]

    message = str(excinfo.value)
    assert "GymAgentTaskRunner" in message
    assert "tasks=" in message, "the message should name the shape that does accept a runner"
    executor.submit.assert_not_called()


@pytest.mark.parametrize(
    "option",
    ["config", "field_mapping", "prompt_template", "metric_bundle_packager"],
)
def test_row_only_options_are_refused_rather_than_silently_dropped(option: str) -> None:
    """A taskset evaluation is configured by its runner, so these four have nowhere to go.

    Accepting them silently would submit a job that ignores what the caller supplied — the failure
    mode is a run that looks fine and used none of the requested configuration.
    """
    evaluator, executor = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        evaluator.submit(tasks=TasksetRef("ts"), target=_runner(), **{option: MagicMock()})

    message = str(excinfo.value)
    assert option in message, "the message should name the option that cannot be honoured"
    assert "target" in message, "and point at what does configure a taskset run"
    executor.submit_agent_eval.assert_not_called()


def test_supplying_both_shapes_is_refused_rather_than_silently_preferring_one() -> None:
    # Picking one for the caller would run an evaluation they did not ask for. Better to stop.
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        # Rejected statically too; the runtime guard is for callers without a type checker.
        evaluator.submit(tasks=TasksetRef("ts"), metric=MagicMock(), dataset=MagicMock(), target=_runner())  # ty: ignore[no-matching-overload]

    assert "not both" in str(excinfo.value)


def test_supplying_neither_shape_says_what_was_expected() -> None:
    # The failure mode of an overloaded entry point is a confusing error deep inside spec
    # construction. This one names both shapes at the boundary.
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        evaluator.submit()  # ty: ignore[no-matching-overload]

    message = str(excinfo.value)
    assert "tasks" in message and "metric" in message


def test_a_taskset_submitted_with_a_non_runner_target_is_refused_by_type() -> None:
    # `target` is overloaded across the two shapes: a Model/Agent for rows, a runner for tasksets.
    # Passing the wrong one is easy, so the error names the type given and what to pass instead.
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        evaluator.submit(tasks=TasksetRef("ts"), target="gpt-5")  # ty: ignore[invalid-argument-type]

    message = str(excinfo.value)
    assert "AgentTaskRunner" in message
    assert "str" in message, "the message should name the type actually supplied"
    assert "GymAgentTaskRunner" in message, "and point at a concrete runner to use"


def test_the_agent_job_resource_does_not_offer_row_evaluation_readers() -> None:
    """The reason this resource is separate rather than shared.

    A row evaluation publishes ``aggregate-scores``, ``row-scores`` and ``artifacts`` results; an
    agent evaluation publishes ``agent-eval-results`` and ``summary``. Inheriting the row job's
    readers would put methods here whose type says "results" and whose behaviour is a 404 against
    routes this job never writes. Status and polling are shared because the platform serves those
    per job, regardless of what produced it.
    """
    from nemo_evaluator.sdk.job_resources import AgentEvaluatorJobResource, EvaluatorJobResource

    agent_surface = {name for name in vars(AgentEvaluatorJobResource) if not name.startswith("_")}

    assert {"get_job_status", "check_if_complete", "wait_until_done", "name", "job"} <= agent_surface
    # Absent on purpose — see the class docstring. If either is added, it must target this job's own
    # artifacts rather than the row job's.
    assert "get_result" not in agent_surface
    assert "download_artifacts" not in agent_surface
    # And it is not related to the row resource by inheritance in either direction.
    assert not issubclass(AgentEvaluatorJobResource, EvaluatorJobResource)
    assert not issubclass(EvaluatorJobResource, AgentEvaluatorJobResource)


def test_a_placement_for_a_runner_it_does_not_fit_is_refused() -> None:
    # Ignoring the placement would submit a job missing the environment the caller asked for.
    evaluator, executor = _evaluator()

    # A real runner, not a mock: a MagicMock satisfies any parameter type, leaving the overloads
    # unexercised.
    harbor = HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=Path("/tmp/harbor-unused")))

    with pytest.raises(TypeError) as excinfo:
        evaluator.submit(tasks=TasksetRef("ts"), target=harbor, placement=GymPlacement())  # ty: ignore[invalid-argument-type]

    assert "GymAgentTaskRunner" in str(excinfo.value)
    executor.submit_agent_eval.assert_not_called()


def test_a_placement_on_the_row_path_is_refused() -> None:
    # A row evaluation has no runner to place; accepting one would silently drop it.
    evaluator, executor = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        evaluator.submit(  # ty: ignore[no-matching-overload]
            metric=MagicMock(), dataset=MagicMock(), placement=GymPlacement()
        )

    assert "row evaluation" in str(excinfo.value)
    executor.submit.assert_not_called()


def test_executor_refuses_placement_without_target() -> None:
    # Saved trials are not executed, so a placement would be silently dropped.
    evaluator = Evaluator(client=EvaluatorClient(base_url="http://test", workspace="default"))
    with pytest.raises(TypeError, match="placement requires target"):
        evaluator._executor.submit_agent_eval(tasks=TasksetRef("default/suite"), trials=[], placement=GymPlacement())
