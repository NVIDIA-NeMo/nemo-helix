# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting a taskset evaluation with a live agent runner, through ``AsyncEvaluator.submit``.

The async mirror of ``test_submit_agent_eval.py``, asserting both surfaces enforce the same rules:
a caller moving a working sync call to ``await`` should not find that the guards differ.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_evaluator.api.fields import TasksetRef
from nemo_evaluator.jobs.agent_spec import GymPlacement
from nemo_evaluator.sdk.resources import AsyncEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient


def _evaluator() -> tuple[AsyncEvaluator, MagicMock]:
    """An AsyncEvaluator with its executor stubbed — these tests are about dispatch, not transport."""
    evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
    executor = MagicMock()
    # Both executor entry points are awaited, so they must return awaitables rather than mocks.
    executor.submit_agent_eval = AsyncMock()
    executor.submit = AsyncMock()
    evaluator._executor = executor
    return evaluator, executor


def _runner() -> GymAgentTaskRunner:
    return GymAgentTaskRunner(
        config=GymRuntimeConfig(agent="simple_agent", agent_config="c.yaml", resources_server="mcqa", num_repeats=3)
    )


async def test_a_taskset_and_a_live_runner_route_to_the_agent_eval_path() -> None:
    evaluator, executor = _evaluator()
    runner = _runner()

    await evaluator.submit(tasks=TasksetRef("my-taskset"), target=runner)

    kwargs = executor.submit_agent_eval.call_args.kwargs
    assert kwargs["target"] is runner, "the live runner must reach the executor, not a copy of it"
    assert kwargs["tasks"].root == "my-taskset"
    executor.submit.assert_not_called()


async def test_a_gym_placement_reaches_the_executor() -> None:
    evaluator, executor = _evaluator()
    placement = GymPlacement(agent_ref_name="mcqa_simple_agent")

    await evaluator.submit(tasks=TasksetRef("ts"), target=_runner(), placement=placement)

    assert executor.submit_agent_eval.call_args.kwargs["placement"] is placement


async def test_the_row_path_is_untouched_by_the_new_overload() -> None:
    evaluator, executor = _evaluator()

    # An explicit packager: a MagicMock metric is not a built-in, and the real resolver refuses to
    # guess a bundling policy for one.
    await evaluator.submit(metric=MagicMock(), dataset=MagicMock(), metric_bundle_packager=MagicMock())

    executor.submit.assert_called_once()
    executor.submit_agent_eval.assert_not_called()


async def test_a_runner_passed_to_the_row_path_is_refused_rather_than_sent_as_an_endpoint() -> None:
    evaluator, executor = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        # Rejected statically too — the runtime guard is for callers without a type checker.
        await evaluator.submit(metric=MagicMock(), dataset=MagicMock(), target=_runner())  # ty: ignore[invalid-argument-type]

    message = str(excinfo.value)
    assert "GymAgentTaskRunner" in message
    assert "tasks=" in message, "the message should name the shape that does accept a runner"
    executor.submit.assert_not_called()


@pytest.mark.parametrize(
    "option",
    ["config", "field_mapping", "prompt_template", "metric_bundle_packager"],
)
async def test_row_only_options_are_refused_rather_than_silently_dropped(option: str) -> None:
    """A taskset evaluation is configured by its runner, so these four have nowhere to go."""
    evaluator, executor = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(tasks=TasksetRef("ts"), target=_runner(), **{option: MagicMock()})

    message = str(excinfo.value)
    assert option in message, "the message should name the option that cannot be honoured"
    assert "target" in message, "and point at what does configure a taskset run"
    executor.submit_agent_eval.assert_not_called()


async def test_supplying_both_shapes_is_refused_rather_than_silently_preferring_one() -> None:
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(tasks=TasksetRef("ts"), metric=MagicMock(), dataset=MagicMock(), target=_runner())  # ty: ignore[no-matching-overload]

    assert "not both" in str(excinfo.value)


async def test_supplying_neither_shape_says_what_was_expected() -> None:
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit()  # ty: ignore[no-matching-overload]

    message = str(excinfo.value)
    assert "tasks" in message and "metric" in message


async def test_a_taskset_submitted_with_a_non_runner_target_is_refused_by_type() -> None:
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(tasks=TasksetRef("ts"), target="gpt-5")  # ty: ignore[invalid-argument-type]

    message = str(excinfo.value)
    assert "AgentTaskRunner" in message
    assert "str" in message, "the message should name the type actually supplied"
    assert "GymAgentTaskRunner" in message, "and point at a concrete runner to use"


async def test_a_placement_for_a_runner_it_does_not_fit_is_refused() -> None:
    evaluator, executor = _evaluator()

    # A real runner, not a mock: a MagicMock satisfies any parameter type, leaving the overloads
    # unexercised.
    harbor = HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=Path("/tmp/harbor-unused")))

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(tasks=TasksetRef("ts"), target=harbor, placement=GymPlacement())  # ty: ignore[invalid-argument-type]

    assert "GymAgentTaskRunner" in str(excinfo.value)
    executor.submit_agent_eval.assert_not_called()


async def test_a_placement_on_the_row_path_is_refused() -> None:
    evaluator, executor = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(  # ty: ignore[no-matching-overload]
            metric=MagicMock(), dataset=MagicMock(), placement=GymPlacement()
        )

    assert "row evaluation" in str(excinfo.value)
    executor.submit.assert_not_called()


def test_the_two_executors_take_the_same_agent_eval_arguments() -> None:
    """Guards the drift this change exists to fix.

    The tests above stub the executor, so a parameter added to one side and not the other would not
    show up there — the mock accepts anything. Comparing the real signatures is what catches it.
    Return annotations legitimately differ (each returns its own job resource), so only the
    parameters are compared.
    """
    import inspect

    from nemo_evaluator.sdk._executor import _AsyncEvaluatorPluginExecutor, _SyncEvaluatorPluginExecutor

    for method in ("create_agent_eval", "submit_agent_eval"):
        sync = inspect.signature(getattr(_SyncEvaluatorPluginExecutor, method)).parameters
        async_ = inspect.signature(getattr(_AsyncEvaluatorPluginExecutor, method)).parameters

        assert list(sync) == list(async_), f"{method}: parameter names diverged"
        for name in sync:
            assert sync[name].kind == async_[name].kind, f"{method}.{name}: parameter kind diverged"
            assert sync[name].default == async_[name].default, f"{method}.{name}: default diverged"


def test_the_async_agent_job_resource_mirrors_the_sync_one() -> None:
    """The async handle mirrors the sync handle's surface, including its deliberate omissions.

    An agent evaluation publishes ``agent-eval-results`` and ``summary`` rather than the row job's
    ``aggregate-scores`` and ``artifacts``, so ``get_result`` and ``download_artifacts`` are absent.
    """
    from nemo_evaluator.sdk.job_resources import (
        AgentEvaluatorJobResource,
        AsyncAgentEvaluatorJobResource,
        AsyncEvaluatorJobResource,
    )

    async_surface = {name for name in vars(AsyncAgentEvaluatorJobResource) if not name.startswith("_")}
    sync_surface = {name for name in vars(AgentEvaluatorJobResource) if not name.startswith("_")}

    assert async_surface == sync_surface, "the two agent-eval handles must offer the same methods"
    assert {"get_job_status", "check_if_complete", "wait_until_done", "name", "job"} <= async_surface
    # Absent on purpose — see the class docstring.
    assert "get_result" not in async_surface
    assert "download_artifacts" not in async_surface
    assert not issubclass(AsyncAgentEvaluatorJobResource, AsyncEvaluatorJobResource)
    assert not issubclass(AsyncEvaluatorJobResource, AsyncAgentEvaluatorJobResource)


@pytest.mark.parametrize("empty", [False, True])
async def test_saved_trials_reach_the_job_spec_without_runner_conversion(monkeypatch, empty: bool) -> None:
    """Offline rescoring has no runner, so nothing may be asked to describe one."""
    evaluator = AsyncEvaluator(client=AsyncEvaluatorClient(base_url="http://test", workspace="default"))
    create_job = AsyncMock()
    monkeypatch.setattr(evaluator._executor, "create_agent_eval", create_job)
    monkeypatch.setattr(
        "nemo_evaluator.sdk._executor.runner_to_target",
        MagicMock(side_effect=AssertionError("Offline submission must not convert a runner")),
    )
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

    job = await evaluator.submit(tasks=tasks, trials=trials)

    assert job is create_job.return_value
    spec = create_job.call_args.kwargs["spec"]
    assert spec.tasks == tasks
    assert spec.trials == trials
    assert spec.target is None


async def test_a_taskset_with_both_a_runner_and_trials_is_refused() -> None:
    """The spec takes exactly one trial source, so accepting both would just fail later."""
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(tasks=TasksetRef("ts"), target=_runner(), trials=[])  # ty: ignore[no-matching-overload]

    assert "exactly one of" in str(excinfo.value)


async def test_trials_without_a_taskset_are_refused() -> None:
    """Trials are scored against the tasks they belong to; alone they name nothing."""
    evaluator, _ = _evaluator()

    with pytest.raises(TypeError) as excinfo:
        await evaluator.submit(trials=[])  # ty: ignore[no-matching-overload]

    assert "requires `tasks=TasksetRef(...)`" in str(excinfo.value)


async def test_a_placement_with_saved_trials_is_refused() -> None:
    """Placement says where a runner runs, and saved trials are not executed at all."""
    # The real executor, not the stub: this guard lives below the resource layer.
    evaluator = AsyncEvaluator(client=AsyncEvaluatorClient(base_url="http://test", workspace="default"))

    with pytest.raises(TypeError, match="placement requires target"):
        await evaluator._executor.submit_agent_eval(
            tasks=TasksetRef("default/suite"), trials=[], placement=GymPlacement()
        )
