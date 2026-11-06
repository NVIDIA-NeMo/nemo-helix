# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task Job reports each Harbor trial's progress as Harbor moves it through its lifecycle."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from harbor.job import DatasetConfig, Job, JobConfig
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import ExceptionInfo, StepResult, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial.hooks import HookCallback, TrialEvent, TrialHookEvent
from nemo_evals.harbor.trial_progress import TrialProgressReporter, trial_error, trial_reward
from nemo_evals.taskset_runs.trials import TRIALS_STATUS_KEY, TrialPhase, TrialProgress

STARTED = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
async def job(tmp_path: Path) -> Job:
    """A planned Harbor job: two attempts at one task, each retried once on errors Harbor retries."""
    task = tmp_path / "dataset" / "hello-world"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('[task]\nname = "trial-rec/hello-world"\n')
    (task / "instruction.md").write_text("Say hello.\n")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    return await Job.create(
        JobConfig(
            job_name="job",
            jobs_dir=tmp_path / "jobs",
            n_attempts=2,
            retry=RetryConfig(max_retries=1),
            agents=[AgentConfig(name="oracle")],
            datasets=[DatasetConfig(path=tmp_path / "dataset")],
        )
    )


class Harbor:
    """The hooks the reporter registers on the job, fired the way Harbor fires them."""

    def __init__(self, job: Job, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hooks: dict[TrialEvent, list[HookCallback]] = {}

        def add_hook(event: TrialEvent, callback: HookCallback) -> Job:
            self.hooks.setdefault(event, []).append(callback)
            return job

        monkeypatch.setattr(job, "add_hook", add_hook)

    async def fire(self, event: TrialEvent, config: TrialConfig, result: TrialResult | None = None) -> None:
        hook_event = TrialHookEvent.model_construct(
            event=event,
            task_name="hello-world",
            config=config,
            result=result or TrialResult.model_construct(started_at=STARTED),
            timestamp=STARTED,
        )
        for hook in self.hooks.get(event, []):
            await hook(hook_event)

    async def run(self, config: TrialConfig, *, error: str | None = None, reward: float | None = 1.0) -> None:
        """One attempt: started, agent, verifier, ended."""
        for event in (TrialEvent.START, TrialEvent.AGENT_START, TrialEvent.VERIFICATION_START):
            await self.fire(event, config)
        await self.fire(TrialEvent.END, config, _result(error=error, reward=reward))


def _result(*, error: str | None = None, reward: float | None = None, trial_name: str = "") -> TrialResult:
    return TrialResult.model_construct(
        trial_name=trial_name,
        started_at=STARTED,
        finished_at=STARTED.replace(minute=5),
        exception_info=_error(error) if error is not None else None,
        verifier_result=VerifierResult(rewards={"reward": reward}) if reward is not None else None,
    )


def _error(kind: str) -> ExceptionInfo:
    return ExceptionInfo(
        exception_type=kind, exception_message=f"{kind}: boom", exception_traceback="", occurred_at=STARTED
    )


class Reports:
    """Every report the reporter published, as the task Job's status details would receive them."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def __call__(self, details: dict[str, Any]) -> None:
        self.published.append(details)

    def last(self) -> list[TrialProgress]:
        return [TrialProgress.model_validate(item) for item in self.published[-1][TRIALS_STATUS_KEY]]

    def phases(self) -> list[TrialPhase]:
        return [trial.phase for trial in self.last()]


@pytest.fixture
def harbor(job: Job, monkeypatch: pytest.MonkeyPatch) -> Harbor:
    return Harbor(job, monkeypatch)


def _reporter(reports: Reports) -> TrialProgressReporter:
    return TrialProgressReporter(reports, reward_key="reward", session_id=lambda trial: f"run-1-task-0:{trial}")


async def test_every_planned_trial_is_reported_pending_with_its_session(job: Job, harbor: Harbor) -> None:
    reports = Reports()
    configs = job._remaining_trial_configs

    async with _reporter(reports).job_scope()(job):
        trials = reports.last()

    assert [trial.trial for trial in trials] == [config.trial_name for config in configs]
    assert {trial.phase for trial in trials} == {TrialPhase.PENDING}
    assert [trial.session_id for trial in trials] == [f"run-1-task-0:{config.trial_name}" for config in configs]


async def test_a_resumed_job_reports_the_trials_it_already_finished(job: Job, harbor: Harbor) -> None:
    """Harbor resumes per trial: finished trials aren't run again, but they're still the task's trials."""
    reports = Reports()
    succeeded, cancelled = job._remaining_trial_configs
    job._existing_trial_results = [
        _result(reward=0.5, trial_name=succeeded.trial_name),
        _result(error="CancelledError", trial_name=cancelled.trial_name),  # how Harbor records a cancelled trial
    ]
    job._remaining_trial_configs = []

    async with _reporter(reports).job_scope()(job):
        pass

    done, stopped = reports.last()
    assert (done.trial, done.phase, done.reward) == (succeeded.trial_name, TrialPhase.SUCCEEDED, 0.5)
    assert (stopped.trial, stopped.phase) == (cancelled.trial_name, TrialPhase.CANCELLED)


async def test_each_lifecycle_event_is_reported_as_it_happens(job: Job, harbor: Harbor) -> None:
    reports = Reports()
    first, second = job._remaining_trial_configs

    async with _reporter(reports).job_scope()(job):
        await harbor.fire(TrialEvent.START, first)
        assert reports.phases() == [TrialPhase.STARTING, TrialPhase.PENDING]

        await harbor.fire(TrialEvent.AGENT_START, first)
        assert reports.phases() == [TrialPhase.AGENT_RUNNING, TrialPhase.PENDING]

        await harbor.fire(TrialEvent.VERIFICATION_START, first)
        assert reports.phases() == [TrialPhase.VERIFYING, TrialPhase.PENDING]

        await harbor.fire(TrialEvent.END, first, _result(reward=0.5))
        await harbor.run(second, reward=1.0)

    done, other = reports.last()
    assert reports.phases() == [TrialPhase.SUCCEEDED, TrialPhase.SUCCEEDED]
    assert (done.attempts, done.reward, done.started_at, done.finished_at) == (
        1,
        0.5,
        STARTED,
        STARTED.replace(minute=5),
    )
    assert other.reward == 1.0
    assert done.error is None and done.error_type is None


async def test_an_error_is_retrying_until_harbor_stops_retrying_it(job: Job, harbor: Harbor) -> None:
    reports = Reports()
    first, second = job._remaining_trial_configs

    async with _reporter(reports).job_scope()(job):
        await harbor.run(first, error="RuntimeError", reward=None)
        retrying = reports.last()[0]

        await harbor.run(first, error="RuntimeError", reward=None)

        # Harbor scores an agent that timed out, and never retries the timeout.
        await harbor.run(second, error="AgentTimeoutError", reward=0.0)

    assert (retrying.phase, retrying.finished_at, retrying.error_type) == (TrialPhase.RETRYING, None, "RuntimeError")

    errored, timed_out = reports.last()
    assert (errored.phase, errored.attempts, errored.error) == (TrialPhase.ERRORED, 2, "RuntimeError: boom")
    assert errored.finished_at is not None
    assert (timed_out.phase, timed_out.attempts, timed_out.reward) == (TrialPhase.ERRORED, 1, 0.0)


async def test_a_cancelled_trial_stays_cancelled_when_its_attempt_ends(job: Job, harbor: Harbor) -> None:
    reports = Reports()
    first, _ = job._remaining_trial_configs

    async with _reporter(reports).job_scope()(job):
        await harbor.fire(TrialEvent.START, first)
        await harbor.fire(TrialEvent.CANCEL, first)
        await harbor.fire(TrialEvent.END, first, _result(error="CancelledError"))

    assert reports.last()[0].phase is TrialPhase.CANCELLED


async def test_trials_left_unsettled_when_the_job_stops_are_reported_cancelled(job: Job, harbor: Harbor) -> None:
    reports = Reports()
    first, _ = job._remaining_trial_configs

    with pytest.raises(RuntimeError, match="stopped"):
        async with _reporter(reports).job_scope()(job):
            await harbor.run(first)
            await harbor.fire(TrialEvent.START, job._remaining_trial_configs[1])
            raise RuntimeError("stopped")

    assert reports.phases() == [TrialPhase.SUCCEEDED, TrialPhase.CANCELLED]
    assert reports.last()[1].finished_at is not None


async def test_a_failed_or_slow_report_never_fails_the_trial(job: Job, harbor: Harbor) -> None:
    first, _ = job._remaining_trial_configs
    calls = 0

    async def failing(details: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls % 2:
            raise RuntimeError("jobs unavailable")
        await asyncio.Event().wait()  # never answers

    reporter = TrialProgressReporter(failing, reward_key="reward", publish_timeout=0.01)

    async with reporter.job_scope()(job):
        await harbor.run(first)

    assert reporter.trials[0].phase is TrialPhase.SUCCEEDED
    assert calls == 6  # planned, started, agent, verifying, ended, then the unrun trial cancelled at exit


def test_reward_and_error_are_read_off_the_attempts_result() -> None:
    def result(rewards: dict[str, float | int] | None) -> TrialResult:
        return TrialResult.model_construct(verifier_result=VerifierResult(rewards=rewards))

    assert trial_reward(result({"reward": 1, "style": 0.2}), "reward") == 1.0
    assert trial_reward(result({"accuracy": 0.25}), "reward") == 0.25  # the only reward
    assert trial_reward(result({"accuracy": 0.25, "style": 1}), "reward") is None

    step = StepResult.model_construct(exception_info=_error("StepError"))
    multi_step = TrialResult.model_construct(exception_info=None, step_results=[StepResult.model_construct(), step])

    assert trial_error(multi_step) == step.exception_info
