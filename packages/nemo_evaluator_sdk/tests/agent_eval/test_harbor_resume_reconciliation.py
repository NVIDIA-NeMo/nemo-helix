# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin Harbor 0.20.0 resume behavior and the aligned SDK cache contract."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor", reason="Harbor tests require the optional harbor extra")

import harbor
from harbor.cli.jobs import jobs_app, resume
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor_fixtures import write_harbor_trial_result
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    _CACHE_IRRELEVANT_OPTIONS,
    HarborAgentTaskRunner,
    HarborRewardMetric,
    HarborRuntimeConfig,
    _all_tasks_cached,
    _build_native_job,
    _cache_stamp,
    _write_cache_stamp,
    build_trials_from_job_dir,
)
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_trial_adapter import _iter_harbor_trial_results
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus

_DATASET_DIR = Path(__file__).resolve().parents[2] / "examples" / "harbor" / "hello_world_dataset"


def _write_harbor_result(trial_dir: Path, *, exception_type: str | None = None, text: str | None = None) -> None:
    if text is not None:
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(text, encoding="utf-8")
        return
    write_harbor_trial_result(
        trial_dir,
        task_name="harbor/hello-world",
        rewards={"reward": 0.0},
        exception=exception_type,
    )


@contextmanager
def _halt_harbor_cli_after_filter() -> Iterator[None]:
    """Let ``jobs resume`` delete matching trials, then stop before Docker/Job.run."""
    from unittest.mock import patch

    def abandon(coro: Any) -> object:
        coro.close()
        return object()

    with (
        patch("harbor.environments.factory.EnvironmentFactory.run_preflight"),
        patch("harbor.cli.jobs.run_async", side_effect=abandon),
        patch("harbor.cli.jobs.print_job_results_tables"),
    ):
        yield


def _cli_job_dir(tmp_path: Path) -> Path:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "resume-job"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(
        JobConfig(job_name="resume-job", jobs_dir=jobs_dir, quiet=True).model_dump_json(),
        encoding="utf-8",
    )
    return job_dir


def _sdk_errored_job(
    tmp_path: Path,
) -> tuple[HarborRuntimeConfig, Path, AgentEvalTask]:
    dataset_path = tmp_path / "dataset"
    task_dir = dataset_path / "t"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[task]\nname = "t"\n')
    jobs_dir = dataset_path / "jobs"
    job_dir = jobs_dir / "cached-job"
    write_harbor_trial_result(
        job_dir / "t__aaa",
        task_name="t",
        rewards={"reward": 0.8},
        exception={"exception_type": "AgentTimeoutError", "exception_message": "timed out"},
    )
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="cached-job")
    task = AgentEvalTask(
        id="t",
        intent="x",
        inputs={"instruction": "x"},
        metrics=[HarborRewardMetric()],
        metadata={"harbor_dataset_path": str(dataset_path), "harbor_task_dir": str(task_dir)},
    )
    _write_cache_stamp(job_dir, _cache_stamp(config, dataset_path, [task]))
    return config, job_dir, task


# --- Harbor CLI ``jobs resume`` -------------------------------------------------


def test_harbor_cli_resume_defaults_to_cancelled_error_and_hides_it() -> None:
    from typer.testing import CliRunner

    assert harbor.__version__ == "0.20.0"
    parameter = inspect.signature(resume).parameters["filter_error_types"]
    assert parameter.default == ["CancelledError"]
    source = inspect.getsource(resume)
    assert "show_default=False" in source
    help_text = CliRunner().invoke(jobs_app, ["resume", "--help"]).output
    assert "--filter-error-type" in help_text
    assert "CancelledError" not in help_text


def test_harbor_cli_resume_exact_matches_exception_type_and_skips_bad_results(
    tmp_path: Path,
) -> None:
    job_dir = _cli_job_dir(tmp_path)
    _write_harbor_result(job_dir / "cancelled", exception_type="CancelledError")
    _write_harbor_result(job_dir / "timeout", exception_type="AgentTimeoutError")
    _write_harbor_result(job_dir / "clean", exception_type=None)
    _write_harbor_result(job_dir / "empty", text="   ")
    _write_harbor_result(job_dir / "invalid", text="{not json")
    (job_dir / "no-result").mkdir()
    source = inspect.getsource(resume)
    assert "trial_result.exception_info.exception_type in filter_error_types_set" in source

    with _halt_harbor_cli_after_filter():
        resume(job_path=job_dir)

    remaining = sorted(path.name for path in job_dir.iterdir() if path.is_dir())
    assert remaining == ["clean", "empty", "invalid", "no-result", "timeout"]


def test_harbor_cli_filter_flag_replaces_the_cancelled_error_default(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    job_dir = _cli_job_dir(tmp_path)
    _write_harbor_result(job_dir / "cancelled", exception_type="CancelledError")
    _write_harbor_result(job_dir / "timeout", exception_type="AgentTimeoutError")
    source = inspect.getsource(resume)
    assert inspect.signature(resume).parameters["filter_error_types"].default == ["CancelledError"]
    assert "filter_error_types_set = set(filter_error_types)" in source

    with _halt_harbor_cli_after_filter():
        result = CliRunner().invoke(
            jobs_app,
            ["resume", "--job-path", str(job_dir), "--filter-error-type", "AgentTimeoutError"],
        )

    assert result.exit_code == 0, result.output
    remaining = sorted(path.name for path in job_dir.iterdir() if path.is_dir())
    assert remaining == ["cancelled"], "passing -f replaces CancelledError rather than appending to it"


def test_harbor_job_create_has_no_filter_error_types() -> None:
    assert "filter_error_type" not in inspect.signature(Job.create).parameters
    assert "filter_error_type" not in inspect.getsource(Job)


def test_empty_resume_filter_deletes_nothing(tmp_path: Path) -> None:
    job_dir = _cli_job_dir(tmp_path)
    _write_harbor_result(job_dir / "cancelled", exception_type="CancelledError")

    with _halt_harbor_cli_after_filter():
        resume(job_path=job_dir, filter_error_types=[])

    assert (job_dir / "cancelled").is_dir()


# --- Harbor Job reconciliation --------------------------------------------------


async def _harbor_job(tmp_path: Path, *, n_attempts: int = 1) -> tuple[Job, Path]:
    jobs_dir = tmp_path / "jobs"
    job_name = "spike"
    job_dir = jobs_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    config = JobConfig(
        job_name=job_name,
        jobs_dir=jobs_dir,
        n_attempts=n_attempts,
        quiet=True,
        datasets=[DatasetConfig(path=_DATASET_DIR, task_names=["hello-world"])],
    )
    (job_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    return await Job.create(config), job_dir


def _plant_errored_trial(job: Any, job_dir: Path, exception_type: str) -> Path:
    planned = job._trial_configs[0]
    trial_dir = job_dir / planned.trial_name
    write_harbor_trial_result(
        trial_dir,
        task_name="harbor/hello-world",
        rewards={"reward": 0.0},
        exception=exception_type,
        config=planned,
    )
    return trial_dir


@pytest.mark.asyncio
async def test_harbor_job_treats_an_errored_result_json_as_complete(tmp_path: Path) -> None:
    job, job_dir = await _harbor_job(tmp_path)
    _plant_errored_trial(job, job_dir, "AgentTimeoutError")
    source = inspect.getsource(Job._maybe_init_existing_job)
    assert "exception_info" not in source

    resumed = await Job.create(job.config)

    [existing_result] = resumed._existing_trial_results
    assert existing_result.exception_info is not None
    assert existing_result.exception_info.exception_type == "AgentTimeoutError"
    assert resumed._remaining_trial_configs == []


@pytest.mark.asyncio
async def test_harbor_job_rmtrees_a_trial_dir_without_result_json_and_reruns_it(
    tmp_path: Path,
) -> None:
    job, job_dir = await _harbor_job(tmp_path)
    orphan = job_dir / "empty-attempt"
    orphan.mkdir()

    resumed = await Job.create(job.config)

    assert not orphan.exists()
    assert resumed._existing_trial_results == []
    assert len(resumed._remaining_trial_configs) == 1


@pytest.mark.asyncio
async def test_harbor_cancelled_error_is_stats_only_and_is_not_rerun(tmp_path: Path) -> None:
    job, job_dir = await _harbor_job(tmp_path)
    trial_dir = _plant_errored_trial(job, job_dir, "CancelledError")
    source = inspect.getsource(Job._init_progress_tracking)
    assert "_is_cancelled_result" in source
    assert "rmtree" not in inspect.getsource(Job._is_cancelled_result)

    resumed = await Job.create(job.config)

    assert resumed._remaining_trial_configs == []
    assert trial_dir.name in resumed._cancelled_trial_names
    assert resumed._existing_stats.n_cancelled_trials == 1
    assert (trial_dir / "result.json").exists()


# --- SDK cache alignment --------------------------------------------------------


def test_all_tasks_cached_accepts_an_errored_only_n_attempts_of_one(tmp_path: Path) -> None:
    config, job_dir, task = _sdk_errored_job(tmp_path)

    assert _all_tasks_cached(job_dir, [task], n_attempts=1) is True
    assert config.job_name == "cached-job"


@pytest.mark.asyncio
async def test_errored_stamped_job_is_served_without_invoking_harbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes import harbor_runtime

    config, job_dir, task = _sdk_errored_job(tmp_path)
    calls: list[bool] = []

    def fake_build(runtime_config, _dataset_path, _task_names, *, job_name=None, force_rerun=None):
        async def run_job() -> None:
            calls.append(bool(force_rerun))

        return runtime_config.jobs_dir / (job_name or "job"), run_job

    monkeypatch.setattr(harbor_runtime, "_build_native_job", fake_build)

    trials = await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert calls == [], "a stamp hit with Harbor-valid errored coverage must be served directly"
    assert [trial.id for trial in trials] == ["t__aaa"]
    assert trials[0].status == AgentEvalTrialStatus.PARTIAL
    assert trials[0].error is not None
    assert trials[0].error.type == "AgentTimeoutError"
    assert trials[0].metadata["reward"] == 0.8


_REQUIRED_RESULT_FIELDS = ("task_name", "trial_name", "trial_uri", "task_id", "task_checksum", "config", "agent_info")
_VALID_RESULT_INPUTS: dict[str, tuple[dict[str, float | int], str | None]] = {
    "valid-clean": ({"reward": 1.0}, None),
    "valid-runtime-error": ({"reward": 0.8}, "RuntimeError"),
    "valid-timeout-without-primary": ({"format_ok": 1.0}, "AgentTimeoutError"),
    "valid-cancelled": ({"reward": 0.0}, "CancelledError"),
}
_INVALID_RESULT_TEXTS = {
    "empty": "",
    "null": "null",
    "array": "[]",
    "scalar": '"result"',
    "fragment": '{"task_name": "t"}',
}
_INVALID_FIELD_VALUES: dict[str, tuple[str, object]] = {
    "invalid-task-name": ("task_name", 17),
    "invalid-task-id": ("task_id", {}),
    "invalid-config": ("config", {}),
    "invalid-agent-info": ("agent_info", {"name": "oracle"}),
    "invalid-exception-info": ("exception_info", {"exception_type": "RuntimeError"}),
    "invalid-verifier-result": ("verifier_result", {"rewards": "not-a-mapping"}),
    "invalid-agent-result": ("agent_result", {"n_input_tokens": {"not": "an integer"}}),
    "invalid-timing": ("environment_setup", {"started_at": []}),
}
_RESULT_VALIDITY_CASES = (
    *_VALID_RESULT_INPUTS,
    "unreadable",
    *_INVALID_RESULT_TEXTS,
    *(f"missing-{field}" for field in _REQUIRED_RESULT_FIELDS),
    *_INVALID_FIELD_VALUES,
)


def _plant_result_validity_case(job_dir: Path, case: str) -> None:
    trial_dir = job_dir / f"t__{case}"
    if case == "unreadable":
        (trial_dir / "result.json").mkdir(parents=True)
        return
    if case in _INVALID_RESULT_TEXTS:
        trial_dir.mkdir(parents=True)
        (trial_dir / "result.json").write_text(_INVALID_RESULT_TEXTS[case], encoding="utf-8")
        return

    if case in _VALID_RESULT_INPUTS:
        rewards, exception = _VALID_RESULT_INPUTS[case]
    else:
        rewards, exception = {"reward": 1.0}, None

    result = write_harbor_trial_result(
        trial_dir,
        task_name="t",
        rewards=rewards,
        exception=exception,
    )
    if case in _VALID_RESULT_INPUTS:
        return

    payload = result.model_dump(mode="json")
    if case.startswith("missing-"):
        del payload[case.removeprefix("missing-")]
    elif case in _INVALID_FIELD_VALUES:
        field, value = _INVALID_FIELD_VALUES[case]
        payload[field] = value
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(f"unhandled result-validity case {case!r}")
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("case", _RESULT_VALIDITY_CASES)
def test_harbor_result_validity_is_shared_by_loader_cache_and_adaptation(tmp_path: Path, case: str) -> None:
    assert harbor.__version__ == "0.20.0"
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _plant_result_validity_case(job_dir, case)
    task = AgentEvalTask(id="t", intent="x", inputs={"instruction": "x"}, metrics=[HarborRewardMetric()])
    expected_valid = case in _VALID_RESULT_INPUTS

    loaded = list(_iter_harbor_trial_results(job_dir))
    cached = _all_tasks_cached(job_dir, [task], n_attempts=1)
    trials = build_trials_from_job_dir(job_dir, [task])

    assert bool(loaded) is expected_valid
    assert cached is expected_valid
    assert bool(trials) is expected_valid

    if case == "valid-clean":
        assert trials[0].status is AgentEvalTrialStatus.COMPLETED
        assert trials[0].metadata["reward"] == 1.0
    elif case == "valid-runtime-error":
        assert trials[0].status is AgentEvalTrialStatus.PARTIAL
        assert trials[0].metadata["reward"] == 0.8
    elif case == "valid-timeout-without-primary":
        assert trials[0].status is AgentEvalTrialStatus.PARTIAL
        assert trials[0].metadata["reward"] is None


def test_sdk_does_not_expose_resume_filter_error_types_today() -> None:
    assert "resume_filter_error_types" not in HarborRuntimeConfig.model_fields
    assert "resume_filter_error_types" not in _CACHE_IRRELEVANT_OPTIONS


# --- Retry vs resume ------------------------------------------------------------


def test_in_run_retry_and_cli_resume_filter_are_separate_surfaces() -> None:
    assert harbor.__version__ == "0.20.0"
    fields = RetryConfig.model_fields
    assert set(fields) >= {"max_retries", "include_exceptions", "exclude_exceptions"}
    default_exclude = RetryConfig().exclude_exceptions
    assert default_exclude is not None
    assert "AgentTimeoutError" in default_exclude
    assert "CancelledError" not in default_exclude

    source = inspect.getsource(_build_native_job)
    assert "RetryConfig(max_retries=config.max_retries)" in source
    assert "include_exceptions" not in source
    assert "exclude_exceptions" not in source
    assert "filter_error_type" not in source
    assert "resume_filter" not in source
