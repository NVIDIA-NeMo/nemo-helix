# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sandbox_k8s status readers: a present-but-partial Harbor ``result.json`` must
not read as ``succeeded``.

Harbor writes ``result.json`` incrementally (and leaves a partial one behind if
the run dies early), so "the file exists" is not "the run finished". Before this
was fixed, a harbor-runner container that exited non-zero with an initial
result.json (trials still pending, ``finished_at`` null) was reported as a
deceptive ``succeeded`` / ``reward=null`` instead of ``failed``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

try:
    from scaled_evals.api.settings import settings
    from scaled_evals.dispatch.sandbox_k8s import (
        _harbor_error_summary,
        _harbor_run_finished,
        build_backend,
        make_sandbox_k8s_docker_status_reader,
        make_sandbox_k8s_status_reader,
    )
    from scaled_evals.models.runtime import LaunchHandle
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

EID = "ev_test123"
JOBS_DIR = "jobs/results"

FINISHED_RESULT: dict[str, Any] = {
    "id": "run-1",
    "finished_at": "2026-06-24T10:13:40.677719",
    "n_total_trials": 1,
    "stats": {"n_completed_trials": 1, "n_errored_trials": 0, "n_pending_trials": 0},
}
PARTIAL_RESULT: dict[str, Any] = {
    "id": "run-1",
    "finished_at": None,
    "n_total_trials": 1,
    "stats": {
        "n_completed_trials": 0,
        "n_errored_trials": 0,
        "n_running_trials": 0,
        "n_pending_trials": 1,
    },
}
# A finished run whose only trial errored (e.g. the sandbox never came up or
# the agent could not reach the model) — distinct from a trial that ran and
# genuinely scored reward 0. exception_stats names the retry-exhausted failure.
ERRORED_RESULT: dict[str, Any] = {
    "id": "run-1",
    "finished_at": "2026-06-24T10:13:40.677719",
    "n_total_trials": 1,
    "stats": {
        "n_completed_trials": 0,
        "n_errored_trials": 1,
        "n_pending_trials": 0,
        "evals": {
            "oracle__adhoc": {
                "n_trials": 1,
                "n_errors": 1,
                "metrics": [],
                "exception_stats": {"EnvironmentStartTimeoutError": 1},
            }
        },
    },
}
# A finished run that scored a genuine reward 0 — every trial completed; nothing
# errored. This must stay ``succeeded`` (a real zero, not an infra failure).
SCORED_ZERO_RESULT: dict[str, Any] = {
    "id": "run-1",
    "finished_at": "2026-06-24T10:13:40.677719",
    "n_total_trials": 1,
    "stats": {
        "n_completed_trials": 1,
        "n_errored_trials": 0,
        "n_pending_trials": 0,
        "evals": {"oracle__adhoc": {"n_trials": 1, "n_errors": 0, "metrics": [{"mean": 0.0}]}},
    },
}


def _write_result(harbor_dir: Path, result: dict[str, Any]) -> None:
    d = harbor_dir / JOBS_DIR / EID
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(result))


def _write_artifact_result(artifact_root: Path, result: dict[str, Any]) -> None:
    d = artifact_root / EID
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(result))


def _handle(**raw: str) -> LaunchHandle:
    return LaunchHandle(backend="sandbox_k8s", external_id=EID, raw=raw)


# --- _harbor_run_finished -----------------------------------------------------


def test_finished_true_when_finished_at_set() -> None:
    assert _harbor_run_finished(FINISHED_RESULT) is True


def test_finished_false_when_partial() -> None:
    assert _harbor_run_finished(PARTIAL_RESULT) is False


def test_finished_true_when_all_trials_accounted_without_finished_at() -> None:
    # Defensive fallback: finished_at absent but every trial is accounted for.
    result = {
        "n_total_trials": 2,
        "stats": {"n_completed_trials": 1, "n_errored_trials": 1, "n_pending_trials": 0},
    }
    assert _harbor_run_finished(result) is True


def test_finished_false_when_no_trials_and_no_finished_at() -> None:
    assert _harbor_run_finished({"stats": {}}) is False


# --- _harbor_error_summary ----------------------------------------------------


def test_error_summary_none_when_no_errored_trials() -> None:
    assert _harbor_error_summary(FINISHED_RESULT) is None


def test_error_summary_none_for_genuinely_scored_zero() -> None:
    # A real reward-0 run completed every trial — not an infra error.
    assert _harbor_error_summary(SCORED_ZERO_RESULT) is None


def test_error_summary_reports_count_and_exception_name() -> None:
    summary = _harbor_error_summary(ERRORED_RESULT)
    assert summary == "1/1 trials errored: EnvironmentStartTimeoutError"


def test_error_summary_counts_errored_trials_without_exception_stats() -> None:
    result = {
        "n_total_trials": 4,
        "stats": {"n_completed_trials": 2, "n_errored_trials": 2},
    }
    assert _harbor_error_summary(result) == "2/4 trials errored"


def test_error_summary_aggregates_exception_counts_across_evals() -> None:
    result = {
        "n_total_trials": 3,
        "stats": {
            "n_completed_trials": 0,
            "n_errored_trials": 3,
            "evals": {
                "a": {"exception_stats": {"SandboxExecutionError": 2}},
                "b": {"exception_stats": {"EnvironmentStartTimeoutError": 1}},
            },
        },
    }
    summary = _harbor_error_summary(result)
    assert summary == "3/3 trials errored: SandboxExecutionError x2, EnvironmentStartTimeoutError"


# --- file reader --------------------------------------------------------------


def test_file_reader_running_when_no_result(tmp_path: Path) -> None:
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)
    assert reader(_handle()).phase == "running"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_file_reader_failed_when_host_runner_exits_without_finished_result(tmp_path: Path, exit_code: int) -> None:
    exit_path = tmp_path / "runner.exit.json"
    exit_path.write_text(
        json.dumps({"token": "claim-token", "exit_code": exit_code, "finished_at": "2026-07-10T00:00:00Z"})
    )
    if exit_code == 0:
        _write_result(tmp_path, PARTIAL_RESULT)
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)

    status = reader(_handle(exit_file=str(exit_path)))

    assert status.phase == "failed"
    assert f"exited {exit_code}" in (status.detail or "")


def test_file_reader_finished_result_wins_over_host_exit(tmp_path: Path) -> None:
    _write_result(tmp_path, FINISHED_RESULT)
    exit_path = tmp_path / "runner.exit.json"
    exit_path.write_text(json.dumps({"token": "claim-token", "exit_code": 1, "finished_at": "2026-07-10T00:00:00Z"}))
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)

    assert reader(_handle(exit_file=str(exit_path))).phase == "succeeded"


def test_file_reader_malformed_terminal_metadata_fails(tmp_path: Path) -> None:
    exit_path = tmp_path / "runner.exit.json"
    exit_path.write_text("not-json")
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)

    status = reader(_handle(exit_file=str(exit_path)))

    assert status.phase == "failed"
    assert "terminal metadata is invalid" in (status.detail or "")


def test_file_reader_running_when_result_partial(tmp_path: Path) -> None:
    _write_result(tmp_path, PARTIAL_RESULT)
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)
    status = reader(_handle())
    assert status.phase == "running"  # was deceptively "succeeded" before the fix


def test_file_reader_succeeded_when_finished(tmp_path: Path) -> None:
    _write_result(tmp_path, FINISHED_RESULT)
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)
    status = reader(_handle())
    assert status.phase == "succeeded"
    assert status.raw == FINISHED_RESULT


def test_file_reader_failed_when_finished_with_errored_trials(tmp_path: Path) -> None:
    # Infra/environment error: the run finished but its trial errored. Must read
    # as failed (was a deceptive succeeded/reward-null before this fix), with the
    # exception named in the detail and the envelope preserved for the worker.
    _write_result(tmp_path, ERRORED_RESULT)
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)
    status = reader(_handle())
    assert status.phase == "failed"
    assert "1/1 trials errored" in (status.detail or "")
    assert "EnvironmentStartTimeoutError" in (status.detail or "")
    assert status.raw == ERRORED_RESULT


def test_file_reader_succeeded_for_genuinely_scored_zero(tmp_path: Path) -> None:
    # A real reward-0 run (every trial completed) must stay succeeded — we do not
    # conflate a genuine zero with an infra error.
    _write_result(tmp_path, SCORED_ZERO_RESULT)
    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)
    assert reader(_handle()).phase == "succeeded"


def test_file_reader_reports_sandbox_oom_kill(tmp_path: Path) -> None:
    _write_result(tmp_path, ERRORED_RESULT)
    status_dir = tmp_path / JOBS_DIR / EID / "task__abc" / "artifacts" / "k8s"
    status_dir.mkdir(parents=True)
    (status_dir / "pod-status.json").write_text(
        json.dumps(
            {
                "container_statuses": [
                    {
                        "name": "sandbox",
                        "state": {
                            "terminated": {
                                "exit_code": 137,
                                "reason": "OOMKilled",
                            }
                        },
                    }
                ]
            }
        )
    )

    reader = make_sandbox_k8s_status_reader(harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR)
    status = reader(_handle())

    assert status.phase == "failed"
    assert status.detail == "harbor sandbox failed (sandbox container was OOMKilled, exit 137)"


# --- docker reader ------------------------------------------------------------


class _FakeContainer:
    def __init__(self, *, running: bool, exit_code: int) -> None:
        self.attrs = {"State": {"Running": running, "ExitCode": exit_code}}

    def reload(self) -> None:  # noqa: D401 - test stub
        pass

    def logs(self, tail: int = 80) -> bytes:  # noqa: ARG002
        return b"boom: FileNotFoundError task.toml"


def _install_fake_docker(monkeypatch: pytest.MonkeyPatch, container: _FakeContainer) -> None:
    docker_mod = types.ModuleType("docker")
    errors_mod = types.ModuleType("docker.errors")

    class NotFound(Exception):
        pass

    errors_mod.NotFound = NotFound  # type: ignore[attr-defined]

    class _Containers:
        def get(self, _name: str) -> _FakeContainer:
            return container

    class _Client:
        containers = _Containers()

    docker_mod.from_env = lambda: _Client()  # type: ignore[attr-defined]
    docker_mod.errors = errors_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docker", docker_mod)
    monkeypatch.setitem(sys.modules, "docker.errors", errors_mod)


def test_docker_reader_failed_when_container_exited_with_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prod regression: container exited 1, result.json present but partial.
    _write_result(tmp_path, PARTIAL_RESULT)
    _install_fake_docker(monkeypatch, _FakeContainer(running=False, exit_code=1))
    reader = make_sandbox_k8s_docker_status_reader(
        harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR, work_dir=str(tmp_path / "work")
    )
    status = reader(_handle())
    assert status.phase == "failed"
    assert "exited 1" in (status.detail or "")


def test_docker_reader_succeeded_when_container_exited_with_finished_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_result(tmp_path, FINISHED_RESULT)
    _install_fake_docker(monkeypatch, _FakeContainer(running=False, exit_code=0))
    reader = make_sandbox_k8s_docker_status_reader(
        harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR, work_dir=str(tmp_path / "work")
    )
    status = reader(_handle())
    assert status.phase == "succeeded"
    assert status.raw == FINISHED_RESULT


def test_docker_reader_uses_explicit_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "harbor-jobs" / "results"
    _write_artifact_result(artifact_root, FINISHED_RESULT)
    _install_fake_docker(monkeypatch, _FakeContainer(running=False, exit_code=0))
    reader = make_sandbox_k8s_docker_status_reader(
        harbor_dir=str(tmp_path / "harbor"),
        jobs_dir=JOBS_DIR,
        work_dir=str(tmp_path / "work"),
        artifact_root=str(artifact_root),
    )

    status = reader(_handle())

    assert status.phase == "succeeded"
    assert status.raw == FINISHED_RESULT


def test_docker_reader_failed_when_finished_result_has_errored_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Container exited cleanly and Harbor completed the run, but a trial errored
    # (infra/environment failure). Deferring to the file reader must surface this
    # as failed, not a succeeded run scored from the trials that did complete.
    _write_result(tmp_path, ERRORED_RESULT)
    _install_fake_docker(monkeypatch, _FakeContainer(running=False, exit_code=0))
    reader = make_sandbox_k8s_docker_status_reader(
        harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR, work_dir=str(tmp_path / "work")
    )
    status = reader(_handle())
    assert status.phase == "failed"
    assert "EnvironmentStartTimeoutError" in (status.detail or "")


def test_docker_reader_running_while_container_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_docker(monkeypatch, _FakeContainer(running=True, exit_code=0))
    reader = make_sandbox_k8s_docker_status_reader(
        harbor_dir=str(tmp_path), jobs_dir=JOBS_DIR, work_dir=str(tmp_path / "work")
    )
    assert reader(_handle()).phase == "running"


def _enable_docker_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal settings to take build_backend's harbor-runner (docker) branch."""
    monkeypatch.setattr(settings, "sandbox_k8s_enabled", True)
    monkeypatch.setattr(settings, "sandbox_k8s_config_path", "/harness/agent-sandbox/configs/x.yaml")
    monkeypatch.setattr(settings, "sandbox_k8s_env_file", "/harness/agent-sandbox/targets/x.env")
    monkeypatch.setattr(settings, "harbor_runner_image", "scaled-evals-harbor-runner:dev")
    monkeypatch.setattr(settings, "kube_config_dir_host", "/home/user/.kube")


def test_build_backend_rejects_relative_jobs_dir_without_host_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The silent-data-loss trap: a relative HARBOR_JOBS_DIR on the docker path
    # with no SCALED_EVALS_HOST_DIR resolves against the worker cwd, so the
    # runner writes result.json where the worker can't read it. Fail loud.
    _enable_docker_runner(monkeypatch)
    monkeypatch.setattr(settings, "harbor_jobs_dir", "./logs")
    monkeypatch.setattr(settings, "scaled_evals_host_dir", "")
    with pytest.raises(RuntimeError, match="SCALED_EVALS_HOST_DIR"):
        build_backend()


def test_build_backend_accepts_relative_jobs_dir_with_host_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_docker_runner(monkeypatch)
    monkeypatch.setattr(settings, "harbor_jobs_dir", "./logs")
    monkeypatch.setattr(settings, "scaled_evals_host_dir", "/host/checkout")
    build_backend()  # anchored -> no raise


def test_build_backend_accepts_absolute_jobs_dir_without_host_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_docker_runner(monkeypatch)
    monkeypatch.setattr(settings, "harbor_jobs_dir", "/var/lib/scaled-evals/jobs")
    monkeypatch.setattr(settings, "scaled_evals_host_dir", "")
    build_backend()  # absolute bind source needs no anchor -> no raise
