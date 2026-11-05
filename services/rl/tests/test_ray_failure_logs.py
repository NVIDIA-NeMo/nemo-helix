# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray session logs are printed on failure, and the usual startup failures are classified."""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx.customization_common.training.errors.classifier import get_classifier
from nhx.rl.tasks.training.backends.nemo_rl import ray_bootstrap
from nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap import RayClusterBootstrap
from nhx.rl.tasks.training.backends.nemo_rl.ray_failure_logs import (
    diagnose_ray_startup,
    read_ray_failure_logs,
    render_ray_failure_logs,
)

# The paragraph Ray prints for every agent death. It names all three causes and is none of them.
_CANNED_HINT = "\n".join(
    [
        "The raylet fate shares with the agent. This can happen because",
        "- The version of `grpcio` doesn't follow Ray's requirement. Agent can segfault with the incorrect `grpcio` version. Check the grpcio version `pip freeze | grep grpcio`.",
        "- The agent failed to start because of unexpected error or port conflict. Read the log `cat /tmp/ray/session_latest/logs/{dashboard_agent|runtime_env_agent}.log`.",
        "- The agent is killed by the OS (e.g., out of memory).",
        'reason_message: "runtime_env_agent failed and raylet fate-shares with it."',
    ]
)


def _logs(tmp_path: Path, **files: str) -> dict[str, list[str]]:
    log_dir = tmp_path / "session_latest" / "logs"
    log_dir.mkdir(parents=True)
    for name, text in files.items():
        (log_dir / name).write_text(text)
    return read_ray_failure_logs(log_dir)


def test_a_port_conflict_in_the_agent_log_is_not_the_canned_hint(tmp_path: Path) -> None:
    logs = _logs(
        tmp_path,
        **{
            "runtime_env_agent.log": "OSError: [Errno 98] Address already in use\n",
            "raylet.out": _CANNED_HINT + "\n",
        },
    )

    diagnosis = diagnose_ray_startup(logs, startup_failure=False)

    assert diagnosis is not None
    assert "port is already in use" in diagnosis
    assert get_classifier().classify_output([diagnosis], 1).cause == "ray_agent_port_conflict"


def test_the_canned_hint_alone_is_the_fate_share_notice(tmp_path: Path) -> None:
    logs = _logs(tmp_path, **{"raylet.out": _CANNED_HINT + "\n"})

    diagnosis = diagnose_ray_startup(logs, startup_failure=False)

    assert diagnosis == "RuntimeError: runtime_env_agent failed and raylet fate-shares with it"
    assert get_classifier().classify_output([diagnosis], 1).cause == "ray_agent_fate_share"


def test_a_segfault_in_the_agent_log_is_a_grpcio_crash(tmp_path: Path) -> None:
    logs = _logs(
        tmp_path,
        **{
            "runtime_env_agent.err": "Segmentation fault (core dumped)\n",
            "raylet.err": _CANNED_HINT + "\n",
        },
    )

    diagnosis = diagnose_ray_startup(logs, startup_failure=False)

    assert diagnosis is not None
    assert "grpcio" in diagnosis
    assert get_classifier().classify_output([diagnosis], 1).cause == "ray_agent_grpcio_mismatch"


def test_an_agent_killed_by_the_os_is_not_the_hint(tmp_path: Path) -> None:
    logs = _logs(
        tmp_path,
        **{
            "dashboard_agent.log": "Killed\n",
            "raylet.out": 'reason_message: "dashboard_agent failed and raylet fate-shares with it."\n',
        },
    )

    diagnosis = diagnose_ray_startup(logs, startup_failure=False)

    assert diagnosis is not None
    assert "killed by the operating system" in diagnosis
    assert get_classifier().classify_output([diagnosis], 1).cause == "ray_agent_killed"


def test_a_stale_port_error_without_a_fate_share_is_not_reported_during_training(tmp_path: Path) -> None:
    logs = _logs(tmp_path, **{"runtime_env_agent.log": "OSError: [Errno 98] Address already in use\n"})

    assert diagnose_ray_startup(logs, startup_failure=False) is None


def test_a_startup_failure_reports_the_agent_exception_without_a_fate_share(tmp_path: Path) -> None:
    logs = _logs(tmp_path, **{"runtime_env_agent.log": "ValueError: worker failed to register\n"})

    diagnosis = diagnose_ray_startup(logs, startup_failure=True)

    assert diagnosis is not None
    assert diagnosis.startswith("RuntimeError: Ray agent failed during startup:")
    assert "worker failed to register" in diagnosis
    assert get_classifier().classify_output([diagnosis], 1).cause == "ray_agent_startup_error"


def test_render_names_each_log_and_an_empty_agent_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "session_latest" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "runtime_env_agent.log").write_text("")
    (log_dir / "raylet.out").write_text("hello\n")
    logs = read_ray_failure_logs(log_dir)

    rendered = render_ray_failure_logs(log_dir, logs)

    assert "runtime_env_agent.log" in rendered
    assert "(empty)" in rendered
    assert "hello" in rendered


def _session(tmp_path: Path, text: str) -> None:
    log_dir = tmp_path / "ray" / "session_latest" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "runtime_env_agent.log").write_text(text)
    (log_dir / "raylet.out").write_text(_CANNED_HINT + "\n")


def _head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RayClusterBootstrap:
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "ray"))
    monkeypatch.setattr(ray_bootstrap, "_pause", lambda _seconds: None)
    bootstrap = RayClusterBootstrap(rank=0, world_size=1, master_addr="127.0.0.1", log_dir=tmp_path / "logs")
    monkeypatch.setattr(bootstrap, "_start_head_background", lambda: True)
    monkeypatch.setattr(bootstrap, "_wait_for_workers", lambda: None)
    monkeypatch.setattr(bootstrap, "_log_ray_status", lambda: None)
    return bootstrap


def test_a_failed_driver_prints_ray_logs_before_ray_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _session(tmp_path, "OSError: [Errno 98] Address already in use\n")
    bootstrap = _head(tmp_path, monkeypatch)
    monkeypatch.setattr(bootstrap, "_run_driver", lambda _script, _args: 1)
    printed_before_cleanup: list[str] = []

    def cleanup() -> None:
        printed_before_cleanup.append(capsys.readouterr().out)

    monkeypatch.setattr(bootstrap, "_cleanup_with_timeout", cleanup)

    assert bootstrap._run_head_with_driver("driver.py", []) == 1

    assert printed_before_cleanup
    output = printed_before_cleanup[0]
    assert "runtime_env_agent.log" in output
    assert "Address already in use" in output
    assert any("port is already in use" in line for line in bootstrap.driver_output)


def test_a_head_that_never_starts_reports_the_agent_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _session(tmp_path, "OSError: [Errno 98] Address already in use\n")
    bootstrap = _head(tmp_path, monkeypatch)
    monkeypatch.setattr(bootstrap, "_start_head_background", lambda: False)
    monkeypatch.setattr(bootstrap, "_cleanup_with_timeout", lambda: None)

    assert bootstrap._run_head_with_driver("driver.py", []) == 1

    assert "Address already in use" in capsys.readouterr().out
    assert any("port is already in use" in line for line in bootstrap.driver_output)


def test_a_worker_that_fails_to_start_prints_its_ray_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _session(tmp_path, "Segmentation fault (core dumped)\n")
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "ray"))
    bootstrap = RayClusterBootstrap(rank=1, world_size=2, master_addr="head", log_dir=tmp_path / "logs")
    monkeypatch.setattr(bootstrap, "_start_worker_background", lambda: False)

    assert bootstrap._run_as_worker() == 1

    output = capsys.readouterr().out
    assert "Segmentation fault" in output
    assert any("grpcio" in line for line in bootstrap.driver_output)


def test_a_worker_prints_its_agent_log_when_that_agent_died(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _session(tmp_path, "OSError: [Errno 98] Address already in use\n")
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "ray"))
    monkeypatch.setattr(ray_bootstrap, "_pause", lambda _seconds: None)
    bootstrap = RayClusterBootstrap(rank=1, world_size=2, master_addr="head", log_dir=tmp_path / "logs")
    bootstrap.ended_file.touch()
    monkeypatch.setattr(bootstrap, "_start_worker_background", lambda: True)
    monkeypatch.setattr(bootstrap, "_stop_ray", lambda *args, **kwargs: None)

    assert bootstrap._run_as_worker() == 0

    assert "Address already in use" in capsys.readouterr().out


def test_a_healthy_worker_shutdown_does_not_dump_ray_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "empty-ray"))
    monkeypatch.setattr(ray_bootstrap, "_pause", lambda _seconds: None)
    bootstrap = RayClusterBootstrap(rank=1, world_size=2, master_addr="head", log_dir=tmp_path / "logs")
    bootstrap.ended_file.touch()
    monkeypatch.setattr(bootstrap, "_start_worker_background", lambda: True)
    monkeypatch.setattr(bootstrap, "_stop_ray", lambda *args, **kwargs: None)

    assert bootstrap._run_as_worker() == 0

    assert "session logs" not in capsys.readouterr().out


def test_a_pause_does_not_dump_ray_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _session(tmp_path, "OSError: [Errno 98] Address already in use\n")
    bootstrap = _head(tmp_path, monkeypatch)
    monkeypatch.setattr(bootstrap, "_run_driver", lambda _script, _args: 75)
    monkeypatch.setattr(bootstrap, "_cleanup_with_timeout", lambda: None)

    assert bootstrap._run_head_with_driver("driver.py", []) == 75

    assert "runtime_env_agent.log" not in capsys.readouterr().out
    assert list(bootstrap.driver_output) == []
