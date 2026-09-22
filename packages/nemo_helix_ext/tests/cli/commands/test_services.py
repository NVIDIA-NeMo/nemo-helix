# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-level tests for ``nemo services`` commands.

Tests use the Typer CliRunner with mocked process internals.  The ``base_dir``
fixture + ``_NHX_STATE_DIR`` env var ensure all state goes to a temp directory.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.local.process import (
    ForegroundInstanceError,
    InstanceDescriptor,
    StopResult,
    acquire_lock,
    instance_dir,
    read_descriptor,
    write_descriptor,
)
from nhx.platform_runner.config import HelixAppConfig
from typer.testing import CliRunner

runner = CliRunner()

_PROCESS_MODULE = "nemo_helix_ext.local.process"
_CLI_MODULE = "nemo_helix_ext.cli.commands.services.cli"


def _seed_stopped_scope(base_dir: Path, scope: str, *, log_content: str = "x\n") -> Path:
    """Create a stopped scope directory with service logs."""
    d = instance_dir(scope, base_dir=base_dir)
    (d / "services.log").write_text(log_content)
    return d


_original_find_spec = __import__("importlib").util.find_spec


def _no_pyleak(name, *a, **kw):
    """Mock find_spec that hides ``pyleak`` while passing everything else through."""
    if name == "pyleak":
        return None
    return _original_find_spec(name, *a, **kw)


@pytest.fixture()
def base_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "state" / "nhx"
    monkeypatch.setenv("_NHX_STATE_DIR", str(d))
    return d


# ---------------------------------------------------------------------------
# Group-level help
# ---------------------------------------------------------------------------


def test_services_group_is_registered():
    result = runner.invoke(app, ["services", "--help"])
    assert result.exit_code == 0
    assert "Run platform services locally." in result.stdout
    assert "run" in result.stdout


def test_services_help_lists_all_commands():
    result = runner.invoke(app, ["services", "--help"])
    assert result.exit_code == 0
    for cmd in ("run", "start", "stop", "restart", "status", "ls", "logs", "rm", "prune"):
        assert cmd in result.stdout, f"'{cmd}' not in help output"


@pytest.mark.parametrize("command", ["run", "start", "restart"])
@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_services_reject_non_positive_keep_alive_timeout(command: str, timeout: str):
    with patch(f"{_CLI_MODULE}.stop_instance") as mock_stop:
        result = runner.invoke(app, ["services", command, "--keep-alive-timeout-seconds", timeout])

    assert result.exit_code != 0
    assert "--keep-alive-timeout-seconds" in f"{result.stdout}{result.stderr}"
    if command == "restart":
        mock_stop.assert_not_called()


# ---------------------------------------------------------------------------
# run (foreground)
# ---------------------------------------------------------------------------


def test_run_shows_services_extra_hint_when_pyleak_missing():
    with patch(f"{_CLI_MODULE}.importlib.util.find_spec", side_effect=_no_pyleak):
        result = runner.invoke(app, ["services", "run"])
    assert result.exit_code == 1
    assert "nemo-helix[all]" in result.stderr


def test_run_invokes_runner(base_dir: Path):
    mock_module = ModuleType("nhx.platform_runner.run")
    mock_run_platform = MagicMock()
    mock_module.run_platform = mock_run_platform

    with (
        patch.dict("sys.modules", {"nhx.platform_runner.run": mock_module}),
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_PROCESS_MODULE}.get_create_time", return_value=1000.0),
    ):
        result = runner.invoke(
            app,
            [
                "services",
                "run",
                "--services",
                "auth,entities",
                "--controllers",
                "jobs,models",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--keep-alive-timeout-seconds",
                "12",
                "--instance",
                "test-run",
            ],
        )

    assert result.exit_code == 0, result.stderr
    mock_run_platform.assert_called_once()
    _, kwargs = mock_run_platform.call_args
    config = kwargs["config"]
    assert config.services == ["auth", "entities"]
    assert config.service_group is None
    assert config.controllers == ["jobs", "models"]
    assert config.controller_group is None
    assert config.sidecars is None
    assert config.config_path is None
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.keep_alive_timeout_seconds == 12
    assert kwargs["on_shutdown"] is not None


def test_run_refuses_when_already_running(base_dir: Path):
    fd = acquire_lock("already-8080", base_dir=base_dir)
    try:
        with (
            patch(f"{_CLI_MODULE}._require_services_extra"),
            patch(f"{_CLI_MODULE}.compute_scope", return_value="already-8080"),
        ):
            result = runner.invoke(app, ["services", "run"])
        assert result.exit_code == 1
        assert "already running" in result.stderr
    finally:
        os.close(fd)


def test_run_writes_descriptor(base_dir: Path):
    mock_module = ModuleType("nhx.platform_runner.run")
    mock_module.run_platform = MagicMock()

    with (
        patch.dict("sys.modules", {"nhx.platform_runner.run": mock_module}),
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_PROCESS_MODULE}.get_create_time", return_value=1000.0),
    ):
        result = runner.invoke(
            app,
            ["services", "run", "--instance", "desc-test", "--port", "9999"],
        )

    assert result.exit_code == 0, result.stderr
    desc = read_descriptor("desc-test", base_dir=base_dir)
    assert desc is not None
    assert desc.mode == "foreground"
    assert desc.config.port == 9999


def test_run_records_background_mode_when_launched_by_start(base_dir: Path):
    mock_module = ModuleType("nhx.platform_runner.run")
    mock_module.run_platform = MagicMock()

    with (
        patch.dict("sys.modules", {"nhx.platform_runner.run": mock_module}),
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_PROCESS_MODULE}.get_create_time", return_value=1000.0),
        patch.dict(os.environ, {"_NHX_LAUNCH_MODE": "background"}),
    ):
        result = runner.invoke(
            app,
            ["services", "run", "--instance", "bg-mode-test", "--port", "9999"],
        )

    assert result.exit_code == 0, result.stderr
    desc = read_descriptor("bg-mode-test", base_dir=base_dir)
    assert desc is not None
    assert desc.mode == "background"


# ---------------------------------------------------------------------------
# start (background)
# ---------------------------------------------------------------------------


def test_start_shows_extra_hint_when_pyleak_missing():
    with patch(f"{_CLI_MODULE}.importlib.util.find_spec", side_effect=_no_pyleak):
        result = runner.invoke(app, ["services", "start"])
    assert result.exit_code == 1
    assert "nemo-helix[all]" in result.stderr


def test_start_launches_background(base_dir: Path):
    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.poll.return_value = None

    with (
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_CLI_MODULE}.start_background", return_value=mock_proc) as mock_start,
        patch(f"{_CLI_MODULE}._wait_for_healthy", return_value=True),
    ):
        result = runner.invoke(
            app,
            ["services", "start", "--instance", "bg-test", "--keep-alive-timeout-seconds", "12"],
        )

    assert result.exit_code == 0
    assert "99999" in result.stdout
    config = mock_start.call_args.args[0]
    assert config.keep_alive_timeout_seconds == 12


def test_start_refuses_when_already_running(base_dir: Path):
    fd = acquire_lock("already-8080", base_dir=base_dir)
    try:
        with (
            patch(f"{_CLI_MODULE}._require_services_extra"),
            patch(f"{_CLI_MODULE}.compute_scope", return_value="already-8080"),
        ):
            result = runner.invoke(app, ["services", "start"])
        assert result.exit_code == 1
        assert "already running" in result.stderr
    finally:
        os.close(fd)


def test_start_exits_early_when_port_occupied_by_foreign_process(base_dir: Path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.listen(1)

        with (
            patch(f"{_CLI_MODULE}._require_services_extra"),
            patch(f"{_CLI_MODULE}.start_background") as mock_start,
            patch(f"{_CLI_MODULE}.compute_scope", return_value=f"port-test-{port}"),
        ):
            result = runner.invoke(
                app,
                ["services", "start", "--port", str(port), "--instance", f"port-test-{port}"],
            )

    assert result.exit_code == 1
    assert "already in use" in result.stderr
    assert "lsof" in result.stderr
    assert "services.log" not in result.stderr
    mock_start.assert_not_called()


def test_run_exits_early_when_port_occupied_by_foreign_process(base_dir: Path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.listen(1)

        with (
            patch(f"{_CLI_MODULE}._require_services_extra"),
            patch(f"{_CLI_MODULE}.compute_scope", return_value=f"run-port-test-{port}"),
        ):
            result = runner.invoke(
                app,
                ["services", "run", "--port", str(port), "--instance", f"run-port-test-{port}"],
            )

    assert result.exit_code == 1
    assert "already in use" in result.stderr
    assert "lsof" in result.stderr


def test_start_reports_failure(base_dir: Path):
    mock_proc = MagicMock()
    mock_proc.pid = 88888
    mock_proc.poll.return_value = 1

    with (
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_CLI_MODULE}.start_background", return_value=mock_proc),
        patch(f"{_CLI_MODULE}._wait_for_healthy", return_value=False),
    ):
        result = runner.invoke(
            app,
            ["services", "start", "--instance", "fail-test"],
        )

    assert result.exit_code == 1
    assert "exited early" in result.stderr


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestServicesStop:
    def test_no_running_services(self, base_dir: Path):
        with patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[])):
            result = runner.invoke(app, ["services", "stop", "--instance", "none"])
        assert result.exit_code == 0
        assert "not running" in result.stdout.lower()

    def test_stops_running_services(self, base_dir: Path):
        with patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[12345])):
            with patch(f"{_CLI_MODULE}.compute_scope", return_value="stop-scope"):
                result = runner.invoke(app, ["services", "stop", "--instance", "stop-scope"])
        assert result.exit_code == 0
        assert "12345" in result.stdout
        assert "nemo services rm stop-scope" in result.stdout

    def test_stop_timeout_option(self, base_dir: Path):
        with patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[])) as mock_stop:
            runner.invoke(app, ["services", "stop", "--timeout", "60", "--instance", "test"])
        _, kwargs = mock_stop.call_args
        assert kwargs["timeout"] == 60.0

    def test_stop_refuses_foreground_instance(self, base_dir: Path):
        with patch(
            f"{_CLI_MODULE}.stop_instance",
            side_effect=ForegroundInstanceError("fg-scope", 12345),
        ):
            result = runner.invoke(app, ["services", "stop", "--instance", "fg-scope"])
        assert result.exit_code == 1
        assert "foreground" in result.stderr

    def test_stop_force_overrides_foreground_protection(self, base_dir: Path):
        with patch(
            f"{_CLI_MODULE}.stop_instance",
            return_value=StopResult(stopped_pids=[12345]),
        ) as mock_stop:
            result = runner.invoke(
                app,
                ["services", "stop", "--instance", "fg-scope", "--force"],
            )
        assert result.exit_code == 0
        _, kwargs = mock_stop.call_args
        assert kwargs["force"] is True


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------


class TestServicesRestart:
    def test_restart_rejects_services_with_service_group(self, base_dir: Path):
        result = runner.invoke(
            app,
            ["services", "restart", "--services", "entities", "--service-group", "default"],
        )
        assert result.exit_code != 0

    def test_restart_rejects_controllers_with_controller_group(self, base_dir: Path):
        result = runner.invoke(
            app,
            ["services", "restart", "--controllers", "jobs", "--controller-group", "default"],
        )
        assert result.exit_code != 0

    def test_restart_checks_extras_before_stopping(self, base_dir: Path):
        with (
            patch(f"{_CLI_MODULE}.importlib.util.find_spec", side_effect=_no_pyleak),
            patch(f"{_CLI_MODULE}.stop_instance") as mock_stop,
        ):
            result = runner.invoke(app, ["services", "restart"])
        assert result.exit_code == 1
        assert "nemo-helix[all]" in result.stderr
        mock_stop.assert_not_called()

    def test_restart_errors_when_no_prior_instance(self, base_dir: Path):
        with patch(f"{_CLI_MODULE}._require_services_extra"):
            result = runner.invoke(
                app,
                ["services", "restart", "--instance", "ghost"],
            )
        assert result.exit_code == 1
        assert "No instance found for scope" in result.stderr
        assert "nemo services start" in result.stderr

    def test_restart_stops_and_starts(self, base_dir: Path):
        scope = "restart-test"
        fd = acquire_lock(scope, base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=os.getpid(),
            config=HelixAppConfig(scope=scope),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None

        try:
            with (
                patch(f"{_CLI_MODULE}._require_services_extra"),
                patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[os.getpid()])),
                patch(f"{_CLI_MODULE}.start_background", return_value=mock_proc),
                patch(f"{_CLI_MODULE}._wait_for_healthy", return_value=True),
            ):
                result = runner.invoke(
                    app,
                    ["services", "restart", "--instance", scope],
                )
        finally:
            os.close(fd)

        assert result.exit_code == 0
        assert "restarted" in result.stdout.lower()

    def test_restart_exits_early_when_port_occupied_by_foreign_process(self, base_dir: Path):
        scope = "restart-foreign-port"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.listen(1)

            desc = InstanceDescriptor(
                pid=99999,
                config=HelixAppConfig(scope=scope, host="127.0.0.1", port=port),
                mode="background",
                create_time=1.0,
            )
            write_descriptor(desc, base_dir=base_dir)

            with (
                patch(f"{_CLI_MODULE}._require_services_extra"),
                patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[99999])),
                patch(f"{_CLI_MODULE}.start_background") as mock_start,
                patch(f"{_CLI_MODULE}.compute_scope", return_value=scope),
            ):
                result = runner.invoke(
                    app,
                    ["services", "restart", "--instance", scope, "--port", str(port)],
                )

        assert result.exit_code == 1
        assert "already in use" in result.stderr
        assert "lsof" in result.stderr
        assert "services.log" not in result.stderr
        mock_start.assert_not_called()

    def test_restart_preserves_previous_args(self, base_dir: Path):
        scope = "preserve-test"
        fd = acquire_lock(scope, base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=os.getpid(),
            config=HelixAppConfig(
                scope=scope,
                services=["entities", "models"],
                controllers=["jobs"],
                host="127.0.0.1",
                port=9000,
                keep_alive_timeout_seconds=12,
            ),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)

        mock_proc = MagicMock()
        mock_proc.pid = 22222
        mock_proc.poll.return_value = None

        try:
            with (
                patch(f"{_CLI_MODULE}._require_services_extra"),
                patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[os.getpid()])),
                patch(f"{_CLI_MODULE}.start_background", return_value=mock_proc) as mock_start,
                patch(f"{_CLI_MODULE}._wait_for_healthy", return_value=True),
            ):
                result = runner.invoke(
                    app,
                    ["services", "restart", "--instance", scope],
                )
        finally:
            os.close(fd)

        assert result.exit_code == 0
        args, _kwargs = mock_start.call_args
        config = args[0]
        assert config.services == ["entities", "models"]
        assert config.controllers == ["jobs"]
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.keep_alive_timeout_seconds == 12

    def test_restart_overrides_previous_keep_alive_timeout(self, base_dir: Path):
        scope = "override-keep-alive-test"
        fd = acquire_lock(scope, base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=os.getpid(),
            config=HelixAppConfig(
                scope=scope,
                host="127.0.0.1",
                port=9000,
                keep_alive_timeout_seconds=12,
            ),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)

        mock_proc = MagicMock()
        mock_proc.pid = 22223
        mock_proc.poll.return_value = None

        try:
            with (
                patch(f"{_CLI_MODULE}._require_services_extra"),
                patch(f"{_CLI_MODULE}.stop_instance", return_value=StopResult(stopped_pids=[os.getpid()])),
                patch(f"{_CLI_MODULE}.start_background", return_value=mock_proc) as mock_start,
                patch(f"{_CLI_MODULE}._wait_for_healthy", return_value=True),
            ):
                result = runner.invoke(
                    app,
                    ["services", "restart", "--instance", scope, "--keep-alive-timeout-seconds", "15"],
                )
        finally:
            os.close(fd)

        assert result.exit_code == 0
        config = mock_start.call_args.args[0]
        assert config.keep_alive_timeout_seconds == 15


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestServicesStatus:
    def test_not_running(self, base_dir: Path):
        result = runner.invoke(app, ["services", "status", "--instance", "none"])
        assert result.exit_code == 0
        assert "No running instance for scope" in result.stdout

    def test_running_instance(self, base_dir: Path):
        scope = "status-test"
        fd = acquire_lock(scope, base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=os.getpid(),
            config=HelixAppConfig(scope=scope),
            mode="foreground",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)

        try:
            with patch(f"{_CLI_MODULE}._wait_for_healthy", return_value=True):
                result = runner.invoke(app, ["services", "status", "--instance", scope])
        finally:
            os.close(fd)

        assert result.exit_code == 0
        assert scope in result.stdout
        assert str(os.getpid()) in result.stdout
        assert "foreground" in result.stdout
        assert "healthy" in result.stdout


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


class TestServicesLs:
    def test_no_instances(self, base_dir: Path):
        result = runner.invoke(app, ["services", "ls"])
        assert result.exit_code == 0
        assert "No running instances" in result.stdout

    def test_lists_running_instance(self, base_dir: Path):
        scope = "ls-test"
        fd = acquire_lock(scope, base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=os.getpid(),
            config=HelixAppConfig(scope=scope),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)

        try:
            result = runner.invoke(app, ["services", "ls"])
        finally:
            os.close(fd)

        assert result.exit_code == 0
        assert scope in result.stdout
        assert "running" in result.stdout

    def test_hides_stopped_by_default(self, base_dir: Path):
        _seed_stopped_scope(base_dir, "stopped-hidden")

        result = runner.invoke(app, ["services", "ls"])
        assert result.exit_code == 0
        assert "stopped-hidden" not in result.stdout
        assert "1 stopped instance directory" in result.stdout
        assert "ls --all" in result.stdout

    def test_all_shows_stopped_with_footer(self, base_dir: Path):
        _seed_stopped_scope(base_dir, "stopped-visible")

        result = runner.invoke(app, ["services", "ls", "--all"])
        assert result.exit_code == 0
        assert "stopped-visible" in result.stdout
        assert "stopped" in result.stdout
        assert "nemo services prune" in result.stdout

    def test_all_no_instances(self, base_dir: Path):
        result = runner.invoke(app, ["services", "ls", "--all"])
        assert result.exit_code == 0
        assert "No instances found" in result.stdout

    def test_mixed_running_and_stopped(self, base_dir: Path):
        running_scope = "running-mixed"
        fd = acquire_lock(running_scope, base_dir=base_dir)
        write_descriptor(
            InstanceDescriptor(
                pid=os.getpid(),
                config=HelixAppConfig(scope=running_scope),
                mode="background",
                create_time=1.0,
            ),
            base_dir=base_dir,
        )
        _seed_stopped_scope(base_dir, "stopped-mixed")

        try:
            default = runner.invoke(app, ["services", "ls"])
            all_rows = runner.invoke(app, ["services", "ls", "--all"])
        finally:
            os.close(fd)

        assert default.exit_code == 0
        assert running_scope in default.stdout
        assert "stopped-mixed" not in default.stdout

        assert all_rows.exit_code == 0
        assert running_scope in all_rows.stdout
        assert "stopped-mixed" in all_rows.stdout
        assert "nemo services prune" in all_rows.stdout


class TestServicesRm:
    @pytest.mark.parametrize(
        ("scope", "args"),
        [
            ("rm-cli", ["services", "rm", "rm-cli"]),
            ("rm-flag", ["services", "rm", "--instance", "rm-flag"]),
        ],
    )
    def test_rm_removes_stopped_scope(self, base_dir: Path, scope: str, args: list[str]) -> None:
        d = _seed_stopped_scope(base_dir, scope)

        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert f"Removed instance '{scope}'" in result.stdout
        assert not d.exists()

    def test_rm_not_found(self, base_dir: Path):
        result = runner.invoke(app, ["services", "rm", "missing-scope"])
        assert result.exit_code == 1
        assert "No stopped instance directory found" in result.stderr

    def test_rm_refuses_running(self, base_dir: Path):
        fd = acquire_lock("running-rm-cli", base_dir=base_dir)
        try:
            result = runner.invoke(app, ["services", "rm", "running-rm-cli"])
        finally:
            os.close(fd)
        assert result.exit_code == 1
        assert "still running" in result.stderr

    def test_rm_requires_scope(self, base_dir: Path):
        result = runner.invoke(app, ["services", "rm"])
        assert result.exit_code == 1
        assert "Scope required" in result.stderr

    def test_rm_rejects_invalid_scope(self, base_dir: Path):
        result = runner.invoke(app, ["services", "rm", "../escape"])
        assert result.exit_code == 1
        assert "Invalid scope" in result.stderr

    def test_rm_rejects_conflicting_scope_args(self, base_dir: Path):
        result = runner.invoke(app, ["services", "rm", "scope-a", "--instance", "scope-b"])
        assert result.exit_code != 0
        assert "Cannot pass different values" in result.stderr


class TestServicesPrune:
    def test_prune_force_removes_stopped(self, base_dir: Path):
        d = _seed_stopped_scope(base_dir, "prune-me")

        result = runner.invoke(app, ["services", "prune", "--force"])
        assert result.exit_code == 0
        assert "prune-me" in result.stdout
        assert not d.exists()

    def test_prune_noop_when_clean(self, base_dir: Path):
        result = runner.invoke(app, ["services", "prune", "--force"])
        assert result.exit_code == 0
        assert "No stopped instance directories" in result.stdout

    def test_prune_cancelled(self, base_dir: Path):
        d = _seed_stopped_scope(base_dir, "keep-me")

        result = runner.invoke(app, ["services", "prune"], input="n\n")
        assert result.exit_code == 0
        assert "Prune cancelled" in result.stdout
        assert d.exists()


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


class TestServicesLogs:
    def test_path_only(self, base_dir: Path):
        result = runner.invoke(
            app,
            ["services", "logs", "--path", "--instance", "log-test"],
        )
        assert result.exit_code == 0
        assert "services.log" in result.stdout

    def test_tail_log(self, base_dir: Path):
        scope = "tail-test"
        d = instance_dir(scope, base_dir=base_dir)
        (d / "services.log").write_text("line1\nline2\nline3\n")

        result = runner.invoke(
            app,
            ["services", "logs", "--instance", scope, "-n", "2"],
        )
        assert result.exit_code == 0
        assert "line2" in result.stdout
        assert "line3" in result.stdout

    def test_no_log_file(self, base_dir: Path):
        result = runner.invoke(
            app,
            ["services", "logs", "--instance", "nolog"],
        )
        assert result.exit_code == 0
        assert "No log file" in result.stdout


# ---------------------------------------------------------------------------
# Default host (127.0.0.1)
# ---------------------------------------------------------------------------


def test_default_host_is_loopback(base_dir: Path):
    mock_module = ModuleType("nhx.platform_runner.run")
    mock_run_platform = MagicMock()
    mock_module.run_platform = mock_run_platform

    with (
        patch.dict("sys.modules", {"nhx.platform_runner.run": mock_module}),
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_PROCESS_MODULE}.get_create_time", return_value=1000.0),
    ):
        result = runner.invoke(
            app,
            ["services", "run", "--instance", "loopback-test"],
        )

    assert result.exit_code == 0, result.stderr
    _, kwargs = mock_run_platform.call_args
    assert kwargs["config"].host == "127.0.0.1"


def test_bind_all_warning(base_dir: Path):
    mock_module = ModuleType("nhx.platform_runner.run")
    mock_module.run_platform = MagicMock()

    with (
        patch.dict("sys.modules", {"nhx.platform_runner.run": mock_module}),
        patch(f"{_CLI_MODULE}._require_services_extra"),
        patch(f"{_PROCESS_MODULE}.get_create_time", return_value=1000.0),
    ):
        result = runner.invoke(
            app,
            ["services", "run", "--host", "0.0.0.0", "--instance", "warn-test"],
        )

    assert result.exit_code == 0
    assert "0.0.0.0" in result.stderr
    assert "network interfaces" in result.stderr


# ---------------------------------------------------------------------------
# start --services / --service-group mutual exclusion
# ---------------------------------------------------------------------------


def test_start_rejects_services_with_service_group(base_dir: Path):
    with patch(f"{_CLI_MODULE}._require_services_extra"):
        result = runner.invoke(
            app,
            ["services", "start", "--services", "a", "--service-group", "b"],
        )
    assert result.exit_code != 0


def test_start_rejects_controllers_with_controller_group(base_dir: Path):
    with patch(f"{_CLI_MODULE}._require_services_extra"):
        result = runner.invoke(
            app,
            ["services", "start", "--controllers", "a", "--controller-group", "b"],
        )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _wait_for_healthy
# ---------------------------------------------------------------------------


class TestWaitForHealthy:
    """Unit tests for the health polling function."""

    def test_returns_true_on_immediate_200(self):
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch(f"{_CLI_MODULE}.httpx.get", return_value=mock_resp):
            result = _wait_for_healthy("127.0.0.1", 8080, timeout=5, poll_interval=0.1)

        assert result is True

    def test_returns_false_on_timeout(self):
        import httpx
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        with patch(f"{_CLI_MODULE}.httpx.get", side_effect=httpx.ConnectError("")):
            result = _wait_for_healthy("127.0.0.1", 8080, timeout=0.3, poll_interval=0.1)

        assert result is False

    def test_retries_until_success(self):
        import httpx
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        call_count = 0

        def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("")
            return mock_resp

        with patch(f"{_CLI_MODULE}.httpx.get", side_effect=fail_then_succeed):
            result = _wait_for_healthy("127.0.0.1", 8080, timeout=5, poll_interval=0.1)

        assert result is True
        assert call_count == 3

    def test_translates_wildcard_host_to_localhost(self):
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        captured_urls: list[str] = []

        def capture_get(url, **kwargs):
            captured_urls.append(url)
            return mock_resp

        with patch(f"{_CLI_MODULE}.httpx.get", side_effect=capture_get):
            _wait_for_healthy("0.0.0.0", 9090, timeout=2, poll_interval=0.1)

        assert "localhost" in captured_urls[0]
        assert "0.0.0.0" not in captured_urls[0]

    def test_non_200_keeps_polling(self):
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_200 = MagicMock()
        resp_200.status_code = 200
        responses = [resp_503, resp_503, resp_200]

        with patch(f"{_CLI_MODULE}.httpx.get", side_effect=responses):
            result = _wait_for_healthy("127.0.0.1", 8080, timeout=5, poll_interval=0.1)

        assert result is True
