# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the flock/psutil-based process lifecycle in _process.py."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest
from nemo_helix_ext.local import process as process_module
from nemo_helix_ext.local.process import (
    ForegroundInstanceError,
    InstanceAlreadyRunningError,
    InstanceDescriptor,
    InstanceStillRunningError,
    _pid_alive,
    _snapshot_children,
    _sweep_orphans,
    acquire_lock,
    compute_scope,
    instance_dir,
    instance_log_bytes,
    is_instance_alive,
    is_removable_ghost,
    list_instances,
    list_stopped_scopes,
    prune_instances,
    read_descriptor,
    remove_descriptor,
    remove_instance,
    rotate_log,
    start_background,
    stop_instance,
    validate_pid,
    write_descriptor,
)
from nhx.platform_runner.config import DEFAULT_LOCAL_SERVICES_BIND_HOST, HelixAppConfig


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    return tmp_path / "state" / "nhx"


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


class TestComputeScope:
    def test_explicit_scope(self) -> None:
        assert compute_scope(port=1234, explicit_scope="myapp") == "myapp"

    def test_default_scope_is_stable_for_port(self) -> None:
        scope = compute_scope(port=8080)

        assert scope == compute_scope(port=8080)
        assert scope.endswith("-8080")


# ---------------------------------------------------------------------------
# Scope directory
# ---------------------------------------------------------------------------


class TestInstanceDir:
    def test_creates_directory(self, base_dir: Path) -> None:
        d = instance_dir("test-scope", base_dir=base_dir)
        assert d.is_dir()
        assert d.name == "test-scope"

    def test_idempotent(self, base_dir: Path) -> None:
        d1 = instance_dir("test-scope", base_dir=base_dir)
        d2 = instance_dir("test-scope", base_dir=base_dir)
        assert d1 == d2


# ---------------------------------------------------------------------------
# flock liveness
# ---------------------------------------------------------------------------


class TestFlockLiveness:
    def test_acquire_and_release(self, base_dir: Path) -> None:
        fd = acquire_lock("test", base_dir=base_dir)
        assert fd >= 0
        assert is_instance_alive("test", base_dir=base_dir)
        os.close(fd)
        assert not is_instance_alive("test", base_dir=base_dir)

    def test_double_acquire_raises(self, base_dir: Path) -> None:
        fd = acquire_lock("test", base_dir=base_dir)
        try:
            with pytest.raises(InstanceAlreadyRunningError, match="test"):
                acquire_lock("test", base_dir=base_dir)
        finally:
            os.close(fd)

    def test_different_scopes_independent(self, base_dir: Path) -> None:
        fd1 = acquire_lock("scope-a", base_dir=base_dir)
        fd2 = acquire_lock("scope-b", base_dir=base_dir)
        assert is_instance_alive("scope-a", base_dir=base_dir)
        assert is_instance_alive("scope-b", base_dir=base_dir)
        os.close(fd1)
        assert not is_instance_alive("scope-a", base_dir=base_dir)
        assert is_instance_alive("scope-b", base_dir=base_dir)
        os.close(fd2)

    def test_not_alive_when_no_lock_file(self, base_dir: Path) -> None:
        assert not is_instance_alive("nonexistent", base_dir=base_dir)


# ---------------------------------------------------------------------------
# Descriptor I/O
# ---------------------------------------------------------------------------


class TestDescriptorRoundTrip:
    def test_write_and_read(self, base_dir: Path) -> None:
        desc = InstanceDescriptor(
            pid=12345,
            config=HelixAppConfig(
                scope="test-8080",
                services=["entities", "models"],
                controllers=["jobs"],
                host="127.0.0.1",
            ),
            mode="background",
            create_time=1000.0,
        )
        write_descriptor(desc, base_dir=base_dir)
        recovered = read_descriptor("test-8080", base_dir=base_dir)

        assert recovered is not None
        assert recovered.pid == 12345
        assert recovered.config.scope == "test-8080"
        assert recovered.config.host == "127.0.0.1"
        assert recovered.config.port == 8080
        assert recovered.mode == "background"
        assert recovered.create_time == 1000.0
        assert recovered.config.services == ["entities", "models"]
        assert recovered.config.controllers == ["jobs"]

    def test_read_missing_returns_none(self, base_dir: Path) -> None:
        assert read_descriptor("no-such-scope", base_dir=base_dir) is None

    def test_read_corrupt_returns_none(self, base_dir: Path) -> None:
        d = instance_dir("corrupt", base_dir=base_dir)
        (d / "instance.json").write_text("not json{{{")
        assert read_descriptor("corrupt", base_dir=base_dir) is None

    def test_remove_descriptor(self, base_dir: Path) -> None:
        desc = InstanceDescriptor(
            pid=1,
            config=HelixAppConfig(scope="rm-test"),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)
        assert read_descriptor("rm-test", base_dir=base_dir) is not None
        remove_descriptor("rm-test", base_dir=base_dir)
        assert read_descriptor("rm-test", base_dir=base_dir) is None

    def test_remove_missing_is_noop(self, base_dir: Path) -> None:
        remove_descriptor("nonexistent", base_dir=base_dir)


# ---------------------------------------------------------------------------
# PID validation via psutil
# ---------------------------------------------------------------------------


class TestValidatePid:
    def test_validates_current_process(self) -> None:
        ct = psutil.Process(os.getpid()).create_time()
        assert validate_pid(os.getpid(), ct)

    def test_rejects_wrong_create_time(self) -> None:
        assert not validate_pid(os.getpid(), 0.0)

    def test_rejects_dead_pid(self) -> None:
        assert not validate_pid(999999999, 0.0)


# ---------------------------------------------------------------------------
# list_instances
# ---------------------------------------------------------------------------


class TestListInstances:
    def test_empty_when_no_instances(self, base_dir: Path) -> None:
        assert list_instances(base_dir=base_dir) == []

    def test_lists_alive_instance(self, base_dir: Path) -> None:
        fd = acquire_lock("alive-one", base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=os.getpid(),
            config=HelixAppConfig(scope="alive-one"),
            mode="foreground",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)
        try:
            instances = list_instances(base_dir=base_dir)
            assert len(instances) == 1
            assert instances[0].scope == "alive-one"
            assert instances[0].alive is True
            assert instances[0].descriptor is not None
        finally:
            os.close(fd)

    def test_cleans_up_dead_descriptor(self, base_dir: Path) -> None:
        d = instance_dir("dead-scope", base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=999999,
            config=HelixAppConfig(scope="dead-scope"),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)

        instances = list_instances(base_dir=base_dir)
        assert instances == []
        assert not d.exists()

    def test_stale_descriptor_with_logs_stays_listed(self, base_dir: Path) -> None:
        d = instance_dir("dead-with-logs", base_dir=base_dir)
        desc = InstanceDescriptor(
            pid=999999,
            config=HelixAppConfig(scope="dead-with-logs"),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)
        (d / "services.log").write_text("crash output\n")

        instances = list_instances(base_dir=base_dir)
        assert len(instances) == 1
        assert instances[0].scope == "dead-with-logs"
        assert instances[0].alive is False
        assert instances[0].descriptor is None
        assert d.exists()

    def test_auto_removes_lock_only_ghost(self, base_dir: Path) -> None:
        d = instance_dir("ghost-lock", base_dir=base_dir)
        (d / "services.lock").touch()

        instances = list_instances(base_dir=base_dir)
        assert instances == []
        assert not d.exists()

    def test_keeps_stopped_instance_with_logs(self, base_dir: Path) -> None:
        d = instance_dir("stopped-logs", base_dir=base_dir)
        (d / "services.lock").touch()
        (d / "services.log").write_text("crash output\n")

        instances = list_instances(base_dir=base_dir)
        assert len(instances) == 1
        assert instances[0].scope == "stopped-logs"
        assert instances[0].alive is False
        assert instances[0].descriptor is None
        assert d.exists()

    def test_keeps_stopped_instance_with_rotated_logs_only(self, base_dir: Path) -> None:
        d = instance_dir("rotated-logs", base_dir=base_dir)
        (d / "services.lock").touch()
        (d / "services.log.20250101T000000Z").write_text("old boot\n")

        instances = list_instances(base_dir=base_dir)
        assert len(instances) == 1
        assert instances[0].scope == "rotated-logs"

    def test_empty_services_log_is_still_a_ghost(self, base_dir: Path) -> None:
        d = instance_dir("empty-log", base_dir=base_dir)
        (d / "services.lock").touch()
        (d / "services.log").touch()

        instances = list_instances(base_dir=base_dir)
        assert instances == []
        assert not d.exists()

    def test_running_instance_not_auto_removed(self, base_dir: Path) -> None:
        fd = acquire_lock("alive-ghost-check", base_dir=base_dir)
        try:
            instances = list_instances(base_dir=base_dir)
            assert len(instances) == 1
            assert instances[0].alive is True
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# remove_instance / prune_instances
# ---------------------------------------------------------------------------


class TestRemoveInstance:
    def test_removes_dead_scope_dir(self, base_dir: Path) -> None:
        d = instance_dir("rm-me", base_dir=base_dir)
        (d / "services.log").write_text("logs\n")

        assert remove_instance("rm-me", base_dir=base_dir) is True
        assert not d.exists()

    def test_missing_scope_returns_false(self, base_dir: Path) -> None:
        assert remove_instance("missing", base_dir=base_dir) is False

    def test_refuses_running_instance(self, base_dir: Path) -> None:
        fd = acquire_lock("running-rm", base_dir=base_dir)
        try:
            with pytest.raises(InstanceStillRunningError, match="running-rm"):
                remove_instance("running-rm", base_dir=base_dir)
        finally:
            os.close(fd)

    def test_rejects_invalid_scope(self, base_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid scope"):
            remove_instance("../escape", base_dir=base_dir)

    def test_returns_false_when_rmtree_fails(self, base_dir: Path) -> None:
        d = instance_dir("rmtree-fail", base_dir=base_dir)
        (d / "services.log").write_text("logs\n")

        with patch(
            "nemo_helix_ext.local.process.shutil.rmtree",
            side_effect=OSError("permission denied"),
        ):
            assert remove_instance("rmtree-fail", base_dir=base_dir) is False
        assert d.exists()


class TestPruneInstances:
    def test_removes_all_stopped_scopes(self, base_dir: Path) -> None:
        alive_fd = acquire_lock("alive-prune", base_dir=base_dir)
        stopped = instance_dir("stopped-prune", base_dir=base_dir)
        (stopped / "services.log").write_text("x\n")
        try:
            removed = prune_instances(base_dir=base_dir)
            assert removed == ["stopped-prune"]
            assert not stopped.exists()
            assert is_instance_alive("alive-prune", base_dir=base_dir)
        finally:
            os.close(alive_fd)

    def test_noop_when_empty(self, base_dir: Path) -> None:
        assert prune_instances(base_dir=base_dir) == []


class TestInstanceLogBytes:
    def test_sums_active_and_rotated_logs(self, base_dir: Path) -> None:
        d = instance_dir("log-bytes", base_dir=base_dir)
        (d / "services.log").write_text("abc")
        (d / "services.log.old").write_text("de")

        assert instance_log_bytes("log-bytes", base_dir=base_dir) == 5


class TestIsRemovableGhost:
    def test_true_for_lock_only(self, base_dir: Path) -> None:
        d = instance_dir("ghost", base_dir=base_dir)
        (d / "services.lock").touch()
        assert is_removable_ghost("ghost", base_dir=base_dir) is True

    def test_false_when_logs_present(self, base_dir: Path) -> None:
        d = instance_dir("not-ghost", base_dir=base_dir)
        (d / "services.log").write_text("data\n")
        assert is_removable_ghost("not-ghost", base_dir=base_dir) is False


class TestListStoppedScopes:
    def test_lists_only_stopped(self, base_dir: Path) -> None:
        fd = acquire_lock("up", base_dir=base_dir)
        down = instance_dir("down", base_dir=base_dir)
        (down / "services.log").write_text("x\n")
        try:
            assert list_stopped_scopes(base_dir=base_dir) == ["down"]
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------


class TestLogRotation:
    def test_rotate_empty_log(self, base_dir: Path) -> None:
        log = rotate_log("test", base_dir=base_dir)
        assert log.name == "services.log"

    def test_rotate_existing_log(self, base_dir: Path) -> None:
        d = instance_dir("test", base_dir=base_dir)
        log = d / "services.log"
        log.write_text("old content\n")

        new_log = rotate_log("test", base_dir=base_dir)
        assert new_log == log
        assert not log.exists()
        rotated = list(d.glob("services.log.*"))
        assert len(rotated) == 1
        assert rotated[0].read_text() == "old content\n"

    def test_keeps_all_rotated_logs(self, base_dir: Path) -> None:
        d = instance_dir("test", base_dir=base_dir)
        for i in range(8):
            (d / f"services.log.2025010{i}T000000Z").write_text(f"log {i}\n")
        log = d / "services.log"
        log.write_text("current\n")

        rotate_log("test", base_dir=base_dir)
        rotated = list(d.glob("services.log.*"))
        assert len(rotated) == 9


# ---------------------------------------------------------------------------
# stop_instance
# ---------------------------------------------------------------------------


class TestStopInstance:
    def test_not_running_returns_empty(self, base_dir: Path) -> None:
        result = stop_instance("nothing", base_dir=base_dir)
        assert result.stopped_pids == []

    def test_stops_running_process(self, base_dir: Path) -> None:
        scope = "stop-test"
        fd: int | None = None
        proc: subprocess.Popen[bytes] | None = None
        try:
            fd = acquire_lock(scope, base_dir=base_dir)
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(fd,),
            )
            desc = InstanceDescriptor(
                pid=proc.pid,
                config=HelixAppConfig(scope=scope),
                mode="background",
                create_time=psutil.Process(proc.pid).create_time(),
            )
            write_descriptor(desc, base_dir=base_dir)

            os.close(fd)
            fd = None

            result = stop_instance(scope, base_dir=base_dir, timeout=5.0)
            assert proc.pid in result.stopped_pids
            proc.wait(timeout=5)
            assert proc.poll() is not None
            assert not is_instance_alive(scope, base_dir=base_dir)
            assert read_descriptor(scope, base_dir=base_dir) is None
        finally:
            if fd is not None:
                os.close(fd)
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_preserves_descriptor_while_lock_remains_held(self, base_dir: Path) -> None:
        scope = "lock-still-held"
        fd: int | None = None
        lock_holder: subprocess.Popen[bytes] | None = None
        target: subprocess.Popen[bytes] | None = None
        try:
            fd = acquire_lock(scope, base_dir=base_dir)
            lock_holder = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(fd,),
            )
            target = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.close(fd)
            fd = None

            desc = InstanceDescriptor(
                pid=target.pid,
                config=HelixAppConfig(scope=scope),
                mode="background",
                create_time=psutil.Process(target.pid).create_time(),
            )
            write_descriptor(desc, base_dir=base_dir)

            result = stop_instance(scope, base_dir=base_dir, timeout=0.5)

            assert target.pid in result.stopped_pids
            assert is_instance_alive(scope, base_dir=base_dir)
            assert read_descriptor(scope, base_dir=base_dir) == desc
        finally:
            if fd is not None:
                os.close(fd)
            for proc in (target, lock_holder):
                if proc is None:
                    continue
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)

    def test_cleans_up_stale_descriptor(self, base_dir: Path) -> None:
        scope = "stale"
        desc = InstanceDescriptor(
            pid=999999999,
            config=HelixAppConfig(scope=scope),
            mode="background",
            create_time=0.0,
        )
        write_descriptor(desc, base_dir=base_dir)
        result = stop_instance(scope, base_dir=base_dir)
        assert result.stopped_pids == []
        assert read_descriptor(scope, base_dir=base_dir) is None

    def test_refuses_to_stop_foreground_instance(self, base_dir: Path) -> None:
        scope = "fg-protect"
        fd = acquire_lock(scope, base_dir=base_dir)
        try:
            desc = InstanceDescriptor(
                pid=os.getpid(),
                config=HelixAppConfig(scope=scope),
                mode="foreground",
                create_time=1.0,
            )
            write_descriptor(desc, base_dir=base_dir)

            with pytest.raises(ForegroundInstanceError, match="foreground"):
                stop_instance(scope, base_dir=base_dir)

            assert is_instance_alive(scope, base_dir=base_dir)
            assert read_descriptor(scope, base_dir=base_dir) is not None
        finally:
            os.close(fd)

    def test_force_stops_foreground_instance(self, base_dir: Path) -> None:
        scope = "fg-force"
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            desc = InstanceDescriptor(
                pid=proc.pid,
                config=HelixAppConfig(scope=scope),
                mode="foreground",
                create_time=psutil.Process(proc.pid).create_time(),
            )
            write_descriptor(desc, base_dir=base_dir)

            result = stop_instance(scope, base_dir=base_dir, force=True)
            assert proc.pid in result.stopped_pids
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_preserves_descriptor_when_sigkill_does_not_stop_parent(self, base_dir: Path, monkeypatch) -> None:
        scope = "sigkill-still-alive"
        desc = InstanceDescriptor(
            pid=12345,
            config=HelixAppConfig(scope=scope),
            mode="background",
            create_time=1.0,
        )
        write_descriptor(desc, base_dir=base_dir)
        kill_signals: list[int] = []

        def fake_kill(_pid: int, sig: int) -> None:
            kill_signals.append(sig)

        monkeypatch.setattr(process_module, "validate_pid", lambda _pid, _create_time: True)
        monkeypatch.setattr(process_module, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(process_module, "_snapshot_children", lambda _pid: [object()])
        monkeypatch.setattr(process_module, "_sweep_orphans", lambda _children: [222])
        monkeypatch.setattr(process_module, "_SIGKILL_WAIT_TIMEOUT", 0.0)
        monkeypatch.setattr(process_module.os, "kill", fake_kill)

        result = stop_instance(scope, base_dir=base_dir, timeout=0.0)

        assert kill_signals == [signal.SIGTERM, signal.SIGKILL]
        assert result.stopped_pids == []
        assert result.swept_children == [222]
        assert read_descriptor(scope, base_dir=base_dir) is not None


# ---------------------------------------------------------------------------
# start_background
# ---------------------------------------------------------------------------


class TestStartBackground:
    def test_uses_default_platform_app_config(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        mock_proc = MagicMock()
        mock_proc.pid = 99998
        captured_args: list[str] = []
        captured_env: dict[str, str] = {}

        def fake_popen(args, **kwargs):
            captured_args.extend(args)
            captured_env.update(kwargs["env"])
            return mock_proc

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            side_effect=fake_popen,
        ):
            proc = start_background()

        assert proc.pid == 99998
        assert captured_args[captured_args.index("--instance") + 1] == "default"
        assert captured_args[captured_args.index("--host") + 1] == "127.0.0.1"
        assert captured_args[captured_args.index("--port") + 1] == "8080"
        assert captured_env["XDG_STATE_HOME"] == str(tmp_path)
        assert "_NHX_STATE_DIR" not in captured_env
        assert (tmp_path / "nhx" / "instances" / "default" / "services.log").exists()

    def test_launches_detached_subprocess(self, base_dir: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            return_value=mock_proc,
        ) as mock_popen:
            proc = start_background(
                HelixAppConfig(
                    scope="bg-test",
                    services=["entities", "models"],
                    controllers=["jobs"],
                    host="127.0.0.1",
                    port=8080,
                    keep_alive_timeout_seconds=12,
                    state_root=base_dir,
                ),
            )

        assert proc.pid == 99999
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["start_new_session"] is True
        assert call_kwargs["stdin"] == subprocess.DEVNULL
        assert call_kwargs["close_fds"] is True
        args = mock_popen.call_args.args[0]
        assert args[args.index("--keep-alive-timeout-seconds") + 1] == "12"

    def test_injects_nhx_data_dir_when_unset(self, base_dir: Path, monkeypatch) -> None:
        monkeypatch.delenv("NHX_DATA_DIR", raising=False)
        mock_proc = MagicMock()
        mock_proc.pid = 4242
        captured_env: dict[str, str] = {}

        def fake_popen(args, **kwargs):
            captured_env.update(kwargs["env"])
            return mock_proc

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            side_effect=fake_popen,
        ):
            start_background(
                HelixAppConfig(
                    scope="data-dir-test",
                    host=DEFAULT_LOCAL_SERVICES_BIND_HOST,
                    state_root=base_dir,
                ),
                data_dir="/chosen/data/dir",
            )

        assert captured_env.get("NHX_DATA_DIR") == "/chosen/data/dir"

    def test_does_not_override_shell_nhx_data_dir(self, base_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv("NHX_DATA_DIR", "/shell/wins")
        mock_proc = MagicMock()
        mock_proc.pid = 4243
        captured_env: dict[str, str] = {}

        def fake_popen(args, **kwargs):
            captured_env.update(kwargs["env"])
            return mock_proc

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            side_effect=fake_popen,
        ):
            start_background(
                HelixAppConfig(
                    scope="shell-env-test",
                    host=DEFAULT_LOCAL_SERVICES_BIND_HOST,
                    state_root=base_dir,
                ),
                data_dir="/chosen/data/dir",
            )

        assert captured_env.get("NHX_DATA_DIR") == "/shell/wins"

    def test_rotates_log_before_start(self, base_dir: Path) -> None:
        d = instance_dir("rotate-test", base_dir=base_dir)
        log = d / "services.log"
        log.write_text("old log content\n")

        mock_proc = MagicMock()
        mock_proc.pid = 5555

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            return_value=mock_proc,
        ):
            start_background(
                HelixAppConfig(
                    scope="rotate-test",
                    host=DEFAULT_LOCAL_SERVICES_BIND_HOST,
                    state_root=base_dir,
                ),
            )

        rotated = list(d.glob("services.log.*"))
        assert len(rotated) == 1
        assert rotated[0].read_text() == "old log content\n"

    def test_forwards_scope_to_child(self, base_dir: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 7777
        captured_args: list[str] = []

        def fake_popen(args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            side_effect=fake_popen,
        ):
            start_background(
                HelixAppConfig(
                    scope="custom-key",
                    services=["entities"],
                    host="127.0.0.1",
                    port=9090,
                    state_root=base_dir,
                ),
            )

        assert "--instance" in captured_args
        idx = captured_args.index("--instance")
        assert captured_args[idx + 1] == "custom-key"

    def test_sets_launch_mode_background_in_child_env(self, base_dir: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 8888
        captured_env: dict[str, str] = {}

        def fake_popen(args, **kwargs):
            captured_env.update(kwargs["env"])
            return mock_proc

        with patch(
            "nemo_helix_ext.local.process.subprocess.Popen",
            side_effect=fake_popen,
        ):
            start_background(
                HelixAppConfig(
                    scope="mode-test",
                    host=DEFAULT_LOCAL_SERVICES_BIND_HOST,
                    state_root=base_dir,
                ),
            )

        assert captured_env.get("_NHX_LAUNCH_MODE") == "background"


# ---------------------------------------------------------------------------
# _snapshot_children / _sweep_orphans
# ---------------------------------------------------------------------------


def _wait_for_children(pid: int, timeout: float = 5.0) -> list[psutil.Process]:
    """Poll until *pid* has at least one child process, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            children = psutil.Process(pid).children(recursive=True)
            if children:
                return children
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    return []


_SPAWN_CHILD_SCRIPT = (
    "import subprocess, sys, time; "
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "time.sleep(60)"
)


class TestSnapshotChildren:
    def test_captures_child_processes(self) -> None:
        parent = subprocess.Popen(
            [sys.executable, "-c", _SPAWN_CHILD_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_children(parent.pid)
            children = _snapshot_children(parent.pid)
            assert len(children) >= 1
            child_pids = [c.pid for c in children]
            assert all(isinstance(p, int) for p in child_pids)
        finally:
            for c in psutil.Process(parent.pid).children(recursive=True):
                c.kill()
            parent.kill()
            parent.wait(timeout=5)

    def test_returns_empty_for_dead_pid(self) -> None:
        assert _snapshot_children(999999999) == []

    def test_returns_empty_for_process_with_no_children(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            children = _snapshot_children(proc.pid)
            assert children == []
        finally:
            proc.kill()
            proc.wait(timeout=5)


class TestSweepOrphans:
    def test_terminates_surviving_children(self) -> None:
        procs = []
        for _ in range(3):
            p = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(p)
        try:
            ps_children = [psutil.Process(p.pid) for p in procs]
            killed = _sweep_orphans(ps_children, timeout=3.0)
            assert len(killed) == 3
            for p in procs:
                p.wait(timeout=5)
                assert p.poll() is not None
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=5)

    def test_noop_when_all_already_exited(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ps_child = psutil.Process(proc.pid)
        proc.kill()
        proc.wait(timeout=5)
        killed = _sweep_orphans([ps_child], timeout=1.0)
        assert killed == []

    def test_escalates_to_sigkill(self) -> None:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.3)
            ps_child = psutil.Process(proc.pid)
            killed = _sweep_orphans([ps_child], timeout=2.0)
            assert proc.pid in killed
            proc.wait(timeout=5)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_warns_when_child_survives_sigkill(self, caplog: pytest.LogCaptureFixture) -> None:
        """A child that outlives SIGKILL is reported rather than silently dropped.

        Nothing survives SIGKILL on demand — the real trigger is a process wedged
        in uninterruptible sleep, which a test cannot arrange — so `wait_procs` is
        stubbed to keep reporting the child as alive. The terminate/kill calls are
        still real; only the observation is simulated.
        """
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.3)
            ps_child = psutil.Process(proc.pid)
            # The first wait drives the SIGTERM -> SIGKILL escalation; the second
            # is the one whose non-empty result trips the warning.
            with (
                patch.object(
                    process_module.psutil,
                    "wait_procs",
                    side_effect=[([], [ps_child]), ([], [ps_child])],
                ) as wait_procs,
                caplog.at_level(logging.WARNING, logger=process_module.logger.name),
            ):
                killed = _sweep_orphans([ps_child], timeout=0.1)

            # Two waits, not one. The second is the post-SIGKILL wait this PR adds,
            # so a single-wait implementation must not be able to satisfy this test.
            assert wait_procs.call_count == 2
            assert proc.pid in killed
            assert f"Orphaned child [{proc.pid}] still alive after SIGKILL" in caplog.text
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# stop_instance — child sweep integration
# ---------------------------------------------------------------------------


class TestStopInstanceChildSweep:
    def test_sweeps_surviving_children(self, base_dir: Path) -> None:
        """Parent + child spawned; stop should kill both and report the child."""
        parent = subprocess.Popen(
            [sys.executable, "-c", _SPAWN_CHILD_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_children(parent.pid)

            scope = "sweep-test"
            fd = acquire_lock(scope, base_dir=base_dir)
            desc = InstanceDescriptor(
                pid=parent.pid,
                config=HelixAppConfig(scope=scope),
                mode="background",
                create_time=psutil.Process(parent.pid).create_time(),
            )
            write_descriptor(desc, base_dir=base_dir)
            os.close(fd)

            result = stop_instance(scope, base_dir=base_dir, timeout=5.0)
            assert parent.pid in result.stopped_pids
            assert len(result.swept_children) >= 1
            for child_pid in result.swept_children:
                assert not _pid_alive(child_pid)
            parent.wait(timeout=5)
        finally:
            try:
                for c in psutil.Process(parent.pid).children(recursive=True):
                    c.kill()
            except psutil.NoSuchProcess:
                pass  # Parent already exited — children cleaned up.
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)

    def test_swept_children_empty_when_no_children(self, base_dir: Path) -> None:
        """Process with no children -- swept_children should be empty."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            scope = "clean-stop"
            fd = acquire_lock(scope, base_dir=base_dir)
            desc = InstanceDescriptor(
                pid=proc.pid,
                config=HelixAppConfig(scope=scope),
                mode="background",
                create_time=psutil.Process(proc.pid).create_time(),
            )
            write_descriptor(desc, base_dir=base_dir)
            os.close(fd)

            result = stop_instance(scope, base_dir=base_dir, timeout=5.0)
            assert proc.pid in result.stopped_pids
            assert result.swept_children == []
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_restart_path_sweeps_children(self, base_dir: Path) -> None:
        """The restart flow calls stop_instance(force=True); verify it sweeps children too."""
        parent = subprocess.Popen(
            [sys.executable, "-c", _SPAWN_CHILD_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_children(parent.pid)
            child_pids_before = [c.pid for c in psutil.Process(parent.pid).children(recursive=True)]
            assert len(child_pids_before) >= 1

            scope = "restart-sweep"
            fd = acquire_lock(scope, base_dir=base_dir)
            desc = InstanceDescriptor(
                pid=parent.pid,
                config=HelixAppConfig(scope=scope),
                mode="foreground",
                create_time=psutil.Process(parent.pid).create_time(),
            )
            write_descriptor(desc, base_dir=base_dir)
            os.close(fd)

            result = stop_instance(scope, base_dir=base_dir, force=True, timeout=5.0)
            assert parent.pid in result.stopped_pids
            assert len(result.swept_children) >= 1
            for child_pid in result.swept_children:
                assert not _pid_alive(child_pid)
            parent.wait(timeout=5)
        finally:
            try:
                for c in psutil.Process(parent.pid).children(recursive=True):
                    c.kill()
            except psutil.NoSuchProcess:
                pass  # Parent already exited — children cleaned up.
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
