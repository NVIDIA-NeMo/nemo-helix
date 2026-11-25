# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandboxed_gym.runtime import container_engine
from sandboxed_gym.runtime.container_engine import ENGINE_ENV, is_kata_guest, prepare_container_engine

KATA_CMDLINE = "console=hvc0 ro rootfstype=ext4 systemd.unit=kata-containers.target agent.cdh_api_timeout=50"
NODE_CMDLINE = "BOOT_IMAGE=/vmlinuz-6.8.0-1041-oracle root=UUID=a664f091 ro console=ttyS0 user_namespace.enable=1"
STATUS_DROPPED = "Name:\tpython\nCapEff:\t0000000000000000\nCapBnd:\t0000000000000000\nNoNewPrivs:\t1\n"
STATUS_SYS_ADMIN = "Name:\tpython\nCapEff:\t0000000000200000\nCapBnd:\t0000000000000000\nNoNewPrivs:\t1\n"
STATUS_FULL_BOUNDING = "Name:\tpython\nCapEff:\t0000000000000000\nCapBnd:\t000001ffffffffff\nNoNewPrivs:\t1\n"
STATUS_PRIVS_ALLOWED = "Name:\tpython\nCapEff:\t0000000000000000\nCapBnd:\t0000000000000000\nNoNewPrivs:\t0\n"


@pytest.mark.parametrize(
    ("cmdline", "expected"),
    [
        (KATA_CMDLINE, True),
        ("console=hvc0 agent.log=debug", True),
        (NODE_CMDLINE, False),
        ("quiet systemd.unit=multi-user.target", False),
        ("", False),
    ],
)
def test_is_kata_guest_recognises_only_a_kata_guest_kernel(cmdline: str, expected: bool) -> None:
    assert is_kata_guest(cmdline) is expected


class _Host:
    """Stands in for the sandbox: kernel files, the setup script and the uid calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, cmdline: str = KATA_CMDLINE) -> None:
        self.files = {
            "/proc/cmdline": cmdline,
            "/proc/self/status": STATUS_DROPPED,
            "/proc/sys/kernel/cap_last_cap": "40\n",
        }
        self.setup_exit = 0
        self.setup_output = "ok\n"
        self.calls: list[tuple[str, object]] = []
        self.euid = 0
        self.log = tmp_path / "setup.log"
        self.files[str(self.log)] = ""
        self.work = tmp_path / "work"
        (self.work / "nested").mkdir(parents=True)
        (self.work / "nested" / "file").write_text("x")
        monkeypatch.setattr(container_engine, "SETUP_LOG", str(self.log))
        monkeypatch.setattr(container_engine, "geteuid", lambda: self.euid)
        monkeypatch.setattr(container_engine, "setgroups", lambda groups: self.calls.append(("setgroups", groups)))
        monkeypatch.setattr(container_engine, "setgid", lambda gid: self.calls.append(("setgid", gid)))
        monkeypatch.setattr(container_engine, "setuid", self._setuid)
        monkeypatch.setattr(container_engine, "chown", self._chown)
        monkeypatch.setattr(
            container_engine,
            "getpwnam",
            lambda name: SimpleNamespace(pw_uid=1000, pw_gid=1000, pw_dir=f"/home/{name}"),
        )

    def _chown(self, target: int | str, uid: int, gid: int, **kwargs: object) -> None:
        # A directory arrives as its open descriptor; an entry as a name relative to one.
        if isinstance(target, int):
            self.calls.append(("chown", ("dir", os.stat(target).st_ino)))
        else:
            assert kwargs == {"dir_fd": kwargs["dir_fd"], "follow_symlinks": False}
            self.calls.append(("chown", ("entry", target)))

    def _setuid(self, uid: int) -> None:
        self.calls.append(("setuid", uid))
        self.euid = uid

    def read_text(self, path: str) -> str:
        return self.files[path]

    def run(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append(("run", argv))
        self.setup_env = kwargs["env"]
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is not subprocess.PIPE, "setup output must not go to a pipe it can hold open"
        self.files[str(self.log)] = self.setup_output
        return subprocess.CompletedProcess(argv, self.setup_exit)

    def prctl(self, option: int, arg: int) -> None:
        self.calls.append(("prctl", (option, arg)))

    def prepare(self, environ: dict[str, str]) -> bool:
        return prepare_container_engine(environ, read_text=self.read_text, run=self.run, prctl=self.prctl)


def test_an_image_without_an_engine_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _Host(monkeypatch, tmp_path)

    assert host.prepare({}) is False
    assert host.calls == []


def test_setup_runs_then_the_host_drops_to_the_container_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _Host(monkeypatch, tmp_path)
    environ = {
        ENGINE_ENV: "rootless-podman",
        "NHX_WORK_PATH": str(host.work),
        "HOME": "/root",
        "NHX_CONTAINER_STORAGE_SIZE": "42949672960",
        "NHX_CONTAINER_STORAGE_DISCARD": "1",
        "NHX_BROKER_TOKEN": "broker-secret",
    }

    assert host.prepare(environ) is True
    # The root-only setup runs with a fixed PATH and its own settings, nothing else from the host.
    assert host.setup_env == {
        "PATH": container_engine._SETUP_PATH,
        "NHX_CONTAINER_STORAGE_SIZE": "42949672960",
        "NHX_CONTAINER_STORAGE_DISCARD": "1",
    }

    names = [name for name, _ in host.calls]
    assert names[0] == "run" and host.calls[0][1] == [container_engine.SETUP_SCRIPT]
    # The work tree is handed over before the uid changes, and the groups go before the uid.
    assert {target for name, target in host.calls if name == "chown"} == {
        ("dir", host.work.stat().st_ino),
        ("dir", (host.work / "nested").stat().st_ino),
        ("entry", "file"),
    }
    assert names[-4:] == ["setgroups", "setgid", "setuid", "prctl"]
    # Non-dumpable after the uid change, whatever fs.suid_dumpable says.
    assert host.calls[-1] == ("prctl", (4, 0))
    # The bounding set (needs CAP_SETPCAP) and no_new_privs go before the uid changes.
    prctls = [args for name, args in host.calls if name == "prctl"]
    assert prctls == [(24, capability) for capability in range(41)] + [(38, 1), (4, 0)]
    assert names.index("setgroups") > max(i for i, name in enumerate(names[:-1]) if name == "prctl")
    assert ("setuid", 1000) in host.calls
    assert environ["HOME"] == "/home/docker-user"
    assert environ["XDG_RUNTIME_DIR"] == "/run/user/1000"


def test_sys_admin_without_a_container_engine_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The isolation extension with the default image would leave the Gym servers holding SYS_ADMIN."""
    host = _Host(monkeypatch, tmp_path)
    host.files["/proc/self/status"] = STATUS_SYS_ADMIN

    with pytest.raises(RuntimeError, match="SYS_ADMIN but no container engine"):
        host.prepare({})
    assert host.calls == []


def test_the_workspace_handover_does_not_follow_links_out_of_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An earlier attempt could leave links in the workspace; root must not change ownership through them."""
    host = _Host(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    (outside / "inner").mkdir(parents=True)
    (outside / "inner" / "secret").write_text("x")
    (host.work / "to-dir").symlink_to(outside / "inner")
    (host.work / "to-file").symlink_to(outside / "inner" / "secret")

    host.prepare({ENGINE_ENV: "rootless-podman", "NHX_WORK_PATH": str(host.work)})

    chowned = {target for name, target in host.calls if name == "chown"}
    assert ("dir", (outside / "inner").stat().st_ino) not in chowned
    assert ("entry", "secret") not in chowned


def test_a_non_kata_sandbox_is_refused_before_any_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _Host(monkeypatch, tmp_path, cmdline=NODE_CMDLINE)

    with pytest.raises(RuntimeError, match="needs a Kata sandbox"):
        host.prepare({ENGINE_ENV: "rootless-podman"})
    assert host.calls == []


def test_a_failed_setup_reports_its_output_and_keeps_gym_from_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    host = _Host(monkeypatch, tmp_path)
    host.setup_exit = 1
    host.setup_output = "mounting storage\npodman's Docker API did not start; service log:\nError: boom\n"

    with pytest.raises(RuntimeError, match="exited 1: .*Error: boom"):
        host.prepare({ENGINE_ENV: "rootless-podman", "NHX_WORK_PATH": str(host.work)})
    assert "setuid" not in [name for name, _ in host.calls]


def test_capabilities_left_after_the_drop_are_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _Host(monkeypatch, tmp_path)
    host.files["/proc/self/status"] = STATUS_SYS_ADMIN

    with pytest.raises(RuntimeError, match="still privileged"):
        host.prepare({ENGINE_ENV: "rootless-podman", "NHX_WORK_PATH": str(host.work)})


@pytest.mark.parametrize("status", [STATUS_FULL_BOUNDING, STATUS_PRIVS_ALLOWED])
def test_a_bounding_set_or_exec_privileges_left_after_the_drop_are_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    """A setuid helper run with a bounding set left would get those capabilities back."""
    host = _Host(monkeypatch, tmp_path)
    host.files["/proc/self/status"] = status

    with pytest.raises(RuntimeError, match="still privileged"):
        host.prepare({ENGINE_ENV: "rootless-podman", "NHX_WORK_PATH": str(host.work)})


def test_a_host_not_started_as_root_cannot_run_the_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _Host(monkeypatch, tmp_path)
    host.euid = 1000

    with pytest.raises(RuntimeError, match="start as root"):
        host.prepare({ENGINE_ENV: "rootless-podman"})
    assert host.calls == []


def test_an_unknown_engine_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _Host(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="not supported"):
        host.prepare({ENGINE_ENV: "dockerd"})
