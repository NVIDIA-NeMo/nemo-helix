# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
from sandboxed_gym.host.entrypoint import (
    DEFAULT_GYM_ACTOR_VENV,
    DEFAULT_GYM_WRITABLE_SRC,
    DEFAULT_IMAGE_GIT_ROOT,
    default_gym_host_entrypoint,
    gym_host_script_path,
    packaged_gym_host_script,
)
from sandboxed_gym.host.models import GymHostSpec, GymHostVolumeMount
from sandboxed_gym.runtime import gym_host_runtime as runtime

#: These two tests instantiate the host provider, which drives the OpenSandbox SDK directly. The
#: SDK is an optional extra (`nemo-sandboxed-gym[opensandbox]`) because only a deployment that
#: provisions real sandboxes needs it, so it is absent from a plain workspace checkout. Everything
#: else in this package works without it -- the broker contract and the sandbox types are vendored
#: (see `sandboxed_gym.wire`).
requires_opensandbox = pytest.mark.skipif(
    importlib.util.find_spec("opensandbox") is None,
    reason="needs the OpenSandbox SDK; install the `opensandbox` extra to run",
)


def test_default_gym_host_entrypoint_uses_sandboxed_actor_venv():
    entrypoint = default_gym_host_entrypoint()
    assert entrypoint[0] == "/bin/sh"
    assert entrypoint[1] == gym_host_script_path()
    assert entrypoint[2] == DEFAULT_GYM_ACTOR_VENV
    assert entrypoint[3] == DEFAULT_IMAGE_GIT_ROOT
    assert entrypoint[4] == DEFAULT_GYM_WRITABLE_SRC
    assert entrypoint[5].endswith("gym_host_runtime.py")
    script = packaged_gym_host_script()
    text = script.read_text(encoding="utf-8")
    assert "cp -a" in text
    assert "gym_host_runtime.py" in text


def test_apply_uv_dirs_fills_defaults(tmp_path, monkeypatch):
    cache = tmp_path / "uv-cache"
    venvs = tmp_path / "gym-venvs"
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", str(venvs))
    cfg: dict = {}
    runtime._apply_uv_dirs(cfg)
    assert cfg["uv_cache_dir"] == str(cache)
    assert cfg["uv_venv_dir"] == str(venvs)


def test_apply_uv_dirs_preserves_config(tmp_path, monkeypatch):
    cache = tmp_path / "from-config-cache"
    venvs = tmp_path / "from-config-venvs"
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "ignored"))
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", str(tmp_path / "ignored-venvs"))
    cfg = {"uv_cache_dir": str(cache), "uv_venv_dir": str(venvs)}
    runtime._apply_uv_dirs(cfg)
    assert cfg["uv_cache_dir"] == str(cache)
    assert cfg["uv_venv_dir"] == str(venvs)


@requires_opensandbox
def test_opensandbox_host_provider_defaults_skip_health_check():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    assert provider._create_options["skip_health_check"] is True


@requires_opensandbox
def test_opensandbox_host_provider_honors_explicit_skip_health_check():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(
        connection={"domain": "x", "api_key": "k"},
        create={"skip_health_check": False},
    )
    assert provider._create_options["skip_health_check"] is False


@requires_opensandbox
def test_opensandbox_host_provider_uses_configured_protocol_for_bare_endpoints():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"protocol": "http"})

    assert provider._absolute_url("10.244.6.40:8080") == "http://10.244.6.40:8080"


def _gym_host_spec(*, entrypoint: tuple[str, ...] | None = None) -> GymHostSpec:
    return GymHostSpec(
        job_id="job-1",
        runtime_image="nmp-gym-host:dev",
        environment_mount=GymHostVolumeMount(
            pvc_claim="job-storage",
            sub_path="environment",
            mount_path="/job/environment",
            read_only=True,
        ),
        workspace_mount=GymHostVolumeMount(
            pvc_claim="job-storage",
            sub_path="work",
            mount_path="/job/work",
        ),
        entrypoint=entrypoint,
    )


@requires_opensandbox
def test_opensandbox_host_provider_supplies_runtime_entrypoint_when_omitted():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    sandbox_spec = provider._to_sandbox_spec(_gym_host_spec())

    assert sandbox_spec.entrypoint == ["python", "-m", "sandboxed_gym.runtime.gym_host_runtime"]


@requires_opensandbox
def test_opensandbox_host_provider_preserves_configured_entrypoint():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    sandbox_spec = provider._to_sandbox_spec(_gym_host_spec(entrypoint=("custom-runtime", "--serve")))

    assert sandbox_spec.entrypoint == ["custom-runtime", "--serve"]


# --------------------------------------------------------------------------------------------
# gym_host.sh — the copy that stages Gym into a writable tree
#
# A caller can configure this explicit bootstrap for a compatible runtime image. The standard
# platform image instead starts the installed runtime module directly.
# --------------------------------------------------------------------------------------------


def _fake_host(tmp_path: Path) -> dict[str, Path]:
    """A tree the script can run against: a Gym source, a venv whose python exits 0, a runtime."""
    gym_tree = tmp_path / "Gym"
    (gym_tree / "nemo_gym").mkdir(parents=True)
    (gym_tree / "nemo_gym" / "__init__.py").write_text("", encoding="utf-8")

    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)

    runtime = tmp_path / "runtime.py"
    runtime.write_text("", encoding="utf-8")
    (tmp_path / "root").mkdir()
    return {"tree": gym_tree, "venv": tmp_path / "venv", "runtime": runtime, "root": tmp_path / "root"}


def _run_gym_host(paths: dict[str, Path], gym_rw: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "/bin/sh",
            str(gym_host_script_path(git_root=str(paths["root"]))),
            str(paths["venv"]),
            str(paths["root"]),
            str(gym_rw),
            str(paths["runtime"]),
        ],
        env={**os.environ, "SANDBOXED_GYM_TREE": str(paths["tree"])},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_gym_tree_is_staged_where_pythonpath_expects_it(tmp_path: Path) -> None:
    paths = _fake_host(tmp_path)
    gym_rw = tmp_path / "gym-src" / "Gym"

    result = _run_gym_host(paths, gym_rw)

    assert result.returncode == 0, result.stderr
    assert (gym_rw / "nemo_gym").is_dir()


def test_a_restart_does_not_nest_the_tree(tmp_path: Path) -> None:
    """`cp -a src dst` copies *into* dst when dst exists.

    A leftover or half-written destination would become `$gym_rw/Gym/nemo_gym`, which leaves the
    guard still true — so each restart adds another layer and `PYTHONPATH` never resolves
    `nemo_gym`. Reproduced by seeding a destination that exists but is incomplete.
    """
    paths = _fake_host(tmp_path)
    gym_rw = tmp_path / "gym-src" / "Gym"
    gym_rw.mkdir(parents=True)
    (gym_rw / "leftover.txt").write_text("from an interrupted run", encoding="utf-8")

    for _ in range(2):
        assert _run_gym_host(paths, gym_rw).returncode == 0

    assert (gym_rw / "nemo_gym").is_dir()
    assert not (gym_rw / "Gym").exists(), "the tree was nested instead of replaced"
    assert not (gym_rw / "leftover.txt").exists(), "a half-written destination must be repaired"
