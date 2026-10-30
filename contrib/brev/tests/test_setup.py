# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the launch script without installing packages or changing the host."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "setup.sh"

STUB = """#!/bin/bash
set -eu
command=${0##*/}
printf '%s %s\\n' "$command" "$*" >> "$HOME/commands"
case "$command" in
    id) if [[ $1 == -u ]]; then echo 1000; else echo brev; fi ;;
    docker) exit "${DOCKER_EXIT:-0}" ;;
    systemctl)
        if [[ $* == *is-active* ]]; then exit "${ACTIVE_EXIT:-1}"; fi
        if [[ $* == *'enable --now'* ]]; then touch "$HOME/started"; fi
        ;;
    curl) [[ -f $HOME/started && ${NEVER_READY:-0} != 1 ]] ;;
    uv)
        mkdir -p "$UV_TOOL_BIN_DIR"
        cp "$0" "$UV_TOOL_BIN_DIR/nemo"
        ;;
    nemo|sudo|loginctl|cc|sleep) ;;
    *) exit 2 ;;
esac
"""


@pytest.fixture
def launch_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("id", "docker", "systemctl", "curl", "uv", "sudo", "loginctl", "cc", "sleep"):
        stub = bin_dir / name
        stub.write_text(STUB)
        stub.chmod(0o755)
    return {
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "NEMO_HELIX_VERSION": "0.8.1",
        "OPENROUTER_API_KEY": "test-secret-do-not-log",
    }


def run_setup(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=15)


def test_install_setup_and_supervise(launch_env: dict[str, str]) -> None:
    result = run_setup(launch_env)
    assert result.returncode == 0, result.stderr
    home = Path(launch_env["HOME"])
    commands = (home / "commands").read_text()
    assert "uv tool install --python 3.12 nemo-helix[all]==0.8.1" in commands
    assert "nemo setup --auto --start-services --ready-timeout 360" in commands
    assert commands.index("nemo services stop") < commands.index("enable --now")
    assert "nemo models list --workspace default" in commands
    unit = (home / ".config/systemd/user/nemo-helix.service").read_text()
    assert "services run --host 0.0.0.0 --port 8080" in unit
    assert "Restart=on-failure" in unit
    assert "NHX_DATA_DIR=%h/.local/share/nemo" in unit
    assert launch_env["OPENROUTER_API_KEY"] not in unit + commands + result.stdout + result.stderr


def test_missing_key_fails_before_install(launch_env: dict[str, str]) -> None:
    del launch_env["OPENROUTER_API_KEY"]
    launch_env["NVIDIA_API_KEY"] = "not-an-auto-provider"
    result = run_setup(launch_env)
    assert result.returncode != 0
    assert "Supply a supported provider key" in result.stderr
    assert "uv tool install" not in (Path(launch_env["HOME"]) / "commands").read_text()


def test_custom_endpoint_requires_generic_key(launch_env: dict[str, str]) -> None:
    launch_env["NEMO_DEFAULT_INFERENCE_BASE_URL"] = "https://inference-api.nvidia.com/v1"
    result = run_setup(launch_env)
    assert result.returncode != 0
    assert "requires NEMO_DEFAULT_INFERENCE_KEY" in result.stderr


@pytest.mark.parametrize("variable", ["DOCKER_EXIT", "ACTIVE_EXIT"])
def test_preflight_refuses_unavailable_docker_or_live_platform(launch_env: dict[str, str], variable: str) -> None:
    launch_env[variable] = "1" if variable == "DOCKER_EXIT" else "0"
    result = run_setup(launch_env)
    assert result.returncode != 0
    assert "uv tool install" not in (Path(launch_env["HOME"]) / "commands").read_text()


def test_readiness_timeout_is_failure(launch_env: dict[str, str]) -> None:
    launch_env["NEVER_READY"] = "1"
    result = run_setup(launch_env)
    assert result.returncode != 0
    assert "Readiness timed out" in result.stderr
    assert "is ready" not in result.stdout


def test_generic_provider(launch_env: dict[str, str]) -> None:
    del launch_env["OPENROUTER_API_KEY"]
    launch_env["NEMO_DEFAULT_INFERENCE_KEY"] = "generic-test-secret"
    launch_env["NEMO_DEFAULT_INFERENCE_BASE_URL"] = "https://inference-api.nvidia.com/v1"
    result = run_setup(launch_env)
    assert result.returncode == 0, result.stderr
    assert "generic-test-secret" not in result.stdout + result.stderr
