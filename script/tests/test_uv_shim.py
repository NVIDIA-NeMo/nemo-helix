# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for script/uv, which picks the uv binary that make and the pre-commit hooks run.

Every flox and uv interaction is driven by stubs on an isolated PATH, so neither tool is required.
"""

import json
import shutil
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM = REPO_ROOT / "script" / "uv"
UV_LOCK_SCRIPT = REPO_ROOT / "script" / "uv-lock.sh"

# Host tools the shim, uv-lock.sh, and the python3 stubs need.
HOST_TOOLS = ("awk", "bash", "dirname", "python3")

STUB = """\
#!/usr/bin/env python3
import json
import os
import sys

name = os.path.basename(sys.argv[0])
with open(os.path.join(os.environ["RECORD_DIR"], name + ".json"), "w") as handle:
    json.dump(sys.argv[1:], handle)
args = sys.argv[1:]
if name == "flox" and args[-2:] == ["uv", "--version"]:
    # The root Flox environment's activate profile writes banners to stdout.
    print("NeMo Helix")
    print("uv 0.13.4")
elif args == ["--version"]:
    print("uv 0.0.0")
"""

FLOX_ACTIVATE = ["-q", "activate", "--dir", str(REPO_ROOT), "--", "uv"]

EnvFactory = Callable[..., dict[str, str]]
Recorder = Callable[[str], list[str] | None]


@pytest.fixture
def stub_env(tmp_path: Path) -> tuple[EnvFactory, Recorder]:
    """Return (env_factory, record_reader) for a PATH holding only the stubs and HOST_TOOLS."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    def make_env(*, tools: tuple[str, ...] = ("flox", "uv"), **extra: str) -> dict[str, str]:
        bin_dir = tmp_path / "bin"
        shutil.rmtree(bin_dir, ignore_errors=True)
        bin_dir.mkdir()
        for host_tool in HOST_TOOLS:
            found = shutil.which(host_tool)
            assert found, f"{host_tool} is required to run these tests"
            (bin_dir / host_tool).symlink_to(found)
        for name in tools:
            stub = bin_dir / name
            stub.write_text(STUB)
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        return {"PATH": str(bin_dir), "RECORD_DIR": str(record_dir), **extra}

    def record(name: str) -> list[str] | None:
        path = record_dir / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    return make_env, record


def run(env: dict[str, str], *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), env=env, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    "extra_env,expected_tool",
    [
        pytest.param({"TOOLCHAIN": "system"}, "uv", id="toolchain-system"),
        pytest.param({"FLOX_ENV_PROJECT": str(REPO_ROOT)}, "uv", id="repo-flox-env-active"),
        pytest.param({}, "flox", id="no-flox-env"),
    ],
)
def test_shim_dispatch(stub_env: tuple[EnvFactory, Recorder], extra_env: dict[str, str], expected_tool: str) -> None:
    make_env, record = stub_env
    result = run(make_env(**extra_env), str(SHIM), "lock", "--check")
    assert result.returncode == 0, result.stderr
    expected = ["lock", "--check"] if expected_tool == "uv" else [*FLOX_ACTIVATE, "lock", "--check"]
    assert record(expected_tool) == expected
    assert record("flox" if expected_tool == "uv" else "uv") is None


def test_flox_env_project_is_compared_after_resolving_symlinks(
    stub_env: tuple[EnvFactory, Recorder], tmp_path: Path
) -> None:
    make_env, record = stub_env
    link = tmp_path / "repo-link"
    link.symlink_to(REPO_ROOT, target_is_directory=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert run(make_env(FLOX_ENV_PROJECT=str(link)), str(SHIM), "--version").returncode == 0
    assert record("flox") is None

    assert run(make_env(FLOX_ENV_PROJECT=str(elsewhere)), str(SHIM), "--version").returncode == 0
    assert record("flox") == [*FLOX_ACTIVATE, "--version"]


@pytest.mark.parametrize(
    "tools,extra_env,expected_hint",
    [
        pytest.param(("flox",), {"TOOLCHAIN": "system"}, "TOOLCHAIN=system requires uv", id="system-without-uv"),
        pytest.param(("uv",), {}, "flox is required", id="without-flox"),
    ],
)
def test_missing_tool_explains_how_to_proceed(
    stub_env: tuple[EnvFactory, Recorder], tools: tuple[str, ...], extra_env: dict[str, str], expected_hint: str
) -> None:
    make_env, _ = stub_env
    result = run(make_env(tools=tools, **extra_env), str(SHIM), "--version")
    assert result.returncode != 0
    assert expected_hint in result.stderr


def test_uv_lock_rejects_version_mismatch(stub_env: tuple[EnvFactory, Recorder]) -> None:
    make_env, _ = stub_env
    result = run(make_env(TOOLCHAIN="system"), "bash", str(UV_LOCK_SCRIPT))
    assert result.returncode != 0
    assert "uv 0.13.4" in result.stderr
    assert "curl" not in result.stderr


def test_uv_lock_tolerates_flox_activate_banner(stub_env: tuple[EnvFactory, Recorder]) -> None:
    make_env, record = stub_env
    result = run(make_env(), "bash", str(UV_LOCK_SCRIPT))
    assert result.returncode == 0, result.stderr
    assert record("flox") == [*FLOX_ACTIVATE, "lock"]
