# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for script/uv, the shared uv shim used by make and the pre-commit hooks.

The shim is the single place that decides which uv binary runs: PATH uv for
TOOLCHAIN=system or an already-activated Flox environment, otherwise the
Flox-pinned uv via `flox -q activate`. Flox is not required here: every flox
interaction is driven by stubs on PATH.
"""

import json
import os
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM = REPO_ROOT / "script" / "uv"
UV_LOCK_SCRIPT = REPO_ROOT / "script" / "uv-lock.sh"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

# Shell scripts whose uv invocations must route through script/uv.
# Value: a literal fragment proving the script references the shim (an explicit
# script/uv path, or the sibling-shim form for scripts that live in script/).
SHELL_SURFACES = {
    "script/generate-openapi-spec.sh": '/uv"',
    "script/uv-lock.sh": '/uv"',
    "tools/lint/run-ty-check.sh": "script/uv",
}

# Pre-commit hook entries whose uv invocations must route through script/uv.
PRE_COMMIT_UV_HOOKS = (
    "config-reference-docs",
    "copyright-fix",
    "skill-frontmatter-lint",
)

STUB = """\
#!/usr/bin/env python3
import json
import os
import sys

record_dir = os.environ["RECORD_DIR"]
name = os.path.basename(sys.argv[0])
with open(os.path.join(record_dir, name + ".json"), "w") as handle:
    json.dump(sys.argv, handle)
if sys.argv[1:] == ["--version"]:
    print("uv 0.0.0")
raise SystemExit(0)
"""


def make_stub_env(
    tmp_path: Path,
) -> tuple[Callable[..., dict[str, str]], Callable[[str], list[str] | None]]:
    """Return (env_factory, record_reader) with `uv` and `flox` stubs on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("uv", "flox"):
        stub = bin_dir / name
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    def make_env(
        toolchain: str | None = None,
        flox_env_project: str | None = None,
        include_flox: bool = True,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env["RECORD_DIR"] = str(record_dir)
        env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
        if toolchain is None:
            env.pop("TOOLCHAIN", None)
        else:
            env["TOOLCHAIN"] = toolchain
        if flox_env_project is None:
            env.pop("FLOX_ENV_PROJECT", None)
        else:
            env["FLOX_ENV_PROJECT"] = flox_env_project
        if not include_flox:
            (bin_dir / "flox").unlink()
        return env

    def record(name: str) -> list[str] | None:
        path = record_dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    return make_env, record


def run_shim(
    env: dict[str, str],
    *args: str,
    cwd: Path | None = None,
    script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script or SHIM), *args],
        env=env,
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def test_toolchain_system_execs_path_uv_without_flox(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(make_env(toolchain="system"), "--version")
    assert result.returncode == 0, result.stderr
    uv_args = record("uv")
    assert uv_args is not None
    assert uv_args[1:] == ["--version"]
    assert record("flox") is None


def test_active_flox_env_execs_path_uv_without_flox(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(make_env(flox_env_project=str(REPO_ROOT)), "--version")
    assert result.returncode == 0, result.stderr
    assert record("uv") is not None
    assert record("flox") is None


def test_active_flox_env_matches_resolved_path(tmp_path: Path) -> None:
    """FLOX_ENV_PROJECT reached through a symlink still counts as the repo."""
    make_env, record = make_stub_env(tmp_path)
    link = tmp_path / "repo-link"
    link.symlink_to(REPO_ROOT, target_is_directory=True)
    result = run_shim(make_env(flox_env_project=str(link)), "--version")
    assert result.returncode == 0, result.stderr
    assert record("uv") is not None
    assert record("flox") is None


def test_unactivated_shell_runs_uv_via_flox(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(make_env(), "lock", "--check")
    assert result.returncode == 0, result.stderr
    flox_args = record("flox")
    assert flox_args is not None
    assert flox_args[1:] == ["-q", "activate", "--dir", str(REPO_ROOT), "--", "uv", "lock", "--check"]
    assert record("uv") is None


def test_missing_flox_reports_how_to_proceed(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(make_env(include_flox=False), "--version")
    assert result.returncode != 0
    assert "flox" in result.stderr
    assert "TOOLCHAIN=system" in result.stderr


def test_makefile_routes_uv_through_shim() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    assert "UV := $(CURDIR)/script/uv" in text
    assert "UV := $(FLOX_EXEC) uv" not in text


def test_uv_lock_version_mismatch_does_not_suggest_curl(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(make_env(toolchain="system"), cwd=REPO_ROOT, script=UV_LOCK_SCRIPT)
    assert result.returncode != 0
    assert "curl" not in result.stderr
    assert "flox -q activate" in result.stderr
    assert "TOOLCHAIN=system" in result.stderr


def test_generate_openapi_spec_routes_through_shim(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(
        make_env(toolchain="system"),
        cwd=REPO_ROOT,
        script=REPO_ROOT / "script" / "generate-openapi-spec.sh",
    )
    assert result.returncode == 0, result.stderr
    uv_args = record("uv")
    assert uv_args is not None
    assert uv_args[1:] == ["run", "--frozen", "python", "-m", "script.generate_openapi_spec"]
    assert record("flox") is None


def tokenize(line: str) -> list[str]:
    return [token for raw in line.split() if (token := raw.strip("'\""))]


def bare_uv_lines(text: str) -> list[int]:
    """Line numbers of comment-stripped lines whose tokens include bare `uv`."""
    offenders = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if "uv" in tokenize(line):
            offenders.append(lineno)
    return offenders


@pytest.mark.parametrize("relpath,reference", sorted(SHELL_SURFACES.items()))
def test_shell_surfaces_do_not_invoke_bare_uv(relpath: str, reference: str) -> None:
    text = (REPO_ROOT / relpath).read_text()
    assert reference in text, f"{relpath} must route uv through script/uv"
    offenders = bare_uv_lines(text)
    assert not offenders, f"{relpath} invokes bare uv on lines {offenders}"


def pre_commit_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("- id: "):
            hook_id = stripped[len("- id: ") :]
            for follow in lines[index + 1 :]:
                follow_stripped = follow.strip()
                if follow_stripped.startswith("- id: "):
                    break
                if follow_stripped.startswith("entry: "):
                    entries[hook_id] = follow_stripped[len("entry: ") :]
        index += 1
    return entries


@pytest.mark.parametrize("hook_id", PRE_COMMIT_UV_HOOKS)
def test_pre_commit_entries_do_not_invoke_bare_uv(hook_id: str) -> None:
    entry = pre_commit_entries(PRE_COMMIT_CONFIG.read_text())[hook_id]
    assert "script/uv" in entry, f"{hook_id} must route uv through script/uv"
    assert "uv" not in tokenize(entry), f"{hook_id} entry invokes bare uv: {entry}"
