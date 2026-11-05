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
import re
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM = REPO_ROOT / "script" / "uv"
UV_LOCK_SCRIPT = REPO_ROOT / "script" / "uv-lock.sh"
GENERATE_OPENAPI_SCRIPT = REPO_ROOT / "script" / "generate-openapi-spec.sh"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

# Shell scripts whose uv invocations must route through script/uv.
# Value: a literal fragment proving the script references the shim (an explicit
# script/uv path, or the sibling-shim form for scripts that live in script/).
SHELL_SURFACES = {
    "script/generate-openapi-spec.sh": '/uv"',
    "script/uv-lock.sh": '/uv"',
    "tools/lint/run-ty-check.sh": "script/uv",
}

# Pre-commit hook entries that must reference script/uv. Every entry in the
# config is additionally scanned for a bare uv command.
PRE_COMMIT_UV_HOOKS = (
    "config-reference-docs",
    "copyright-fix",
    "skill-frontmatter-lint",
)

# `uv` in command position: line start, after a shell command separator, after
# `(` (which covers command substitution like $(uv ...)), or after a command
# keyword. Catches the bare-uv shapes without flagging prose or identifiers.
# Each leading VAR=value assignment must be followed by whitespace, which keeps the
# repetition unambiguous (no exponential backtracking).
BARE_UV_COMMAND_RE = re.compile(
    r"""(?:^|[;&|(]|\b(?:exec|command|do|then|else)\s+)"""
    r"""\s*(?:[A-Za-z_][A-Za-z_0-9]*=(?:"[^"]*"|'[^']*'|[^\s"']*)\s+)*"""
    r"""\buv(?:\s|$)"""
)

# Shell metacharacters that can precede a command, used to split pre-commit
# entry strings such as `bash -c 'uv run ...'` into command tokens.
ENTRY_TOKEN_SPLIT_RE = re.compile(r"""[\s;|&()$'"]+""")

STUB = """\
#!/usr/bin/env python3
import json
import os
import sys

record_dir = os.environ["RECORD_DIR"]
name = os.path.basename(sys.argv[0])
with open(os.path.join(record_dir, name + ".json"), "w") as handle:
    json.dump(sys.argv, handle)
args = sys.argv[1:]
if name == "flox" and args[-2:] == ["uv", "--version"]:
    # The root Flox environment's activate profile writes banners to stdout.
    print("NeMo Helix")
    print("uv 0.13.4")
elif args == ["--version"]:
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
        include_uv: bool = True,
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
        # Drop every PATH entry that provides a tool under test as missing, so a host
        # install cannot leak in; keep the rest so `env` still finds bash and python3.
        for name, include in (("flox", include_flox), ("uv", include_uv)):
            if not include:
                (bin_dir / name).unlink()
                env["PATH"] = os.pathsep.join(d for d in env["PATH"].split(os.pathsep) if not (Path(d) / name).exists())
        return env

    def record(name: str) -> list[str] | None:
        path = record_dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    return make_env, record


def run_shim(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    # Exec the shim directly, the way make ($(CURDIR)/script/uv) and the
    # pre-commit entries (script/uv) do: a non-executable blob fails here too.
    return subprocess.run([str(SHIM), *args], env=env, capture_output=True, text=True, check=False)


def run_script(
    env: dict[str, str],
    script: Path,
    *args: str,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        env=env,
        cwd=str(cwd),
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


def test_toolchain_system_without_uv_reports_how_to_proceed(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_shim(make_env(toolchain="system", include_uv=False), "--version")
    assert result.returncode != 0
    assert "TOOLCHAIN=system" in result.stderr
    assert "uv" in result.stderr
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


def test_other_flox_env_still_activates_repo_env(tmp_path: Path) -> None:
    """FLOX_ENV_PROJECT pointing at a different Flox env must not shortcut."""
    make_env, record = make_stub_env(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = run_shim(make_env(flox_env_project=str(elsewhere)), "--version")
    assert result.returncode == 0, result.stderr
    flox_args = record("flox")
    assert flox_args is not None
    assert flox_args[1:] == ["-q", "activate", "--dir", str(REPO_ROOT), "--", "uv", "--version"]
    assert record("uv") is None


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


def test_uv_lock_version_mismatch_does_not_suggest_curl(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_script(make_env(toolchain="system"), UV_LOCK_SCRIPT)
    assert result.returncode != 0
    assert "curl" not in result.stderr
    assert "uv 0.13.4" in result.stderr
    assert "flox -q activate" in result.stderr
    assert "TOOLCHAIN=system" in result.stderr


def test_uv_lock_tolerates_flox_activate_stdout(tmp_path: Path) -> None:
    """The version gate must survive Flox profile banners around uv output."""
    make_env, record = make_stub_env(tmp_path)
    result = run_script(make_env(), UV_LOCK_SCRIPT)
    assert result.returncode == 0, result.stderr
    # The version check ran through flox, then the lock ran through flox again.
    flox_args = record("flox")
    assert flox_args is not None
    assert flox_args[1:] == ["-q", "activate", "--dir", str(REPO_ROOT), "--", "uv", "lock"]
    assert record("uv") is None


def test_generate_openapi_spec_routes_through_shim(tmp_path: Path) -> None:
    make_env, record = make_stub_env(tmp_path)
    result = run_script(make_env(toolchain="system"), GENERATE_OPENAPI_SCRIPT)
    assert result.returncode == 0, result.stderr
    uv_args = record("uv")
    assert uv_args is not None
    assert uv_args[1:] == ["run", "--frozen", "python", "-m", "script.generate_openapi_spec"]
    assert record("flox") is None


def test_makefile_routes_uv_through_shim() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    assert 'UV := "$(CURDIR)/script/uv"' in text
    assert "UV := $(FLOX_EXEC) uv" not in text


def bare_uv_lines(text: str) -> list[int]:
    """Line numbers of comment-stripped lines with `uv` in command position."""
    offenders = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if BARE_UV_COMMAND_RE.search(line):
            offenders.append(lineno)
    return offenders


@pytest.mark.parametrize(
    "line,is_bare",
    [
        ("uv run pytest", True),
        ("  uv sync", True),
        ("foo && uv lock", True),
        ("out=$(uv --version)", True),
        ("then uv run x", True),
        ('UV_FROZEN=1 FOO="a b" uv run x', True),
        ('"${uv_shim}" run --frozen x', False),
        ("echo use uv to install", False),
        ("uv_shim=script/uv", False),
        ("A=" + " A=" * 5000, False),
    ],
)
def test_bare_uv_command_re(line: str, is_bare: bool) -> None:
    assert bool(BARE_UV_COMMAND_RE.search(line)) is is_bare


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
def test_pre_commit_uv_hooks_reference_shim(hook_id: str) -> None:
    entry = pre_commit_entries(PRE_COMMIT_CONFIG.read_text())[hook_id]
    assert "script/uv" in entry, f"{hook_id} must route uv through script/uv"


def test_no_pre_commit_entry_invokes_bare_uv() -> None:
    """Guards every entry, not just today's known uv hooks."""
    entries = pre_commit_entries(PRE_COMMIT_CONFIG.read_text())
    assert entries, "pre-commit entry parsing found no entries"
    offenders = [(hook_id, entry) for hook_id, entry in entries.items() if "uv" in ENTRY_TOKEN_SPLIT_RE.split(entry)]
    assert not offenders, f"pre-commit entries invoke bare uv: {offenders}"
