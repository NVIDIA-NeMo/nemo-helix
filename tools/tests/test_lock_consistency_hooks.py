# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SOURCE_LOCK = Path("docker/locks/nhx-gym-host/uv.lock")
BUNDLED_LOCK = Path("plugins/nemo-evals/src/nemo_evals/gym_registered_agent/nhx-gym-host.uv.lock")


def copy_hook(root: Path, name: str) -> Path:
    script = root / "tools/lint" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "tools/lint" / name, script)
    return script


@pytest.mark.parametrize("state", ["matching", "stale", "missing-source", "missing-bundle"])
def test_gym_hook_checks_exact_bytes_and_prints_the_repair(tmp_path: Path, state: str) -> None:
    script = copy_hook(tmp_path, "check-gym-host-lock.sh")
    for path, missing in ((SOURCE_LOCK, "missing-source"), (BUNDLED_LOCK, "missing-bundle")):
        if state == missing:
            continue
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stale\n" if state == "stale" and path == BUNDLED_LOCK else b"lock\n")

    result = subprocess.run(["bash", str(script)], cwd=tmp_path / "tools", capture_output=True, text=True)

    assert result.returncode == (0 if state == "matching" else 1)
    if state != "matching":
        assert f"cp {SOURCE_LOCK} {BUNDLED_LOCK}" in result.stderr
    if state == "stale":
        assert (tmp_path / BUNDLED_LOCK).read_bytes() == b"stale\n"  # check-only, no automatic repair


@pytest.mark.parametrize("exit_code", [0, 1])
def test_pnpm_hook_uses_frozen_check_only_mode(tmp_path: Path, exit_code: int) -> None:
    script = copy_hook(tmp_path, "check-pnpm-lock.sh")
    (tmp_path / "web").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pnpm = bin_dir / "pnpm"
    pnpm.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\nexit "$PNPM_TEST_EXIT"\n')
    pnpm.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path / "tools",
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "PNPM_TEST_EXIT": str(exit_code)},
    )

    assert result.returncode == exit_code
    assert result.stdout.splitlines() == [
        str(tmp_path / "web"),
        "install",
        "--lockfile-only",
        "--frozen-lockfile",
        "--ignore-scripts",
    ]
    if exit_code:
        assert "pnpm install --lockfile-only --ignore-scripts" in result.stderr


@pytest.mark.parametrize(
    ("hook_id", "included", "excluded"),
    [
        (
            "gym-host-lock-consistency",
            [
                str(SOURCE_LOCK),
                str(BUNDLED_LOCK),
                "docker/locks/nhx-gym-host/pyproject.toml",
                "tools/lint/check-gym-host-lock.sh",
                ".pre-commit-config.yaml",
            ],
            ["uv.lock", "docker/locks/nhx-gym-tasks/uv.lock", "README.md"],
        ),
        (
            "pnpm-lock-consistency",
            [
                "web/package.json",
                "web/packages/studio/package.json",
                "web/pnpm-workspace.yaml",
                "web/pnpm-lock.yaml",
                "tools/lint/check-pnpm-lock.sh",
                ".pre-commit-config.yaml",
            ],
            ["pyproject.toml", "web/packages/studio/src/index.tsx", "README.md"],
        ),
    ],
)
def test_hooks_trigger_on_their_inputs_only(hook_id: str, included: list[str], excluded: list[str]) -> None:
    config = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
    hook = next(hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == hook_id)
    assert hook["pass_filenames"] is False
    assert hook["language"] == "system"
    assert all(re.search(hook["files"], path) for path in included)
    assert all(not re.search(hook["files"], path) for path in excluded)
