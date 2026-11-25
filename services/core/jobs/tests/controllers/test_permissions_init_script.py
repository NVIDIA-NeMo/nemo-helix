# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fix-permissions init container's script, run under ``sh`` with fake ``mkdir`` and ``chmod``.

PATH holds only the fakes and the real ``grep`` the script needs, so no real file is created or changed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from nhx.core.jobs.controllers.backends.kubernetes.common import (
    PERMISSIONS_INIT_SCRIPT,
    build_permissions_init_container,
)

JOB_DIR = "/vol/jobs/default/job-1"
BUSYBOX_GONE = f"chmod: {JOB_DIR}/workspace/f12: No such file or directory"
GNU_GONE = f"chmod: cannot access '{JOB_DIR}/workspace/f13': No such file or directory"
DENIED = f"chmod: {JOB_DIR}/workspace/ro: Operation not permitted"


def _run(
    tmp_path: Path, *, chmod_output: str = "", chmod_exit: int = 0, mkdir_exit: int = 0, with_grep: bool = True
) -> tuple[int, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if with_grep:
        (bin_dir / "grep").symlink_to(shutil.which("grep") or pytest.fail("grep is needed to run the script"))
    for tool, exit_code, output in (("mkdir", mkdir_exit, ""), ("chmod", chmod_exit, chmod_output)):
        fake = bin_dir / tool
        (tmp_path / f"{tool}.out").write_text(output)
        # Shell builtins only: PATH has nothing else for the fakes to use.
        fake.write_text(
            f'#!/bin/sh\necho "$@" >> {tmp_path}/{tool}.calls\n'
            f"while IFS= read -r line; do printf '%s\\n' \"$line\" >&2; done < {tmp_path}/{tool}.out\n"
            f"exit {exit_code}\n"
        )
        fake.chmod(0o755)
    sh = shutil.which("sh") or pytest.fail("sh is needed to run the script")
    result = subprocess.run(
        [sh, "-c", PERMISSIONS_INIT_SCRIPT, "fix-permissions", JOB_DIR],
        env={"PATH": str(bin_dir)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stderr


def test_the_container_passes_the_job_directory_as_the_scripts_argument() -> None:
    container = build_permissions_init_container("job-storage", "default", "job-1", "busybox")

    assert container.command == ["sh", "-c", PERMISSIONS_INIT_SCRIPT, "fix-permissions", JOB_DIR]


def test_a_clean_chmod_succeeds(tmp_path: Path) -> None:
    code, _ = _run(tmp_path)

    assert code == 0
    assert (tmp_path / "mkdir.calls").read_text() == f"-p {JOB_DIR}\n"
    assert (tmp_path / "chmod.calls").read_text() == f"-R 777 {JOB_DIR}\n"


@pytest.mark.parametrize("output", [BUSYBOX_GONE, GNU_GONE, f"{BUSYBOX_GONE}\n{GNU_GONE}"])
def test_entries_removed_during_the_walk_are_tolerated(tmp_path: Path, output: str) -> None:
    """Something still running for the job deleted them between the listing and the chmod."""
    code, stderr = _run(tmp_path, chmod_output=output + "\n", chmod_exit=1)

    assert code == 0
    assert output in stderr


@pytest.mark.parametrize("output", [DENIED, f"{BUSYBOX_GONE}\n{DENIED}", ""])
def test_any_other_chmod_failure_fails_the_container(tmp_path: Path, output: str) -> None:
    code, stderr = _run(tmp_path, chmod_output=output + "\n" if output else "", chmod_exit=1)

    assert code == 1
    assert output in stderr


def test_a_failed_mkdir_fails_before_chmod(tmp_path: Path) -> None:
    code, _ = _run(tmp_path, mkdir_exit=1)

    assert code == 1
    assert not (tmp_path / "chmod.calls").exists()


def test_a_grep_that_cannot_run_fails_closed(tmp_path: Path) -> None:
    """Without grep the ENOENT check can't be made, so even ENOENT-only output fails the container."""
    code, _ = _run(tmp_path, chmod_output=BUSYBOX_GONE + "\n", chmod_exit=1, with_grep=False)

    assert code == 1
