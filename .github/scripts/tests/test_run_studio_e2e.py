# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Studio runner's readiness, exit-status and cleanup contract."""

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run-studio-e2e.sh"


def executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n")
    path.chmod(0o755)


@pytest.mark.parametrize("test_exit,ready", [(0, True), (7, True), (0, False)])
def test_studio_runner(tmp_path: Path, test_exit: int, ready: bool) -> None:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8080)) == 0:
            pytest.skip("Studio runner reserves port 8080")

    script = tmp_path / ".github/scripts/run-studio-e2e.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    (tmp_path / "web").mkdir()
    tools = tmp_path / "bin"
    tools.mkdir()
    executable(
        tools / "uv",
        'echo "$NHX_DATA_DIR" > "$TEST_STATE/data-dir"\n'
        'echo "platform startup diagnostic"\n'
        "trap 'echo stopped > \"$TEST_STATE/stopped\"; exit 0' TERM\n"
        "while true; do /bin/sleep 0.01; done",
    )
    executable(tools / "curl", "exit 0" if ready else "exit 22")
    executable(tools / "sleep", "/bin/sleep 0.01")
    executable(
        tools / "pnpm",
        'printf "%s\\n" "$@" > "$TEST_STATE/test-args"\n'
        'test "$VSERVICE_URL_STUDIO_UI" = "http://127.0.0.1:8080/studio/"\n'
        'test "$VITE_NHX_BASE_URL" = "$NHX_BASE_URL"\n'
        f"exit {test_exit}",
    )
    result = subprocess.run(
        ["bash", str(script), "--retries=0"],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "TEST_STATE": str(tmp_path),
            "STUDIO_E2E_LOG_DIR": str(tmp_path / "logs"),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == (test_exit if ready else 1)
    assert (tmp_path / "stopped").read_text().strip() == "stopped"
    assert not Path((tmp_path / "data-dir").read_text().strip()).exists()
    if ready:
        assert (tmp_path / "test-args").read_text().splitlines() == [
            "--filter",
            "nemo-studio-ui",
            "test:e2e",
            "--retries=0",
        ]
    else:
        assert not (tmp_path / "test-args").exists()
        assert "platform startup diagnostic" in result.stderr
