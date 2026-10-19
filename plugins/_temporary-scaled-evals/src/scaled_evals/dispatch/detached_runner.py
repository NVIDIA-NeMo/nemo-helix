# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable foreground wrapper for detached dispatch subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(pid_path: Path, exit_path: Path, token: str, argv: list[str]) -> int:
    """Wait for the parent identity record, run argv, and persist terminal state."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            identity = json.loads(pid_path.read_text())
        except (OSError, ValueError):
            time.sleep(0.01)
            continue
        if identity.get("token") == token and identity.get("pid") == os.getpid():
            break
        time.sleep(0.01)
    else:
        return 125

    try:
        try:
            completed = subprocess.run(argv, check=False)
            exit_code = completed.returncode
            error = None
        except OSError as exc:
            exit_code = 127
            error = str(exc)
        payload: dict[str, object] = {
            "token": token,
            "exit_code": exit_code,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        if error is not None:
            payload["error"] = error
        _atomic_json(exit_path, payload)
        return exit_code
    finally:
        try:
            if json.loads(pid_path.read_text()).get("token") == token:
                pid_path.unlink()
        except (OSError, ValueError):
            pass


def main() -> int:
    if len(sys.argv) < 6 or sys.argv[1] != "run":
        raise SystemExit("usage: detached_runner run PID_PATH EXIT_PATH TOKEN COMMAND [ARG ...]")
    return run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])


if __name__ == "__main__":
    raise SystemExit(main())
