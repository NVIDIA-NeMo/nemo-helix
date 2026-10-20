# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Analyst stack (ClickHouse + Platform model routing on :8080).

State lives under $RUNNER_TEMP/state.
`--verify` only re-checks both health endpoints and writes the summary line.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PLATFORM_ROOT = Path(__file__).resolve().parents[4]
CLICKHOUSE_URL = "http://localhost:8123"
_CLICKHOUSE_LABEL_FILTERS = (
    "label=nhx.nvidia.com/managed-by=nemo-helix",
    "label=nhx.nvidia.com/component=intake-clickhouse",
)


def _wait(url: str, attempts: int, delay: float) -> bool:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except OSError:
            time.sleep(delay)
    return False


def _fail_with_log(log: Path, message: str) -> None:
    print(message)
    if log.exists():
        print("".join(log.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-50:]))
    sys.exit(1)


def _intake_clickhouse_container(clickhouse_url: str = CLICKHOUSE_URL) -> str:
    """Resolve the labeled container publishing the stack's local ClickHouse URL."""
    parsed_url = urlsplit(clickhouse_url)
    if parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(f"stack ClickHouse URL is not local: {clickhouse_url}")
    host_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    command = ["docker", "ps"]
    for filter_value in (*_CLICKHOUSE_LABEL_FILTERS, f"publish={host_port}"):
        command.extend(("--filter", filter_value))
    command.extend(("--format", "{{.Names}}"))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    names = [name for name in result.stdout.splitlines() if name]
    if len(names) != 1:
        detail = "none found" if not names else f"found {', '.join(names)}"
        raise RuntimeError(f"expected one labeled Intake ClickHouse container publishing port {host_port}; {detail}")
    return names[0]


def verify() -> None:
    for url in (f"{CLICKHOUSE_URL}/ping", "http://localhost:8080/health/ready"):
        if not _wait(url, attempts=1, delay=0):
            sys.exit(f"verify failed: {url}")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
        fh.write("### stack-check ✓ ClickHouse + platform ready\n")


def main() -> None:
    if "--verify" in sys.argv[1:]:
        verify()
        return
    runner_temp = Path(os.environ["RUNNER_TEMP"])
    state = runner_temp / "state"
    (state / "clickhouse").mkdir(parents=True, exist_ok=True)
    (state / "nhx").mkdir(parents=True, exist_ok=True)
    log = runner_temp / "platform.log"

    subprocess.run(
        [str(PLATFORM_ROOT / "services/intake/scripts/spans/run_clickhouse.sh")],
        check=True,
        env={
            **os.environ,
            "CLICKHOUSE_DATA_DIR": str(state / "clickhouse"),
            "NHX_INTAKE_CLICKHOUSE_URL": CLICKHOUSE_URL,
        },
    )
    if not _wait(f"{CLICKHOUSE_URL}/ping", attempts=30, delay=2):
        sys.exit("ClickHouse never became ready")
    clickhouse_container = _intake_clickhouse_container()
    subprocess.run(
        ["docker", "exec", clickhouse_container, "clickhouse-client", "--query", "SYSTEM STOP TTL MERGES"],
        check=True,
    )

    subprocess.run(["uv", "sync"], check=True, cwd=PLATFORM_ROOT)
    with open(log, "ab") as fh:
        subprocess.Popen(
            [
                "uv",
                "run",
                "nemo",
                "services",
                "run",
                "--services",
                "auth,entities,intake,models,inference-gateway,secrets",
                "--controllers",
                "models",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
            ],
            cwd=PLATFORM_ROOT,
            env={
                **os.environ,
                "NHX_DATA_DIR": str(state / "nhx"),
                "NHX_INTAKE_CLICKHOUSE_URL": CLICKHOUSE_URL,
            },
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if not _wait("http://localhost:8080/health/ready", attempts=60, delay=5):
        _fail_with_log(log, "platform never became ready; last log lines:")


if __name__ == "__main__":
    main()
