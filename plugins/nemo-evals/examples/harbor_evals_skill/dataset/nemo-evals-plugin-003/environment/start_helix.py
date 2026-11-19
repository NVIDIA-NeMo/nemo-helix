# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the inner Helix and leave bounded diagnostics for the Harbor trial."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

STATE = Path("/var/lib/helix-task")
SERVICE = STATE / "service"
URL = "http://127.0.0.1:8080"


def ready() -> bool:
    try:
        with urlopen(URL + "/health/ready", timeout=2) as response:
            if json.load(response)["status"] != "ready":
                return False
        with urlopen(URL + "/apis/jobs/v2/execution-profiles", timeout=2) as response:
            return any(
                profile.get("provider") == "subprocess" and profile.get("profile") == "default"
                for profile in json.load(response)
            )
    except (URLError, TimeoutError, ValueError, KeyError):
        return False


def main() -> None:
    begin = time.monotonic()
    log = STATE / "root/server.log"
    if ready():
        return
    try:
        with log.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                [
                    "/app/.venv/bin/nemo",
                    "services",
                    "run",
                    "--services",
                    "auth,entities,files,jobs,secrets,models,inference-gateway,evals",
                    "--controllers",
                    "jobs,entities,models",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8080",
                ],
                user="helix-task",
                group="helix-task",
                extra_groups=[],
                umask=0o077,
                cwd=SERVICE,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env={
                    "HOME": str(SERVICE),
                    "XDG_CONFIG_HOME": str(SERVICE / ".config"),
                    "PATH": "/app/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "NHX_DATA_DIR": str(SERVICE),
                    "DATABASE_DIALECT": "sqlite",
                    "DATABASE_PATH": str(SERVICE / "helix.db"),
                    "NHX_CONFIG_FILE_PATH": "/opt/helix-task/helix-config.yaml",
                    "NHX_BASE_URL": URL,
                    "NEMO_BASE_URL": URL,
                    "NEMO_TELEMETRY_ENABLED": "false",
                    "NSS_DISABLE_AUTO_LOGGING": "1",
                    "OMP_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1",
                    "TIKTOKEN_CACHE_DIR": "/app/tiktoken-cache",
                },
            )
            while process.poll() is None and time.monotonic() - begin < 240:
                if ready():
                    print(
                        f"Inner Helix ready after {time.monotonic() - begin:.1f}s (service pid {process.pid})",
                        flush=True,
                    )
                    return
                time.sleep(1)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise SystemExit(f"Inner Helix did not become ready (process exit: {process.poll()})")
    finally:
        with log.open("rb") as source, (STATE / "evidence/startup.log").open("wb") as artifact:
            source.seek(max(0, log.stat().st_size - 65_536))
            artifact.write(source.read(65_536))
            artifact.write(f"\nHealthcheck elapsed seconds: {time.monotonic() - begin:.1f}\n".encode())
            peak = Path("/sys/fs/cgroup/memory.peak")
            if peak.is_file():
                artifact.write(f"Sandbox memory.peak bytes: {peak.read_text().strip()}\n".encode())


if __name__ == "__main__":
    main()
