# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job entrypoint. The controller downloads this module from the run fileset."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from barrier import BarrierFiles, live_fileset, wait_for_barrier
from hammer import Operations, live_operations, run_hammer
from history import HistoryLog

RESULT_PATH = Path("/tmp/service-stress.jsonl")


def main() -> None:
    kind = os.environ["STRESS_KIND"]
    duration = float(os.environ.get("STRESS_DURATION_SECONDS", "0"))
    timeout = float(os.environ.get("STRESS_TIMEOUT_SECONDS", "0"))
    service = os.environ.get("STRESS_SERVICE", "")
    rate = float(os.environ.get("STRESS_RATE", "1"))
    payload = b"x" * int(os.environ.get("STRESS_PAYLOAD_BYTES", "1024"))
    log = HistoryLog(RESULT_PATH)
    code = run(
        kind=kind,
        duration=duration,
        timeout=timeout,
        service=service,
        rate=rate,
        payload=payload,
        job=os.environ.get("NEMO_JOB_ID", ""),
        step=os.environ.get("STRESS_STEP_ID", ""),
        fetch=float(os.environ.get("STRESS_CODE_FETCH_SECONDS", "0")),
        clock=time.monotonic,
        sleep=time.sleep,
        log=log,
        publish=lambda: publish_result(RESULT_PATH),
    )
    if code:
        sys.exit(code)


def run(
    *,
    kind: str,
    duration: float,
    timeout: float = 0,
    service: str,
    rate: float,
    payload: bytes,
    job: str,
    step: str,
    fetch: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    log: HistoryLog,
    publish: Callable[[], None],
    calls: Operations | None = None,
    files: BarrierFiles | None = None,
    barrier_poll: float = 1.0,
) -> int:
    started = clock()
    achieved = 0.0
    reason = ""
    code = 0
    try:
        if kind == "idle":
            sleep(duration)
        elif kind == "barrier":
            port = files if files is not None else live_fileset()
            reason = wait_for_barrier(
                step=step,
                job=job,
                timeout=timeout,
                clock=clock,
                sleep=sleep,
                files=port,
                poll=barrier_poll,
            )
            code = 0 if reason == "barrier" else 1
        elif kind == "hammer":
            achieved = run_hammer(
                service=service,
                duration=duration,
                rate=rate,
                payload=payload,
                job=job,
                step=step,
                clock=clock,
                sleep=sleep,
                log=log,
                calls=live_operations() if calls is None else calls,
                publish=publish,
            )
    finally:
        log.write(
            {
                "record": "header",
                "kind": kind,
                "started_at": started,
                "ended_at": clock(),
                "code_fetch_seconds": fetch,
                "achieved_rate": achieved,
                "exit_reason": reason,
            }
        )
        log.close()
        publish()
    return code


def publish_result(path: Path) -> None:
    from nemo_helix_plugin.client_provider import get_task_nemo_client
    from nemo_helix_plugin.files.client import FilesClient
    from nemo_helix_plugin.jobs.client import JobsClient
    from nemo_helix_plugin.jobs.result_manager import result_manager_factory

    workspace = os.environ["NEMO_JOB_WORKSPACE"]
    client = get_task_nemo_client("jobs", workspace=workspace)
    manager = result_manager_factory(
        os.environ["NEMO_JOB_ID"],
        attempt_id=os.environ["NEMO_JOB_ATTEMPT_ID"],
        workspace=workspace,
        files_client=FilesClient.from_client(client),
        jobs_client=JobsClient.from_client(client),
    )
    manager.create_result("service-stress", path)
