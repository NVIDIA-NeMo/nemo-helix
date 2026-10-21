# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job entrypoint. The controller downloads this module from the run fileset."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from hammer import Operations, live_operations, run_hammer
from history import HistoryLog

RESULT_PATH = Path("/tmp/service-stress.jsonl")


def main() -> None:
    kind = os.environ["STRESS_KIND"]
    duration = float(os.environ.get("STRESS_DURATION_SECONDS", "0"))
    service = os.environ.get("STRESS_SERVICE", "")
    rate = float(os.environ.get("STRESS_RATE", "1"))
    payload = b"x" * int(os.environ.get("STRESS_PAYLOAD_BYTES", "1024"))
    log = HistoryLog(RESULT_PATH)
    calls = live_operations() if kind == "hammer" else _Unused()
    run(
        kind=kind,
        duration=duration,
        service=service,
        rate=rate,
        payload=payload,
        job=os.environ.get("NEMO_JOB_ID", ""),
        step=os.environ.get("STRESS_STEP_ID", ""),
        fetch=float(os.environ.get("STRESS_CODE_FETCH_SECONDS", "0")),
        clock=time.monotonic,
        sleep=time.sleep,
        log=log,
        calls=calls,
        publish=lambda: publish_result(RESULT_PATH),
    )


def run(
    *,
    kind: str,
    duration: float,
    service: str,
    rate: float,
    payload: bytes,
    job: str,
    step: str,
    fetch: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    log: HistoryLog,
    calls: Operations,
    publish: Callable[[], None],
) -> None:
    started = clock()
    achieved = 0.0
    try:
        if kind == "idle":
            sleep(duration)
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
                calls=calls,
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
            }
        )
        log.close()
        publish()


def publish_result(path: Path) -> None:
    from nemo_helix_plugin.client_provider import get_task_nemo_client
    from nemo_helix_plugin.files.client import FilesClient
    from nemo_helix_plugin.jobs.client import JobsClient
    from nemo_helix_plugin.jobs.result_manager import result_manager_factory

    workspace = os.environ["NEMO_JOB_WORKSPACE"]
    client = get_task_nemo_client("jobs", workspace=workspace)
    manager = result_manager_factory(
        os.environ["NEMO_JOB_ID"],
        attempt_id=os.environ.get("NEMO_JOB_ATTEMPT_ID"),
        workspace=workspace,
        files_client=FilesClient.from_client(client),
        jobs_client=JobsClient.from_client(client),
    )
    manager.create_result("service-stress", path)


class _Unused:
    """Exit and idle jobs do not call a service."""

    def fileset_create(self, name: str) -> int:
        return 0

    def fileset_get(self, name: str) -> int:
        return 0

    def fileset_list(self) -> int:
        return 0

    def fileset_delete(self, name: str) -> int:
        return 0

    def file_upload(self, name: str, payload: bytes) -> int:
        return 0

    def file_download(self, name: str) -> int:
        return 0

    def file_list(self) -> int:
        return 0

    def file_delete(self, name: str) -> int:
        return 0

    def secret_create(self, name: str) -> int:
        return 0

    def secret_get(self, name: str) -> int:
        return 0

    def secret_list(self) -> int:
        return 0

    def secret_update(self, name: str) -> int:
        return 0

    def secret_delete(self, name: str) -> int:
        return 0

    def secret_value(self, name: str) -> int:
        return 0

    def discovery(self) -> int:
        return 0

    def authz(self) -> int:
        return 0

    def access_key_create(self) -> tuple[int, str]:
        return 0, ""

    def access_key_list(self) -> int:
        return 0

    def access_key_revoke(self, jti: str) -> int:
        return 0

    def cleanup(self) -> None:
        return None
