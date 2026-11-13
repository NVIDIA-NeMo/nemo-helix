# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait until the controller writes a fileset release, or time out."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol


class BarrierFiles(Protocol):
    def write(self, path: str, data: bytes) -> None:
        """Write one fileset object."""

    def read(self, path: str) -> bytes | None:
        """Return one fileset object, or None when it is absent."""


def ready_path(step_id: str, job: str) -> str:
    return f"barrier/{step_id}/ready/{job}"


def release_path(step_id: str) -> str:
    return f"barrier/{step_id}/release"


def exit_path(step_id: str, job: str) -> str:
    return f"barrier/{step_id}/exit/{job}"


def wait_for_barrier(
    *,
    step: str,
    job: str,
    timeout: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    files: BarrierFiles,
    poll: float = 1.0,
) -> str:
    """Write ready, then return ``barrier`` or ``timeout`` after recording the exit reason."""
    files.write(ready_path(step, job), b"ready\n")
    deadline = clock() + timeout
    while True:
        if files.read(release_path(step)) is not None:
            files.write(exit_path(step, job), b"barrier\n")
            return "barrier"
        if clock() >= deadline:
            files.write(exit_path(step, job), b"timeout\n")
            return "timeout"
        sleep(poll)


class LiveFileset:
    """Fileset reads and writes from inside a platform job."""

    def __init__(self, workspace: str, fileset: str) -> None:
        from nemo_helix_plugin.client_provider import get_task_nemo_client
        from nemo_helix_plugin.files.client import FilesClient

        client = get_task_nemo_client("jobs", workspace=workspace)
        self._files = FilesClient.from_client(client)
        self.workspace = workspace
        self.fileset = fileset

    def write(self, path: str, data: bytes) -> None:
        self._files.upload_file(workspace=self.workspace, name=self.fileset, path=path, content=data).data()

    def read(self, path: str) -> bytes | None:
        from nemo_helix_plugin.client.errors import NemoHTTPError

        try:
            return self._files.download_file(workspace=self.workspace, name=self.fileset, path=path).read()
        except NemoHTTPError as exc:
            if exc.status_code == 404:
                return None
            raise


def live_fileset() -> LiveFileset:
    return LiveFileset(os.environ["NEMO_JOB_WORKSPACE"], os.environ["STRESS_FILESET"])
