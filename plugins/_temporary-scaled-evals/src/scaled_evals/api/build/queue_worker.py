# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable database-leased task image build worker."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

from scaled_evals.api.build.buildkit import build_revision_image
from scaled_evals.api.build.cloud_build import build_revision_image as build_cloud_revision_image
from scaled_evals.api.build.image_builder_service import resolve_uploaded_revision_image
from scaled_evals.api.build.task_image_identity import resolve_task_image
from scaled_evals.api.repositories.build_repository import TaskBuildJob, TaskBuildRepository
from scaled_evals.api.settings import settings

LOG = logging.getLogger(__name__)
ConnFactory = Callable[[], AbstractContextManager[psycopg.Connection]]


@contextmanager
def _default_connect() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(settings.resolved_database_url(), row_factory=dict_row)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


@dataclass
class TaskBuildWorker:
    connect: ConnFactory = _default_connect
    sleep: Callable[[float], None] = time.sleep
    claim_timeout: float = 90.0
    heartbeat_interval: float = 15.0
    max_attempts: int = 3
    retry_delay: float = 30.0
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}")

    def claim_next(self) -> TaskBuildJob | None:
        with self.connect() as conn:
            return TaskBuildRepository(conn).claim_next(
                worker_id=self.worker_id,
                claim_timeout=self.claim_timeout,
                max_attempts=self.max_attempts,
            )

    def work_once(self) -> bool:
        if settings.platform_build_jobs_enabled:
            return False
        job = self.claim_next()
        if job is None:
            return False
        self.run(job)
        return True

    def work_forever(self, *, idle_sleep: float = 2.0) -> None:
        next_presence_heartbeat = 0.0
        while True:
            try:
                now = time.monotonic()
                if now >= next_presence_heartbeat:
                    self._heartbeat_presence()
                    next_presence_heartbeat = now + self.heartbeat_interval
                worked = self.work_once()
            except Exception:  # noqa: BLE001 - transient DB failures must not kill the worker
                LOG.exception("task build queue poll failed; retrying")
                self.sleep(idle_sleep)
                continue
            if not worked:
                self.sleep(idle_sleep)

    def run(self, job: TaskBuildJob) -> bool:
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_forever,
            args=(job, stop),
            name=f"build-heartbeat-{job.task_id}-{job.revision}",
            daemon=True,
        )
        heartbeat.start()
        try:
            image_ref, image_digest = self._execute(job)
        except Exception as exc:  # noqa: BLE001 - persisted for retry/diagnosis
            error = str(exc)[:4000]
            LOG.exception(
                "task build failed task=%s revision=%s attempt=%s",
                job.task_id,
                job.revision,
                job.attempt,
            )
            with self.connect() as conn:
                TaskBuildRepository(conn).retry_or_fail(
                    job.task_id,
                    job.revision,
                    worker_id=self.worker_id,
                    build_error=error,
                    attempt=job.attempt,
                    max_attempts=self.max_attempts,
                    retry_delay=self.retry_delay,
                )
            return False
        else:
            with self.connect() as conn:
                completed = TaskBuildRepository(conn).complete(
                    job.task_id,
                    job.revision,
                    worker_id=self.worker_id,
                    image_ref=image_ref,
                    image_digest=image_digest,
                )
            if not completed:
                LOG.warning(
                    "discarded task build result after lease loss task=%s revision=%s",
                    job.task_id,
                    job.revision,
                )
            return completed
        finally:
            stop.set()
            heartbeat.join(timeout=max(1.0, self.heartbeat_interval * 2))

    def _execute(self, job: TaskBuildJob) -> tuple[str, str]:
        return execute_task_build(job)

    def _heartbeat_forever(self, job: TaskBuildJob, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_interval):
            try:
                with self.connect() as conn:
                    repo = TaskBuildRepository(conn)
                    repo.heartbeat_worker(self.worker_id)
                    alive = repo.heartbeat(
                        job.task_id,
                        job.revision,
                        worker_id=self.worker_id,
                    )
                if not alive:
                    LOG.warning(
                        "task build lease lost task=%s revision=%s",
                        job.task_id,
                        job.revision,
                    )
                    return
            except Exception:  # noqa: BLE001 - next heartbeat may recover
                LOG.exception(
                    "task build heartbeat failed task=%s revision=%s",
                    job.task_id,
                    job.revision,
                )

    def _heartbeat_presence(self) -> None:
        with self.connect() as conn:
            TaskBuildRepository(conn).heartbeat_worker(self.worker_id)


def execute_task_build(job: TaskBuildJob) -> tuple[str, str]:
    """Execute one claimed task-image build through its frozen backend."""
    if job.backend == "buildkit":
        return asyncio.run(build_revision_image(job.task_id, job.revision, job.object_key))
    if job.backend == "cloudbuild":
        image_ref, builder_digest = build_cloud_revision_image(job.task_id, job.revision, job.object_key)
        resolved = resolve_task_image(image_ref, expected_digest=builder_digest)
        return resolved.runtime_ref, resolved.digest
    if job.backend == "prebuilt":
        payload = job.payload
        resolved = resolve_task_image(
            str(payload["image_ref"]),
            expected_digest=str(payload.get("expected_digest") or "") or None,
        )
        return resolved.runtime_ref, resolved.digest
    if job.backend != "image_builder_service":
        raise ValueError(f"unsupported task build backend: {job.backend}")

    payload = job.payload
    image_ref, builder_digest = resolve_uploaded_revision_image(
        tarball_object_key=job.object_key,
        context_path=str(payload.get("context_path") or "."),
        builder_source_commit=str(payload.get("builder_source_commit") or settings.image_builder_source_commit),
    )
    resolved = resolve_task_image(image_ref, expected_digest=builder_digest)
    return resolved.runtime_ref, resolved.digest
