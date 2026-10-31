# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-run archives, embedded in their run's entity.

Every lease change rewrites the run under a compare-and-swap; claims,
heartbeats and publication are fenced by the claim token.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from nemo_scaled_evals_plugin.entities import ScaledEvalsBenchmarkArchive, ScaledEvalsBenchmarkRun

from scaled_evals.api import entity_store
from scaled_evals.api.benchmark_run_store import BenchmarkRunStore
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.repositories.base_repository import Conflict, NotFound
from scaled_evals.api.resource_access import INTERNAL_ACCESS, ResourceAccess
from scaled_evals.models.benchmark_archives import BenchmarkArchiveMember

_MAX_ATTEMPTS = 3
_STATUSES = ("queued", "building", "ready", "failed")
T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _row(entity: ScaledEvalsBenchmarkRun) -> dict[str, Any] | None:
    if entity.archive is None:
        return None
    return {"benchmark_run_id": entity.name, **entity.archive.model_dump()}


def _claimable(archive: ScaledEvalsBenchmarkArchive | None, now: datetime, claim_timeout: float) -> bool:
    if archive is None:
        return False
    if archive.status == "queued":
        return True
    return (
        archive.status == "building"
        and archive.claimed_at is not None
        and archive.claimed_at < now - timedelta(seconds=claim_timeout)
    )


def _owns(entity: ScaledEvalsBenchmarkRun, token: str) -> ScaledEvalsBenchmarkArchive | None:
    archive = entity.archive
    return archive if archive is not None and archive.claim_token == token and archive.status == "building" else None


def _members(run_id: str) -> list[dict]:
    """Return the run's live member evaluations by id, with the task slug and name they ran."""
    rows = sorted(EvaluationStore().list_for_runs([run_id]), key=lambda row: row["id"])
    for row in rows:
        task = (row.get("execution_snapshot") or {}).get("task") or {}
        row["task_slug"], row["task_name"] = task.get("slug"), task.get("name")
    return rows


class BenchmarkArchiveStore:
    def __init__(self, *, access: ResourceAccess = INTERNAL_ACCESS) -> None:
        self._runs = BenchmarkRunStore(access=access)

    def _runs_with_archive(self, *statuses: str) -> list[ScaledEvalsBenchmarkRun]:
        """Return runs whose archive has one of `statuses`, deleted runs included."""
        return entity_store.list_all(self._runs._client, ScaledEvalsBenchmarkRun, {"archive_status": statuses})

    def _mutate(
        self,
        run_id: str,
        change: Callable[[ScaledEvalsBenchmarkRun], tuple[T, bool]],
        *,
        include_deleted: bool = False,
    ) -> T | None:
        def synced(entity: ScaledEvalsBenchmarkRun) -> tuple[T, bool]:
            result, write = change(entity)
            entity.archive_status = None if entity.archive is None else entity.archive.status
            return result, write

        return self._runs._mutate(run_id, synced, include_deleted=include_deleted)

    def get(self, run_id: str) -> dict | None:
        entity = self._runs._get(run_id)
        return None if entity is None else _row(entity)

    def request(self, run_id: str, *, force: bool = False) -> dict:
        run = self._runs.get_for_mutation(run_id)
        if run is None:
            raise NotFound("benchmark_run", "benchmark run not found")
        if run["framework"] != "harbor":
            raise Conflict("unsupported_framework", "benchmark archives require Harbor")
        existing = self.get(run_id)
        if existing and (existing["status"] in {"queued", "building"} or (existing["status"] == "ready" and not force)):
            return existing
        # ponytail: members are read before the write, so one may change first.
        # validate_members re-checks them before publishing.
        rows = _members(run_id)
        if not rows:
            raise Conflict("empty_run", "benchmark run has no evaluations")
        for row in rows:
            if row["status"] not in {"succeeded", "failed", "cancelled"}:
                raise Conflict("run_not_terminal", "all member evaluations must be terminal")
            if (
                row["archive_status"] != "ready"
                or not row["archive_object_key"]
                or row["evidence_status"] != "ready"
                or row["cancel_teardown_status"] in {"pending", "failed"}
            ):
                raise Conflict(
                    "member_artifacts_not_ready",
                    f"evaluation {row['id']} must finish evidence, teardown and archive generation before export",
                )
        members = [BenchmarkArchiveMember.model_validate(row).model_dump(mode="json") for row in rows]

        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[dict | None, bool]:
            current = entity.archive
            if current and (current.status in {"queued", "building"} or (current.status == "ready" and not force)):
                return _row(entity), False
            entity.archive = ScaledEvalsBenchmarkArchive(
                generation=str(uuid4()),
                status="queued",
                members=members,
                requested_at=_now(),
                cleanup_checked_at=current.cleanup_checked_at if current else None,
            )
            return _row(entity), True

        row = self._mutate(run_id, change)
        if row is None:
            raise NotFound("benchmark_run", "benchmark run not found")
        return row

    def list_claimable(self, *, claim_timeout: float, limit: int) -> list[dict]:
        """Return archives a builder may claim, without claiming them."""
        now = _now()
        rows = [
            row
            for entity in self._runs_with_archive("queued", "building")
            if entity.deleted_at is None
            and _claimable(entity.archive, now, claim_timeout)
            and (row := _row(entity)) is not None
        ]
        return sorted(rows, key=lambda row: row["requested_at"])[:limit]

    def claim(self, *, claim_timeout: float, run_id: str) -> dict | None:
        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[dict | None, bool]:
            archive = entity.archive
            now = _now()
            if archive is None or not _claimable(archive, now, claim_timeout):
                return None, False
            if archive.attempts >= _MAX_ATTEMPTS:
                archive.status, archive.error = "failed", "archive worker lease expired"
            else:
                archive.status, archive.error = "building", None
            archive.claimed_at, archive.claim_token = now, str(uuid4())
            archive.attempts += 1
            return _row(entity), True

        return self._mutate(run_id, change)

    def skip_attempt(self, run_id: str, *, generation: str, attempts: int) -> None:
        """Count an attempt whose builder ended before claiming it, failing the third."""

        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[None, bool]:
            archive = entity.archive
            if (
                archive is None
                or archive.generation != generation
                or archive.attempts != attempts
                or archive.status not in {"queued", "building"}
            ):
                return None, False
            archive.attempts += 1
            if archive.attempts >= _MAX_ATTEMPTS:
                archive.status, archive.error = "failed", "archive build job ended before claiming"
            return None, True

        self._mutate(run_id, change)

    def claim_cleanup(self, *, interval_seconds: float) -> str | None:
        """Schedule another bounded sweep, including failed or soft-deleted runs.

        The timestamp is a throttle, not a completion marker: crashes, transient
        delete failures and uploads completing late are retried on later sweeps.
        """
        epoch = datetime.min.replace(tzinfo=UTC)

        def due(archive: ScaledEvalsBenchmarkArchive | None, now: datetime) -> bool:
            return archive is not None and (
                archive.cleanup_checked_at is None
                or archive.cleanup_checked_at < now - timedelta(seconds=interval_seconds)
            )

        now = _now()
        candidates = sorted(
            (
                (entity.archive.cleanup_checked_at or epoch, entity.archive.requested_at, entity.name)
                for entity in self._runs_with_archive(*_STATUSES)
                if entity.archive is not None and due(entity.archive, now)
            ),
        )

        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[bool, bool]:
            moment = _now()
            if entity.archive is None or not due(entity.archive, moment):
                return False, False
            entity.archive.cleanup_checked_at = moment
            return True, True

        for *_, run_id in candidates:
            if self._mutate(run_id, change, include_deleted=True):
                return run_id
        return None

    def for_cleanup(self, run_id: str) -> dict | None:
        """Return the archive's current references, deleted runs included."""
        entity = self._runs._get(run_id, include_deleted=True)
        row = None if entity is None else _row(entity)
        if entity is None or row is None:
            return None
        return {**row, "run_deleted": entity.deleted_at is not None}

    def heartbeat(self, run_id: str, token: str) -> bool:
        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[bool, bool]:
            archive = _owns(entity, token)
            if archive is None:
                return False, False
            archive.claimed_at = _now()
            return True, True

        return bool(self._mutate(run_id, change))

    def validate_members(self, run_id: str, members: list[dict]) -> None:
        """Called before reading sources and again just before publication."""
        actual = [
            BenchmarkArchiveMember.model_validate(row).model_dump(mode="json", exclude={"task_slug", "task_name"})
            for row in _members(run_id)
            if row["archive_status"] == "ready"
        ]
        expected = [{k: v for k, v in member.items() if k not in {"task_slug", "task_name"}} for member in members]
        if actual != expected:
            raise Conflict("members_changed", "member executions or archives changed; rebuild export")

    def finish(
        self,
        job: dict,
        *,
        object_key: str,
        size_bytes: int,
        sha256: str | None = None,
        partial: bool = False,
    ) -> bool:
        """Return whether this claim published; a write error remains uncertain."""
        self.validate_members(job["benchmark_run_id"], job["members"])

        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[bool, bool]:
            archive = _owns(entity, job["claim_token"])
            if archive is None:
                return False, False
            archive.status, archive.object_key, archive.size_bytes = "ready", object_key, size_bytes
            archive.sha256, archive.partial, archive.built_at = sha256, partial, _now()
            archive.error, archive.claimed_at = None, None
            return True, True

        return bool(self._mutate(job["benchmark_run_id"], change))

    def fail(self, job: dict, error: str) -> None:
        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[None, bool]:
            archive = _owns(entity, job["claim_token"])
            if archive is None:
                return None, False
            archive.status, archive.error, archive.claimed_at = "failed", error, None
            return None, True

        self._mutate(job["benchmark_run_id"], change)
