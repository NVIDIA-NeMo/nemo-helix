# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pause, resume, and storage-retention decisions for the jobs service."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, Protocol

from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.constants import (
    IMAGE_DIGEST,
    IMAGE_DIGEST_AT_SAVE,
    IMAGE_DIGEST_COMPARISON_SKIPPED,
    IMAGE_DIGEST_WARNING,
    NON_RESUMABLE_REASON,
    PAUSE_EXIT_CODE,
    PAUSE_REQUESTED_AT,
    POD_PHASES,
    RERUN_EXCLUDED_STATUS_DETAILS,
    RERUN_WARNING,
    RESUMABLE,
    RESUMED_AT,
    STOPPED_AT,
)


class _PauseDeadline(Protocol):
    pause_deadline_seconds: int


class _StepLifecycleSpec(Protocol):
    lifecycle: _PauseDeadline | None


class StatusDetailsStep(Protocol):
    """A step the lifecycle helpers can read without owning it."""

    @property
    def status_details(self) -> Mapping[str, Any] | None: ...


class LifecycleStep(StatusDetailsStep, Protocol):
    @property
    def status(self) -> HelixJobStatus: ...


class PausableStep(StatusDetailsStep, Protocol):
    @property
    def step_spec(self) -> _StepLifecycleSpec | None: ...


class TimedStep(StatusDetailsStep, Protocol):
    @property
    def created_at(self) -> datetime.datetime | None: ...


def _resolve_now(now: datetime.datetime | None) -> datetime.datetime:
    current = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=datetime.timezone.utc)
    return current


def parse_timestamp(value: Any) -> datetime.datetime | None:
    """Parse an ISO-8601 timestamp, assuming UTC when the value has no timezone."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def ensure_aware(value: datetime.datetime | None) -> datetime.datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value


def configured_pause_deadline_seconds(step: PausableStep) -> int:
    if step.step_spec is None or step.step_spec.lifecycle is None:
        return 0
    return step.step_spec.lifecycle.pause_deadline_seconds


def pause_deadline_exceeded(step: PausableStep, now: datetime.datetime | None = None) -> bool:
    """Return whether a pausing step has used up its pause deadline.

    A missing ``pause_requested_at`` is treated as already exceeded so a pausing
    step cannot hold compute indefinitely.
    """
    requested_at = parse_timestamp((step.status_details or {}).get(PAUSE_REQUESTED_AT))
    if requested_at is None:
        return True
    elapsed = (_resolve_now(now) - requested_at).total_seconds()
    return elapsed > configured_pause_deadline_seconds(step)


def status_for_paused_exit(exit_code: int) -> HelixJobStatus:
    """Map a finished pausing workload to paused, completed, or error.

    Exit ``75`` means the pause checkpoint is written. Exit ``0`` means training
    finished. Any other code is an error.
    """
    if exit_code == PAUSE_EXIT_CODE:
        return HelixJobStatus.PAUSED
    if exit_code == 0:
        return HelixJobStatus.COMPLETED
    return HelixJobStatus.ERROR


def describe_paused_exit(
    exit_code: int, *, paused_message: str = "Job is paused"
) -> tuple[HelixJobStatus, str, dict[str, str]]:
    """Return the status, message, and error details for a finished pausing workload.

    Callers delete the workload themselves when the status is paused. Exit 0 leaves
    the workload in place so the pipeline can continue.
    """
    status = status_for_paused_exit(exit_code)
    if status == HelixJobStatus.COMPLETED:
        return status, "Job completed while a pause was in progress", {}
    if status == HelixJobStatus.ERROR:
        message = f"Job exited with code {exit_code} while pausing"
        return status, message, {"message": message}
    return status, paused_message, {}


def with_lifecycle_timestamps(
    step: LifecycleStep,
    status: HelixJobStatus,
    status_details: Mapping[str, Any] | None,
    now: datetime.datetime | None = None,
) -> dict[str, Any] | None:
    """Stamp resume and stop clocks, and snapshot the image digest at save time.

    ``stopped_at`` and ``resumed_at`` are overwritten on each new occurrence.
    A same-status write does not move them.
    """
    details = dict(status_details or {})
    if step.status == status:
        return details or None
    stamp = _resolve_now(now).isoformat()
    if status in (HelixJobStatus.PAUSED, HelixJobStatus.ERROR):
        details[STOPPED_AT] = stamp
        existing = step.status_details or {}
        digest = details.get(IMAGE_DIGEST)
        if not isinstance(digest, str) or not digest:
            digest = existing.get(IMAGE_DIGEST)
        if isinstance(digest, str) and digest:
            details[IMAGE_DIGEST_AT_SAVE] = digest
    elif status == HelixJobStatus.RESUMING:
        details[RESUMED_AT] = stamp
    return details or None


def image_digest_updates(existing: dict[str, Any] | None, observed: str | None) -> dict[str, Any]:
    """Record the live image digest, and compare it once a saved digest exists.

    The saved digest is ``image_digest_at_save``, copied when the step enters
    ``paused`` or ``error``. Until that exists, the live digest is overwritten
    as pods start. The comparison runs on the first reconcile that sees the
    new pod.
    """
    current = existing or {}
    saved = current.get(IMAGE_DIGEST_AT_SAVE)
    has_saved = isinstance(saved, str) and bool(saved)
    if not has_saved:
        if observed and current.get(IMAGE_DIGEST) != observed:
            return {IMAGE_DIGEST: observed}
        return {}

    updates: dict[str, Any] = {}
    if observed and current.get(IMAGE_DIGEST) != observed:
        updates[IMAGE_DIGEST] = observed
    if IMAGE_DIGEST_WARNING in current or IMAGE_DIGEST_COMPARISON_SKIPPED in current:
        return updates
    if not observed:
        updates[IMAGE_DIGEST_COMPARISON_SKIPPED] = "Image digest comparison skipped because the new digest is missing."
        return updates
    if observed != saved:
        updates[IMAGE_DIGEST_WARNING] = (
            "The training image changed since this state was saved "
            f"(was {saved}, now {observed}). Training may behave differently or fail to resume."
        )
    return updates


def storage_window_expired(
    stopped_at: datetime.datetime | None, ttl_seconds: int, now: datetime.datetime | None = None
) -> bool:
    """Return whether ``stopped_at`` is older than ``ttl_seconds``. A non-positive TTL is already expired."""
    if ttl_seconds <= 0:
        return True
    anchor = ensure_aware(stopped_at)
    if anchor is None:
        return True
    return anchor + datetime.timedelta(seconds=ttl_seconds) < _resolve_now(now)


def active_ttl_anchor(step: TimedStep) -> datetime.datetime | None:
    """Anchor the active TTL on the later of creation and the latest resume.

    Time spent paused does not count. ``resumed_at`` is written when the step
    enters ``resuming``.
    """
    created_at = ensure_aware(step.created_at)
    resumed_at = parse_timestamp((step.status_details or {}).get(RESUMED_AT))
    if created_at is None:
        return resumed_at
    if resumed_at is None:
        return created_at
    return max(created_at, resumed_at)


def effective_pause_ttl_seconds(requested: int | None, *, default_seconds: int, maximum_seconds: int) -> int:
    """Return the pause window a job actually gets.

    An unset request uses the platform default. Both are capped by the platform
    maximum, so lowering that maximum shortens windows that were saved earlier.
    """
    chosen = default_seconds if requested is None else requested
    return min(chosen, maximum_seconds)


def rerun_status_details(previous: dict[str, Any] | None) -> dict[str, Any]:
    """Copy progress scalars and the metrics series onto a new attempt.

    ``resumable``, pod details, and ``storage_reclaimed_at`` stay behind. A
    non-resumable failure is reported as ``rerun_warning`` without copying the flag.
    ``image_digest_at_save`` is carried so the new attempt can compare images.
    """
    source = previous or {}
    copied: dict[str, Any] = {}
    metrics = source.get("metrics")
    if isinstance(metrics, dict):
        copied["metrics"] = metrics
    saved = source.get(IMAGE_DIGEST_AT_SAVE)
    if isinstance(saved, str) and saved:
        copied[IMAGE_DIGEST_AT_SAVE] = saved
    for key, value in source.items():
        if key in RERUN_EXCLUDED_STATUS_DETAILS or key in ("metrics", IMAGE_DIGEST_AT_SAVE):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            copied[key] = value
        elif key == "phase" and isinstance(value, str) and value not in POD_PHASES:
            copied[key] = value
    if source.get(RESUMABLE) is False:
        reason = source.get(NON_RESUMABLE_REASON) or "the previous attempt failed in a way a rerun would repeat"
        copied[RERUN_WARNING] = str(reason)
    return copied
