# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ATIF ingest process-pool offload.

The route-level tests call ``ingest_atif`` directly with fakes, so they exercise
the real offload wiring (event-loop await, pool submission, insert commit point)
without ClickHouse. Worker fakes live at module scope because the process pool
pickles them by reference.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from typing import Any, NoReturn

import pytest
from fastapi import HTTPException
from fastapi import Response as FastAPIResponse
from nhx.intake.config import IntakeConfig
from nhx.intake.dependencies import get_atif_ingest_executor
from nhx.intake.repository.clickhouse.evaluator_results import evaluator_result_rows_for_insert
from nhx.intake.repository.clickhouse.span import span_rows_for_insert
from nhx.intake.service import ATIF_INGEST_POOL_START_METHOD, ATIF_INGEST_POOL_WORKERS
from nhx.intake.spans.ingest import atif
from nhx.intake.spans.ingest.atif import AtifIngestRequest, ingest_atif
from nhx.intake.spans.ingest.atif_mapping import (
    AtifTrajectoryDepthError,
    trajectory_to_evaluator_results,
    trajectory_to_spans,
)
from nhx.intake.spans.ingest.atif_offload import PreparedAtifBatch, prepare_atif_batch
from nhx.intake.spans.storage import utc_now

# The route-level tests mirror the production wiring by building the same pool
# the Intake service owns: two spawn-method workers.


# ---------------------------------------------------------------------------
# Module-level worker functions. They are pickled by reference, so they must
# stay importable from this module without any test fixtures.
# ---------------------------------------------------------------------------


def _noop() -> None:
    """Instant worker task used to pre-spawn (and finish importing) pool workers."""


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _slow_prepare(
    workspace: str,
    trajectory: Any,
    ingested_at: Any,
    max_subagent_depth: int,
) -> PreparedAtifBatch:
    """Stand-in for the real mapper with one wall-clock second of worker time."""
    time.sleep(1.0)
    return PreparedAtifBatch(span_rows=[["sentinel"]], evaluator_result_rows=[])


def _depth_error_prepare(
    workspace: str,
    trajectory: Any,
    ingested_at: Any,
    max_subagent_depth: int,
) -> NoReturn:
    raise AtifTrajectoryDepthError("ATIF trajectory depth 3 exceeds configured maximum 2")


def _worker_suicide(
    workspace: str,
    trajectory: Any,
    ingested_at: Any,
    max_subagent_depth: int,
) -> NoReturn:
    """Simulate a worker dying mid-mapping (OOM kill, crash)."""
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("worker process survived SIGKILL")


async def _skip_evaluation_context_validation(**_kwargs: Any) -> None:
    """No-op stand-in for the entity-backed evaluation-context validation."""


# ---------------------------------------------------------------------------
# Fakes for the route's loop-side dependencies.
# ---------------------------------------------------------------------------


class _RecordingSpansService:
    """Records prepared inserts; optionally raises to simulate an insert error."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.batches: list[tuple[list[list[Any]], list[list[Any]]]] = []

    async def ingest_prepared_batch(
        self, *, span_rows: list[list[Any]], evaluator_result_rows: list[list[Any]]
    ) -> None:
        if self._error is not None:
            raise self._error
        self.batches.append((span_rows, evaluator_result_rows))


class _RecordingDenormalizer:
    def __init__(self) -> None:
        self.dirty: list[tuple[str, str]] = []

    def mark_dirty(self, *, workspace: str, evaluation_name: str) -> None:
        self.dirty.append((workspace, evaluation_name))


@pytest.fixture(scope="module")
def atif_pool():
    executor = ProcessPoolExecutor(
        max_workers=ATIF_INGEST_POOL_WORKERS, mp_context=get_context(ATIF_INGEST_POOL_START_METHOD)
    )
    yield executor
    executor.shutdown(wait=True)


async def _prewarm_workers(executor: ProcessPoolExecutor) -> None:
    """Force both pool workers to spawn (and finish their imports) before timing.

    The first task sleeps so the second submit cannot land on the same idle
    worker; the pool then spawns its full complement of two workers.
    """
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(executor, _sleep, 0.25),
        loop.run_in_executor(executor, _noop),
    )


def _ingest_body(evaluation_context: dict[str, str] | None = None) -> AtifIngestRequest:
    payload: dict[str, Any] = {
        "schema_version": "ATIF-v1.6",
        "session_id": "atif-offload-session",
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "hello"},
            {"step_id": 2, "source": "agent", "message": "hi back"},
        ],
    }
    if evaluation_context is not None:
        payload["evaluation_context"] = evaluation_context
    return AtifIngestRequest.model_validate(payload)


async def _ingest(
    executor: ProcessPoolExecutor,
    body: AtifIngestRequest,
    *,
    service: _RecordingSpansService,
    denormalizer: _RecordingDenormalizer | None = None,
) -> FastAPIResponse:
    return await ingest_atif(
        workspace="default",
        body=body,
        config=IntakeConfig(),
        service=service,  # ty: ignore[invalid-argument-type]
        entity_client=object(),  # ty: ignore[invalid-argument-type]
        denormalizer=denormalizer,  # ty: ignore[invalid-argument-type]
        executor=executor,
    )


async def _heartbeat() -> float:
    """Yield briefly and report when the coroutine actually got to run."""
    await asyncio.sleep(0.1)
    return time.monotonic()


def test_atif_ingest_executor_dependency_returns_503_after_shutdown() -> None:
    """Requests that arrive while the pool is gone get a 503, not a silent hang."""

    class _StoppedIntakeService:
        atif_ingest_executor = None

    with pytest.raises(HTTPException) as exc_info:
        get_atif_ingest_executor(_StoppedIntakeService())  # ty: ignore[invalid-argument-type]

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_prepare_atif_batch_pool_output_matches_inline_mapping(atif_pool) -> None:
    """The worker produces exactly the rows the on-loop path built before."""
    body = _ingest_body()
    trajectory = body.to_trajectory()
    ingested_at = utc_now()
    max_subagent_depth = IntakeConfig().atif_max_subagent_depth

    loop = asyncio.get_running_loop()
    prepared = await loop.run_in_executor(
        atif_pool, prepare_atif_batch, "default", trajectory, ingested_at, max_subagent_depth
    )

    spans = trajectory_to_spans(
        workspace="default", trajectory=trajectory, ingested_at=ingested_at, max_subagent_depth=max_subagent_depth
    )
    evaluator_results = trajectory_to_evaluator_results(
        workspace="default",
        trajectory=trajectory,
        spans=spans,
        ingested_at=ingested_at,
        max_subagent_depth=max_subagent_depth,
    )

    assert isinstance(prepared, PreparedAtifBatch)
    assert prepared.span_rows == span_rows_for_insert(spans)
    assert prepared.evaluator_result_rows == evaluator_result_rows_for_insert(evaluator_results)
    assert prepared.span_rows, "the representative trajectory must map to at least one span row"


@pytest.mark.asyncio
async def test_overlapping_ingests_keep_event_loop_responsive_and_parallelize(
    atif_pool: ProcessPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four one-second mappers on two workers finish nearer two seconds than four.

    While the pool works, a heartbeat coroutine on the event loop still runs on
    time, proving the mapping no longer blocks the request loop.
    """
    await _prewarm_workers(atif_pool)
    monkeypatch.setattr(atif, "prepare_atif_batch", _slow_prepare)
    service = _RecordingSpansService()

    start = time.monotonic()
    tasks = [asyncio.create_task(_ingest(atif_pool, _ingest_body(), service=service)) for _ in range(4)]
    heartbeat = asyncio.create_task(_heartbeat())
    responses = await asyncio.gather(*tasks)
    heartbeat_ran_at = await heartbeat
    elapsed = time.monotonic() - start

    assert all(response.status_code == 201 for response in responses)
    assert len(service.batches) == 4
    # Two rounds of one-second work, not four serialized ones.
    assert 1.5 <= elapsed < 3.0, f"elapsed={elapsed:.2f}s"
    # The event loop stayed responsive during the pool wait.
    assert heartbeat_ran_at - start < 1.0


@pytest.mark.asyncio
async def test_worker_depth_error_fails_request_without_insert(
    atif_pool: ProcessPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An AtifTrajectoryDepthError in the worker still maps to a 422 and inserts nothing."""
    monkeypatch.setattr(atif, "prepare_atif_batch", _depth_error_prepare)
    service = _RecordingSpansService()

    with pytest.raises(HTTPException) as exc_info:
        await _ingest(atif_pool, _ingest_body(), service=service)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "ATIF trajectory depth 3 exceeds configured maximum 2"
    assert service.batches == []


@pytest.mark.asyncio
async def test_dead_worker_fails_request_without_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that dies mid-mapping fails the request; nothing is inserted."""
    executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context(ATIF_INGEST_POOL_START_METHOD))
    try:
        monkeypatch.setattr(atif, "prepare_atif_batch", _worker_suicide)
        service = _RecordingSpansService()

        with pytest.raises(BrokenProcessPool):
            await _ingest(executor, _ingest_body(), service=service)

        assert service.batches == []
    finally:
        executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_insert_error_does_not_mark_denormalizer_dirty(
    atif_pool: ProcessPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An insert failure propagates (no 201) and the denormalizer stays clean."""
    monkeypatch.setattr(atif, "validate_evaluation_context", _skip_evaluation_context_validation)
    service = _RecordingSpansService(error=RuntimeError("clickhouse insert failed"))
    denormalizer = _RecordingDenormalizer()

    with pytest.raises(RuntimeError, match="clickhouse insert failed"):
        await _ingest(
            atif_pool,
            _ingest_body({"evaluation_name": "eval-x"}),
            service=service,
            denormalizer=denormalizer,
        )

    assert denormalizer.dirty == []


@pytest.mark.asyncio
async def test_successful_ingest_marks_denormalizer_dirty_after_insert(
    atif_pool: ProcessPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mark_dirty only runs after the awaited insert has committed."""
    monkeypatch.setattr(atif, "validate_evaluation_context", _skip_evaluation_context_validation)
    service = _RecordingSpansService()
    denormalizer = _RecordingDenormalizer()

    response = await _ingest(
        atif_pool,
        _ingest_body({"evaluation_name": "eval-x"}),
        service=service,
        denormalizer=denormalizer,
    )

    assert response.status_code == 201
    assert denormalizer.dirty == [("default", "eval-x")]
    assert len(service.batches) == 1
    span_rows, evaluator_result_rows = service.batches[0]
    assert span_rows, "the offload must return built span rows"
    assert evaluator_result_rows == []
