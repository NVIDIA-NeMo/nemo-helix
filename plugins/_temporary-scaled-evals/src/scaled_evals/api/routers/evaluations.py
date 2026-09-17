# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from threading import Lock
from time import monotonic
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from scaled_evals.api import s3
from scaled_evals.api.agent_bundle_registry import accessible_bundle_for_run
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import Database, get_db, get_stream_database_factory
from scaled_evals.api.evaluation_logs import collect_log_lines
from scaled_evals.api.repositories.runtime_resource_repository import switchyard_lease_from_row
from scaled_evals.api.runnability import BlockedPreflight, preflight_evaluation
from scaled_evals.api.schemas.common import (
    DeleteResponse,
    ListEnvelope,
    encode_cursor,
    page_from_rows,
)
from scaled_evals.api.schemas.evaluations import (
    BuildArchiveRequest,
    CreateEvaluationRequest,
    Evaluation,
    EvaluationArchiveResponse,
    EvaluationArtifact,
    EvaluationEvent,
    EvaluationExecutionTelemetry,
    EvaluationLinks,
    EvaluationLogResponse,
    EvaluationResponse,
    EvaluationStatus,
    EvaluationTelemetryArtifacts,
    EvaluationTelemetryCleanup,
    EvaluationTelemetryCost,
    EvaluationTelemetryFailure,
    EvaluationTelemetryIntake,
    EvaluationTelemetryInteractions,
    EvaluationTelemetryPhaseTiming,
    EvaluationTelemetryRawArtifact,
    EvaluationTelemetryResponse,
    EvaluationTelemetryUsage,
    ReproduceEvaluationResponse,
)
from scaled_evals.api.schemas.runnability import RunnabilityReport
from scaled_evals.api.settings import settings
from scaled_evals.api.utils import list_envelope, make_id
from scaled_evals.dispatch.registry import get_backend as _resolve_backend
from scaled_evals.dispatch.registry import get_backend_capabilities
from scaled_evals.dispatch.runtime_backend import LaunchHandle
from scaled_evals.harbor_viewer import (
    harbor_viewer_archive_available_from_result,
    harbor_viewer_upload_url_from_result,
    harbor_viewer_url_from_result,
)
from scaled_evals.models.provenance import MANIFEST_FILE_NAME
from scaled_evals.models.sbom import SBOM_FILE_NAME

router = APIRouter(prefix="/evaluations", tags=["evaluations"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]
StreamDatabaseFactory = Annotated[Callable[[], AbstractContextManager[Database]], Depends(get_stream_database_factory)]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _evaluation_reads(db: Database) -> Any:
    """Return the source for evaluation list/get reads.

    Entity Store reads are served by the plugin's projection. The import is
    deferred so this package keeps working without the plugin installed, and
    Postgres stays both the default and the fallback.
    """
    if not settings.entity_store_reads_enabled:
        return db.evaluations
    from nemo_scaled_evals_plugin.projection import evaluation_reader  # noqa: PLC0415

    return evaluation_reader()


_EVENT_STREAM_BATCH_SIZE = 100
_sse_connection_lock = Lock()
_sse_active_connections = 0


def _acquire_sse_connection() -> bool:
    global _sse_active_connections
    with _sse_connection_lock:
        if _sse_active_connections >= settings.api_sse_max_connections:
            return False
        _sse_active_connections += 1
        return True


def _release_sse_connection() -> None:
    global _sse_active_connections
    with _sse_connection_lock:
        _sse_active_connections = max(0, _sse_active_connections - 1)


def _ensure_stream_evaluation_exists(
    database_factory: Callable[[], AbstractContextManager[Database]], evaluation_id: str
) -> None:
    with database_factory() as db:
        _ensure_evaluation_exists(db, evaluation_id)


def _load_stream_log_snapshot(
    database_factory: Callable[[], AbstractContextManager[Database]], evaluation_id: str
) -> tuple[Mapping[str, Any], list[str]]:
    with database_factory() as db:
        row = _load_observability_row(db, evaluation_id)
        return row, collect_log_lines(row, tail_lines=None)


def _load_stream_event_batch(
    database_factory: Callable[[], AbstractContextManager[Database]],
    evaluation_id: str,
    last_created_at: datetime | None,
    last_id: int | None,
) -> list[Mapping[str, Any]]:
    with database_factory() as db:
        return db.evaluations.load_event_batch(
            evaluation_id,
            after_created_at=last_created_at,
            after_id=last_id,
            limit=_EVENT_STREAM_BATCH_SIZE,
        )


def _load_stream_status(
    database_factory: Callable[[], AbstractContextManager[Database]], evaluation_id: str
) -> str | None:
    with database_factory() as db:
        return db.evaluations.load_stream_status(evaluation_id)


def _sse_response(generate: Any) -> StreamingResponse:
    return StreamingResponse(
        generate,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


def _eval_links(ev_id: str) -> EvaluationLinks:
    base = f"/evaluations/{ev_id}"
    return EvaluationLinks(
        self=base,
        logs=f"{base}/logs",
        logs_stream=f"{base}/logs/stream",
        events_stream=f"{base}/events/stream",
        telemetry=f"{base}/telemetry",
        artifacts=f"{base}/artifacts",
        provenance=f"{base}/artifacts/{MANIFEST_FILE_NAME}",
        sbom=f"{base}/artifacts/{SBOM_FILE_NAME}",
        reproduce=f"{base}/reproduce",
        retry=f"{base}/retry",
        archive=f"{base}/archive",
        cancel=f"{base}/cancel",
    )


def _rerun_name(name: str) -> str:
    candidate = f"rerun of {name}"
    return candidate[:200]


def _reproduce_request(row: Mapping[str, Any]) -> CreateEvaluationRequest:
    runner_metadata = dict(row.get("runner_metadata") or {})
    raw_agent_bundle = runner_metadata.get("agent_bundle")
    agent_bundle_id = (
        str(raw_agent_bundle["bundle_id"])
        if isinstance(raw_agent_bundle, Mapping) and raw_agent_bundle.get("bundle_id")
        else None
    )
    return CreateEvaluationRequest(
        name=_rerun_name(str(row["name"])),
        task_id=str(row["task_id"]),
        task_revision=int(row["task_revision"]),
        framework=row.get("framework") or "harbor",
        framework_version=row.get("framework_version"),
        framework_profile_id=row.get("framework_profile_id"),
        switchyard_profile_id=row.get("switchyard_profile_id"),
        intake_profile_id=row.get("intake_profile_id"),
        credentials=dict(row.get("credentials") or {}),
        agent_bundle_id=agent_bundle_id,
        extra_skill_object_keys=list(row.get("extra_skill_object_keys") or []),
        instruction_prefix=row.get("instruction_prefix"),
        instruction_postfix=row.get("instruction_postfix"),
        initial_user_turns=list(row.get("initial_user_turns") or []),
        runtime=str(row["runtime"]),
        network_policy=row.get("network_policy") or "unrestricted",
        network_policy_config=dict(row.get("network_policy_config") or {}),
        n_attempts=int(row.get("n_attempts") or 1),
        parallelism=int(row["parallelism"]),
        visibility=row.get("visibility") or "private",
    )


def _create_command(body: CreateEvaluationRequest) -> list[str]:
    command = [
        "scaled-evals",
        "evaluation",
        "create",
        "--name",
        body.name,
        "--task-id",
        body.task_id,
        "--task-revision",
        str(body.task_revision),
        "--framework",
        body.framework,
        "--runtime",
        body.runtime,
        "--network-policy",
        body.network_policy,
        "--parallelism",
        str(body.parallelism),
        "--visibility",
        body.visibility,
    ]
    if body.framework_version:
        command.extend(["--framework-version", body.framework_version])
    if body.network_policy_config:
        command.extend(
            [
                "--network-policy-config",
                json.dumps(body.network_policy_config, separators=(",", ":")),
            ]
        )
    for option, value in (
        ("--framework-profile-id", body.framework_profile_id),
        ("--switchyard-profile-id", body.switchyard_profile_id),
        ("--intake-profile-id", body.intake_profile_id),
    ):
        if value:
            command.extend([option, value])
    for role, credential_id in sorted(body.credentials.items()):
        command.extend(["--credential", f"{role}={credential_id}"])
    if body.agent_bundle_id is not None:
        command.extend(["--agent-bundle", body.agent_bundle_id])
    for object_key in body.extra_skill_object_keys:
        command.extend(["--extra-skill-object-key", object_key])
    if body.instruction_prefix is not None:
        command.extend(["--instruction-prefix", body.instruction_prefix])
    if body.instruction_postfix is not None:
        command.extend(["--instruction-postfix", body.instruction_postfix])
    for turn in body.initial_user_turns:
        command.extend(["--initial-user-turn", turn])
    command.extend(["--n-attempts", str(body.n_attempts)])
    return command


def _reproduce_notes(row: Mapping[str, Any], body: CreateEvaluationRequest) -> list[str]:
    notes = [
        "Secret material is not exported; credential values remain credential ids.",
        "The rerun will fail validation if referenced credentials, profiles, task revision, "
        "or runtime backend are unavailable.",
    ]
    if row.get("image_ref"):
        notes.append(f"Captured task image: {row['image_ref']}")
    if row.get("image_digest"):
        notes.append(f"Captured task image digest: {row['image_digest']}")
    if row.get("framework_version"):
        notes.append(f"Resolved framework version: {row['framework_version']}")
    if row.get("runner_image_ref"):
        notes.append(f"Resolved runner artifact: {row['runner_image_ref']}")
    if row.get("benchmark_run_id"):
        notes.append(
            f"Source evaluation belongs to benchmark run {row['benchmark_run_id']}; "
            "this request reruns only this member task."
        )
    if body.credentials:
        roles = ", ".join(sorted(body.credentials))
        notes.append(f"Credential references required for roles: {roles}.")
    return notes


def _response(row: dict[str, Any]) -> EvaluationResponse:
    body = {k: v for k, v in row.items() if k != "backend_handle"}
    links = _eval_links(row["id"])
    result = row.get("result")
    links.harbor_viewer = harbor_viewer_url_from_result(result)
    if harbor_viewer_archive_available_from_result(result):
        links.harbor_viewer_archive = f"/evaluations/{row['id']}/harbor-viewer/archive"
    links.harbor_viewer_upload = harbor_viewer_upload_url_from_result(result)
    return EvaluationResponse(**body, links=links)


def _load_observability_row(db: Database, evaluation_id: str) -> dict[str, Any]:
    row = db.evaluations.load_observability_row(evaluation_id)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    return row


def _ensure_evaluation_exists(db: Database, evaluation_id: str) -> None:
    if not db.evaluations.exists(evaluation_id):
        raise _http_error(404, "not_found", "evaluation not found")


def _load_archive_row(db: Database, evaluation_id: str) -> dict[str, Any]:
    row = db.evaluations.load_archive_row(evaluation_id)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    return row


def _archive_response(row: Mapping[str, Any]) -> EvaluationArchiveResponse:
    status = row.get("archive_status") or "missing"
    object_key = row.get("archive_object_key")
    download = None
    if status == "ready" and object_key:
        if s3.can_presign_get():
            download = {"method": "GET", "url": s3.presign_get(object_key)}
        else:
            download = {
                "method": "GET",
                "url": f"/evaluations/{row['id']}/archive/download",
            }
    return EvaluationArchiveResponse(
        evaluation_id=row["id"],
        status=status,
        size_bytes=row.get("archive_size_bytes"),
        built_at=row.get("archive_built_at"),
        error=row.get("archive_error"),
        download=download,
    )


def _status_event(row: Mapping[str, Any]) -> EvaluationEvent:
    at = row.get("finished_at") or row.get("updated_at") or row.get("created_at")
    return EvaluationEvent(
        evaluation_id=row["id"],
        type="status",
        status=row["status"],
        detail=row.get("status_detail"),
        at=at.isoformat() if isinstance(at, datetime) else at,
    )


def _event_payload(row: Mapping[str, Any], evaluation_id: str) -> EvaluationEvent:
    at = row.get("created_at")
    return EvaluationEvent(
        evaluation_id=evaluation_id,
        type=row.get("type") or "status",
        status=row["status"],
        detail=row.get("detail"),
        at=at.isoformat() if isinstance(at, datetime) else at,
    )


def _logs_payload(row: Mapping[str, Any], tail_lines: int) -> EvaluationLogResponse:
    return EvaluationLogResponse(
        evaluation_id=row["id"],
        lines=collect_log_lines(row, tail_lines=tail_lines),
        status=row["status"],
        complete=row["status"] in _TERMINAL_STATUSES,
    )


def _events_payload(events: list[dict[str, Any]], evaluation_id: str) -> ListEnvelope[EvaluationEvent]:
    return ListEnvelope(
        data=[_event_payload(event, evaluation_id) for event in events],
        next_cursor=None,
    )


def _events_page_payload(events: list[dict[str, Any]], evaluation_id: str, limit: int) -> ListEnvelope[EvaluationEvent]:
    page = events[:limit]
    next_cursor = None
    if len(events) > limit and page:
        last = page[-1]
        next_cursor = encode_cursor(last["created_at"], str(last["id"]))
    return ListEnvelope(
        data=[_event_payload(event, evaluation_id) for event in page],
        next_cursor=next_cursor,
    )


def _launch_handle_from_row(row: Mapping[str, Any]) -> LaunchHandle | None:
    raw_handle = row.get("backend_handle")
    if not raw_handle:
        return None
    data = json.loads(raw_handle) if isinstance(raw_handle, str) else raw_handle
    if not isinstance(data, Mapping):
        raise ValueError("backend_handle must be an object")
    external_id = data.get("external_id")
    if not external_id:
        raise ValueError("backend_handle missing external_id")
    raw = data.get("raw")
    return LaunchHandle(
        backend=str(data.get("backend") or row["runtime"]),
        external_id=str(external_id),
        raw=raw if isinstance(raw, dict) else {},
    )


def _record_cancel_teardown_failure(
    db: Database,
    evaluation_id: str,
    detail: str,
) -> dict[str, Any]:
    row = db.evaluations.record_cancel_teardown_failure(evaluation_id, detail)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    return row


def _record_cancel_teardown_succeeded(db: Database, evaluation_id: str) -> dict[str, Any]:
    row = db.evaluations.record_cancel_teardown_succeeded(evaluation_id)
    if row is None:
        current = db.evaluations.get(evaluation_id)
        if current is None:
            raise _http_error(404, "not_found", "evaluation not found")
        return current
    return row


def teardown_cancelled_evaluation(db: Database, row: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    handle: LaunchHandle | None = None
    try:
        handle = _launch_handle_from_row(row)
        # A durable evaluation Job owns process-local runtime cleanup. The API
        # may be a replacement pod after a rollout and must never signal a PID
        # from another pod; the Job observes the durable cancelled state and
        # invokes backend teardown in its own process namespace.
        if handle is not None and not row.get("dispatch_job_name"):
            capabilities = get_backend_capabilities(row["runtime"])
            if capabilities.supports_teardown:
                backend = _resolve_backend(row["runtime"])
                backend.teardown(handle)
    except Exception as exc:  # noqa: BLE001 — cancellation must remain durable
        failures.append(f"evaluation-runtime cleanup failed: {exc}")
    try:
        has_switchyard = bool(row.get("switchyard_profile_id")) or bool(
            handle is not None and handle.raw.get("switchyard")
        )
        if has_switchyard:
            execution_number = int(row.get("current_execution") or 1)
            switchyard_row = db.runtime_resources.get_switchyard(
                row["id"],
                execution_number,
            )
            lease = switchyard_lease_from_row(switchyard_row)
            if lease is not None:
                drain_seconds = (
                    lease.drain_seconds if lease.drain_seconds is not None else settings.switchyard_drain_seconds
                )
                marked = db.runtime_resources.mark_switchyard_draining(
                    row["id"],
                    execution_number,
                    drain_seconds=drain_seconds,
                )
                if marked is not None:
                    drain_until = marked.get("drain_until")
                    drain_text = drain_until.isoformat() if hasattr(drain_until, "isoformat") else drain_until
                    db.evaluations.append_event(
                        row["id"],
                        status="cancelled",
                        type="switchyard",
                        detail=f"switchyard draining until {drain_text}: {lease.name}",
                    )
    except Exception as exc:  # noqa: BLE001 — cancellation must remain durable
        failures.append(f"switchyard drain mark failed: {exc}")
    if failures:
        return _record_cancel_teardown_failure(
            db,
            row["id"],
            "cancelled; " + "; ".join(failures),
        )
    if row.get("dispatch_job_name"):
        return row
    return _record_cancel_teardown_succeeded(db, row["id"])


@router.post("/preflight", response_model=RunnabilityReport)
def preflight_evaluation_request(
    body: CreateEvaluationRequest,
    db: Db,
    current: Principal,
) -> RunnabilityReport:
    """Check whether an evaluation request is runnable without creating it."""
    result = preflight_evaluation(
        db,
        body,
        current,
        object_exists=s3.object_exists,
        resolve_bundle=accessible_bundle_for_run,
    )
    return result.report


@router.post("", status_code=202, response_model=EvaluationResponse)
def create_evaluation(
    body: CreateEvaluationRequest,
    db: Db,
    current: Principal,
) -> EvaluationResponse:
    """Start an evaluation run over a single task.

    Validates that the target task revision exists and is `ready` (409
    `task_not_ready` otherwise; 404 if unknown), that referenced profile/credential
    ids are well-formed and live (422 `invalid_reference`), inserts the row at
    `queued`, and returns 202 with the new row + `links`. The out-of-process
    dispatch worker claims queued rows from Postgres. To aggregate one evaluation
    per benchmark member task, use POST /benchmark-runs.
    """
    preflight = preflight_evaluation(
        db,
        body,
        current,
        object_exists=s3.object_exists,
        resolve_bundle=accessible_bundle_for_run,
    )
    if isinstance(preflight, BlockedPreflight):
        raise _http_error(
            preflight.blocker.status_code,
            preflight.blocker.code,
            preflight.blocker.message,
        )
    runner = preflight.runner
    runner_metadata = preflight.runner_metadata

    ev_id = make_id("ev")
    row = db.evaluations.create(
        ev_id,
        name=body.name,
        framework=body.framework,
        requested_framework_version=runner.requested_version,
        framework_version=runner.version,
        runner_image_ref=runner.image_ref,
        runner_image_digest=runner.image_digest,
        framework_adapter_version=runner.adapter_version,
        sandbox_k8s_version=runner.sandbox_k8s_version,
        runner_metadata=runner_metadata,
        task_id=body.task_id,
        task_revision=body.task_revision,
        framework_profile_id=body.framework_profile_id,
        harbor_profile_id=body.harbor_profile_id,
        switchyard_profile_id=body.switchyard_profile_id,
        intake_profile_id=body.intake_profile_id,
        credentials=body.credentials,
        extra_skill_object_keys=body.extra_skill_object_keys,
        instruction_prefix=body.instruction_prefix,
        instruction_postfix=body.instruction_postfix,
        initial_user_turns=body.initial_user_turns,
        runtime=body.runtime,
        network_policy=body.network_policy,
        network_policy_config=body.network_policy_config,
        n_attempts=body.n_attempts,
        parallelism=body.parallelism,
        visibility=body.visibility,
        owner_id=current.owner_id,
    )

    # Dispatch loads this row on its own connection, so commit before returning —
    # get_conn otherwise defers COMMIT past the response and the worker sees nothing.
    db.commit()
    return _response(row)


@router.get("", response_model=ListEnvelope[Evaluation])
def list_evaluations(
    db: Db,
    current: Principal,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    status: EvaluationStatus | None = None,
    task_id: str | None = None,
    benchmark_run_id: str | None = None,
    team_id: str | None = None,
    mine: bool = False,
    shared: bool = False,
    q: str | None = Query(default=None, min_length=1, max_length=200),
) -> ListEnvelope[Evaluation]:
    """List live evaluations, newest first.

    Filters: `status`, `task_id`, and `shared` (visibility other than
    private) are applied. By default benchmark-run member evaluations are
    hidden (a benchmark run is listed under /benchmark-runs); pass
    `benchmark_run_id` to list a given run's member evaluations instead. `mine`
    filters by the authenticated caller's owner id. `team_id` remains accepted
    but inert until team ownership is implemented. Results are
    ordered by `created_at` plus `id` tiebreaker and return `next_cursor` when
    more rows exist.
    """
    _ = team_id  # Teams are deliberately deferred.
    rows = _evaluation_reads(db).list(
        limit=limit,
        cursor=cursor,
        order=order,
        status=status,
        task_id=task_id,
        shared=shared,
        benchmark_run_id=benchmark_run_id,
        owner_id=current.owner_id if mine else None,
        q=q,
    )
    return page_from_rows(rows, limit, Evaluation)


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
def get_evaluation(evaluation_id: str, db: Db) -> EvaluationResponse:
    """Fetch one evaluation by id, including its result envelope.

    Returns the row plus the typed scalar reward summary
    (`reward`/`n_trials`/`n_errored`/`finished_at`) and the full framework-typed
    `result` JSON once the run has reached a terminal state (all null/None until
    then). 404 if not found or soft-deleted.
    """
    row = _evaluation_reads(db).get(evaluation_id)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    return _response(row)


def _phase_timings(row: Mapping[str, Any]) -> list[EvaluationTelemetryPhaseTiming]:
    provisioning = row.get("provisioning_started_at")
    running = row.get("running_started_at")
    terminal = row.get("terminal_at")
    timings: list[EvaluationTelemetryPhaseTiming] = []

    def add(phase: str, started_at: datetime | None, ended_at: datetime | None) -> None:
        if started_at is None:
            return
        duration = max(0.0, (ended_at - started_at).total_seconds()) if ended_at else None
        timings.append(
            EvaluationTelemetryPhaseTiming(
                phase=phase,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration,
            )
        )

    add("provisioning", provisioning, running or terminal)
    add("running", running, terminal)
    add("total", provisioning or running, terminal)
    return timings


def _execution_telemetry(
    evaluation_id: str,
    rows: list[Mapping[str, Any]],
) -> list[EvaluationExecutionTelemetry]:
    executions = []
    for row in rows:
        raw_artifacts = []
        for artifact in row.get("raw_artifact_refs") or []:
            if not isinstance(artifact, dict) or not artifact.get("path"):
                continue
            path = str(artifact["path"])
            raw_artifacts.append(
                EvaluationTelemetryRawArtifact(
                    relation=artifact.get("relation") or "result",
                    path=path,
                    download=f"/evaluations/{evaluation_id}/artifacts/{quote(path, safe='/')}",
                )
            )
        executions.append(
            EvaluationExecutionTelemetry(
                execution_number=int(row["execution_number"]),
                terminal_status=row.get("terminal_status"),
                failure_phase=row.get("failure_phase"),
                phase_timings=_phase_timings(row),
                usage=EvaluationTelemetryUsage(
                    input_tokens=row.get("input_tokens"),
                    output_tokens=row.get("output_tokens"),
                    cached_tokens=row.get("cached_tokens"),
                    cache_creation_tokens=row.get("cache_creation_tokens"),
                    source=row.get("usage_source") or "unknown",
                ),
                interactions=EvaluationTelemetryInteractions(
                    turns=row.get("turn_count"),
                    tool_calls=row.get("tool_call_count"),
                ),
                cost=EvaluationTelemetryCost(
                    value_usd=row.get("cost_usd"),
                    source=row.get("cost_source") or "unknown",
                ),
                raw_artifacts=raw_artifacts,
            )
        )
    return executions


@router.get("/{evaluation_id}/telemetry", response_model=EvaluationTelemetryResponse)
def get_evaluation_telemetry(evaluation_id: str, db: Db) -> EvaluationTelemetryResponse:
    """Return portable attempt-aware telemetry without interpreting raw transcripts."""
    row = db.evaluations.get(evaluation_id)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    links = _eval_links(evaluation_id)
    intake_profile_id = row.get("intake_profile_id")
    execution_rows = db.execution_telemetry.list_for_evaluation(evaluation_id)
    current_execution = int(row.get("current_execution") or 1)
    current = next(
        (item for item in execution_rows if int(item["execution_number"]) == current_execution),
        {},
    )
    intake_status = current.get("intake_status") or ("pending" if intake_profile_id is not None else "disabled")
    archive_required = get_backend_capabilities(str(row["runtime"])).supports_archive
    artifact_sync_status = current.get("artifact_sync_status") or "pending"
    evidence_status = row.get("evidence_status") or "missing"
    archive_status = row.get("archive_status") or "missing"
    return EvaluationTelemetryResponse(
        evaluation_id=evaluation_id,
        status=row["status"],
        current_execution=current_execution,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row.get("finished_at"),
        failure=EvaluationTelemetryFailure(
            phase=current.get("failure_phase"),
            code=row.get("last_failure_code"),
            category=row.get("last_failure_category"),
        ),
        cleanup=EvaluationTelemetryCleanup(
            cancellation_status=row.get("cancel_teardown_status") or "not_requested",
            error=row.get("cancel_teardown_error"),
            updated_at=row.get("cancel_teardown_updated_at"),
            executions=db.execution_cleanups.list_for_evaluation(evaluation_id),
        ),
        intake=EvaluationTelemetryIntake(
            enabled=intake_profile_id is not None,
            profile_id=intake_profile_id,
            status=intake_status,
            experiment_ref=current.get("intake_experiment_ref"),
            run_refs=current.get("intake_run_refs") or [],
            expected_records=current.get("intake_expected_records"),
            uploaded_records=current.get("intake_uploaded_records"),
            complete=intake_status in {"disabled", "succeeded", "no_records"},
            error=current.get("intake_error"),
            diagnostic_artifact=(f"{links.artifacts}/intake-upload.json" if intake_profile_id is not None else None),
        ),
        artifacts=EvaluationTelemetryArtifacts(
            listing=links.artifacts,
            archive=links.archive,
            provenance=links.provenance,
            sbom=links.sbom,
            artifact_sync_status=artifact_sync_status,
            artifact_sync_file_count=current.get("artifact_sync_file_count"),
            artifact_sync_error=current.get("artifact_sync_error"),
            evidence_status=evidence_status,
            evidence_error=row.get("evidence_error"),
            archive_status=archive_status,
            archive_required=archive_required,
            archive_error=row.get("archive_error"),
            terminal_sync_complete=(
                artifact_sync_status == "succeeded"
                and evidence_status == "ready"
                and (not archive_required or archive_status == "ready")
            ),
        ),
        executions=_execution_telemetry(evaluation_id, execution_rows),
        resource_usage=db.resource_usage.list_for_evaluation(evaluation_id),
    )


@router.get("/{evaluation_id}/reproduce", response_model=ReproduceEvaluationResponse)
def reproduce_evaluation(evaluation_id: str, db: Db) -> ReproduceEvaluationResponse:
    """Return a safe rerun request and CLI command for an existing evaluation."""
    row = db.evaluations.get(evaluation_id)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    body = _reproduce_request(row)
    return ReproduceEvaluationResponse(
        evaluation_id=evaluation_id,
        source_status=row["status"],
        request=body,
        cli_command=_create_command(body),
        notes=_reproduce_notes(row, body),
    )


@router.post("/{evaluation_id}/retry", status_code=202, response_model=EvaluationResponse)
def retry_evaluation(evaluation_id: str, db: Db) -> EvaluationResponse:
    """Queue another execution of a failed evaluation.

    The evaluation id and benchmark_run_id remain unchanged, so retrying a
    benchmark member updates that benchmark run's derived aggregate. Detailed
    result and artifact paths are reused for the newest execution.
    """
    row = db.evaluations.retry_failed(evaluation_id)
    if row is None:
        current = db.evaluations.get(evaluation_id)
        if current is None:
            raise _http_error(404, "not_found", "evaluation not found")
        if current["status"] != "failed":
            raise _http_error(
                409,
                "evaluation_not_failed",
                f"evaluation status is '{current['status']}', must be 'failed'",
            )
        block_reason = db.evaluations.retry_block_reason(evaluation_id)
        if block_reason == "terminal_artifacts_finalizing":
            raise _http_error(
                409,
                "evaluation_retry_pending",
                "evaluation cannot be retried while terminal evidence or archive generation is in progress",
            )
        if block_reason == "benchmark_unavailable":
            raise _http_error(
                409,
                "evaluation_not_retryable",
                "evaluation cannot be retried because its benchmark run is cancelled or unavailable",
            )
        raise _http_error(409, "evaluation_not_retryable", "evaluation cannot be retried right now")
    db.commit()
    return _response(row)


@router.post("/{evaluation_id}/cancel", response_model=EvaluationResponse)
def cancel_evaluation(evaluation_id: str, db: Db) -> EvaluationResponse:
    """Cancel an in-flight run.

    Flips a non-terminal run to `cancelled`; a run that already reached a
    terminal status is returned unchanged (idempotent). 404 if unknown.
    If the run has a persisted backend handle, runtime cleanup is attempted after the
    cancellation is durable; cleanup failures are recorded in `status_detail`.
    """
    row, cancelled_now = db.evaluations.cancel(evaluation_id)
    if row is None:
        raise _http_error(404, "not_found", "evaluation not found")
    if cancelled_now:
        db.commit()
        row = teardown_cancelled_evaluation(db, row)
    return _response(row)


@router.delete("/{evaluation_id}", response_model=DeleteResponse)
def delete_evaluation(evaluation_id: str, db: Db) -> DeleteResponse:
    """Soft-delete an evaluation's metadata (artifacts are retained).

    Sets `deleted_at`; 404 if unknown or already deleted. This does not append
    an evaluation event because the evaluation's status is unchanged and events
    model status transitions, not metadata visibility changes.
    """
    if not db.evaluations.soft_delete(evaluation_id):
        raise _http_error(404, "not_found", "evaluation not found")
    return DeleteResponse(id=evaluation_id)


# --------------------------------------------------------------------------
# Per-evaluation artifacts are synced to object storage by the dispatch worker
# under the stable prefix from scaled_evals.api.s3.evaluation_artifact_prefix.
# Archive endpoints expose status and signed downloads for results.tar.gz bundles.
# --------------------------------------------------------------------------


@router.get("/{evaluation_id}/logs", response_model=EvaluationLogResponse)
def get_logs(evaluation_id: str, db: Db, tail_lines: int = 100) -> EvaluationLogResponse:
    row = _load_observability_row(db, evaluation_id)
    return _logs_payload(row, tail_lines)


@router.get("/{evaluation_id}/logs/stream")
async def logs_stream(evaluation_id: str, request: Request, database_factory: StreamDatabaseFactory) -> Any:
    await run_in_threadpool(_ensure_stream_evaluation_exists, database_factory, evaluation_id)
    if not _acquire_sse_connection():
        raise _http_error(429, "stream_limit_exceeded", "too many active event streams")

    async def generate():
        try:
            sent = 0
            deadline = monotonic() + settings.api_sse_max_duration_seconds
            while monotonic() < deadline:
                if await request.is_disconnected():
                    break
                row, lines = await run_in_threadpool(_load_stream_log_snapshot, database_factory, evaluation_id)
                for line in lines[sent:]:
                    yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"
                sent = len(lines)
                yield f"event: status\ndata: {_status_event(row).model_dump_json()}\n\n"
                if row["status"] in _TERMINAL_STATUSES:
                    break
                yield "event: ping\ndata: {}\n\n"
                await asyncio.sleep(settings.api_sse_poll_interval_seconds)
        finally:
            _release_sse_connection()

    return _sse_response(generate())


@router.get("/{evaluation_id}/events", response_model=ListEnvelope[EvaluationEvent])
def list_events(
    evaluation_id: str,
    db: Db,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    cursor: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ListEnvelope[EvaluationEvent]:
    """List persisted event history in stable chronological order.

    Ordering is `(created_at ASC, id ASC)`. `cursor` encodes that pair for
    stable pagination; `offset` is retained for compatibility and ad-hoc reads.
    """
    _ensure_evaluation_exists(db, evaluation_id)
    try:
        events = db.evaluations.list_events(
            evaluation_id,
            limit=limit,
            cursor=cursor,
            offset=offset,
        )
    except ValueError:
        raise _http_error(400, "invalid_cursor", "invalid cursor") from None
    return _events_page_payload(events, evaluation_id, limit)


@router.get("/{evaluation_id}/events/stream")
async def events_stream(evaluation_id: str, request: Request, database_factory: StreamDatabaseFactory) -> Any:
    await run_in_threadpool(_ensure_stream_evaluation_exists, database_factory, evaluation_id)
    if not _acquire_sse_connection():
        raise _http_error(429, "stream_limit_exceeded", "too many active event streams")

    async def generate():
        try:
            last_created_at: datetime | None = None
            last_id: int | None = None
            deadline = monotonic() + settings.api_sse_max_duration_seconds
            while monotonic() < deadline:
                if await request.is_disconnected():
                    break
                events = await run_in_threadpool(
                    _load_stream_event_batch,
                    database_factory,
                    evaluation_id,
                    last_created_at,
                    last_id,
                )
                terminal_seen = False
                for row in events:
                    last_created_at = row["created_at"]
                    last_id = int(row["id"])
                    event = _event_payload(row, evaluation_id)
                    event_name = event.type or "status"
                    yield f"event: {event_name}\ndata: {event.model_dump_json()}\n\n"
                    is_terminal_status_event = (row.get("type") or "status") == "status" and row[
                        "status"
                    ] in _TERMINAL_STATUSES
                    if is_terminal_status_event:
                        terminal_seen = True
                        break
                if terminal_seen:
                    break
                if events:
                    continue

                status = await run_in_threadpool(_load_stream_status, database_factory, evaluation_id)
                if status is None:
                    break
                if status in _TERMINAL_STATUSES:
                    terminal_events = await run_in_threadpool(
                        _load_stream_event_batch,
                        database_factory,
                        evaluation_id,
                        last_created_at,
                        last_id,
                    )
                    for row in terminal_events:
                        event = _event_payload(row, evaluation_id)
                        event_name = event.type or "status"
                        yield f"event: {event_name}\ndata: {event.model_dump_json()}\n\n"
                        is_terminal_status_event = (row.get("type") or "status") == "status" and row[
                            "status"
                        ] in _TERMINAL_STATUSES
                        if is_terminal_status_event:
                            break
                    break
                yield "event: ping\ndata: {}\n\n"
                await asyncio.sleep(settings.api_sse_poll_interval_seconds)
        finally:
            _release_sse_connection()

    return _sse_response(generate())


@router.get("/{evaluation_id}/artifacts", response_model=ListEnvelope[EvaluationArtifact])
def list_artifacts(evaluation_id: str, db: Db, prefix: str = "") -> ListEnvelope[EvaluationArtifact]:
    _ensure_evaluation_exists(db, evaluation_id)
    base_prefix = s3.evaluation_artifact_prefix(evaluation_id)
    object_prefix = f"{base_prefix}{prefix.lstrip('/')}"
    data = []
    for item in s3.list_objects(object_prefix):
        if not item["key"].startswith(base_prefix):
            continue
        path = item["key"][len(base_prefix) :]
        data.append(
            {
                "path": path,
                "size_bytes": item["size_bytes"],
                "updated_at": item["updated_at"],
                "links": {
                    "download": f"/evaluations/{evaluation_id}/artifacts/{path}",
                },
            }
        )
    return list_envelope(data)


@router.get("/{evaluation_id}/artifacts/{path:path}")
def get_artifact(evaluation_id: str, path: str, database_factory: StreamDatabaseFactory) -> StreamingResponse:
    try:
        object_key = s3.evaluation_artifact_key(evaluation_id, path)
    except ValueError:
        raise _http_error(404, "not_found", "not found") from None
    # Short-lived checkout: a request-scoped Db dependency would hold the pooled
    # connection for the entire stream, and enough concurrent slow downloads then
    # exhaust the pool and stall unrelated requests.
    with database_factory() as db:
        _ensure_evaluation_exists(db, evaluation_id)
    filename = path.rsplit("/", 1)[-1]
    return StreamingResponse(
        s3.stream_object(object_key),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{evaluation_id}/archive/download")
def download_archive(evaluation_id: str, database_factory: StreamDatabaseFactory) -> StreamingResponse:
    with database_factory() as db:
        row = _load_archive_row(db, evaluation_id)
    if row.get("archive_status") != "ready" or not row.get("archive_object_key"):
        raise _http_error(404, "not_found", "archive not ready")
    return StreamingResponse(
        s3.stream_object(row["archive_object_key"]),
        media_type="application/x-tar",
        headers={"Content-Disposition": 'attachment; filename="results.tar.gz"'},
    )


@router.get("/{evaluation_id}/harbor-viewer/archive")
def download_harbor_viewer_archive(evaluation_id: str, database_factory: StreamDatabaseFactory) -> StreamingResponse:
    with database_factory() as db:
        row = _load_observability_row(db, evaluation_id)
    if not harbor_viewer_archive_available_from_result(row.get("result")):
        raise _http_error(404, "not_found", "Harbor Viewer archive not ready")
    return StreamingResponse(
        s3.stream_object(s3.evaluation_harbor_viewer_archive_key(evaluation_id)),
        media_type="application/gzip",
        headers={"Content-Disposition": (f'attachment; filename="{evaluation_id}-harbor-viewer.tar.gz"')},
    )


@router.get("/{evaluation_id}/archive", response_model=EvaluationArchiveResponse)
def get_archive(evaluation_id: str, db: Db) -> EvaluationArchiveResponse:
    return _archive_response(_load_archive_row(db, evaluation_id))


@router.post("/{evaluation_id}/archive", response_model=EvaluationArchiveResponse, status_code=202)
def build_archive(
    evaluation_id: str,
    body: BuildArchiveRequest,
    db: Db,
    response: Response,
) -> EvaluationArchiveResponse:
    row = _load_archive_row(db, evaluation_id)
    if row["archive_status"] == "ready" and not body.force:
        response.status_code = 200
        return _archive_response(row)

    updated = db.evaluations.request_archive_build(evaluation_id, force=body.force)
    if updated is None:
        raise _http_error(404, "not_found", "evaluation not found")
    db.commit()
    return _archive_response(updated)
