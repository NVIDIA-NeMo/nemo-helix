# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from scaled_evals.api import s3
from scaled_evals.api.agent_bundle_registry import accessible_bundle_for_run
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import Database, get_db, get_stream_database_factory
from scaled_evals.api.repositories.base_repository import Conflict, NotFound
from scaled_evals.api.repositories.benchmark_run_repository import derive_run_view
from scaled_evals.api.routers.evaluations import teardown_cancelled_evaluation
from scaled_evals.api.runnability import BlockedPreflight, preflight_benchmark_run
from scaled_evals.api.schemas.benchmark_archives import BenchmarkArchiveRequest, BenchmarkArchiveResponse
from scaled_evals.api.schemas.benchmark_runs import (
    BenchmarkRun,
    BenchmarkRunLinks,
    BenchmarkRunResponse,
    CreateBenchmarkRunRequest,
    ReproduceBenchmarkRunResponse,
)
from scaled_evals.api.schemas.common import DeleteResponse, ListEnvelope, page_from_rows
from scaled_evals.api.schemas.evaluations import Evaluation
from scaled_evals.api.schemas.runnability import RunnabilityReport
from scaled_evals.api.utils import make_id

router = APIRouter(prefix="/benchmark-runs", tags=["benchmark-runs"])

Db = Annotated[Database, Depends(get_db)]
StreamDatabaseFactory = Annotated[Callable[[], AbstractContextManager[Database]], Depends(get_stream_database_factory)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


def _links(run_id: str) -> BenchmarkRunLinks:
    base = f"/benchmark-runs/{run_id}"
    return BenchmarkRunLinks(
        self=base,
        evaluations=f"{base}/evaluations",
        reproduce=f"{base}/reproduce",
        cancel=f"{base}/cancel",
        archive=f"{base}/archive",
    )


def _rerun_name(name: str) -> str:
    return f"rerun of {name}"[:200]


def _agent_bundle_id(runner_metadata: Mapping[str, Any]) -> str | None:
    bundle = runner_metadata.get("agent_bundle")
    if isinstance(bundle, Mapping) and bundle.get("bundle_id"):
        return str(bundle["bundle_id"])
    return None


def _reproduce_request(run: Mapping[str, Any], member: Mapping[str, Any]) -> CreateBenchmarkRunRequest:
    runner_metadata = run.get("runner_metadata") or {}
    return CreateBenchmarkRunRequest(
        name=_rerun_name(str(run["name"])),
        benchmark_id=str(run["benchmark_id"]),
        benchmark_revision=int(run["benchmark_revision"]),
        framework=run.get("framework") or "harbor",
        framework_version=run.get("framework_version"),
        framework_profile_id=run.get("framework_profile_id"),
        member_framework_profile_ids=dict(runner_metadata.get("member_framework_profile_ids") or {}),
        switchyard_profile_id=run.get("switchyard_profile_id"),
        intake_profile_id=run.get("intake_profile_id"),
        credentials=dict(run.get("credentials") or {}),
        agent_bundle_id=_agent_bundle_id(runner_metadata),
        extra_skill_object_keys=list(member.get("extra_skill_object_keys") or []),
        instruction_prefix=member.get("instruction_prefix"),
        instruction_postfix=member.get("instruction_postfix"),
        initial_user_turns=list(member.get("initial_user_turns") or []),
        runtime=str(run["runtime"]),
        network_policy=run.get("network_policy") or "unrestricted",
        network_policy_config=dict(run.get("network_policy_config") or {}),
        n_attempts=int(member.get("n_attempts") or 1),
        parallelism=int(run["parallelism"]),
        max_concurrent_members=run.get("max_concurrent_members"),
        visibility=run.get("visibility") or "private",
    )


def _create_command(body: CreateBenchmarkRunRequest) -> list[str]:
    command = [
        "scaled-evals",
        "benchmark-run",
        "create",
        "--name",
        body.name,
        "--benchmark-id",
        body.benchmark_id,
        "--benchmark-revision",
        str(body.benchmark_revision),
        "--framework",
        body.framework,
        "--runtime",
        body.runtime,
        "--network-policy",
        body.network_policy,
        "--n-attempts",
        str(body.n_attempts),
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
        ("--agent-bundle", body.agent_bundle_id),
    ):
        if value:
            command.extend([option, value])
    for task_id, profile_id in sorted(body.member_framework_profile_ids.items()):
        command.extend(["--member-framework-profile", f"{task_id}={profile_id}"])
    for role, credential_id in sorted(body.credentials.items()):
        command.extend(["--credential", f"{role}={credential_id}"])
    for object_key in body.extra_skill_object_keys:
        command.extend(["--extra-skill-object-key", object_key])
    if body.instruction_prefix is not None:
        command.extend(["--instruction-prefix", body.instruction_prefix])
    if body.instruction_postfix is not None:
        command.extend(["--instruction-postfix", body.instruction_postfix])
    for turn in body.initial_user_turns:
        command.extend(["--initial-user-turn", turn])
    if body.max_concurrent_members is not None:
        command.extend(["--max-concurrent-members", str(body.max_concurrent_members)])
    return command


def _response(db: Database, run: dict[str, Any]) -> BenchmarkRunResponse:
    """Derive the run's status/reward/breakdown from its members, then build the response."""
    members = db.benchmark_runs.members_for_runs([run["id"]])
    view = derive_run_view(run, members)
    return BenchmarkRunResponse(**view, links=_links(run["id"]))


@router.post("/preflight", response_model=RunnabilityReport)
def preflight_benchmark_run_request(
    body: CreateBenchmarkRunRequest,
    db: Db,
    current: Principal,
) -> RunnabilityReport:
    """Check whether a benchmark-run request is runnable without creating it."""
    result = preflight_benchmark_run(
        db,
        body,
        current,
        object_exists=s3.object_exists,
        resolve_bundle=accessible_bundle_for_run,
    )
    return result.report


@router.post("", status_code=202, response_model=BenchmarkRunResponse)
def create_benchmark_run(body: CreateBenchmarkRunRequest, db: Db, current: Principal) -> BenchmarkRunResponse:
    """Run a benchmark by aggregating one evaluation per member task.

    Resolves the benchmark revision (defaulting to current), confirms it has
    members and every member's resolved task revision is `ready`, then inserts a
    `benchmark_runs` row plus one member evaluation per task. The member
    evaluations are claimed and run independently by the dispatch worker pool;
    the run reaches a terminal state by fan-in once they finish. 404 unknown
    benchmark/revision, 422 empty benchmark / bad reference, 409 a member not
    `ready`.
    """
    preflight = preflight_benchmark_run(
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
    revision = preflight.revision
    members = preflight.members
    runner_metadata = preflight.runner_metadata
    if body.member_framework_profile_ids:
        runner_metadata["member_framework_profile_ids"] = dict(body.member_framework_profile_ids)

    run_id = make_id("bmr")
    spawn = [
        {
            "id": make_id("ev"),
            "task_id": m["task_id"],
            "task_revision": m["task_revision"],
            "task_slug": m.get("task_slug"),
            "framework_profile_id": body.member_framework_profile_ids.get(str(m["task_id"])),
        }
        for m in members
    ]
    row = db.benchmark_runs.create_run(
        run_id,
        name=body.name,
        framework=body.framework,
        requested_framework_version=runner.requested_version,
        framework_version=runner.version,
        runner_image_ref=runner.image_ref,
        runner_image_digest=runner.image_digest,
        framework_adapter_version=runner.adapter_version,
        sandbox_k8s_version=runner.sandbox_k8s_version,
        runner_metadata=runner_metadata,
        benchmark_id=body.benchmark_id,
        benchmark_revision=revision,
        members=spawn,
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
        max_concurrent_members=(
            body.max_concurrent_members
            if body.max_concurrent_members is not None
            else (len(members) if body.switchyard_profile_id is not None else None)
        ),
        visibility=body.visibility,
        owner_id=current.owner_id,
    )
    # Commit so the dispatch worker (its own connection) sees the queued members.
    db.commit()
    return _response(db, row)


@router.get("", response_model=ListEnvelope[BenchmarkRun])
def list_benchmark_runs(
    db: Db,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    benchmark_id: str | None = None,
    shared: bool = False,
    q: str | None = Query(default=None, min_length=1, max_length=200),
) -> ListEnvelope[BenchmarkRun]:
    """List benchmark runs, newest first (optionally filtered by benchmark_id).

    Each run's status/reward is derived from its member evaluations on read — one
    extra query fetches the members for the whole page, then they're grouped and
    rolled up per run (no per-run query).
    """
    rows = db.benchmark_runs.list(
        limit=limit,
        cursor=cursor,
        order=order,
        benchmark_id=benchmark_id,
        shared=shared,
        q=q,
    )
    page = rows[:limit]
    members_by_run: dict[str, list[dict]] = {}
    for member in db.benchmark_runs.members_for_runs([r["id"] for r in page]):
        members_by_run.setdefault(member["benchmark_run_id"], []).append(member)
    derived = [derive_run_view(r, members_by_run.get(r["id"], [])) for r in rows]
    return page_from_rows(derived, limit, BenchmarkRun)


@router.get("/{run_id}", response_model=BenchmarkRunResponse)
def get_benchmark_run(run_id: str, db: Db) -> BenchmarkRunResponse:
    """Fetch one benchmark run; status/reward/breakdown derived from its members."""
    row = db.benchmark_runs.get(run_id)
    if row is None:
        raise _http_error(404, "not_found", "benchmark run not found")
    return _response(db, row)


@router.get("/{run_id}/reproduce", response_model=ReproduceBenchmarkRunResponse)
def reproduce_benchmark_run(run_id: str, db: Db) -> ReproduceBenchmarkRunResponse:
    """Return a safe, complete request and CLI command that reruns a benchmark run."""
    row = db.benchmark_runs.get(run_id)
    if row is None:
        raise _http_error(404, "not_found", "benchmark run not found")
    members = db.benchmark_runs.members_for_runs([run_id])
    if not members:
        raise _http_error(
            409,
            "reproduce_unavailable",
            "benchmark run has no available member evaluation snapshot",
        )
    source = db.evaluations.get(members[0]["id"])
    if source is None:
        raise _http_error(
            409,
            "reproduce_unavailable",
            "benchmark run member execution settings are unavailable",
        )
    body = _reproduce_request(row, source)
    view = derive_run_view(row, members)
    return ReproduceBenchmarkRunResponse(
        benchmark_run_id=run_id,
        source_status=view["status"],
        request=body,
        cli_command=_create_command(body),
        notes=[
            "Secret material is not exported; credential values remain credential ids.",
            "The rerun uses the frozen benchmark revision and member execution settings.",
            "The rerun will fail validation if referenced credentials, profiles, benchmark "
            "revision, agent bundle, or runtime backend are unavailable.",
        ],
    )


@router.get("/{run_id}/evaluations", response_model=ListEnvelope[Evaluation])
def list_run_evaluations(
    run_id: str,
    db: Db,
    limit: int = Query(default=100, ge=1, le=200),
    cursor: str | None = None,
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> ListEnvelope[Evaluation]:
    """List a benchmark run's member task evaluations."""
    if not db.benchmark_runs.exists(run_id):
        raise _http_error(404, "not_found", "benchmark run not found")
    rows = db.evaluations.list(
        limit=limit,
        cursor=cursor,
        order=order,
        status=None,
        task_id=None,
        shared=False,
        benchmark_run_id=run_id,
    )
    return page_from_rows(rows, limit, Evaluation)


@router.post("/{run_id}/cancel", response_model=BenchmarkRunResponse)
def cancel_benchmark_run(run_id: str, db: Db) -> BenchmarkRunResponse:
    """Cancel a benchmark run and its still-active member evaluations (soft cancel).

    Idempotent: a run already terminal is returned unchanged. The member
    evaluations are flipped to `cancelled`, then launched members follow the
    same runtime and Switchyard teardown path as single-evaluation cancellation.
    """
    row, _cancelled_now, cancelled_members = db.benchmark_runs.cancel(run_id)
    if row is None:
        raise _http_error(404, "not_found", "benchmark run not found")
    db.commit()
    for member in cancelled_members:
        teardown_cancelled_evaluation(db, member)
    return _response(db, row)


@router.delete("/{run_id}", response_model=DeleteResponse)
def delete_benchmark_run(run_id: str, db: Db) -> DeleteResponse:
    """Soft-delete a benchmark run (cascades to its member evaluations)."""
    if not db.benchmark_runs.soft_delete(run_id):
        raise _http_error(404, "not_found", "benchmark run not found")
    db.commit()
    return DeleteResponse(id=run_id)


def _benchmark_archive_response(run_id: str, row: dict | None) -> BenchmarkArchiveResponse:
    if row is None:
        return BenchmarkArchiveResponse(benchmark_run_id=run_id, status="missing")
    return BenchmarkArchiveResponse(
        **{key: value for key, value in row.items() if key != "download"},
        download=(f"/benchmark-runs/{run_id}/archive/download" if row["status"] == "ready" else None),
    )


@router.get("/{run_id}/archive", response_model=BenchmarkArchiveResponse)
def get_benchmark_archive(run_id: str, db: Db) -> BenchmarkArchiveResponse:
    if not db.benchmark_runs.exists(run_id):
        raise _http_error(404, "not_found", "benchmark run not found")
    return _benchmark_archive_response(run_id, db.benchmark_archives.get(run_id))


@router.post("/{run_id}/archive", response_model=BenchmarkArchiveResponse, status_code=202)
def request_benchmark_archive(run_id: str, body: BenchmarkArchiveRequest, db: Db) -> BenchmarkArchiveResponse:
    try:
        row = db.benchmark_archives.request(run_id, force=body.force)
    except NotFound as exc:
        raise _http_error(404, "not_found", exc.message) from exc
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    db.commit()
    return _benchmark_archive_response(run_id, row)


@router.get("/{run_id}/archive/download")
def download_benchmark_archive(run_id: str, database_factory: StreamDatabaseFactory) -> StreamingResponse:
    # Short-lived checkout: a request-scoped Db dependency would hold the pooled
    # connection for the entire archive stream and can exhaust the pool.
    with database_factory() as db:
        row = db.benchmark_archives.get(run_id)
    if row is None or row["status"] != "ready" or not row["object_key"]:
        raise _http_error(404, "not_found", "benchmark archive not ready")
    return StreamingResponse(
        s3.stream_object(row["object_key"]),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-results.tar.gz"'},
    )
