# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter

import httpx
from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from scaled_evals.api import dispatch_health, s3
from scaled_evals.api.build import buildkit, registry
from scaled_evals.api.db import pooled_connection
from scaled_evals.api.repositories.ops_repository import OperationsRepository
from scaled_evals.api.schemas.common import HealthStatus, ReadyzResponse
from scaled_evals.api.settings import settings

router = APIRouter(tags=["operations"])

PROMETHEUS_TEXT_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_HTTP_REQUESTS: Counter[tuple[str, str, str]] = Counter()
_HTTP_REQUEST_DURATION_SECONDS: Counter[tuple[str, str, str]] = Counter()
_HTTP_REQUEST_DURATION_COUNT: Counter[tuple[str, str, str]] = Counter()

_KNOWN_EVALUATION_STATUSES = (
    "blocked",
    "queued",
    "provisioning",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)


@dataclass
class ReadinessSnapshot:
    checks: dict[str, str] = field(default_factory=dict)


_READINESS_SNAPSHOT = ReadinessSnapshot()


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # noqa: ANN001
        started_at = perf_counter()
        response = await call_next(request)
        elapsed = perf_counter() - started_at

        observe_request(request, response.status_code, elapsed)
        return response


def _required_check(checks: dict[str, str], name: str, probe: Callable[[], None]) -> bool:
    try:
        probe()
        checks[name] = "ok"
        return True
    except Exception as exc:  # noqa: BLE001 — surface dependency name only
        checks[name] = f"fail: {type(exc).__name__}"
        return False


def _enabled_required_check(
    checks: dict[str, str],
    name: str,
    enabled: bool,
    probe: Callable[[], None],
) -> bool:
    if not enabled:
        checks[name] = "skipped: disabled"
        return True
    return _required_check(checks, name, probe)


@router.get("/healthz", response_model=HealthStatus)
def healthz() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get("/metrics")
def metrics() -> Response:
    return Response(
        content=render_prometheus_metrics(),
        media_type=PROMETHEUS_TEXT_MEDIA_TYPE,
    )


@router.get("/readyz", response_model=ReadyzResponse)
def readyz(response: Response) -> ReadyzResponse:
    checks, required_ok = _run_dependency_checks()
    _READINESS_SNAPSHOT.checks = dict(checks)
    if not required_ok:
        response.status_code = 503
    return ReadyzResponse(status="ok" if required_ok else "degraded", checks=checks)


def _postgres_probe() -> None:
    with pooled_connection(timeout=3) as conn:
        OperationsRepository(conn).ping()


def _schema_probe() -> None:
    with pooled_connection(timeout=3) as conn:
        OperationsRepository(conn).assert_schema_compatible()


def _build_worker_probe() -> None:
    with pooled_connection(timeout=3) as conn:
        if not OperationsRepository(conn).has_fresh_service_heartbeat(
            "build_worker", stale_seconds=settings.build_worker_stale_seconds
        ):
            raise RuntimeError("no fresh build worker heartbeat")


def _platform_jobs_controller_probe() -> None:
    with pooled_connection(timeout=3) as conn:
        if not OperationsRepository(conn).has_fresh_service_heartbeat(
            "platform_jobs_controller", stale_seconds=settings.build_worker_stale_seconds
        ):
            raise RuntimeError("no fresh Platform Jobs controller heartbeat")


def _dispatch_worker_probe() -> None:
    response = httpx.get(settings.dispatch_worker_health_url, timeout=3)
    response.raise_for_status()


def render_prometheus_metrics() -> str:
    evaluation_counts = _evaluation_status_counts()
    fleet_totals = _fleet_totals()
    dispatch_snapshot = _dispatch_observability_snapshot()
    task_pack_snapshot = _task_pack_observability_snapshot()
    checks = dict(_READINESS_SNAPSHOT.checks)

    samples: list[str] = []
    samples.extend(
        _metric_family(
            "scaled_evals_up",
            "gauge",
            "Whether the scaled-evals API process is serving metrics.",
            [((), 1)],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_http_requests_total",
            "counter",
            "HTTP requests handled by method, route, and status code.",
            [
                (
                    (
                        ("method", method),
                        ("route", route),
                        ("status_code", status_code),
                    ),
                    count,
                )
                for (method, route, status_code), count in sorted(_HTTP_REQUESTS.items())
            ],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_http_request_duration_seconds",
            "summary",
            "HTTP request handling time by method, route, and status code.",
            [
                (
                    (
                        ("method", method),
                        ("route", route),
                        ("status_code", status_code),
                    ),
                    total_seconds,
                )
                for (method, route, status_code), total_seconds in sorted(_HTTP_REQUEST_DURATION_SECONDS.items())
            ],
            suffix="_sum",
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_http_request_duration_seconds",
            "summary",
            "HTTP request handling time by method, route, and status code.",
            [
                (
                    (
                        ("method", method),
                        ("route", route),
                        ("status_code", status_code),
                    ),
                    count,
                )
                for (method, route, status_code), count in sorted(_HTTP_REQUEST_DURATION_COUNT.items())
            ],
            suffix="_count",
            include_header=False,
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_evaluations_by_status",
            "gauge",
            "Current non-deleted evaluations grouped by lifecycle status.",
            [((("status", status),), evaluation_counts.get(status, 0)) for status in evaluation_counts],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_fleet_resource_count",
            "gauge",
            "All-time control-plane resource totals, including soft-deleted records.",
            [((("resource", resource),), count) for resource, count in sorted(fleet_totals.items())],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_queue_depth",
            "gauge",
            "Current non-deleted evaluations waiting in queued status.",
            [((), evaluation_counts.get("queued", 0))],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_oldest_queued_seconds",
            "gauge",
            "Age in seconds of the oldest non-deleted queued evaluation.",
            [((), dispatch_snapshot["oldest_queued_seconds"])],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_unclaimed_queued",
            "gauge",
            "Queued evaluations not currently owned by a dispatch worker lease.",
            [((), dispatch_snapshot["unclaimed_queued"])],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_worker_liveness",
            "gauge",
            "Distinct dispatch workers with fresh or stale active leases.",
            [
                ((("state", "live"),), dispatch_snapshot["live_workers"]),
                ((("state", "stale"),), dispatch_snapshot["stale_workers"]),
            ],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_oldest_worker_lease_seconds",
            "gauge",
            "Age in seconds of the oldest active dispatch worker lease heartbeat.",
            [((), dispatch_snapshot["oldest_worker_lease_seconds"])],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_stuck_evaluations",
            "gauge",
            "Active evaluations past production stuck-job thresholds by status and runtime.",
            [
                (
                    (("status", row["status"]), ("runtime", row["runtime"])),
                    row["count"],
                )
                for row in dispatch_snapshot["stuck_jobs"]
            ],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dispatch_backend_failures_last_hour",
            "gauge",
            "Failed evaluations in the last hour with backend launch/status error details.",
            [((("runtime", row["runtime"]),), row["count"]) for row in dispatch_snapshot["backend_failures"]],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_task_pack_missing_ready_revisions",
            "gauge",
            "Ready task revisions whose referenced task-pack object cannot be read.",
            [((), task_pack_snapshot["missing_ready_revisions"])],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_task_pack_ready_revisions_checked",
            "gauge",
            "Ready task revisions checked for task-pack object readability during metrics scrape.",
            [((), task_pack_snapshot["checked_ready_revisions"])],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_switchyard_teardown_resources",
            "gauge",
            "Switchyard runtime resources pending drain or retry after delete failure.",
            [
                (
                    (("state", "draining"),),
                    dispatch_snapshot["switchyard_teardown"].get("draining", 0),
                ),
                (
                    (("state", "delete_failed"),),
                    dispatch_snapshot["switchyard_teardown"].get("delete_failed", 0),
                ),
            ],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dependency_ready",
            "gauge",
            "Required dependency readiness from the latest readiness probe.",
            [
                ((("dependency", name),), 1 if _dependency_state(status) == "ok" else 0)
                for name, status in checks.items()
                if _dependency_is_required(name)
            ],
        )
    )
    samples.extend(
        _metric_family(
            "scaled_evals_dependency_state",
            "gauge",
            "Dependency state from the latest readiness probe.",
            [
                (
                    (
                        ("dependency", name),
                        ("state", _dependency_state(status)),
                    ),
                    1,
                )
                for name, status in checks.items()
            ],
        )
    )
    samples.append("")
    return "\n".join(samples)


def observe_request(
    request: Request,
    status_code: int,
    elapsed: float,
    *,
    route_override: str | None = None,
) -> None:
    route = route_override or _route_template(request)
    if route is None:
        return
    key = (request.method, route, str(status_code))
    _HTTP_REQUESTS[key] += 1
    _HTTP_REQUEST_DURATION_SECONDS[key] += elapsed
    _HTTP_REQUEST_DURATION_COUNT[key] += 1


def _route_template(request: Request) -> str | None:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if not isinstance(path, str):
        return None
    return path


def _run_dependency_checks() -> tuple[dict[str, str], bool]:
    checks: dict[str, str] = {}
    required_ok = True
    required_ok &= _required_check(checks, "postgres", lambda: _postgres_probe())
    required_ok &= _required_check(checks, "schema", _schema_probe)
    required_ok &= _required_check(checks, "object_store", s3.check_bucket)
    required_ok &= _enabled_required_check(
        checks,
        "dispatch_worker",
        bool(settings.dispatch_worker_health_url),
        _dispatch_worker_probe,
    )
    required_ok &= _enabled_required_check(
        checks,
        "platform_jobs_controller",
        settings.platform_build_jobs_enabled or settings.platform_evaluation_jobs_enabled,
        _platform_jobs_controller_probe,
    )
    required_ok &= _enabled_required_check(checks, "build_worker", settings.build_worker_required, _build_worker_probe)
    required_ok &= _enabled_required_check(checks, "buildkit", settings.buildkit_enabled, buildkit.check_buildkit)
    required_ok &= _enabled_required_check(checks, "registry", settings.registry_enabled, registry.check_registry)
    checks["gym_dispatch"] = dispatch_health.check_gym_dispatch()
    checks["sandbox_k8s_dispatch"] = dispatch_health.check_sandbox_k8s_dispatch()
    return checks, required_ok


def _dependency_is_required(name: str) -> bool:
    if name == "dispatch_worker":
        return bool(settings.dispatch_worker_health_url)
    if name == "platform_jobs_controller":
        return settings.platform_build_jobs_enabled or settings.platform_evaluation_jobs_enabled
    if name == "build_worker":
        return settings.build_worker_required
    if name == "buildkit":
        return settings.buildkit_enabled
    if name == "registry":
        return settings.registry_enabled
    return name in {"postgres", "schema", "object_store"}


def _dependency_state(status: str) -> str:
    if status == "ok" or status.startswith("ok:"):
        return "ok"
    if status.startswith("skipped:"):
        return "skipped"
    if status.startswith("fail:"):
        return "fail"
    if status.startswith("pending:"):
        return "pending"
    return "unknown"


def _evaluation_status_counts() -> dict[str, int]:
    counts = dict.fromkeys(_KNOWN_EVALUATION_STATUSES, 0)
    try:
        with pooled_connection(timeout=3) as conn:
            counts.update(OperationsRepository(conn).evaluation_status_counts())
    except Exception:  # noqa: BLE001 — keep metrics scrape available on DB errors
        pass
    return counts


def _fleet_totals() -> dict[str, int]:
    try:
        with pooled_connection(timeout=3) as conn:
            return OperationsRepository(conn).fleet_totals()
    except Exception:  # noqa: BLE001 — keep metrics scrape available on DB errors
        return {}


def _dispatch_observability_snapshot() -> dict:
    empty = {
        "oldest_queued_seconds": 0.0,
        "unclaimed_queued": 0,
        "live_workers": 0,
        "stale_workers": 0,
        "oldest_worker_lease_seconds": 0.0,
        "stuck_jobs": [],
        "backend_failures": [],
        "switchyard_teardown": {"draining": 0, "delete_failed": 0},
    }
    try:
        with pooled_connection(timeout=3) as conn:
            return OperationsRepository(conn).dispatch_observability_snapshot(
                stuck_queued_seconds=settings.observability_stuck_queued_seconds,
                stuck_provisioning_seconds=settings.observability_stuck_provisioning_seconds,
                stuck_running_seconds=settings.observability_stuck_running_seconds,
                stale_worker_seconds=settings.observability_worker_stale_seconds,
            )
    except Exception:  # noqa: BLE001 — keep metrics scrape available on DB errors
        return empty


def _task_pack_observability_snapshot() -> dict[str, int]:
    snapshot = {"checked_ready_revisions": 0, "missing_ready_revisions": 0}
    try:
        with pooled_connection(timeout=3) as conn:
            rows = OperationsRepository(conn).ready_task_pack_revisions(
                limit=settings.observability_task_pack_scan_limit
            )
    except Exception:  # noqa: BLE001 — keep metrics scrape available on DB errors
        return snapshot
    for row in rows:
        object_key = row.get("tarball_object_key")
        if not object_key:
            continue
        snapshot["checked_ready_revisions"] += 1
        try:
            if not s3.object_exists(str(object_key)):
                snapshot["missing_ready_revisions"] += 1
        except Exception:  # noqa: BLE001 — unreadable is alertable, scrape must continue
            snapshot["missing_ready_revisions"] += 1
    return snapshot


def _metric_family(
    name: str,
    metric_type: str,
    help_text: str,
    samples: Iterable[tuple[tuple[tuple[str, str], ...], int | float]],
    *,
    suffix: str = "",
    include_header: bool = True,
) -> list[str]:
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"] if include_header else []
    sample_name = f"{name}{suffix}"
    lines.extend(f"{sample_name}{_labels(labels)} {_format_value(value)}" for labels, value in samples)
    return lines


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    pairs = ",".join(f'{key}="{_escape_label_value(value)}"' for key, value in labels)
    return f"{{{pairs}}}"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", r"\\").replace("\n", r"\n").replace('"', r"\"")


def _format_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"
