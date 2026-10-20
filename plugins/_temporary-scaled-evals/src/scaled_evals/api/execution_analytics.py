# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal-evaluation analytics recorded in, and read back from, Intake.

Each terminal evaluation emits one span to Intake carrying its outcome,
failure classification, and resource-usage totals. The admin usage, compute,
and failure views aggregate those spans over a time window, so they no longer
depend on the evaluation store keeping every historical row.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from nemo_helix_plugin.client_provider import get_nemo_client
from nemo_helix_plugin.intake.client import IntakeClient
from nemo_helix_plugin.intake.types import DirectSpanInput, DirectSpansIngestRequest

from scaled_evals.api.redaction import redact_secret_text

logger = logging.getLogger(__name__)

ANALYTICS_SOURCE = "scaled-evals-executions"
_PREFIX = "scaled_evals."
_SPAN_STATUS = {"succeeded": "success", "failed": "error", "cancelled": "cancelled"}
_USAGE_KEYS = (
    "cpu_samples",
    "cpu_sum",
    "peak_cpu_cores",
    "cpu_request_cores",
    "cpu_limit_cores",
    "memory_samples",
    "memory_sum",
    "peak_memory_bytes",
    "memory_request_bytes",
    "memory_limit_bytes",
    "gpu_request",
    "gpu_samples",
)


def _intake() -> IntakeClient:
    return IntakeClient.from_client(get_nemo_client(as_service="scaled-evals", internal=True))


def failure_category(*, code: str | None, category: str | None, detail: str | None, exception_counts: Any) -> str:
    """Classify a failed evaluation into one operational failure category."""
    parts = [code, category, detail, json.dumps(exception_counts) if exception_counts else None]
    text = " ".join(part for part in parts if part).lower()

    def has(pattern: str) -> bool:
        return re.search(pattern, text) is not None

    if has(r"(^|[^0-9])504([^0-9]|$)|gateway timeout"):
        return "inference_http_504"
    if has(r"ratelimiterror|rate[ -]?limit|(^|[^0-9])429([^0-9]|$)"):
        return "inference_rate_limit"
    if has(r"apitimeouterror|provider_timeout") or (
        has(r"inference|provider|model|openai|anthropic|litellm") and has(r"timed out|timeout")
    ):
        return "inference_timeout"
    if has(r"sandboxcreationerror|environmentstarttimeouterror|imagepull|rhacs|admission") or (
        has(r"sandbox|environment") and has(r"start|create|provision")
    ):
        return "sandbox_startup"
    if has(r"cleanup failed|cleanup command failed|cleanup left|cleanup unavailable"):
        return "runtime_cleanup"
    if has(
        r"poll_timeout|deadline exceeded|timed out waiting|execution timeout|agenttimeouterror|verifiertimeouterror"
    ):
        return "evaluation_timeout"
    if has(r"cancellederror|cancelled trial"):
        return "trial_cancelled"
    if has(r"object_store_unavailable|object store|rustfs|(^|[^a-z])s3([^a-z]|$)"):
        return "object_storage"
    if has(r"nonzeroagentexitcodeerror|agent exited|agent process"):
        return "agent_exit"
    if has(r"runner_disappeared|kubernetesjoberror|sandboxexecutionerror|connectionerror") or (
        category == "infrastructure"
    ):
        return "runtime_infrastructure"
    if (
        has(r"task_object_missing|task object|task pack|invalidreference|validationerror")
        or has(r"addtestsdirerror|registry.+not approved|allowed registries")
        or category == "task"
    ):
        return "task_configuration"
    if has(r"evaluation already terminal"):
        return "control_plane_state"
    return "other"


def execution_span(row: Mapping[str, Any], usage: Mapping[str, Any]) -> DirectSpanInput:
    """Build the analytics span for one terminal evaluation row."""
    occurred_at: datetime = row.get("finished_at") or row.get("updated_at") or datetime.now(UTC)
    created_at: datetime | None = row.get("created_at")
    status = str(row["status"])
    detail = row.get("status_detail")
    attributes: dict[str, Any] = {
        "evaluation_id": row["id"],
        "evaluation_name": row.get("name"),
        "task_id": row.get("task_id"),
        "benchmark_run_id": row.get("benchmark_run_id"),
        "owner_id": row.get("owner_id"),
        "runtime": row.get("runtime"),
        "status": status,
        "failure_code": row.get("last_failure_code"),
        "failure_category": failure_category(
            code=row.get("last_failure_code"),
            category=row.get("last_failure_category"),
            detail=detail,
            exception_counts=row.get("exception_counts"),
        )
        if status == "failed"
        else None,
        "detail": redact_secret_text(str(detail)) if detail is not None else None,
        "created_at": created_at.isoformat() if created_at else None,
        "occurred_at": occurred_at.isoformat(),
        "runtime_seconds": (occurred_at - created_at).total_seconds() if created_at else None,
        "n_trials": row.get("n_trials"),
        "parallelism": row.get("parallelism"),
        "executions": row.get("current_execution"),
        **{key: usage.get(key) for key in _USAGE_KEYS},
    }
    return DirectSpanInput(
        span_id=f"{row['id']}:{row.get('current_execution') or 0}",
        trace_id=str(row["id"]),
        session_id=str(row["id"]),
        name="scaled_evals.evaluation",
        status=_SPAN_STATUS.get(status, "unknown"),
        started_at=occurred_at,
        ended_at=occurred_at,
        attributes={_PREFIX + key: value for key, value in attributes.items()},
    )


def emit_execution_span(row: Mapping[str, Any], usage: Mapping[str, Any], *, workspace: str) -> None:
    """Record one terminal evaluation in Intake; failures are logged, never raised."""
    try:
        _intake().create_spans(
            workspace=workspace,
            body=DirectSpansIngestRequest(source=ANALYTICS_SOURCE, spans=[execution_span(row, usage)]),
        )
    except Exception:
        logger.warning("could not record analytics span for evaluation %s", row.get("id"), exc_info=True)


def load_records(*, workspace: str, window_start: datetime, window_end: datetime) -> list[dict[str, Any]]:
    """Return one record per terminal evaluation whose outcome falls in the window."""
    # ponytail: lists every span in the window per request (O(n), 90-day cap);
    # move to Intake-side aggregation if admin windows grow past tens of thousands.
    spans = _intake().list_spans(
        workspace=workspace,
        query_params={
            "mode": "detailed",
            "page_size": 1000,
            "filter": {
                "source": ANALYTICS_SOURCE,
                "started_at": {"$gte": window_start.isoformat(), "$lte": window_end.isoformat()},
            },
        },
    )
    latest: dict[str, dict[str, Any]] = {}
    for span in spans.items():
        raw = json.loads(span.raw_attributes or "{}")
        record = {key.removeprefix(_PREFIX): value for key, value in raw.items() if key.startswith(_PREFIX)}
        if not record.get("evaluation_id"):
            continue
        record["occurred_at"] = datetime.fromisoformat(record["occurred_at"])
        if record.get("created_at"):
            record["created_at"] = datetime.fromisoformat(record["created_at"])
        previous = latest.get(record["evaluation_id"])
        if previous is None or record["occurred_at"] >= previous["occurred_at"]:
            latest[record["evaluation_id"]] = record
    return [record for record in latest.values() if window_start <= record["occurred_at"] < window_end]


def _days(window_start: datetime, window_end: datetime) -> list[date]:
    first, last = window_start.date(), window_end.date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _num(record: Mapping[str, Any], key: str) -> float:
    return float(record.get(key) or 0)


def _present(records: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: float, denominator: float, scale: float = 1.0) -> float | None:
    return scale * numerator / denominator if denominator else None


def _compute(records: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_req = [r for r in records if _num(r, "cpu_request_cores") > 0]
    mem_req = [r for r in records if _num(r, "memory_request_bytes") > 0]
    peak_cpu = _present(records, "peak_cpu_cores")
    peak_mem = _present(records, "peak_memory_bytes")
    gpus = _present(records, "gpu_request")
    return {
        "evaluations": len(records),
        "sampled_evaluations": sum(
            1 for r in records if _num(r, "cpu_samples") or _num(r, "memory_samples") or _num(r, "gpu_samples")
        ),
        "samples": int(
            sum(max(_num(r, "cpu_samples"), _num(r, "memory_samples"), _num(r, "gpu_samples")) for r in records)
        ),
        "avg_cpu_cores": _ratio(sum(_num(r, "cpu_sum") for r in records), sum(_num(r, "cpu_samples") for r in records)),
        "peak_cpu_cores": max(peak_cpu) if peak_cpu else None,
        "avg_cpu_request_cores": _mean(_present(records, "cpu_request_cores")),
        "avg_cpu_limit_cores": _mean(_present(records, "cpu_limit_cores")),
        "avg_cpu_request_utilization_percent": _ratio(
            sum(_num(r, "cpu_sum") for r in cpu_req),
            sum(_num(r, "cpu_samples") * _num(r, "cpu_request_cores") for r in cpu_req),
            100.0,
        ),
        "avg_memory_bytes": _ratio(
            sum(_num(r, "memory_sum") for r in records), sum(_num(r, "memory_samples") for r in records)
        ),
        "peak_memory_bytes": int(max(peak_mem)) if peak_mem else None,
        "avg_memory_request_bytes": _mean(_present(records, "memory_request_bytes")),
        "avg_memory_limit_bytes": _mean(_present(records, "memory_limit_bytes")),
        "avg_memory_request_utilization_percent": _ratio(
            sum(_num(r, "memory_sum") for r in mem_req),
            sum(_num(r, "memory_samples") * _num(r, "memory_request_bytes") for r in mem_req),
            100.0,
        ),
        "requested_gpus": sum(gpus) if gpus else None,
    }


_DAILY_COMPUTE_KEYS = (
    "evaluations",
    "sampled_evaluations",
    "samples",
    "avg_cpu_cores",
    "peak_cpu_cores",
    "avg_cpu_request_cores",
    "avg_memory_bytes",
    "peak_memory_bytes",
    "avg_memory_request_bytes",
    "requested_gpus",
)


def compute_summary(records: list[dict[str, Any]], *, window_start: datetime, window_end: datetime) -> dict:
    """Aggregate resource usage overall, per runtime, and per day."""
    by_runtime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_runtime[str(record.get("runtime"))].append(record)
        by_day[record["occurred_at"].date()].append(record)
    runtimes = [{"runtime": runtime, **_compute(rows)} for runtime, rows in by_runtime.items()]
    runtimes.sort(key=lambda row: (-row["sampled_evaluations"], -row["evaluations"], row["runtime"]))
    timeline = []
    for day in _days(window_start, window_end):
        point = _compute(by_day.get(day, []))
        timeline.append({"date": day, **{key: point[key] for key in _DAILY_COMPUTE_KEYS}})
    return {
        "runtime": "all",
        **_compute(records),
        "runtimes": runtimes,
        "timeline": timeline,
        "gpu_utilization_available": any(_num(r, "gpu_samples") for r in records),
    }


def failure_summary(
    records: list[dict[str, Any]], *, window_start: datetime, window_end: datetime, examples_per_category: int
) -> dict:
    """Group failed evaluations by category with recent examples and a daily timeline."""
    failed = [r for r in records if r.get("status") == "failed"]
    failed.sort(key=lambda r: (r["occurred_at"], r["evaluation_id"]), reverse=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in failed:
        grouped[record.get("failure_category") or "other"].append(record)
    categories = [
        {
            "key": key,
            "count": len(rows),
            "examples": [
                {
                    "evaluation_id": r["evaluation_id"],
                    "evaluation_name": r.get("evaluation_name"),
                    "task_id": r.get("task_id"),
                    "owner_id": r.get("owner_id"),
                    "owner_label": r.get("owner_id"),
                    "runtime": r.get("runtime"),
                    "failure_code": r.get("failure_code"),
                    "detail": r.get("detail"),
                    "occurred_at": r["occurred_at"],
                }
                for r in rows[:examples_per_category]
            ],
        }
        for key, rows in grouped.items()
    ]
    categories.sort(key=lambda row: (-row["count"], row["key"]))
    daily: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for record in failed:
        daily[record["occurred_at"].date()].append(record)
    timeline = []
    for day in _days(window_start, window_end):
        counts: dict[str, int] = defaultdict(int)
        codes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record in daily.get(day, []):
            category = record.get("failure_category") or "other"
            counts[category] += 1
            codes[category][record.get("failure_code") or "unknown"] += 1
        timeline.append(
            {
                "date": day,
                "total": sum(counts.values()),
                "counts": dict(counts),
                "codes": {category: dict(by_code) for category, by_code in codes.items()},
            }
        )
    return {"total_failures": len(failed), "categories": categories, "timeline": timeline}


def _run_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = _present(records, "runtime_seconds")
    return {
        "total_runs": len(records),
        "succeeded_runs": sum(1 for r in records if r.get("status") == "succeeded"),
        "failed_runs": sum(1 for r in records if r.get("status") == "failed"),
        "cancelled_runs": sum(1 for r in records if r.get("status") == "cancelled"),
        "total_parallelism": int(sum(_num(r, "parallelism") for r in records)),
        "avg_runtime_seconds": _mean(runtimes),
        "max_runtime_seconds": max(runtimes) if runtimes else None,
    }


def usage_summary(records: list[dict[str, Any]], live: Mapping[str, Any], *, limit: int) -> dict:
    """Combine windowed terminal runs with live queued and active counts, per owner."""
    by_owner: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_owner[record.get("owner_id")].append(record)
    live_by_owner = {row["owner_id"]: row for row in live["actors"]}
    actors = []
    for owner_id in set(by_owner) | set(live_by_owner):
        rows = by_owner.get(owner_id, [])
        owner_live = live_by_owner.get(owner_id, {})
        stats = _run_stats(rows)
        queued, active = int(owner_live.get("queued_runs", 0)), int(owner_live.get("active_runs", 0))
        created = [r["created_at"] for r in rows if r.get("created_at")]
        if owner_live.get("last_run_at"):
            created.append(owner_live["last_run_at"])
        actors.append(
            {
                **stats,
                "owner_id": owner_id,
                "total_runs": stats["total_runs"] + queued + active,
                "queued_runs": queued,
                "active_runs": active,
                "total_parallelism": stats["total_parallelism"] + int(owner_live.get("total_parallelism", 0)),
                "last_run_at": max(created) if created else None,
            }
        )
    epoch = datetime.min.replace(tzinfo=UTC)
    actors.sort(key=lambda a: (a["total_runs"], a["last_run_at"] or epoch, a["owner_id"] or ""), reverse=True)
    stats = _run_stats(records)
    live_runs = int(live["queued_runs"]) + int(live["active_runs"])
    return {
        **stats,
        "total_runs": stats["total_runs"] + live_runs,
        "total_evaluation_jobs": stats["total_runs"] + live_runs,
        "total_tasks": int(live["total_tasks"]),
        "total_tasks_run": len({r.get("task_id") for r in records if r.get("task_id")}),
        "total_executions": int(sum(_num(r, "executions") for r in records)),
        "total_trials": int(sum(_num(r, "n_trials") for r in records)),
        "total_benchmark_runs": len({r.get("benchmark_run_id") for r in records if r.get("benchmark_run_id")}),
        "queued_runs": int(live["queued_runs"]),
        "active_runs": int(live["active_runs"]),
        "total_parallelism": stats["total_parallelism"] + int(live["total_parallelism"]),
        "actors": actors[:limit],
    }
