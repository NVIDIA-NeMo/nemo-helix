# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


def validate_scoped_egress_config(config: dict[str, Any]) -> None:
    """Require explicit public CIDR destinations for user-scoped sandbox egress."""
    if set(config) != {"egress"}:
        raise ValueError("scoped_egress network_policy_config must contain exactly an 'egress' list")
    egress = config.get("egress")
    if not isinstance(egress, list) or not egress:
        raise ValueError("scoped_egress requires at least one egress rule")
    for rule_index, rule in enumerate(egress):
        if not isinstance(rule, dict):
            raise ValueError("scoped_egress egress rules must be objects")
        if not set(rule) <= {"to", "ports"} or not isinstance(rule.get("to"), list):
            raise ValueError(
                f"scoped_egress rule {rule_index} must contain only a non-empty 'to' list and optional 'ports'"
            )
        destinations = rule["to"]
        if not destinations:
            raise ValueError(f"scoped_egress rule {rule_index} must declare a destination")
        for destination_index, destination in enumerate(destinations):
            if not isinstance(destination, dict) or set(destination) != {"ipBlock"}:
                raise ValueError(
                    "scoped_egress destinations must be explicit public ipBlock entries; "
                    "podSelector and namespaceSelector are not allowed"
                )
            ip_block = destination["ipBlock"]
            if not isinstance(ip_block, dict) or not set(ip_block) <= {"cidr", "except"}:
                raise ValueError(f"scoped_egress destination {rule_index}.{destination_index} has an invalid ipBlock")
            cidr = ip_block.get("cidr")
            if not isinstance(cidr, str):
                raise ValueError("scoped_egress ipBlock.cidr must be a string")
            try:
                network = ipaddress.ip_network(cidr, strict=True)
            except ValueError as exc:
                raise ValueError(f"invalid scoped_egress CIDR {cidr!r}") from exc
            if network.prefixlen == 0 or not network.is_global:
                raise ValueError(f"scoped_egress CIDR {cidr!r} must be a globally routable non-default route")
            excluded = ip_block.get("except", [])
            if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
                raise ValueError("scoped_egress ipBlock.except must be a list of CIDR strings")
            seen: set[str] = set()
            for excluded_cidr in excluded:
                try:
                    excluded_network = ipaddress.ip_network(excluded_cidr, strict=True)
                except ValueError as exc:
                    raise ValueError(f"invalid scoped_egress excluded CIDR {excluded_cidr!r}") from exc
                if excluded_network.version != network.version or not excluded_network.subnet_of(network):
                    raise ValueError(f"scoped_egress excluded CIDR {excluded_cidr!r} must be inside {cidr!r}")
                normalized = str(excluded_network)
                if normalized in seen:
                    raise ValueError(f"scoped_egress excluded CIDR {excluded_cidr!r} is duplicated")
                seen.add(normalized)


class HealthStatus(BaseModel):
    status: str


class ReadyzResponse(BaseModel):
    status: str
    checks: dict[str, str]


class ApiError(BaseModel):
    code: str
    message: str
    # Open-ended by design: each error code can attach its own structured
    # debugging context without widening every response model.
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ApiError


class ListEnvelope[T](BaseModel):
    data: list[T]
    next_cursor: str | None = None


class DeleteResponse(BaseModel):
    id: str
    deleted: bool = True


class StubEvaluationSummary(BaseModel):
    active: int = 0
    queued: int | None = None
    failed_24h: int | None = None
    succeeded_24h: int | None = None


class StubTaskSummary(BaseModel):
    ready: int = 0
    building: int = 0


class StubRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Stub endpoints intentionally tolerate extra fields until the backing
    # product/domain models replace them.
    id: str
    name: str = "stub"
    created_at: str
    stub: bool = True


class UserSummaryResponse(BaseModel):
    evaluations: StubEvaluationSummary
    tasks: StubTaskSummary | None = None
    recent: list[ActivityRecord] = Field(default_factory=list)
    stub: bool = False


class ActivityRecord(BaseModel):
    id: str
    kind: str
    name: str
    status: str | None = None
    created_at: datetime


class AdminUserRecord(BaseModel):
    id: str
    email: str | None = None
    username: str | None = None
    display_name: str | None = None
    created_at: datetime
    last_seen_at: datetime


class TeamSummaryResponse(BaseModel):
    team_id: str
    evaluations: StubEvaluationSummary
    stub: bool = True


class AdminUserSummaryResponse(BaseModel):
    user_id: str
    evaluations: StubEvaluationSummary
    tasks: StubTaskSummary | None = None
    stub: bool = False


class AdminCapacityResponse(BaseModel):
    active_runs: int = 0
    queued_runs: int = 0
    active_slots: int = 0
    cluster_limit: int
    per_user_limit: int
    stub: bool = False


class AdminUsageActor(BaseModel):
    owner_id: str | None = None
    email: str | None = None
    username: str | None = None
    display_name: str | None = None
    total_runs: int = 0
    queued_runs: int = 0
    active_runs: int = 0
    succeeded_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    total_parallelism: int = 0
    avg_runtime_seconds: float | None = None
    max_runtime_seconds: float | None = None
    last_run_at: datetime | None = None


class AdminUsageSummaryResponse(BaseModel):
    total_runs: int = 0
    total_tasks: int = 0
    total_tasks_run: int = 0
    total_evaluation_jobs: int = 0
    total_executions: int = 0
    total_trials: int = 0
    total_benchmark_runs: int = 0
    queued_runs: int = 0
    active_runs: int = 0
    succeeded_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    total_parallelism: int = 0
    avg_runtime_seconds: float | None = None
    max_runtime_seconds: float | None = None
    actors: list[AdminUsageActor] = Field(default_factory=list)
    stub: bool = False


class AdminComputeRuntime(BaseModel):
    runtime: str
    evaluations: int = 0
    sampled_evaluations: int = 0
    samples: int = 0
    avg_cpu_cores: float | None = None
    peak_cpu_cores: float | None = None
    avg_cpu_request_cores: float | None = None
    avg_cpu_limit_cores: float | None = None
    avg_cpu_request_utilization_percent: float | None = None
    avg_memory_bytes: float | None = None
    peak_memory_bytes: int | None = None
    avg_memory_request_bytes: float | None = None
    avg_memory_limit_bytes: float | None = None
    avg_memory_request_utilization_percent: float | None = None
    requested_gpus: float | None = None


class AdminComputeDailyPoint(BaseModel):
    date: date
    evaluations: int = 0
    sampled_evaluations: int = 0
    samples: int = 0
    avg_cpu_cores: float | None = None
    peak_cpu_cores: float | None = None
    avg_cpu_request_cores: float | None = None
    avg_memory_bytes: float | None = None
    peak_memory_bytes: int | None = None
    avg_memory_request_bytes: float | None = None
    requested_gpus: float | None = None


class AdminComputeSummaryResponse(AdminComputeRuntime):
    runtime: str = "all"
    window_days: int
    window_start: datetime
    window_end: datetime
    runtimes: list[AdminComputeRuntime] = Field(default_factory=list)
    timeline: list[AdminComputeDailyPoint] = Field(default_factory=list)
    gpu_utilization_available: bool = False


class AdminFailureExample(BaseModel):
    evaluation_id: str
    evaluation_name: str
    task_id: str
    owner_id: str | None = None
    owner_label: str | None = None
    runtime: str
    failure_code: str | None = None
    detail: str | None = None
    occurred_at: datetime


class AdminFailureCategory(BaseModel):
    key: str
    label: str
    description: str
    count: int = 0
    examples: list[AdminFailureExample] = Field(default_factory=list)


class AdminFailureDailyPoint(BaseModel):
    date: date
    total: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    codes: dict[str, dict[str, int]] = Field(default_factory=dict)


class AdminFailureSummaryResponse(BaseModel):
    window_days: int
    window_start: datetime
    window_end: datetime
    total_failures: int = 0
    categories: list[AdminFailureCategory] = Field(default_factory=list)
    timeline: list[AdminFailureDailyPoint] = Field(default_factory=list)


@dataclass(frozen=True)
class CursorPosition:
    created_at: datetime
    id: str


def _invalid_cursor() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"error": {"code": "invalid_cursor", "message": "invalid cursor", "details": {}}},
    )


def encode_cursor(created_at: datetime, item_id: str) -> str:
    payload = json.dumps(
        {"created_at": created_at.isoformat(), "id": item_id},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> CursorPosition | None:
    if cursor is None:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        created_at = payload["created_at"]
        item_id = payload["id"]
        if not isinstance(created_at, str) or not isinstance(item_id, str) or not item_id:
            raise ValueError
        return CursorPosition(created_at=datetime.fromisoformat(created_at), id=item_id)
    except (binascii.Error, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise _invalid_cursor() from None


def page_from_rows[T](rows: list[dict[str, Any]], limit: int, model: type[T]) -> ListEnvelope[T]:
    page_rows = rows[:limit]
    next_cursor = None
    if len(rows) > limit and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(last["created_at"], last["id"])
    return ListEnvelope(data=[model(**row) for row in page_rows], next_cursor=next_cursor)
