# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from scaled_evals.api.auth import CurrentPrincipal
from scaled_evals.api.build.task_image_identity import (
    TaskImageIdentityError,
    validate_task_image_request,
)
from scaled_evals.api.db import Database
from scaled_evals.api.framework_versions import (
    ResolvedFrameworkRunner,
    resolve_framework_runner,
)
from scaled_evals.api.repositories.base_repository import InvalidReference
from scaled_evals.api.schemas.benchmark_runs import CreateBenchmarkRunRequest
from scaled_evals.api.schemas.evaluations import CreateEvaluationRequest
from scaled_evals.api.schemas.runnability import (
    BenchmarkMemberFailure,
    BenchmarkMemberSummary,
    RunnabilityCheck,
    RunnabilityReport,
)
from scaled_evals.api.settings import settings
from scaled_evals.api.tenancy import is_admin
from scaled_evals.dispatch.harbor_dataset_images import effective_image_mode
from scaled_evals.dispatch.registry import get_backend_capabilities

ObjectExists = Callable[[str], bool]
BundleResolver = Callable[[Database, CurrentPrincipal, str], dict[str, Any] | None]
_MAX_MEMBER_FAILURES = 50


@dataclass(frozen=True)
class PreflightBlocker:
    status_code: int
    code: str
    message: str


@dataclass(frozen=True)
class BlockedPreflight:
    report: RunnabilityReport
    blocker: PreflightBlocker


@dataclass(frozen=True)
class ReadyEvaluationPreflight:
    report: RunnabilityReport
    runner: ResolvedFrameworkRunner
    task_revision: dict[str, Any]
    runner_metadata: dict[str, Any]


@dataclass(frozen=True)
class ReadyBenchmarkRunPreflight:
    report: RunnabilityReport
    runner: ResolvedFrameworkRunner
    revision: int
    members: list[dict[str, Any]]
    runner_metadata: dict[str, Any]


EvaluationPreflight = BlockedPreflight | ReadyEvaluationPreflight
BenchmarkRunPreflight = BlockedPreflight | ReadyBenchmarkRunPreflight


def _check(
    prerequisite: str,
    state: str,
    message: str,
    *,
    blocking: bool = False,
    code: str | None = None,
    details: dict[str, Any] | None = None,
) -> RunnabilityCheck:
    return RunnabilityCheck(
        prerequisite=prerequisite,
        state=state,
        blocking=blocking,
        code=code,
        message=message,
        details=details or {},
    )


def _report(
    kind: str,
    checks: list[RunnabilityCheck],
    member_summary: BenchmarkMemberSummary | None = None,
) -> RunnabilityReport:
    return RunnabilityReport(
        kind=kind,
        runnable=not any(item.blocking for item in checks),
        checked_at=datetime.now(UTC),
        checks=checks,
        member_summary=member_summary,
    )


def _blocking(
    kind: str,
    checks: list[RunnabilityCheck],
    *,
    status_code: int,
    code: str,
    message: str,
    member_summary: BenchmarkMemberSummary | None = None,
) -> tuple[RunnabilityReport, PreflightBlocker]:
    return (
        _report(kind, checks, member_summary),
        PreflightBlocker(status_code=status_code, code=code, message=message),
    )


def _framework_profile_type(framework: str) -> str:
    return "gym" if framework == "nemo_gym" else framework


def _reference_shape_error(body: CreateEvaluationRequest | CreateBenchmarkRunRequest) -> str | None:
    for field_name, value in (
        ("framework_profile_id", body.framework_profile_id),
        ("harbor_profile_id", body.harbor_profile_id),
        ("intake_profile_id", body.intake_profile_id),
    ):
        if value is not None and not value.startswith("cfg_"):
            return f"{field_name} must be a cfg_ id"
    if isinstance(body, CreateBenchmarkRunRequest):
        for task_id, profile_id in body.member_framework_profile_ids.items():
            if not task_id.startswith("task_"):
                return f"member_framework_profile_ids key must be a task_ id: {task_id}"
            if not profile_id.startswith("cfg_"):
                return f"member_framework_profile_ids[{task_id}] must be a cfg_ id"
    for role, credential_id in body.credentials.items():
        if not credential_id.startswith("cred_"):
            return f"credentials[{role}] must be a cred_ id"
    return None


def _append_reference_checks(
    db: Database,
    body: CreateEvaluationRequest | CreateBenchmarkRunRequest,
    current: CurrentPrincipal,
    checks: list[RunnabilityCheck],
) -> PreflightBlocker | None:
    profile_slots = [
        (body.framework_profile_id, _framework_profile_type(body.framework)),
        (body.intake_profile_id, "intake"),
    ]
    if isinstance(body, CreateBenchmarkRunRequest):
        profile_slots.extend(
            (profile_id, _framework_profile_type(body.framework))
            for profile_id in body.member_framework_profile_ids.values()
        )
    profile_slots = [(profile_id, kind) for profile_id, kind in profile_slots if profile_id]
    try:
        db.evaluations.validate_profile_references(profile_slots)
    except InvalidReference as exc:
        checks.append(
            _check(
                "config_profiles",
                "unavailable",
                exc.message,
                blocking=True,
                code="invalid_reference",
            )
        )
        return PreflightBlocker(422, "invalid_reference", exc.message)
    checks.append(
        _check(
            "config_profiles",
            "ready" if profile_slots else "not_applicable",
            "referenced config profiles are accessible" if profile_slots else "no config profiles were requested",
        )
    )

    framework_profile_ids = {body.framework_profile_id} if body.framework_profile_id else set()
    if isinstance(body, CreateBenchmarkRunRequest):
        framework_profile_ids.update(body.member_framework_profile_ids.values())
    for framework_profile_id in sorted(framework_profile_ids) if body.framework == "harbor" else ():
        profile_row = db.evaluations.load_framework_profile(framework_profile_id)
        profile = profile_row.get("config") if profile_row else None
        if isinstance(profile, dict) and profile.get("dataset_only") is True:
            try:
                effective_image_mode(profile)
            except ValueError as exc:
                message = str(exc)
                checks.append(
                    _check(
                        "harbor_dataset_images",
                        "incompatible",
                        message,
                        blocking=True,
                        code="managed_dataset_images_required",
                    )
                )
                return PreflightBlocker(422, "managed_dataset_images_required", message)
            if (
                not settings.image_builder_service_url
                and not settings.cloud_build_enabled
                and not settings.buildkit_enabled
            ):
                message = "managed Harbor dataset images require an enabled image builder"
                checks.append(
                    _check(
                        "harbor_dataset_images",
                        "unavailable",
                        message,
                        blocking=True,
                        code="managed_dataset_builder_unavailable",
                    )
                )
                return PreflightBlocker(503, "managed_dataset_builder_unavailable", message)
            checks.append(
                _check(
                    "harbor_dataset_images",
                    "ready",
                    "dataset task images will be materialized through managed task revisions",
                )
            )

    credential_ids = sorted(set(body.credentials.values()))
    try:
        db.evaluations.validate_credential_references(
            credential_ids,
            owner_id=current.owner_id,
            include_unowned=current.source == "disabled" or is_admin(current),
        )
    except InvalidReference as exc:
        checks.append(
            _check(
                "credentials",
                "unavailable",
                exc.message,
                blocking=True,
                code="invalid_reference",
            )
        )
        return PreflightBlocker(422, "invalid_reference", exc.message)
    checks.append(
        _check(
            "credentials",
            "ready" if credential_ids else "not_applicable",
            "referenced credentials are accessible" if credential_ids else "no credentials were requested",
            details={"roles": sorted(body.credentials)},
        )
    )
    if credential_ids:
        checks.append(
            _check(
                "credential_verification",
                "unverified",
                "credential usability is not persisted and was not probed by this preflight",
                details={"roles": sorted(body.credentials)},
            )
        )
    return None


def _append_runtime_checks(
    body: CreateEvaluationRequest | CreateBenchmarkRunRequest,
    checks: list[RunnabilityCheck],
) -> tuple[ResolvedFrameworkRunner | None, PreflightBlocker | None]:
    try:
        capabilities = get_backend_capabilities(body.runtime)
    except ValueError as exc:
        message = str(exc)
        checks.append(
            _check(
                "runtime_network_policy",
                "incompatible",
                message,
                blocking=True,
                code="unsupported_network_policy",
            )
        )
        return None, PreflightBlocker(422, "unsupported_network_policy", message)
    if body.network_policy not in capabilities.supported_network_policies:
        message = f"runtime {body.runtime!r} does not support {body.network_policy!r} network policy"
        checks.append(
            _check(
                "runtime_network_policy",
                "incompatible",
                message,
                blocking=True,
                code="unsupported_network_policy",
            )
        )
        return None, PreflightBlocker(422, "unsupported_network_policy", message)
    checks.append(
        _check(
            "runtime_network_policy",
            "ready",
            "runtime supports the requested network policy",
            details={"runtime": body.runtime, "network_policy": body.network_policy},
        )
    )
    try:
        runner = resolve_framework_runner(
            body.framework,
            body.framework_version,
            runtime=body.runtime,
        )
    except ValueError as exc:
        message = str(exc)
        checks.append(
            _check(
                "framework_runner",
                "incompatible",
                message,
                blocking=True,
                code="unsupported_framework_version",
            )
        )
        return None, PreflightBlocker(422, "unsupported_framework_version", message)
    checks.append(
        _check(
            "framework_runner",
            "ready",
            "framework runner resolved",
            details={"framework": body.framework, "version": runner.version},
        )
    )
    qualification = runner.metadata.get("qualification")
    checks.append(
        _check(
            "framework_qualification",
            "ready" if qualification else "unverified",
            "framework runner includes qualification evidence"
            if qualification
            else "framework runner has no qualification evidence",
            details={"evidence_present": bool(qualification)},
        )
    )
    return runner, None


def _append_task_pack_check(
    row: dict[str, Any],
    checks: list[RunnabilityCheck],
    object_exists: ObjectExists,
    *,
    member_prefix: str = "",
) -> PreflightBlocker | None:
    object_key = row.get("tarball_object_key")
    if not object_key:
        checks.append(_check("task_pack", "not_applicable", f"{member_prefix}uses no task pack"))
        return None
    try:
        exists = object_exists(str(object_key))
    except Exception:  # noqa: BLE001 - report dependency failure without leaking internals
        message = (
            "could not verify member task-pack objects before queuing this benchmark; retry later"
            if member_prefix
            else "could not verify the task pack object before queuing this evaluation; retry later"
        )
        checks.append(
            _check(
                "task_pack",
                "unavailable",
                message,
                blocking=True,
                code="object_store_unavailable",
            )
        )
        return PreflightBlocker(503, "object_store_unavailable", message)
    if not exists:
        message = (
            f"{member_prefix}is not runnable because its task-pack object is missing"
            if member_prefix
            else "task revision is not runnable because its task pack object is missing"
        )
        checks.append(
            _check(
                "task_pack",
                "unavailable",
                message,
                blocking=True,
                code="task_object_missing",
            )
        )
        return PreflightBlocker(409, "task_object_missing", message)
    checks.append(_check("task_pack", "ready", f"{member_prefix}task pack object exists"))
    return None


def _append_task_image_check(
    row: dict[str, Any],
    checks: list[RunnabilityCheck],
    *,
    member_prefix: str = "",
) -> PreflightBlocker | None:
    image_ref = row.get("image_ref")
    if settings.task_image_validation_mode == "disabled" or not image_ref:
        checks.append(
            _check(
                "task_image_identity",
                "not_applicable" if not image_ref else "unverified",
                f"{member_prefix}task image identity validation is not applicable"
                if not image_ref
                else f"{member_prefix}task image identity validation is disabled",
            )
        )
        return None
    try:
        validate_task_image_request(image_ref, row.get("image_digest"))
    except TaskImageIdentityError as exc:
        message = f"{member_prefix}{exc}"
        checks.append(
            _check(
                "task_image_identity",
                "incompatible",
                message,
                blocking=True,
                code="invalid_task_image",
            )
        )
        return PreflightBlocker(422, "invalid_task_image", message)
    checks.append(
        _check(
            "task_image_identity",
            "ready",
            f"{member_prefix}task image identity satisfies request policy",
        )
    )
    return None


def _append_bundle_check(
    db: Database,
    current: CurrentPrincipal,
    bundle_id: str | None,
    checks: list[RunnabilityCheck],
    resolve_bundle: BundleResolver,
    runner_metadata: dict[str, Any],
) -> PreflightBlocker | None:
    if bundle_id is None:
        checks.append(_check("agent_bundle", "not_applicable", "no agent bundle was requested"))
        return None
    bundle = resolve_bundle(db, current, bundle_id)
    if bundle is None:
        message = "agent bundle not found or inaccessible"
        checks.append(
            _check(
                "agent_bundle",
                "unavailable",
                message,
                blocking=True,
                code="agent_bundle_not_found",
            )
        )
        return PreflightBlocker(404, "agent_bundle_not_found", message)
    runner_metadata["agent_bundle"] = bundle
    checks.append(_check("agent_bundle", "ready", "agent bundle is accessible"))
    qualification = str(bundle.get("qualification_status") or "unverified")
    checks.append(
        _check(
            "agent_bundle_qualification",
            "ready" if qualification == "qualified" else "unverified",
            f"agent bundle qualification status is {qualification!r}",
            details={"qualification_status": qualification},
        )
    )
    return None


def _append_advisory_checks(checks: list[RunnabilityCheck]) -> None:
    checks.extend(
        [
            _check(
                "runtime_availability",
                "unverified",
                "runtime health is not probed by deterministic preflight",
            ),
            _check(
                "target_image_admission",
                "unverified",
                "target-specific image admission is not proven by image identity metadata",
            ),
        ]
    )


def preflight_evaluation(
    db: Database,
    body: CreateEvaluationRequest,
    current: CurrentPrincipal,
    *,
    object_exists: ObjectExists,
    resolve_bundle: BundleResolver,
) -> EvaluationPreflight:
    checks: list[RunnabilityCheck] = []
    shape_error = _reference_shape_error(body)
    if shape_error:
        checks.append(
            _check(
                "reference_shape",
                "incompatible",
                shape_error,
                blocking=True,
                code="invalid_reference",
            )
        )
        report, blocker = _blocking(
            "evaluation", checks, status_code=422, code="invalid_reference", message=shape_error
        )
        return BlockedPreflight(report, blocker)
    checks.append(_check("reference_shape", "ready", "reference ids are well formed"))

    runner, blocker = _append_runtime_checks(body, checks)
    if blocker:
        return BlockedPreflight(_report("evaluation", checks), blocker)

    task_revision = db.evaluations.task_revision_for_evaluation(body.task_id, body.task_revision)
    if task_revision is None:
        message = f"task revision not found: {body.task_id} rev {body.task_revision}"
        checks.append(
            _check(
                "task_revision",
                "unavailable",
                message,
                blocking=True,
                code="not_found",
            )
        )
        return BlockedPreflight(_report("evaluation", checks), PreflightBlocker(404, "not_found", message))
    status = str(task_revision["status"])
    if status != "ready":
        message = f"task revision is '{status}', must be 'ready'"
        checks.append(
            _check(
                "task_revision",
                "unavailable",
                message,
                blocking=True,
                code="task_not_ready",
            )
        )
        return BlockedPreflight(
            _report("evaluation", checks),
            PreflightBlocker(409, "task_not_ready", message),
        )
    checks.append(_check("task_revision", "ready", "task revision is ready"))

    blocker = _append_task_pack_check(task_revision, checks, object_exists)
    if blocker:
        return BlockedPreflight(_report("evaluation", checks), blocker)
    blocker = _append_task_image_check(task_revision, checks)
    if blocker:
        return BlockedPreflight(_report("evaluation", checks), blocker)
    blocker = _append_reference_checks(db, body, current, checks)
    if blocker:
        return BlockedPreflight(_report("evaluation", checks), blocker)

    runner_metadata = dict(runner.metadata)
    blocker = _append_bundle_check(db, current, body.agent_bundle_id, checks, resolve_bundle, runner_metadata)
    if blocker:
        return BlockedPreflight(_report("evaluation", checks), blocker)
    _append_advisory_checks(checks)
    return ReadyEvaluationPreflight(_report("evaluation", checks), runner, task_revision, runner_metadata)


def preflight_benchmark_run(
    db: Database,
    body: CreateBenchmarkRunRequest,
    current: CurrentPrincipal,
    *,
    object_exists: ObjectExists,
    resolve_bundle: BundleResolver,
) -> BenchmarkRunPreflight:
    checks: list[RunnabilityCheck] = []
    shape_error = _reference_shape_error(body)
    if shape_error:
        checks.append(
            _check(
                "reference_shape",
                "incompatible",
                shape_error,
                blocking=True,
                code="invalid_reference",
            )
        )
        return BlockedPreflight(
            _report("benchmark_run", checks),
            PreflightBlocker(422, "invalid_reference", shape_error),
        )
    checks.append(_check("reference_shape", "ready", "reference ids are well formed"))

    blocker = _append_reference_checks(db, body, current, checks)
    if blocker:
        return BlockedPreflight(_report("benchmark_run", checks), blocker)
    runner, blocker = _append_runtime_checks(body, checks)
    if blocker:
        return BlockedPreflight(_report("benchmark_run", checks), blocker)

    benchmark_revision = db.benchmark_runs.benchmark_revision_for_run(body.benchmark_id, body.benchmark_revision)
    if benchmark_revision is None:
        target = (
            f"{body.benchmark_id} rev {body.benchmark_revision}"
            if body.benchmark_revision is not None
            else body.benchmark_id
        )
        message = f"benchmark revision not found: {target}"
        checks.append(
            _check(
                "benchmark_revision",
                "unavailable",
                message,
                blocking=True,
                code="not_found",
            )
        )
        return BlockedPreflight(_report("benchmark_run", checks), PreflightBlocker(404, "not_found", message))
    revision = int(benchmark_revision["revision"])
    checks.append(
        _check(
            "benchmark_revision",
            "ready",
            "benchmark revision resolved",
            details={"benchmark_id": body.benchmark_id, "revision": revision},
        )
    )

    members = db.benchmark_runs.load_members(body.benchmark_id, revision)
    if not members:
        message = "benchmark revision has no member tasks"
        checks.append(
            _check(
                "benchmark_members",
                "incompatible",
                message,
                blocking=True,
                code="invalid_reference",
            )
        )
        summary = BenchmarkMemberSummary(total=0, ready=0, blocked=0)
        return BlockedPreflight(
            _report("benchmark_run", checks, summary),
            PreflightBlocker(422, "invalid_reference", message),
        )
    unknown_profile_tasks = sorted(
        set(body.member_framework_profile_ids) - {str(member["task_id"]) for member in members}
    )
    if unknown_profile_tasks:
        message = "member framework profile override references task(s) outside the benchmark: " + ", ".join(
            unknown_profile_tasks
        )
        checks.append(
            _check(
                "benchmark_members",
                "incompatible",
                message,
                blocking=True,
                code="invalid_reference",
            )
        )
        summary = BenchmarkMemberSummary(total=len(members), ready=0, blocked=len(members))
        return BlockedPreflight(
            _report("benchmark_run", checks, summary),
            PreflightBlocker(422, "invalid_reference", message),
        )

    failures: list[BenchmarkMemberFailure] = []
    first_blocker: PreflightBlocker | None = None
    blocked_count = 0
    for member in members:
        member_revision = member.get("task_revision")
        status = member.get("revision_status")
        prefix = f"member task {member['task_id']} rev {member_revision}: "
        if member_revision is None or status is None:
            message = f"member task has no revision to run: {member['task_id']}"
            candidate = PreflightBlocker(404, "not_found", message)
            prerequisite = "task_revision"
        elif status != "ready":
            message = f"member task {member['task_id']} rev {member_revision} is '{status}', must be 'ready'"
            candidate = PreflightBlocker(409, "task_not_ready", message)
            prerequisite = "task_revision"
        else:
            member_checks: list[RunnabilityCheck] = []
            candidate = _append_task_pack_check(member, member_checks, object_exists, member_prefix=prefix)
            if candidate is None:
                candidate = _append_task_image_check(member, member_checks, member_prefix=prefix)
            prerequisite = member_checks[-1].prerequisite
            message = candidate.message if candidate else ""
        if candidate is None:
            continue
        blocked_count += 1
        first_blocker = first_blocker or candidate
        if len(failures) < _MAX_MEMBER_FAILURES:
            failures.append(
                BenchmarkMemberFailure(
                    task_id=str(member["task_id"]),
                    task_revision=member_revision,
                    prerequisite=prerequisite,
                    code=candidate.code,
                    message=message,
                )
            )

    summary = BenchmarkMemberSummary(
        total=len(members),
        ready=len(members) - blocked_count,
        blocked=blocked_count,
        failures=failures,
        failures_truncated=blocked_count > len(failures),
    )
    if first_blocker:
        checks.append(
            _check(
                "benchmark_members",
                "unavailable",
                f"{summary.blocked} benchmark member(s) are not runnable",
                blocking=True,
                code=first_blocker.code,
            )
        )
        return BlockedPreflight(
            _report("benchmark_run", checks, summary),
            first_blocker,
        )
    checks.append(
        _check(
            "benchmark_members",
            "ready",
            f"all {len(members)} benchmark members are runnable",
        )
    )
    summary = BenchmarkMemberSummary(total=len(members), ready=len(members), blocked=0)

    runner_metadata = dict(runner.metadata)
    blocker = _append_bundle_check(db, current, body.agent_bundle_id, checks, resolve_bundle, runner_metadata)
    if blocker:
        return BlockedPreflight(
            _report("benchmark_run", checks, summary),
            blocker,
        )
    qualification_status = str(benchmark_revision.get("qualification_status") or "unverified")
    qualification_evidence = benchmark_revision.get("qualification_evidence") or {}
    checks.append(
        _check(
            "benchmark_qualification",
            "ready" if qualification_status == "qualified" else "unverified",
            f"benchmark qualification status is {qualification_status!r}; it is advisory",
            details={
                "qualification_status": qualification_status,
                "evidence_present": bool(qualification_evidence),
                "qualified_at": benchmark_revision.get("qualified_at"),
            },
        )
    )
    _append_advisory_checks(checks)
    return ReadyBenchmarkRunPreflight(
        _report("benchmark_run", checks, summary),
        runner,
        revision,
        members,
        runner_metadata,
    )
