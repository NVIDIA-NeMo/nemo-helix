# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from scaled_evals.api.schemas.common import validate_scoped_egress_config
from scaled_evals.api.schemas.tasks import Visibility
from scaled_evals.models.evaluations import RewardValue

# Mirrors the evaluation_status enum in db/schema/evaluations.sql.
EvaluationStatus = Literal[
    "blocked",
    "queued",
    "provisioning",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]
ArchiveStatus = Literal["missing", "building", "ready"]

Framework = Literal["harbor", "nemo_gym"]
NetworkPolicyMode = Literal["unrestricted", "default_deny", "scoped_egress"]
OutcomeCategory = Literal[
    "in_progress",
    "completed",
    "completed_zero_reward",
    "completed_with_failed_solves",
    "trial_errors",
    "infrastructure_failed",
    "cancelled",
]


class EvaluationOutcomeSummary(BaseModel):
    category: OutcomeCategory
    reward: RewardValue | None = None
    n_trials: int | None = None
    n_completed: int | None = None
    n_errored: int | None = None
    n_failed_solve: int | None = None
    exception_counts: dict[str, int] = Field(default_factory=dict)


def _outcome_from_row(value: dict[str, Any]) -> EvaluationOutcomeSummary:
    status = value.get("status")
    reward = value.get("reward")
    exceptions = value.get("exception_counts") or {}
    n_errored = value.get("n_errored")
    n_failed_solve = value.get("n_failed_solve")
    detail = str(value.get("status_detail") or "").lower()
    legacy_trial_error = "trials errored" in detail or "rollouts errored" in detail
    if status not in {"succeeded", "failed", "cancelled"}:
        category: OutcomeCategory = "in_progress"
    elif status == "cancelled":
        category = "cancelled"
    elif (isinstance(n_errored, int) and n_errored > 0) or exceptions or legacy_trial_error:
        category = "trial_errors"
    elif status == "failed":
        category = "infrastructure_failed"
    elif reward is False or (isinstance(reward, int | float) and reward == 0):
        category = "completed_zero_reward"
    elif isinstance(n_failed_solve, int) and n_failed_solve > 0:
        category = "completed_with_failed_solves"
    else:
        category = "completed"
    return EvaluationOutcomeSummary(
        category=category,
        reward=reward,
        n_trials=value.get("n_trials"),
        n_completed=value.get("n_completed"),
        n_errored=n_errored,
        n_failed_solve=n_failed_solve,
        exception_counts=exceptions,
    )


# Hypermedia links returned on single-evaluation responses.
class EvaluationLinks(BaseModel):
    self: str
    logs: str
    logs_stream: str
    events_stream: str
    telemetry: str
    artifacts: str
    provenance: str
    sbom: str
    harbor_viewer: str | None = None
    harbor_viewer_archive: str | None = None
    harbor_viewer_upload: str | None = None
    reproduce: str
    retry: str
    archive: str
    cancel: str


# Request body: POST /v1/evaluations
class CreateEvaluationRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    # An evaluation runs one task revision. To aggregate one evaluation per
    # benchmark member task, use POST /benchmark-runs instead.
    task_id: str = Field(min_length=1)
    task_revision: int = Field(ge=1)
    framework: Framework = Field(
        default="harbor",
        description=("Evaluation framework runner. Use 'harbor' for Harbor packs and 'nemo_gym' for NeMo Gym packs."),
    )
    framework_version: str | None = Field(
        default=None,
        description=(
            "Exact supported framework version or a documented alias such as 'stable'. "
            "Omitted Harbor versions resolve to the service default before queueing."
        ),
    )
    framework_profile_id: str | None = Field(
        default=None,
        description=(
            "Generic framework config profile id. Must reference a live "
            "'harbor' profile when framework='harbor' and a live 'gym' "
            "profile when framework='nemo_gym'."
        ),
    )
    harbor_profile_id: str | None = Field(
        default=None,
        description=(
            "Compatibility alias for framework_profile_id on Harbor requests "
            "only. If both fields are supplied for framework='harbor', they "
            "must match."
        ),
    )
    intake_profile_id: str | None = Field(
        default=None,
        description="Optional intake config profile id.",
    )
    # role -> credential id, e.g. {"anthropic": "cred_…", "intake": "cred_…"}.
    credentials: dict[str, str] = Field(default_factory=dict)
    agent_bundle_id: str | None = Field(
        default=None,
        description=(
            "Optional accessible agent-bundle id. The service snapshots its immutable "
            "identity and provenance independently of the task image."
        ),
    )
    extra_skill_object_keys: list[str] = Field(default_factory=list)
    instruction_prefix: str | None = None
    instruction_postfix: str | None = None
    initial_user_turns: list[str] = Field(default_factory=list)
    # Selects the dispatch RuntimeBackend (see scaled_evals.dispatch.runtime_backend).
    runtime: str = "sandbox_k8s"
    network_policy: NetworkPolicyMode = Field(
        default="unrestricted",
        description=("Direct sandbox egress policy."),
    )
    network_policy_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider policy details for scoped_egress. For sandbox_k8s this is a "
            "Kubernetes NetworkPolicy spec fragment containing egress rules."
        ),
    )
    n_attempts: int = Field(default=1, ge=1, le=256)
    # One evaluation cannot consume more than the per-user sandbox-slot ceiling.
    # Larger suites remain supported as queued benchmark members.
    parallelism: int = Field(default=1, ge=1, le=50)
    visibility: Visibility = "private"

    @field_validator("initial_user_turns")
    @classmethod
    def _validate_initial_user_turns(cls, turns: list[str]) -> list[str]:
        if any(not turn.strip() for turn in turns):
            raise ValueError("initial_user_turns must not contain blank turns")
        return turns

    @model_validator(mode="after")
    def _resolve_framework_profile_alias(self) -> "CreateEvaluationRequest":
        if self.agent_bundle_id is not None and (self.framework != "harbor" or self.runtime != "sandbox_k8s"):
            raise ValueError("agent_bundle currently requires framework='harbor' and runtime='sandbox_k8s'")
        if self.agent_bundle_id is not None and self.framework_profile_id is None:
            raise ValueError("agent_bundle_id requires a Harbor framework_profile_id")
        if self.initial_user_turns and self.framework != "harbor":
            raise ValueError('initial_user_turns is only valid for framework="harbor"')
        if self.framework == "harbor":
            if self.framework_profile_id is not None and self.harbor_profile_id is not None:
                if self.framework_profile_id != self.harbor_profile_id:
                    raise ValueError('framework_profile_id and harbor_profile_id must match for framework="harbor"')
            elif self.framework_profile_id is None:
                self.framework_profile_id = self.harbor_profile_id
            else:
                self.harbor_profile_id = self.framework_profile_id
        elif self.harbor_profile_id is not None:
            raise ValueError('harbor_profile_id is only valid for framework="harbor"')
        self._validate_network_policy_config()
        return self

    def _validate_network_policy_config(self) -> None:
        if self.network_policy != "scoped_egress":
            if self.network_policy_config:
                raise ValueError("network_policy_config is only valid with network_policy='scoped_egress'")
            return
        validate_scoped_egress_config(self.network_policy_config)


class ReproduceEvaluationResponse(BaseModel):
    evaluation_id: str
    source_status: EvaluationStatus
    request: CreateEvaluationRequest
    cli_command: list[str]
    notes: list[str] = Field(default_factory=list)


# Response: GET /v1/evaluations/{id} and list items (the DB row).
class Evaluation(BaseModel):
    id: str
    name: str
    framework: str = Field(description="Evaluation framework runner, for example 'harbor' or 'nemo_gym'.")
    requested_framework_version: str | None = None
    framework_version: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    framework_adapter_version: str | None = None
    sandbox_k8s_version: str | None = None
    runner_metadata: dict[str, Any] = Field(default_factory=dict)
    task_id: str
    task_revision: int
    # Set when this evaluation is a member of a benchmark run. Null for
    # standalone single-task runs.
    benchmark_run_id: str | None = None
    framework_profile_id: str | None = Field(
        default=None,
        description="Generic framework config profile id used by this evaluation.",
    )
    harbor_profile_id: str | None = Field(description="Deprecated Harbor compatibility alias for framework_profile_id.")
    intake_profile_id: str | None
    credentials: dict[str, str]
    runtime: str
    network_policy: NetworkPolicyMode = "unrestricted"
    network_policy_config: dict[str, Any] = Field(default_factory=dict)
    n_attempts: int
    parallelism: int
    visibility: Visibility
    status: EvaluationStatus
    status_detail: str | None
    cancel_teardown_status: Literal["not_requested", "pending", "succeeded", "failed"] = "not_requested"
    cancel_teardown_error: str | None = None
    cancel_teardown_updated_at: datetime | None = None
    current_execution: int = Field(default=1, ge=1)
    max_executions: int = Field(default=3, ge=1)
    infrastructure_retries: int = Field(default=0, ge=0)
    max_infrastructure_retries: int = Field(default=2, ge=0)
    next_retry_at: datetime | None = None
    last_failure_code: str | None = None
    last_failure_category: (
        Literal["infrastructure", "provider", "task", "unknown", "retryable_task", "non_retryable"] | None
    ) = None
    # Result summary: reward preserves the backend's JSON scalar type
    # (bool/int/float/str). Existing Harbor/Gym rewards remain numeric. The DB
    # keeps a separate numeric projection for legacy query paths.
    reward: RewardValue | None = None
    n_trials: int | None = None
    n_completed: int | None = None
    n_errored: int | None = None
    n_failed_solve: int | None = None
    exception_counts: dict[str, int] = Field(default_factory=dict)
    outcome: EvaluationOutcomeSummary
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _project_typed_reward(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        reward_value = value.get("reward_value")
        projected = dict(value)
        if reward_value is not None:
            projected["reward"] = reward_value
        projected["outcome"] = _outcome_from_row(projected)
        return projected


# Single-evaluation responses add hypermedia links and the full framework-typed
# result envelope (the Harbor result.json). None until the run is terminal;
# omitted on list items (Evaluation) to keep the listing lightweight.
class EvaluationResponse(Evaluation):
    links: EvaluationLinks
    result: dict[str, Any] | None = None


class EvaluationLogResponse(BaseModel):
    evaluation_id: str
    lines: list[str]
    status: EvaluationStatus
    complete: bool


class EvaluationEvent(BaseModel):
    evaluation_id: str
    type: str
    status: EvaluationStatus
    detail: str | None = None
    at: str | None = None


class EvaluationResourceUsage(BaseModel):
    execution_number: int = Field(ge=1)
    component: str
    source: str
    collection_status: str
    collection_error: str | None = None
    sample_count: int = Field(ge=0)
    first_observed_at: datetime
    last_observed_at: datetime
    cpu_sample_count: int = Field(ge=0)
    avg_cpu_cores: float | None = Field(default=None, ge=0)
    peak_cpu_cores: float | None = Field(default=None, ge=0)
    memory_sample_count: int = Field(ge=0)
    avg_memory_bytes: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    cpu_request_cores: float | None = Field(default=None, ge=0)
    cpu_limit_cores: float | None = Field(default=None, ge=0)
    memory_request_bytes: int | None = Field(default=None, ge=0)
    memory_limit_bytes: int | None = Field(default=None, ge=0)
    gpu_request: float | None = Field(default=None, ge=0)
    gpu_sample_count: int = Field(ge=0)
    avg_gpu_usage_percent: float | None = Field(default=None, ge=0)
    peak_gpu_usage_percent: float | None = Field(default=None, ge=0)
    gpu_memory_sample_count: int = Field(ge=0)
    avg_gpu_memory_usage_bytes: float | None = Field(default=None, ge=0)
    peak_gpu_memory_usage_bytes: int | None = Field(default=None, ge=0)


class EvaluationTelemetryFailure(BaseModel):
    phase: str | None = None
    code: str | None = None
    category: str | None = None


class EvaluationTelemetryExecutionCleanup(BaseModel):
    execution_number: int = Field(ge=1)
    runtime: str
    failure_code: str
    retry_after_cleanup: bool
    status: Literal["pending", "deleting", "delete_failed", "deleted"]
    teardown_attempts: int = Field(ge=0)
    delete_error: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class EvaluationTelemetryCleanup(BaseModel):
    cancellation_status: Literal["not_requested", "pending", "succeeded", "failed"]
    error: str | None = None
    updated_at: datetime | None = None
    executions: list[EvaluationTelemetryExecutionCleanup] = Field(default_factory=list)


class EvaluationTelemetryIntake(BaseModel):
    enabled: bool
    profile_id: str | None = None
    status: Literal["disabled", "pending", "succeeded", "failed", "no_records"]
    experiment_ref: str | None = None
    run_refs: list[str] = Field(default_factory=list)
    expected_records: int | None = Field(default=None, ge=0)
    uploaded_records: int | None = Field(default=None, ge=0)
    complete: bool
    error: str | None = None
    diagnostic_artifact: str | None = None


class EvaluationTelemetryArtifacts(BaseModel):
    listing: str
    archive: str
    provenance: str
    sbom: str
    artifact_sync_status: Literal["pending", "succeeded", "failed"]
    artifact_sync_file_count: int | None = Field(default=None, ge=0)
    artifact_sync_error: str | None = None
    evidence_status: Literal["missing", "building", "ready"]
    evidence_error: str | None = None
    archive_status: Literal["missing", "building", "ready"]
    archive_required: bool
    archive_error: str | None = None
    terminal_sync_complete: bool


class EvaluationTelemetryPhaseTiming(BaseModel):
    phase: Literal["provisioning", "running", "total"]
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)


class EvaluationTelemetryUsage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)
    source: str


class EvaluationTelemetryInteractions(BaseModel):
    turns: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)


class EvaluationTelemetryCost(BaseModel):
    value_usd: float | None = Field(default=None, ge=0)
    source: Literal["provider", "estimated", "unknown"]


class EvaluationTelemetryRawArtifact(BaseModel):
    relation: Literal[
        "result",
        "trajectory",
        "native_trajectory",
        "transcript",
        "intake_diagnostic",
    ]
    path: str
    download: str


class EvaluationExecutionTelemetry(BaseModel):
    execution_number: int = Field(ge=1)
    terminal_status: Literal["succeeded", "failed", "cancelled"] | None = None
    failure_phase: str | None = None
    phase_timings: list[EvaluationTelemetryPhaseTiming] = Field(default_factory=list)
    usage: EvaluationTelemetryUsage
    interactions: EvaluationTelemetryInteractions
    cost: EvaluationTelemetryCost
    raw_artifacts: list[EvaluationTelemetryRawArtifact] = Field(default_factory=list)


class EvaluationTelemetryResponse(BaseModel):
    """Portable, factual run telemetry; absent samples mean unavailable, not zero."""

    schema_version: Literal["scaled-evals-evaluation-telemetry-v1"] = "scaled-evals-evaluation-telemetry-v1"
    evaluation_id: str
    status: EvaluationStatus
    current_execution: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    failure: EvaluationTelemetryFailure
    cleanup: EvaluationTelemetryCleanup
    intake: EvaluationTelemetryIntake
    artifacts: EvaluationTelemetryArtifacts
    executions: list[EvaluationExecutionTelemetry] = Field(default_factory=list)
    resource_usage: list[EvaluationResourceUsage] = Field(default_factory=list)


class EvaluationArtifact(BaseModel):
    path: str
    size_bytes: int
    updated_at: str | None = None
    links: dict[str, str]


class BuildArchiveRequest(BaseModel):
    force: bool = False


class ArchiveDownload(BaseModel):
    method: Literal["GET"] = "GET"
    url: str


class EvaluationArchiveResponse(BaseModel):
    evaluation_id: str
    status: ArchiveStatus
    format: Literal["tar.gz"] = "tar.gz"
    size_bytes: int | None = None
    built_at: datetime | None = None
    error: str | None = None
    download: ArchiveDownload | None = None
