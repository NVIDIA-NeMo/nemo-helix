# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity Store entities for scaled-evals metadata.

Every scaled-evals resource lives only here. The store offers per-entity
optimistic locking but no multi-entity transaction and no atomic dequeue, so
each evaluation carries its own children (status history, per-execution
telemetry, resource usage and runtime cleanups) and queues are claimed by a
compare-and-swap on the evaluation itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nemo_helix_plugin.entity import NemoEntity
from pydantic import BaseModel, Field, model_validator


class ScaledEvalsEvaluationEvent(BaseModel):
    """One status transition or retry note, numbered from 1 within its evaluation."""

    id: int
    type: str = "status"
    status: str
    detail: str | None = None
    created_at: datetime


class ScaledEvalsExecutionTelemetry(BaseModel):
    """Portable facts about one execution of an evaluation."""

    execution_number: int
    provisioning_started_at: datetime | None = None
    running_started_at: datetime | None = None
    terminal_at: datetime | None = None
    terminal_status: str | None = None
    failure_phase: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    usage_source: str = "unknown"
    turn_count: int | None = None
    tool_call_count: int | None = None
    cost_usd: float | None = None
    cost_source: str = "unknown"
    raw_artifact_refs: list[Any] = Field(default_factory=list)
    intake_experiment_ref: str | None = None
    intake_run_refs: list[str] = Field(default_factory=list)
    intake_status: str = "disabled"
    intake_expected_records: int | None = None
    intake_uploaded_records: int | None = None
    intake_error: str | None = None
    artifact_sync_status: str = "pending"
    artifact_sync_file_count: int | None = None
    artifact_sync_error: str | None = None
    created_at: datetime
    updated_at: datetime


class ScaledEvalsResourceUsage(BaseModel):
    """Running sums and peaks of one component's samples within one execution."""

    execution_number: int
    component: str
    source: str
    collection_status: str = "sampled"
    collection_error: str | None = None
    sample_count: int = 0
    first_observed_at: datetime
    last_observed_at: datetime
    cpu_sample_count: int = 0
    cpu_usage_cores_sum: float = 0
    cpu_usage_cores_max: float = 0
    memory_sample_count: int = 0
    memory_usage_bytes_sum: float = 0
    memory_usage_bytes_max: int = 0
    cpu_request_cores: float | None = None
    cpu_limit_cores: float | None = None
    memory_request_bytes: int | None = None
    memory_limit_bytes: int | None = None
    gpu_request: float | None = None
    gpu_sample_count: int = 0
    gpu_usage_percent_sum: float = 0
    gpu_usage_percent_max: float = 0
    gpu_memory_sample_count: int = 0
    gpu_memory_usage_bytes_sum: float = 0
    gpu_memory_usage_bytes_max: int = 0


class ScaledEvalsExecutionCleanup(BaseModel):
    """Teardown of the runtime an execution left behind when its runner was lost."""

    execution_number: int
    runtime: str
    backend_handle: str
    dispatch_job_name: str
    failure_code: str
    failure_detail: str
    retry_after_cleanup: bool
    status: str = "pending"
    next_attempt_at: datetime
    teardown_claimed_at: datetime | None = None
    teardown_claimed_by: str | None = None
    teardown_attempts: int = 0
    delete_error: str | None = None
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ScaledEvalsEvaluation(NemoEntity, entity_type="scaled_evals_evaluation"):
    """One logical evaluation, named by its evaluation id.

    `changed_at` is the evaluation's own `updated_at`: telemetry and resource
    samples also write the entity, and they must not make a stuck run look
    fresh or delay its artifact builds.
    """

    owner_id: str | None = None
    display_name: str
    framework: str
    requested_framework_version: str | None = None
    framework_version: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    framework_adapter_version: str | None = None
    sandbox_k8s_version: str | None = None
    runner_metadata: dict[str, Any] = Field(default_factory=dict)
    task_id: str
    task_revision: int
    benchmark_run_id: str | None = None
    framework_profile_id: str | None = None
    harbor_profile_id: str | None = None
    switchyard_profile_id: str | None = None
    intake_profile_id: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    extra_skill_object_keys: list[str] = Field(default_factory=list)
    instruction_prefix: str | None = None
    instruction_postfix: str | None = None
    initial_user_turns: list[str] = Field(default_factory=list)
    runtime: str
    network_policy: str = "unrestricted"
    network_policy_config: dict[str, Any] = Field(default_factory=dict)
    n_attempts: int = 1
    parallelism: int
    visibility: str = "private"
    status: str
    status_detail: str | None = None
    cancel_teardown_status: str = "not_requested"
    cancel_teardown_error: str | None = None
    cancel_teardown_updated_at: datetime | None = None
    backend_handle: str | None = None
    dispatch_job_name: str | None = None
    dispatch_job_uid: str | None = None
    dispatch_claimed_at: datetime | None = None
    dispatch_claimed_by: str | None = None
    dispatch_attempts: int = 0
    dispatch_reconcile_claimed_at: datetime | None = None
    dispatch_reconcile_claimed_by: str | None = None
    current_execution: int = 1
    max_executions: int = 3
    infrastructure_retries: int = 0
    max_infrastructure_retries: int = 2
    next_retry_at: datetime | None = None
    last_failure_code: str | None = None
    last_failure_category: str | None = None
    result: dict[str, Any] | None = None
    reward: float | None = None
    reward_value: Any = None
    n_trials: int | None = None
    n_completed: int | None = None
    n_errored: int | None = None
    n_failed_solve: int | None = None
    exception_counts: dict[str, int] = Field(default_factory=dict)
    finished_at: datetime | None = None
    evidence_status: str = "missing"
    evidence_requested_at: datetime | None = None
    evidence_built_at: datetime | None = None
    evidence_error: str | None = None
    evidence_claimed_at: datetime | None = None
    evidence_claimed_by: str | None = None
    evidence_build_attempts: int = 0
    archive_status: str = "missing"
    archive_object_key: str | None = None
    archive_size_bytes: int | None = None
    archive_requested_at: datetime | None = None
    archive_built_at: datetime | None = None
    archive_error: str | None = None
    archive_claimed_at: datetime | None = None
    archive_claimed_by: str | None = None
    archive_build_attempts: int = 0
    execution_snapshot: dict[str, Any] | None = None
    deleted_at: datetime | None = None
    changed_at: datetime
    # Whether any cleanup is not yet deleted, so the cleanup queue lists only
    # evaluations that have work instead of every evaluation ever run.
    cleanup_open: bool = False
    # Filterable mirrors of `benchmark_run_id is None` and `deleted_at is not
    # None`: the store does not promise to match a JSON null. Derived on every
    # load; code that sets `deleted_at` on a loaded entity sets `deleted` too.
    standalone: bool = True
    deleted: bool = False
    events: list[ScaledEvalsEvaluationEvent] = Field(default_factory=list)
    telemetry: list[ScaledEvalsExecutionTelemetry] = Field(default_factory=list)
    resource_usage: list[ScaledEvalsResourceUsage] = Field(default_factory=list)
    cleanups: list[ScaledEvalsExecutionCleanup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_filter_flags(self) -> ScaledEvalsEvaluation:
        self.standalone = self.benchmark_run_id is None
        self.deleted = self.deleted_at is not None
        return self


class ScaledEvalsBenchmarkArchive(BaseModel):
    """A benchmark run's combined Harbor archive and its build lease."""

    generation: str
    status: str
    members: list[dict[str, Any]] = Field(default_factory=list)
    requested_at: datetime
    claimed_at: datetime | None = None
    claim_token: str | None = None
    attempts: int = 0
    object_key: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    partial: bool | None = None
    built_at: datetime | None = None
    error: str | None = None
    cleanup_checked_at: datetime | None = None


class ScaledEvalsBenchmarkRun(NemoEntity, entity_type="scaled_evals_benchmark_run"):
    """A benchmark run's identity and config, named by its run id.

    Status, reward and the per-task breakdown are derived from the member
    evaluations on read, never stored here.
    """

    owner_id: str | None = None
    display_name: str
    framework: str
    requested_framework_version: str | None = None
    framework_version: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    framework_adapter_version: str | None = None
    sandbox_k8s_version: str | None = None
    runner_metadata: dict[str, Any] = Field(default_factory=dict)
    benchmark_id: str
    benchmark_revision: int
    framework_profile_id: str | None = None
    harbor_profile_id: str | None = None
    intake_profile_id: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    runtime: str
    network_policy: str = "unrestricted"
    network_policy_config: dict[str, Any] = Field(default_factory=dict)
    parallelism: int
    max_concurrent_members: int | None = None
    visibility: str = "private"
    cancelled_at: datetime | None = None
    deleted_at: datetime | None = None
    archive: ScaledEvalsBenchmarkArchive | None = None


class ScaledEvaluationExecution(NemoEntity, entity_type="scaled_evals_execution"):
    """The immutable launch inputs of one evaluation execution.

    Named after the execution's Platform Job and written once, before that Job
    is created, so the Job launches from inputs a later retry cannot change. It
    carries prompt content and the private execution snapshot, so it lives in a
    workspace only the scaled-evals service can read. Credentials appear only as provider to
    platform-secret references; the Job resolves the values at launch.
    """

    evaluation_id: str
    execution_number: int
    inputs: dict[str, Any] = Field(default_factory=dict)


class ScaledEvalsConfigProfile(NemoEntity, entity_type="scaled_evals_config_profile"):
    """A reusable framework or intake config profile, named by its profile id."""

    owner_id: str | None = None
    display_name: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class ScaledEvalsAgentBundle(NemoEntity, entity_type="scaled_evals_agent_bundle"):
    """A registered agent runtime bundle, named by its bundle id."""

    owner_id: str
    bundle_name: str
    agent_name: str
    agent_version: str
    image_ref: str
    image_digest: str
    entrypoint: str
    platform: str = "linux/amd64"
    runtime_abi: str = "glibc"
    bundle_layout_version: int = 1
    builder_profile: str = "node22-npm-v1"
    source_lock_digest: str
    fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    visibility: str = "private"
    qualification_status: str = "registered"
    qualification_evidence: dict[str, Any] = Field(default_factory=dict)
    qualified_at: datetime | None = None
    qualified_by: str | None = None


class ScaledEvalsTaskRevision(BaseModel):
    """One immutable task-pack revision and its image build state."""

    revision: int
    status: str = "uploading"
    tarball_object_key: str
    tarball_size_bytes: int | None = None
    tarball_sha256: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    verifier_image_ref: str | None = None
    verifier_image_digest: str | None = None
    build_backend: str | None = None
    build_payload: dict[str, Any] = Field(default_factory=dict)
    build_credentials: dict[str, str] = Field(default_factory=dict)
    build_error: str | None = None
    build_attempts: int = 0
    build_claimed_by: str | None = None
    build_claimed_at: datetime | None = None
    build_first_claimed_at: datetime | None = None
    build_next_attempt_at: datetime | None = None
    build_started_at: datetime | None = None
    build_completed_at: datetime | None = None
    created_at: datetime


class ScaledEvalsTask(NemoEntity, entity_type="scaled_evals_task"):
    """A task, named by its task id, with every revision embedded.

    Embedding keeps revision numbering, finalize, and build-state transitions
    to a single versioned write on one entity. Soft-deleted tasks are kept so
    evaluations that ran them still resolve their revision.
    """

    owner_id: str | None = None
    display_name: str
    slug: str
    description: str | None = None
    visibility: str = "private"
    current_revision: int | None = None
    deleted_at: datetime | None = None
    revisions: list[ScaledEvalsTaskRevision] = Field(default_factory=list)


class ScaledEvalsBenchmarkMember(BaseModel):
    """One member task of a benchmark revision; a None `task_revision` floats to the latest."""

    task_id: str
    task_revision: int | None = None
    position: int


class ScaledEvalsBenchmarkRevision(BaseModel):
    """One immutable benchmark revision: its member set, lineage and policy."""

    revision: int
    description: str | None = None
    derived_from_benchmark_id: str | None = None
    derived_from_revision: int | None = None
    operational_policy: dict[str, Any] = Field(default_factory=dict)
    members: list[ScaledEvalsBenchmarkMember] = Field(default_factory=list)
    created_at: datetime


class ScaledEvalsBenchmark(NemoEntity, entity_type="scaled_evals_benchmark"):
    """A benchmark, named by its benchmark id, with every revision embedded."""

    owner_id: str | None = None
    display_name: str
    slug: str
    description: str | None = None
    visibility: str = "private"
    qualification_status: str = "registered"
    qualification_evidence: dict[str, Any] = Field(default_factory=dict)
    qualified_at: datetime | None = None
    qualified_by: str | None = None
    current_revision: int | None = None
    deleted_at: datetime | None = None
    revisions: list[ScaledEvalsBenchmarkRevision] = Field(default_factory=list)


class ScaledEvalsBenchmarkImportTask(BaseModel):
    """One manifest task of an import and the task revision it materialized into."""

    position: int
    slug: str
    task_id: str
    task_revision: int
    pack_path: str
    pack_sha256: str
    image_ref: str | None = None
    image_digest: str | None = None
    image_metadata: dict[str, Any] = Field(default_factory=dict)


class ScaledEvalsBenchmarkImportBenchmark(BaseModel):
    """One manifest benchmark of an import and the revision it published to."""

    position: int
    slug: str
    name: str
    task_slugs: list[str] = Field(default_factory=list)
    benchmark_id: str | None = None
    benchmark_revision: int | None = None


class ScaledEvalsBenchmarkImport(NemoEntity, entity_type="scaled_evals_benchmark_import"):
    """A materialized benchmark import, named after its (owner, manifest, visibility) identity.

    The deterministic name makes a repeated create of the same manifest
    conflict instead of starting a second import.
    """

    owner_id: str
    manifest_sha256: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    visibility: str = "private"
    description: str | None = None
    tasks: list[ScaledEvalsBenchmarkImportTask] = Field(default_factory=list)
    benchmarks: list[ScaledEvalsBenchmarkImportBenchmark] = Field(default_factory=list)
