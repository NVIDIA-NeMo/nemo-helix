# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scaled_evals.models.evaluations import EvaluationResultSummary


class LaunchSpec(BaseModel):
    """Everything a backend needs to start one evaluation."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str
    benchmark_run_id: str | None = None
    name: str
    task_slug: str | None = None
    framework: str
    framework_version: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    runner_source_revision: str | None = None
    runner_package_version: str | None = None
    allow_live_runner_fallback: bool = False
    framework_adapter_version: str | None = None
    sandbox_k8s_version: str | None = None
    agent_bundle: dict[str, Any] | None = None
    harbor_dir: str | None = None
    image_ref: str
    image_digest: str | None = None
    n_attempts: int = 1
    parallelism: int
    network_policy: str = "unrestricted"
    network_policy_config: dict[str, Any] = Field(default_factory=dict)
    # Object-store key of the task revision's uploaded tarball (the same
    # pack the BuildKit build downloads). Lets dispatch source the Harbor task
    # tree (task.toml / tests/ / solution/ / instruction.md) from the upload
    # per-eval, rather than only the task trees baked into the harbor-runner
    # image. ``None`` for legacy revisions with no recorded key → dispatch falls
    # back to the baked/global task path.
    tarball_object_key: str | None = None
    # S3 object keys for extra skill files to inject into the staged task tree's
    # environment/skills/ directory before Harbor uploads skills to the sandbox.
    # Optional skill objects injected into an agent session at runtime.
    extra_skill_object_keys: list[str] = Field(default_factory=list)
    # Optional text prepended/appended to instruction.md at dispatch time without
    # rebuilding the benchmark image.
    instruction_prefix: str | None = None
    instruction_postfix: str | None = None
    # Metadata-only benchmark variant policy: raise staged task.toml
    # [agent].timeout_sec to at least this floor.
    agent_timeout_floor_sec: int | None = None
    # Ordered user messages sent before the benchmark instruction in the same
    # agent session. Supported by the Claude Code Harbor adapter.
    initial_user_turns: list[str] = Field(default_factory=list)
    harbor_profile_id: str | None = None
    framework_config: dict[str, Any] = Field(default_factory=dict)
    harbor_config: dict[str, Any] = Field(default_factory=dict)
    harbor_dataset_image_imports: list[dict[str, Any]] = Field(default_factory=list)
    switchyard_profile_id: str | None = None
    switchyard_config: dict[str, Any] = Field(default_factory=dict)
    switchyard: SwitchyardLease | None = None
    intake_profile_id: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    credential_env: dict[str, str] = Field(default_factory=dict)


class LaunchHandle(BaseModel):
    """Opaque reference to a launched run, returned by a runtime backend."""

    model_config = ConfigDict(frozen=True)

    backend: str
    external_id: str
    raw: dict[str, Any] = Field(default_factory=dict)


class SwitchyardLease(BaseModel):
    """Per-evaluation managed resource or external endpoint identity.

    Secret material is intentionally absent. The lease is safe to persist in
    Postgres, pass through LaunchSpec, and record in provenance.
    """

    model_config = ConfigDict(frozen=True)

    profile_id: str
    benchmark_run_id: str | None = None
    mode: Literal["managed", "external"] = "managed"
    namespace: str | None = None
    name: str | None = None
    service_name: str | None = None
    config_map_name: str | None = None
    secret_name: str | None = None
    network_policy_name: str | None = None
    endpoint: str
    openai_base_url: str
    anthropic_base_url: str
    inbound: str
    port: int
    book_mode: str | None = None
    resource_labels: dict[str, str] = Field(default_factory=dict)
    endpoint_identity: str | None = None
    trust_warning: str | None = None
    manifest_hash: str | None = None
    config_hash: str | None = None
    drain_seconds: float | None = None
    routing_stats_path: str = "/v1/routing/stats"
    routing_stats_max_bytes: int = 1_048_576
    artifact_path: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    source_project: str | None = None
    source_ref: str | None = None
    source_commit: str | None = None
    context_path: str | None = None
    dockerfile_path: str | None = None
    dockerfile_sha256: str | None = None
    context_hash: str | None = None


class RuntimeStatus(BaseModel):
    """Backend-reported run state normalized to the evaluation lifecycle."""

    model_config = ConfigDict(frozen=True)

    phase: str
    detail: str | None = None
    failure_code: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


ResultSummary = EvaluationResultSummary
