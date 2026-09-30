# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

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
    intake_profile_id: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    credential_env: dict[str, str] = Field(default_factory=dict)


class LaunchHandle(BaseModel):
    """Opaque reference to a launched run, returned by a runtime backend."""

    model_config = ConfigDict(frozen=True)

    backend: str
    external_id: str
    raw: dict[str, Any] = Field(default_factory=dict)


class RuntimeStatus(BaseModel):
    """Backend-reported run state normalized to the evaluation lifecycle."""

    model_config = ConfigDict(frozen=True)

    phase: str
    detail: str | None = None
    failure_code: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


ResultSummary = EvaluationResultSummary
