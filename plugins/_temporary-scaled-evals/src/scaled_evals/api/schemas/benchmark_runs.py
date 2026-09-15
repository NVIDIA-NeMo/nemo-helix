# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from scaled_evals.api.schemas.common import validate_scoped_egress_config
from scaled_evals.api.schemas.evaluations import EvaluationStatus, Framework, NetworkPolicyMode
from scaled_evals.api.schemas.tasks import Visibility


# Request body: POST /v1/benchmark-runs
class CreateBenchmarkRunRequest(BaseModel):
    """Run a benchmark by aggregating one evaluation per member task.

    Spawns one member evaluation per member task of the resolved benchmark
    revision; the run aggregates their rewards by fan-in.
    """

    name: str = Field(min_length=1, max_length=200)
    benchmark_id: str = Field(min_length=1)
    benchmark_revision: int | None = Field(
        default=None,
        ge=1,
        description="Benchmark revision to run; defaults to the benchmark's current revision.",
    )
    framework: Framework = "harbor"
    framework_version: str | None = Field(
        default=None,
        description="Exact supported framework version or documented alias.",
    )
    framework_profile_id: str | None = None
    member_framework_profile_ids: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional task_id to framework profile mapping. A member override takes precedence "
            "over framework_profile_id for that task."
        ),
    )
    harbor_profile_id: str | None = Field(
        default=None,
        description="Compatibility alias for framework_profile_id on Harbor requests.",
    )
    switchyard_profile_id: str | None = None
    intake_profile_id: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    agent_bundle_id: str | None = None
    extra_skill_object_keys: list[str] = Field(default_factory=list)
    instruction_prefix: str | None = None
    instruction_postfix: str | None = None
    initial_user_turns: list[str] = Field(default_factory=list)
    runtime: str = "sandbox_k8s"
    network_policy: NetworkPolicyMode = "unrestricted"
    network_policy_config: dict[str, Any] = Field(default_factory=dict)
    n_attempts: int = Field(default=1, ge=1, le=256)
    # Trials within each member task; cross-task concurrency comes from the
    # dispatch worker pool, not this knob.
    parallelism: int = Field(default=1, ge=1, le=256)
    max_concurrent_members: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description=(
            "Benchmark-run member concurrency cap. With a Switchyard profile, the same cap "
            "also limits members sharing the managed Switchyard gateway."
        ),
    )
    visibility: Visibility = "private"

    @field_validator("initial_user_turns")
    @classmethod
    def _validate_initial_user_turns(cls, turns: list[str]) -> list[str]:
        if any(not turn.strip() for turn in turns):
            raise ValueError("initial_user_turns must not contain blank turns")
        return turns

    @model_validator(mode="after")
    def _resolve_framework_profile_alias(self) -> "CreateBenchmarkRunRequest":
        if self.agent_bundle_id is not None and (self.framework != "harbor" or self.runtime != "sandbox_k8s"):
            raise ValueError("agent_bundle currently requires framework='harbor' and runtime='sandbox_k8s'")
        if self.agent_bundle_id is not None and self.framework_profile_id is None:
            raise ValueError("agent_bundle_id requires a Harbor framework_profile_id")
        if self.initial_user_turns and self.framework != "harbor":
            raise ValueError('initial_user_turns is only valid for framework="harbor"')
        if self.switchyard_profile_id is not None:
            if self.runtime != "sandbox_k8s":
                raise ValueError("shared Switchyard campaigns require runtime='sandbox_k8s'")
            if self.parallelism != 1:
                raise ValueError("shared Switchyard campaigns require parallelism=1")
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


class BenchmarkRunLinks(BaseModel):
    self: str
    evaluations: str
    reproduce: str
    cancel: str
    archive: str | None = None


class ReproduceBenchmarkRunResponse(BaseModel):
    benchmark_run_id: str
    source_status: EvaluationStatus
    request: CreateBenchmarkRunRequest
    cli_command: list[str]
    notes: list[str] = Field(default_factory=list)


class BenchmarkRun(BaseModel):
    """A benchmark run row aggregating one evaluation per member task."""

    id: str
    name: str
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
    switchyard_profile_id: str | None = None
    intake_profile_id: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    runtime: str
    network_policy: NetworkPolicyMode = "unrestricted"
    network_policy_config: dict[str, Any] = Field(default_factory=dict)
    parallelism: int
    max_concurrent_members: int | None = None
    visibility: Visibility
    status: EvaluationStatus
    status_detail: str | None = None
    # Aggregate summary, written by the worker once all members finish.
    reward: float | None = None
    n_trials: int | None = None
    n_completed: int | None = None
    n_errored: int | None = None
    n_failed_solve: int | None = None
    exception_counts: dict[str, int] = Field(default_factory=dict)
    n_teardown_pending: int = 0
    n_teardown_failed: int = 0
    n_retryable_failures: int = 0
    n_recovered: int = 0
    failure_counts: dict[str, int] = Field(default_factory=dict)
    recovered_counts: dict[str, int] = Field(default_factory=dict)
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


# Single-run responses add links + the full aggregate result envelope.
class BenchmarkRunResponse(BenchmarkRun):
    links: BenchmarkRunLinks
    result: dict[str, Any] | None = None
