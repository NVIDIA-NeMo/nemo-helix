# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve / orchestrator configuration (no NeMo-RL / GRPO fields)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sandboxed_gym.config import K8S_LABEL_VALUE_RE, EpisodeBrokerConfig
from sandboxed_gym.host.models import (
    DEFAULT_JOB_ID,
    GymHostEgressRule,
    SandboxConfig,
)


class SandboxedGymServeConfig(BaseModel):
    """Trusted-side config to start the episode broker + Gym host.

    ``gym_global_config`` is opaque Gym JSON (injected as ``NMP_GYM_GLOBAL_CONFIG``).
    Callers own policy_model_name / policy_base_url / config_paths; this package does
    not inject RL training knobs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = DEFAULT_JOB_ID
    host_provider: str = "opensandbox"
    environment_path: str | None = None
    environment_offline: bool = False
    sandbox: SandboxConfig
    episode_broker: EpisodeBrokerConfig | dict[str, Any] = Field(default_factory=dict)
    gym_global_config: dict[str, Any] = Field(default_factory=dict)
    # Environment variables the caller's job needs inside the host, merged into the bootstrap
    # environment. Repr-suppressed: an `env_secrets` value resolved by the caller lands here.
    host_env: dict[str, str] = Field(default_factory=dict, repr=False)
    # Extra egress targets beyond broker + parsed policy URLs (e.g. model servers).
    egress_extra: tuple[GymHostEgressRule, ...] = ()
    policy_base_urls: tuple[str, ...] = ()
    rollout_auth_token: str | None = Field(default=None, repr=False)
    serve_mode: Literal["orchestrator", "host-urls"] = "orchestrator"

    @field_validator("job_id", mode="before")
    @classmethod
    def _default_job_id(cls, value: str | None) -> str:
        if value is None or value == "":
            return DEFAULT_JOB_ID
        if not K8S_LABEL_VALUE_RE.match(str(value)):
            raise ValueError(f"job_id must be a valid Kubernetes label value: {value!r}")
        return str(value)

    def broker_config(self) -> EpisodeBrokerConfig:
        raw = self.episode_broker
        if isinstance(raw, EpisodeBrokerConfig):
            data = raw.model_dump()
        else:
            data = dict(raw)
        data.setdefault("job_id", self.job_id)
        data["job_id"] = self.job_id
        return EpisodeBrokerConfig.model_validate(data)


class SandboxedGymSessionDescriptor(BaseModel):
    """Endpoint handoff for cross-job clients (Job A → Job B)."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    mode: Literal["orchestrator", "host-urls"]
    orchestrator_url: str | None = None
    health_url: str
    rollout_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    broker_url: str
    broker_token: str = Field(repr=False)
    rollout_auth_token: str | None = Field(default=None, repr=False)
