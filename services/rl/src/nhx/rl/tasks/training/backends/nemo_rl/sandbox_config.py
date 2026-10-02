# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandbox and Gym-host bootstrap helpers used when compiling GRPO configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GymHostEgressRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str
    port: int = Field(ge=1, le=65535)


#: NeMo-RL's ``DEFAULT_ROLLOUT_CHUNK_SIZE`` (``environments/sandbox/host/models.py``).
DEFAULT_ROLLOUT_CHUNK_SIZE = 8
#: NeMo-RL's ``DEFAULT_ROLLOUT_MAX_IN_FLIGHT``. Concurrent POSTs; effective in-flight
#: rollouts are ``rollout_chunk_size * rollout_max_in_flight``.
DEFAULT_ROLLOUT_MAX_IN_FLIGHT = 8


class SandboxNetworkPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    egress_allow: list[GymHostEgressRule] = Field(default_factory=list)
    # Extra DNS suffixes or FQDNs, consulted by NeMo-RL only when allow_internet is true.
    # NeMo-RL's own default list is ("*.com", "*.org"), so an environment whose package index
    # lives anywhere else -- hub.primeintellect.ai, for one -- is denied at DNS without this.
    # Operator-scoped like allow_internet: sourced from platformConfig.rl.sandbox_public_dns_allow,
    # never from a job submission.
    public_dns_allow: tuple[str, ...] = ()


class SandboxConfig(BaseModel):
    """Mirror of NeMo-RL's ``env.nemo_gym.sandbox`` schema.

    Field names and requiredness must track
    ``nemo_rl/environments/sandbox/host/models.py``: that model is
    ``extra="forbid"`` and declares ``environment_pvc_claim`` and
    ``workspace_pvc_claim`` with no defaults, so omitting either makes the Gym
    host fail validation at provisioning time.
    """

    model_config = ConfigDict(extra="forbid")

    image: str
    # Sourced from platformConfig.rl.sandbox_allow_internet, so it is per-deployment, never
    # per-job. Opens NeMo-RL's default public DNS suffixes; network_policy.public_dns_allow
    # adds to them. Both are operator-scoped, so the job schema has no path to widen egress.
    allow_internet: bool = False
    env_mount_path: str = "/job/environment"
    dataset_mount_path: str = "/job/dataset"
    work_mount_path: str = "/job/work"
    max_request_bytes: int = Field(default=268_435_456, gt=0)
    max_response_bytes: int = Field(default=268_435_456, gt=0)
    ttl_s: int = 14_400
    # Rollouts carried by one POST to the sandbox. A chunk's wall time scales with the tokens
    # its rollouts ask for, and the OpenSandbox proxy caps how long one request may stay open,
    # so long generations can need a smaller chunk than NeMo-RL's default.
    rollout_chunk_size: int = Field(default=DEFAULT_ROLLOUT_CHUNK_SIZE, gt=0)
    # Concurrent chunk POSTs. Lowering rollout_chunk_size without raising this throttles
    # the step: in-flight rollouts are the product of the two.
    rollout_max_in_flight: int = Field(default=DEFAULT_ROLLOUT_MAX_IN_FLIGHT, gt=0)
    network_policy: SandboxNetworkPolicy = Field(default_factory=SandboxNetworkPolicy)
    resources: dict[str, str] | None = None
    # Forwarded verbatim to NeMo-RL's host provider constructor (connection / create /
    # probe / operations). ``connection.protocol`` is the one that matters in practice:
    # it sets the scheme for BOTH the OpenSandbox SDK connection and the health/rollout
    # proxy URLs, and NeMo-RL defaults it to https. Against a plain-HTTP in-cluster
    # server the health poll then stalls in the TLS handshake until ready_timeout_s.
    host_provider_options: dict[str, Any] = Field(default_factory=dict)
    # PVC claim + sub-path triples. The sandbox mounts the same job-storage claim the
    # training container uses, so the environment and dataset it reads are the ones the
    # file_io download step already materialized.
    environment_pvc_claim: str = Field(min_length=1)
    environment_sub_path: str = ""
    dataset_pvc_claim: str | None = None
    dataset_sub_path: str = ""
    workspace_pvc_claim: str = Field(min_length=1)
    workspace_sub_path: str = ""


class NemoGymSandboxedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sandboxed: bool = True
    host_provider: str = "opensandbox"
    environment_path: str | None = None
    # Whether that package vendors a complete closure. NeMo-RL forwards it to the sandbox as
    # NHX_ENVIRONMENT_OFFLINE so uv resolves from wheels/ instead of an index it cannot reach.
    environment_offline: bool = False
    sandbox: SandboxConfig | None = None
    # Stamped onto every sandbox pod as a label. Upstream defaults this to a shared
    # constant when absent, which makes concurrent jobs indistinguishable.
    job_id: str | None = None


def resolve_job_storage_pvc_claim() -> str | None:
    """Job-storage PVC claim name injected by the compiler, if any."""
    from nhx.rl.app.constants import NHX_JOB_STORAGE_PVC_ENVVAR

    return os.environ.get(NHX_JOB_STORAGE_PVC_ENVVAR) or None


class SandboxMounts(BaseModel):
    """PVC claim + sub-path triples for the Gym sandbox's three mounts."""

    model_config = ConfigDict(frozen=True)

    environment_pvc_claim: str
    environment_sub_path: str
    dataset_pvc_claim: str
    dataset_sub_path: str
    workspace_pvc_claim: str
    workspace_sub_path: str


def build_sandbox_mounts(
    *,
    pvc_claim: str,
    workspace: str,
    job_id: str,
    storage_root: Path,
    environment_path: str,
    dataset_path: str,
) -> SandboxMounts:
    """Map job-storage directories to PVC-relative sub-paths for the Gym sandbox.

    The training container sees ``environment_path`` / ``dataset_path`` under the
    job-storage mount; the sandbox mounts the same claim directly, so each path has
    to be re-expressed relative to the PVC root.
    """
    from nemo_helix_plugin.jobs.constants import job_storage_subpath

    prefix = job_storage_subpath(workspace, job_id)

    def _sub_path(path: str) -> str:
        leaf = Path(path).relative_to(storage_root)
        return f"{prefix}/{leaf.as_posix()}"

    return SandboxMounts(
        environment_pvc_claim=pvc_claim,
        environment_sub_path=_sub_path(environment_path),
        dataset_pvc_claim=pvc_claim,
        dataset_sub_path=_sub_path(dataset_path),
        workspace_pvc_claim=pvc_claim,
        # Writable scratch for the Gym host, kept beside the job's own data so it is
        # reclaimed by the same job-storage cleanup.
        workspace_sub_path=f"{prefix}/gym-work",
    )


def assemble_master_egress_allow(
    *,
    vllm_host: str | None = None,
    vllm_port: int | None = None,
    broker_host: str | None = None,
    broker_port: int | None = None,
) -> list[GymHostEgressRule]:
    """Build OpenSandbox egress rules from live vLLM and broker endpoints.

    Explicit kwargs win; otherwise read ``NHX_*_SERVICE_*`` env vars, then
    fall back to localhost defaults suitable for single-node bring-up.
    """
    from nhx.rl.app.constants import (
        NHX_BROKER_HOST_ENVVAR,
        NHX_BROKER_PORT_ENVVAR,
        NHX_VLLM_HOST_ENVVAR,
        NHX_VLLM_PORT_ENVVAR,
    )

    host_vllm = vllm_host or os.environ.get(NHX_VLLM_HOST_ENVVAR) or "127.0.0.1"
    port_vllm = vllm_port if vllm_port is not None else int(os.environ.get(NHX_VLLM_PORT_ENVVAR, "8000"))
    host_broker = broker_host or os.environ.get(NHX_BROKER_HOST_ENVVAR) or "127.0.0.1"
    port_broker = broker_port if broker_port is not None else int(os.environ.get(NHX_BROKER_PORT_ENVVAR, "51234"))
    return [
        GymHostEgressRule(host=host_vllm, port=port_vllm),
        GymHostEgressRule(host=host_broker, port=port_broker),
    ]


def apply_master_egress_to_sandbox_config(sandbox: SandboxConfig) -> SandboxConfig:
    """Refresh ``network_policy.egress_allow`` from the current master endpoints.

    Only ``egress_allow`` is recomputed. ``public_dns_allow`` is operator configuration that
    the master cannot re-derive, so it is carried over -- rebuilding the whole policy here
    would drop it and the sandbox would lose its package index mid-compile.
    """
    return sandbox.model_copy(
        update={
            "network_policy": sandbox.network_policy.model_copy(
                update={"egress_allow": assemble_master_egress_allow()},
            ),
        }
    )
