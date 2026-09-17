# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base remote-submit training job for customization backends.

Both backends submit a 4-step ``PlatformJobSpec`` (download → train → upload →
model-entity) executed on the platform GPU cluster. ``to_spec`` and the
runtime guards are shared here; ``compile`` genuinely diverges (compiler
call convention, schema validation, profile resolution) and stays per-backend.
"""

from __future__ import annotations

from typing import ClassVar, Generic, TypeVar

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.capabilities import probe_docker
from nemo_platform_plugin.config import NemoPlatformConfig, Runtime
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nmp.customization_common.service.platform_client import (
    AsyncCustomizationPlatformClients,
    async_customization_platform_clients_from_platform,
)
from pydantic import BaseModel

JobInputT = TypeVar("JobInputT", bound=BaseModel)
JobOutputT = TypeVar("JobOutputT", bound=BaseModel)


def require_container_runtime(backend_label: str, *, num_nodes: int = 1) -> None:
    """Refuse to compile unless the platform can run the requested container job.

    SFT backends (automodel / unsloth) build a container ``PlatformJobSpec`` the
    platform runs on either supported target:

    - **Kubernetes** — the platform schedules GPU pods, including multi-node
      ``gpu_distributed`` jobs via Volcano.
    - **Docker** — the platform's local Docker GPU executor (single host),
      detected via :func:`~nemo_platform_plugin.capabilities.probe_docker`
      rather than treating ``Runtime.DOCKER`` / ``Runtime.NONE`` as the
      capability signal (AIRCORE-971).

    Single-node jobs accept Kubernetes topology or a reachable Docker daemon.
    **Multi-node jobs** (``num_nodes > 1``) compile to a ``gpu_distributed``
    executor that only the Volcano (Kubernetes) backend can place — Docker has
    no multi-node/``gpu_distributed`` backend — so they require
    ``platform.runtime: kubernetes``. Failing here surfaces the misconfiguration at
    compile time instead of as an opaque "no backend found" scheduling error.
    """
    platform_config = NemoPlatformConfig.get()
    runtime = platform_config.runtime

    if num_nodes > 1 and runtime != Runtime.KUBERNETES:
        raise PlatformJobCompilationError(
            f"{backend_label} multi-node training (num_nodes={num_nodes}) requires "
            "platform.runtime: kubernetes — multi-node jobs run on the Volcano "
            "(gpu_distributed) backend, which has no Docker equivalent. "
            f"Current runtime: {runtime.value}.",
        )

    if runtime == Runtime.KUBERNETES:
        return

    # Capability probe — independent of Runtime.DOCKER vs soft-downgraded NONE.
    # Use the process cache so compile agrees with the jobs registry boot probe.
    # Mid-process "start Docker then retry compile" requires a platform restart.
    result = probe_docker()
    if result.available:
        return

    detail = result.detail or "Docker daemon is unavailable"
    raise PlatformJobCompilationError(
        f"{backend_label} training requires a reachable Docker daemon (or platform.runtime: kubernetes). {detail}",
    )


def require_distributed_runtime(backend_label: str) -> None:
    """Refuse to compile when the platform isn't a remote Kubernetes cluster.

    Sibling to :func:`require_container_runtime` for backends that provision a Ray
    cluster (e.g. NeMo-RL DPO). Unlike the SFT backends, these accept **only**
    Kubernetes (no Docker fallback): they need the platform's Kubernetes/Volcano
    scheduler to place GPU pods and inject the distributed env
    (``RANK``/``WORLD_SIZE``/``MASTER_ADDR``). Surface the misconfiguration before
    the Jobs API rejects the spec.
    """
    platform_config = NemoPlatformConfig.get()
    if platform_config.runtime != Runtime.KUBERNETES:
        raise PlatformJobCompilationError(
            f"{backend_label} training requires platform.runtime: kubernetes — it provisions a Ray "
            "cluster on the remote GPU cluster and has no local Docker fallback.",
        )


class BaseSubmitJob(NemoJob, Generic[JobInputT, JobOutputT]):
    """Shared submit-only job scaffold.

    Subclasses set the ``NemoJob`` ClassVars (``name``, ``description``,
    ``job_collection_path``, ``input_spec_schema``, ``spec_schema``), implement
    :meth:`_transform` and :meth:`compile`, and may set :attr:`runtime_label`.
    """

    dependencies: ClassVar[list[str]] = ["entities", "auth", "jobs", "secrets", "files", "models"]
    #: Human-readable backend name used in the runtime guard messages.
    runtime_label: ClassVar[str] = "Training"

    @classmethod
    def _job_input_schema(cls) -> type[JobInputT]:
        """Return the concrete submitter-facing schema for this backend."""
        raise PlatformJobCompilationError(f"{cls.__name__} is missing an input_spec_schema.")

    @classmethod
    async def _transform(
        cls,
        job_input: JobInputT,
        workspace: str,
        platform: AsyncCustomizationPlatformClients,
    ) -> JobOutputT:
        """Validate platform refs and return the canonical output spec. Per backend."""
        raise NotImplementedError

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool,
    ) -> JobOutputT:
        """Validate platform refs, resolve naming, return the canonical spec."""
        del entity_client, is_local
        schema = cls._job_input_schema()
        job_input = input_spec if isinstance(input_spec, schema) else schema.model_validate(input_spec.model_dump())
        platform = async_customization_platform_clients_from_platform(async_sdk)
        return await cls._transform(job_input, workspace, platform)
