# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-host provider protocol and registry."""

from typing import Protocol, TypeVar

from sandboxed_gym.host.models import GymHostHandle, GymHostSpec

TProvider = TypeVar("TProvider")


class SandboxedGymHostProvider(Protocol[TProvider]):
    """Provisions and tears down the job-level Gym host sandbox."""

    name: str
    # Concrete provider implementation stored on ``GymHostHandle.provider``.
    provider_class: type[TProvider]

    async def create_host(self, spec: GymHostSpec) -> GymHostHandle[TProvider]:
        """Create the host and return actor-reachable health/rollout URLs."""

    async def wait_ready(self, handle: GymHostHandle[TProvider], timeout_s: float) -> None:
        """Block until ``GET health_url`` reports ready, or raise on timeout."""

    async def destroy_host(self, handle: GymHostHandle[TProvider]) -> None:
        """Terminate the host. Best-effort; must not raise after a successful destroy."""

    async def destroy_job_sandboxes(self, job_id: str) -> tuple[str, ...]:
        """Destroy every sandbox this provider still has for ``job_id``.

        Cancel and process exit do not hold the handles ``destroy_host`` needs. The
        implementation finds them by the job id stamped at create time.
        """
        ...


def get_host_provider(name: str, options: dict | None = None) -> SandboxedGymHostProvider:
    """Construct a registered job-host provider by name."""
    options = options or {}
    if name == "opensandbox":
        from sandboxed_gym.host.opensandbox import (
            OpenSandboxGymHostProvider,
        )

        return OpenSandboxGymHostProvider(**options)
    if name == "docker":
        # Local execution only -- see the module docstring: a container is not an isolation
        # boundary and this provider applies no egress policy.
        from sandboxed_gym.host.docker import DockerGymHostProvider

        return DockerGymHostProvider(**options)
    raise ValueError(f"Unknown sandboxed gym host provider: {name!r}")
