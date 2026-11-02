# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick a task step's executor from the execution profiles the platform registered."""

from __future__ import annotations

from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import InternalServerError, NemoResponseValidationError, NemoTransportError
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    ExecutorSpec,
    SubprocessExecutionProviderSpec,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.image import get_qualified_image


async def resolve_task_executor(
    *,
    profile: str,
    async_sdk: AsyncNemoClient,
    image: str,
    entrypoint: list[str],
    command: list[str],
    step: str,
) -> ExecutorSpec:
    """Prefer a ``subprocess`` profile named *profile*; otherwise run *image* on the ``cpu`` profile.

    Local platforms register a subprocess backend, so the step runs in the host's own venv. Helm and
    Minikube register none, so the step runs in a container on whichever backend the deployment
    registered as ``cpu/<profile>`` (docker or kubernetes_job).
    """
    try:
        profiles = (await client_from_platform(async_sdk, AsyncJobsClient).get_execution_profiles()).data()
    except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
        raise HelixJobDependencyUnavailableError(
            f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
            "unavailable.  Retry the submission."
        ) from exc

    if any(
        isinstance(candidate, SubprocessJobExecutionProfile) and candidate.profile == profile for candidate in profiles
    ):
        return SubprocessExecutionProviderSpec(provider="subprocess", profile=profile, command=[*entrypoint, *command])

    if any(candidate.provider == "cpu" and candidate.profile == profile for candidate in profiles):
        return CPUExecutionProviderSpec(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(image=get_qualified_image(image), entrypoint=entrypoint, command=command),
        )

    available = sorted({f"{candidate.provider}/{candidate.profile}" for candidate in profiles})
    raise HelixJobCompilationError(
        f"No 'subprocess' or 'cpu' execution profile named {profile!r} is registered, so the "
        f"{step} step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )
