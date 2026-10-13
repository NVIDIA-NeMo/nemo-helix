# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker-specific job validation (e.g. GPU availability for Docker backends)."""

import logging
from collections.abc import Mapping
from typing import Any

from nemo_helix_plugin.capabilities import probe_docker
from nemo_helix_plugin.config import Configuration, NemoHelixConfig, Runtime
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from pydantic import ValidationError

logger = logging.getLogger(__name__)

HelixJobSpecLike = HelixJobSpec | Mapping[str, Any]


def get_platform_config() -> NemoHelixConfig:
    return Configuration.get_service_config(NemoHelixConfig)


_GPU_UNAVAILABLE_MSG = (
    "This job requires a GPU. The platform is running on Docker with no GPUs "
    "configured. Configure GPUs in platform config or use a GPU-enabled environment."
)


def spec_has_gpu_step(job: HelixJobSpecLike) -> bool:
    """Return True if the job spec has at least one step requiring a GPU."""
    platform_job = HelixJobSpec.model_validate(job)
    for step in platform_job.steps:
        if step.executor.provider in ("gpu", "gpu_distributed"):
            return True
    return False


def _docker_gpu_validation_applies(runtime: Runtime) -> bool:
    """Whether reserved-GPU checks apply for this platform runtime.

    ``Runtime.DOCKER`` always applies. Soft-downgraded ``Runtime.NONE`` still
    runs Docker-backed jobs when the daemon is reachable, so reserved-GPU
    checks must not be skipped solely because runtime is not DOCKER.
    """
    if runtime == Runtime.DOCKER:
        return True
    if runtime == Runtime.NONE:
        return probe_docker().available
    return False


def validate_gpu_available_for_docker(job: HelixJobSpecLike) -> None:
    """Fail fast when job requires GPU but platform Docker has no GPUs configured.

    platform_config.docker.get_reserved_gpu_ids() returns:
    - None when reserved_gpu_device_ids is "all" (auto-detect GPUs)
    - [] when reserved_gpu_device_ids is "none" or empty (GPUs disabled)
    - list[int] when reserved_gpu_device_ids is explicit IDs (e.g. "0,1,2")

    We only run GPU validation when reserved_ids is an empty list (no GPUs configured).
    When reserved_ids is None (auto-detect) or a non-empty list (explicit IDs), we
    skip validation and assume GPUs are available. If the job has a GPU step and
    reserved_ids is [], we raise HelixJobCompilationError so the user gets a
    clear 422 at create time instead of a job that fails with no logs.
    If platform config cannot be loaded, skip validation (e.g. environments without
    this config).
    """
    try:
        platform_config = get_platform_config()
    except (ValueError, OSError, ValidationError):
        logger.debug("Skipping GPU availability validation: could not load platform config")
        return

    if not _docker_gpu_validation_applies(platform_config.runtime):
        return

    # None = "all" (auto-detect), [] = "none"/empty (no GPUs), list[int] = explicit IDs.
    # Skip validation when auto-detect (None) or when GPUs are configured (non-empty list).
    # Only when reserved_ids is [] do we continue and potentially raise.
    reserved_ids = platform_config.docker.get_reserved_gpu_ids()
    if reserved_ids is None:
        return
    if len(reserved_ids) > 0:
        return

    if not spec_has_gpu_step(job):
        return

    raise HelixJobCompilationError(_GPU_UNAVAILABLE_MSG)
