# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker-specific job validation compatibility wrapper."""

from collections.abc import Mapping
from typing import Any

from nemo_helix_plugin.jobs.docker import spec_has_gpu_step as spec_has_gpu_step
from nemo_helix_plugin.jobs.docker import validate_gpu_available_for_docker as _plugin_validate
from nemo_helix_plugin.jobs.spec import HelixJobSpec

HelixJobSpecLike = HelixJobSpec | Mapping[str, Any]


def validate_gpu_available_for_docker(job: HelixJobSpecLike) -> None:
    """Fail fast when job requires GPU but platform Docker has no GPUs configured.

    Delegates to :func:`nemo_helix_plugin.jobs.docker.validate_gpu_available_for_docker`
    so soft-downgraded ``Runtime.NONE`` with a reachable Docker daemon still
    enforces reserved-GPU checks (AIRCORE-971).
    """
    _plugin_validate(job)
