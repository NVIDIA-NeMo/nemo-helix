# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile
from nhx.core.jobs.controllers.backends.kubernetes import (
    KubernetesJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nhx.core.jobs.controllers.backends.subprocess import SubprocessJobExecutionProfile
from nhx.core.jobs.controllers.backends.test import E2EJobExecutionProfile

ExecutionProfileT = Union[
    DockerJobExecutionProfile,
    KubernetesJobExecutionProfile,
    VolcanoJobExecutionProfile,
    SubprocessJobExecutionProfile,
    E2EJobExecutionProfile,
]
