# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress reporting for RL training tasks.

Thin subclass of the shared
:class:`nhx.customization_common.training.progress.JobsServiceProgressReporter`
that bakes in the RL ``SERVICE_NAME`` so callers keep the
``JobsServiceProgressReporter(job_ctx)`` constructor. Mirrors the equivalent
modules in the unsloth and automodel services.
"""

from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.progress import (
    JobsServiceProgressReporter as _BaseJobsServiceProgressReporter,
)
from nhx.rl.app.constants import SERVICE_NAME

__all__ = ["JobsServiceProgressReporter"]


class JobsServiceProgressReporter(_BaseJobsServiceProgressReporter):
    """RL training progress reporter (binds the RL service name)."""

    def __init__(self, job_ctx: NHXJobContext):
        super().__init__(job_ctx, service_name=SERVICE_NAME)
