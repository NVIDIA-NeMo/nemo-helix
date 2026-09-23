# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress reporting for Unsloth training tasks.

Thin subclass of the shared
:class:`nhx.customization_common.training.progress.JobsServiceProgressReporter`
that bakes in the unsloth ``SERVICE_NAME`` so callers keep the
``JobsServiceProgressReporter(job_ctx)`` constructor.
"""

from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.progress import (
    JobsServiceProgressReporter as _BaseJobsServiceProgressReporter,
)
from nhx.unsloth.app.constants import SERVICE_NAME

__all__ = ["JobsServiceProgressReporter"]


class JobsServiceProgressReporter(_BaseJobsServiceProgressReporter):
    """Unsloth training progress reporter (binds the unsloth service name)."""

    def __init__(self, job_ctx: NHXJobContext):
        super().__init__(job_ctx, service_name=SERVICE_NAME)
