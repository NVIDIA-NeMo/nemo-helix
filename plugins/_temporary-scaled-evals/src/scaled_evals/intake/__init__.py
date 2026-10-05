# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Helix Intake integration.

Post-run ATIF upload is handled directly by dispatch: each finished job creates
its run's Experiment (one per benchmark run) and POSTs the trials' ATIF
trajectories tagged with that experiment.
"""

from scaled_evals.intake.config import IntakeTarget, resolve_intake_target
from scaled_evals.intake.experiments import (
    ExperimentRequest,
    build_experiment_name,
    ensure_experiment,
)
from scaled_evals.intake.upload import UploadResult, upload_job_atif, upload_job_atif_warn

__all__ = [
    "ExperimentRequest",
    "IntakeTarget",
    "UploadResult",
    "build_experiment_name",
    "ensure_experiment",
    "resolve_intake_target",
    "upload_job_atif",
    "upload_job_atif_warn",
]
