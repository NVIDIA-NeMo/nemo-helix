# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI hooks for the Automodel customization contributor."""

from __future__ import annotations

import typer
from nemo_helix_plugin.job import NemoJob

from nemo_automodel_plugin.cli.inputs import apply_automodel_job_cli_overrides
from nemo_automodel_plugin.jobs.jobs import AutomodelJob


class AutomodelContributorCLI:
    """Passed to ``add_job_commands`` to override job submit with job-file args."""

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        if job_cls is AutomodelJob:
            apply_automodel_job_cli_overrides(group)
