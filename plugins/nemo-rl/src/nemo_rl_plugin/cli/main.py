# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI hooks for the NeMo-RL customization contributor.

The plugin's CLI surface is auto-mounted by the customization hub via
:meth:`RlContributor.get_cli`. This class provides the ``add_job_commands``
integration hook for any caller that builds the CLI through that helper instead.
"""

from __future__ import annotations

import typer
from nemo_helix_plugin.job import NemoJob

from nemo_rl_plugin.cli.inputs import apply_rl_job_cli_overrides
from nemo_rl_plugin.jobs.jobs import RlJob


class RlContributorCLI:
    """Passed to ``add_job_commands`` to override submit with job-file args."""

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        if job_cls is RlJob:
            apply_rl_job_cli_overrides(group)
