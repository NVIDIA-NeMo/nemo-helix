# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI hooks for the Unsloth customization contributor.

The plugin's CLI surface is auto-mounted by the customization hub via
:meth:`UnslothContributor.get_cli`. This class provides the
``add_job_commands`` integration hook for any caller that builds the
CLI through that helper instead — both shapes apply the same overrides.
"""

from __future__ import annotations

import typer
from nemo_helix_plugin.job import NemoJob

from nemo_unsloth_plugin.cli.inputs import apply_unsloth_job_cli_overrides
from nemo_unsloth_plugin.jobs.jobs import UnslothJob


class UnslothContributorCLI:
    """Passed to ``add_job_commands`` to override submit with job-file args."""

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        if job_cls is UnslothJob:
            apply_unsloth_job_cli_overrides(group)
