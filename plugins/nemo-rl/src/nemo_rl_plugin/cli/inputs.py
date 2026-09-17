# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI overrides for the NeMo-RL contributor.

The override machinery is shared in :mod:`nmp.customization_common.cli.overrides`;
this module supplies the RL specifics: the ``RlJobInput`` schema (via
``load_job_json``) and the ``JOB_JSON`` help text.
"""

import json
from pathlib import Path

import typer
from nmp.customization_common.cli.overrides import apply_job_cli_overrides

from nemo_rl_plugin.schema import RlJobInput

_JOB_JSON_HELP = "Path to NeMo-RL job JSON (RlJobInput schema)."


_SUBMIT_HELP = """Submit a NeMo-RL training job to the platform.

Pass the path to a job JSON file holding one RlJobInput object: the base
model, the dataset, and how to align it. Set training.type to 'dpo' or 'grpo';
GRPO also needs an environment fileset. Submit fails immediately if the
platform has no kubernetes_job backend.

Submit validates the file before creating the job, so an invalid field is
reported immediately. The platform then creates the job and runs it on the
execution profile resolved for this backend.

Submit prints the created job as JSON on stdout. The 'name' field is the job
id. Track the job with 'nemo jobs watch <job id>', or check its status with
'nemo jobs get-status <job id>'.

Run 'nemo customization rl explain' to print the job JSON schema."""


def load_job_json(path: Path) -> str:
    """Load and validate job JSON; return canonical JSON string for ``--spec``."""
    data = json.loads(path.read_text())
    validated = RlJobInput.model_validate(data)
    return validated.model_dump_json()


def apply_rl_job_cli_overrides(group: typer.Typer) -> None:
    """Flat ``rl`` CLI: ``submit JOB.json``."""
    apply_job_cli_overrides(
        group,
        load_job_json=load_job_json,
        job_json_help=_JOB_JSON_HELP,
        submit_help=_SUBMIT_HELP,
    )
