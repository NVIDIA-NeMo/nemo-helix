# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI overrides for the Unsloth contributor.

The override machinery is shared in :mod:`nmp.customization_common.cli.overrides`; this
module supplies the Unsloth specifics: the ``UnslothJobInput`` schema (via
``load_job_json``) and the ``JOB_JSON`` help text.
"""

import json
from pathlib import Path

import typer
from nmp.customization_common.cli.overrides import apply_job_cli_overrides

from nemo_unsloth_plugin.schema import UnslothJobInput

_JOB_JSON_HELP = "Path to Unsloth job JSON (UnslothJobInput schema)."


_SUBMIT_HELP = """Submit an Unsloth training job to the platform.

Pass the path to a job JSON file holding one UnslothJobInput object: the base
model, the dataset, and how to train it.

Submit validates the file before creating the job, so an invalid field is
reported immediately. The platform then creates the job and runs it on the
execution profile resolved for this backend.

Submit prints the created job as JSON on stdout. The 'name' field is the job
id. Track the job with 'nemo jobs watch <job id>', or check its status with
'nemo jobs get-status <job id>'.

Run 'nemo customization unsloth explain' to print the job JSON schema."""


def load_job_json(path: Path) -> str:
    """Load and validate job JSON; return canonical JSON string for ``--spec``."""
    data = json.loads(path.read_text())
    validated = UnslothJobInput.model_validate(data)
    return validated.model_dump_json()


def apply_unsloth_job_cli_overrides(group: typer.Typer) -> None:
    """Flat ``unsloth`` CLI: ``submit JOB.json``."""
    apply_job_cli_overrides(
        group,
        load_job_json=load_job_json,
        job_json_help=_JOB_JSON_HELP,
        submit_help=_SUBMIT_HELP,
    )
