# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI overrides for the Automodel contributor.

The override machinery is shared in :mod:`nmp.customization_common.cli.overrides`; this
module supplies the Automodel specifics: the ``AutomodelJobInput`` schema (via
``load_job_json``) and the ``JOB_JSON`` help text.
"""

import json
from pathlib import Path

import typer
from nmp.customization_common.cli.overrides import apply_job_cli_overrides

from nemo_automodel_plugin.schema import AutomodelJobInput

_JOB_JSON_HELP = "Path to Automodel job JSON (AutomodelJobInput schema)."


def load_job_json(path: Path) -> str:
    """Load and validate job JSON; return canonical JSON string for ``--spec``.

    ``exclude_unset`` is load-bearing: ``with_resolved_recipe`` fills recipe
    defaults only for fields absent from ``model_fields_set``, so a full dump
    presents every schema default as an explicit choice and suppresses them.
    """
    data = json.loads(path.read_text())
    validated = AutomodelJobInput.model_validate(data)
    return validated.model_dump_json(exclude_unset=True)


def apply_automodel_job_cli_overrides(group: typer.Typer) -> None:
    """Flat ``automodel`` CLI: ``submit JOB.json``."""
    apply_job_cli_overrides(
        group,
        load_job_json=load_job_json,
        job_json_help=_JOB_JSON_HELP,
    )
