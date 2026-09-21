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
from nmp.customization_common.cli.uploads import SpecRefs

from nemo_automodel_plugin.schema import AutomodelJobInput

_JOB_JSON_HELP = "Path to Automodel job JSON (AutomodelJobInput schema)."


_SUBMIT_HELP = """Submit an Automodel training job to the platform.

Pass the path to a job JSON file holding one AutomodelJobInput object: the
base model, the dataset, and how to train it.

Submit validates the file before creating the job, so an invalid field is
reported immediately. The platform then creates the job and runs it on the
execution profile resolved for this backend.

Submit prints the created job as JSON on stdout. The 'name' field is the job
id. Track the job with 'nemo jobs watch <job id>', or check its status with
'nemo jobs get-status <job id>'.

Pass --wait to stay with the job until it finishes, or --watch to do the same
and print its logs as they arrive. Both report the final status, and exit
non-zero when the job does not complete.

Pass --upload-model with a local path or a HuggingFace repo id, and
--upload-dataset with a local path, to create those resources as part of this
submit. Either flag can be used on its own. Submit creates the fileset, uploads
the files, registers the model entity, and fills the reference into the job it
sends. Your job JSON file is not modified.

Each flag takes one file or one directory, the same as 'nemo files upload'. A
directory is uploaded whole, so put several files in one when a backend reads
more than one.

A reference must be set in one place only. Submit refuses when the job JSON
already names the model or dataset and the matching flag is passed, rather than
choosing between the two for you. Remove the field from the job JSON to create
it here, or drop the flag to use what the file names.

Run 'nemo customization --upload-model ... --upload-dataset ...' instead to
create the resources and print their references without submitting anything.

One --upload-dataset fills both dataset.training and dataset.validation with
the new fileset. Automodel finds the train and validation files inside it by
name, so upload a directory holding both.

Creating a resource that already exists fails unless you pass --exist-ok,
which applies to whichever sources you gave.

Important: with --exist-ok the existing fileset is reused and its files are not
re-uploaded. Edits you made to a local file after that fileset was created are
not part of this job, and nothing fails to tell you so. Delete the fileset, or
upload into it yourself, when you need the new contents.

Run 'nemo customization automodel explain' to print the job JSON schema."""


_SPEC_REFS = SpecRefs(
    model=("model",),
    dataset=("dataset", "training"),
    dataset_validation=("dataset", "validation"),
)


def validate_job_spec(data: dict) -> str:
    """Validate a job spec; return the canonical JSON string for ``--spec``.

    ``exclude_unset`` matters here: ``with_resolved_recipe`` fills recipe
    defaults only for fields absent from ``model_fields_set``, so a full dump
    presents every schema default as an explicit choice and suppresses them.
    """
    return AutomodelJobInput.model_validate(data).model_dump_json(exclude_unset=True)


def load_job_json(path: Path) -> str:
    """Load and validate job JSON; return canonical JSON string for ``--spec``."""
    return validate_job_spec(json.loads(path.read_text()))


def apply_automodel_job_cli_overrides(group: typer.Typer) -> None:
    """Flat ``automodel`` CLI: ``submit JOB.json``."""
    apply_job_cli_overrides(
        group,
        backend="automodel",
        load_job_json=load_job_json,
        validate_job_spec=validate_job_spec,
        spec_refs=_SPEC_REFS,
        job_json_help=_JOB_JSON_HELP,
        submit_help=_SUBMIT_HELP,
    )
