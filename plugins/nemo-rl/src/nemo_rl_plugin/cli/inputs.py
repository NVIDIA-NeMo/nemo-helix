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
from nmp.customization_common.cli.uploads import SpecRefs

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

Uploaded files keep their local names. NeMo-RL reads both splits from one
fileset, so point --upload-dataset at a local directory holding training.jsonl
and validation.jsonl. For GRPO, --upload-environment takes the environment
directory. Build and validate that package first; submit only uploads it.

Creating a resource that already exists fails unless you pass --exist-ok,
which applies to whichever sources you gave.

Important: with --exist-ok the existing fileset is reused and its files are not
re-uploaded. Edits you made to a local file after that fileset was created are
not part of this job, and nothing fails to tell you so. Delete the fileset, or
upload into it yourself, when you need the new contents.

Run 'nemo customization rl explain' to print the job JSON schema."""


# NeMo-RL reads both splits from one dataset fileset, so a single ref covers both.
_SPEC_REFS = SpecRefs(
    model=("model",),
    dataset=("dataset",),
    environment=("environment",),
)


def validate_job_spec(data: dict) -> str:
    """Validate a job spec; return the canonical JSON string for ``--spec``."""
    return RlJobInput.model_validate(data).model_dump_json()


def load_job_json(path: Path) -> str:
    """Load and validate job JSON; return canonical JSON string for ``--spec``."""
    return validate_job_spec(json.loads(path.read_text()))


def apply_rl_job_cli_overrides(group: typer.Typer) -> None:
    """Flat ``rl`` CLI: ``submit JOB.json``."""
    apply_job_cli_overrides(
        group,
        backend="rl",
        load_job_json=load_job_json,
        validate_job_spec=validate_job_spec,
        spec_refs=_SPEC_REFS,
        job_json_help=_JOB_JSON_HELP,
        submit_help=_SUBMIT_HELP,
    )
