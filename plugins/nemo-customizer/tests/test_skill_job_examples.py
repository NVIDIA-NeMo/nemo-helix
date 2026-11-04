# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the skills' job JSON examples against the job schemas.

An agent copies these examples into job JSON, so an example the schema rejects fails
at submit. Each one goes through the backends' own ``--job-json`` validators, and the
schemas forbid unknown keys, so a renamed or removed field fails here first.
"""

import copy
import json
from collections.abc import Callable, Iterator

import pytest
from nemo_automodel_plugin.cli.inputs import validate_job_spec as validate_automodel
from nemo_automodel_plugin.schema import AutomodelJobInput
from nemo_customizer.skills import get_skills_path
from nemo_rl_plugin.cli.inputs import validate_job_spec as validate_rl
from nemo_rl_plugin.schema import RlJobInput
from nemo_unsloth_plugin.cli.inputs import validate_job_spec as validate_unsloth
from nemo_unsloth_plugin.schema import UnslothJobInput

SKILLS = ["nemo-customizer", "nemo-gym-environments"]

# The smallest spec each backend accepts. An example is merged over it, so a fragment
# such as `"batch": {...}` is checked inside a complete job.
BACKENDS: dict[str, tuple[Callable[[dict], str], dict]] = {
    "automodel": (
        validate_automodel,
        {"model": "default/base", "dataset": {"training": "default/data"}, "training": {"training_type": "sft"}},
    ),
    "unsloth": (validate_unsloth, {"model": {"name": "default/base"}, "dataset": {"path": "default/data"}}),
    "rl": (validate_rl, {"model": "default/base", "dataset": "default/data", "training": {"type": "dpo"}}),
}

JOB_FIELDS = set(AutomodelJobInput.model_fields) | set(UnslothJobInput.model_fields) | set(RlJobInput.model_fields)


def _json_blocks() -> Iterator[tuple[str, str]]:
    """Yield ``(location, body)`` for every ```json fence in the skills."""
    for skill in SKILLS:
        skill_dir = get_skills_path() / skill
        for doc in [skill_dir / "SKILL.md", *sorted((skill_dir / "references").glob("*.md"))]:
            start: int | None = None
            body: list[str] = []
            for number, line in enumerate(doc.read_text().splitlines(), start=1):
                if start is None:
                    if line.strip() == "```json":
                        start, body = number, []
                elif line.strip() == "```":
                    yield f"{skill}/{doc.relative_to(skill_dir)}:{start}", "\n".join(body)
                    start = None
                else:
                    body.append(line)


def _parse(body: str) -> object:
    """Parse one value, a fragment of top-level pairs such as `"batch": {...}`, or JSONL rows."""
    for text in (body, "{" + body + "}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _merge(base: dict, example: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in example.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(example: dict) -> dict[str, str]:
    """Return each backend's error; empty when any backend accepts the example."""
    errors: dict[str, str] = {}
    for backend, (validate, base) in BACKENDS.items():
        try:
            validate(_merge(base, example))
        except ValueError as exc:
            errors[backend] = str(exc)
        else:
            return {}
    return errors


BLOCKS = list(_json_blocks())
JOB_EXAMPLES = [
    (location, example)
    for location, body in BLOCKS
    if isinstance(example := _parse(body), dict) and example and set(example) <= JOB_FIELDS
]


@pytest.mark.parametrize("body", [body for _, body in BLOCKS], ids=[location for location, _ in BLOCKS])
def test_json_example_parses(body: str) -> None:
    _parse(body)


@pytest.mark.parametrize("example", [e for _, e in JOB_EXAMPLES], ids=[location for location, _ in JOB_EXAMPLES])
def test_job_example_matches_a_backend_schema(example: dict) -> None:
    errors = _validate(example)
    assert not errors, "No backend accepts this example:\n" + "\n".join(f"{b}: {e}" for b, e in errors.items())


def test_every_backend_has_a_full_example() -> None:
    """Guards the collection itself: a parsing regression would otherwise pass vacuously."""
    covered = set()
    for _, example in JOB_EXAMPLES:
        if {"model", "training"} <= set(example):
            for backend, (validate, _) in BACKENDS.items():
                try:
                    validate(example)
                except ValueError:
                    continue
                covered.add(backend)
    assert covered == set(BACKENDS)


def test_unknown_field_is_rejected() -> None:
    """The check relies on the schemas rejecting unknown keys."""
    assert _validate({"training": {"no_such_field": 1}})
