# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create NeMo Helix Intake Experiments (and their Groups) for scaled-evals runs.

Each benchmark run (or standalone evaluation) maps to one Experiment; its member
tasks are ``test_case_id``s and each trial is a session. One Group per benchmark.
The Experiment name is ``{benchmark}-{run-id}`` — readable, workspace-unique,
and stable across member tasks; structured detail lives in the entity
``metadata`` (str->str).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scaled_evals.intake.client import create_evaluation, create_experiment_group

# Version stamp of the nemo-helix intake contract this upload targets. Bump on contract changes;
# stamped onto every Evaluation so a run records which contract produced it.
INTAKE_CONTRACT_REF = "nemo-helix@3df96dd4d7cadeb6fbe5049696b40cfa54bc5c5b"
MAX_INTAKE_EVALUATION_NAME_LENGTH = 63


def str_metadata(values: Mapping[str, Any]) -> dict[str, str]:
    """Coerce a metadata mapping to the entity's ``dict[str, str]`` contract, dropping empties."""
    out: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        text = str(value)
        if text:
            out[key] = text
    return out


@dataclass(frozen=True)
class ExperimentRequest:
    """Run-context inputs for a model-independent Experiment identity."""

    benchmark: str  # groups the experiment and names its dataset
    run_key: str  # full benchmark_run_id or evaluation id; cross-member dedup key
    dataset_version: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    group_metadata: dict[str, str] = field(default_factory=dict)


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "unknown"


def _intake_name_slug(value: str) -> str:
    text = slugify(value)
    if not text[0].isalpha():
        text = f"x-{text}"
    return text


def _truncate_slug(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip("-") or value[:max_length].strip("-") or "x"


def build_experiment_name(benchmark: str, run_key: str) -> str:
    benchmark_slug = _intake_name_slug(benchmark)
    run_slug = slugify(run_key)
    suffix = f"-{run_slug}"
    benchmark_budget = MAX_INTAKE_EVALUATION_NAME_LENGTH - len(suffix)
    if benchmark_budget < 1:
        raise ValueError(f"run key is too long for an Intake Experiment name: {run_key!r}")
    benchmark_slug = _truncate_slug(benchmark_slug, benchmark_budget)
    return f"{benchmark_slug}{suffix}"


def ensure_experiment(
    base_url: str,
    workspace: str,
    request: ExperimentRequest,
    timeout: float,
) -> str:
    """Ensure the Group and Experiment exist; return the Experiment name (the ``evaluation_id``).

    Idempotent across a benchmark run's members: every member uses the same full
    run key, so the second and later creates get a harmless 409.
    """
    group_id = create_experiment_group(base_url, workspace, slugify(request.benchmark), request.group_metadata, timeout)
    name = build_experiment_name(request.benchmark, request.run_key)
    body: dict[str, object] = {
        "name": name,
        "experiment_ids": [group_id],
        "dataset_name": request.benchmark,
        # Always record the intake contract version this run was produced against.
        "metadata": {**request.metadata, "intake_contract_ref": INTAKE_CONTRACT_REF},
    }
    if request.dataset_version:
        body["dataset_version"] = request.dataset_version
    create_evaluation(base_url, workspace, body, timeout)
    return name
