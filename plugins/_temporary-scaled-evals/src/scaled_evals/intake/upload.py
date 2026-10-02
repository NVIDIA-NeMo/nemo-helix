# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run Harbor ATIF upload to NeMo Helix Intake.

Creates the run's Experiment (idempotent, one per benchmark run) then POSTs each
trial's ATIF trajectory tagged with that experiment via ``evaluation_context``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.intake.atif_payload import (
    IntakeError,
    load_json_if_exists,
    switchyard_model_metrics,
    trial_payloads,
)
from scaled_evals.intake.client import post_atif_payload
from scaled_evals.intake.config import IntakeTarget
from scaled_evals.intake.experiments import (
    ExperimentRequest,
    build_experiment_name,
    ensure_experiment,
)

LOG = logging.getLogger(__name__)
_ATTEMPT_COUNT_RE = re.compile(r"(?:after |attempts=)(\d+) attempts?\b|attempts=(\d+)\b")


@dataclass(frozen=True)
class UploadResult:
    uploaded: int
    job_dir: Path
    experiment_ref: str
    run_refs: tuple[str, ...]


def upload_job_atif(
    job_dir: Path,
    target: IntakeTarget,
    *,
    evaluation_run_id: str,
    experiment: ExperimentRequest,
    test_case_id: str | None = None,
    timeout: float = 30.0,
) -> UploadResult:
    """Create the run's Experiment, then POST every trial ATIF trajectory to Intake.

    Native trial trajectories are uploaded as-is. Missing or unreadable
    trajectories produce minimal task-scoped ATIF records so every completed
    task contributes to the Experiment denominator.
    """
    if not job_dir.is_dir():
        raise IntakeError(f"Harbor job directory does not exist: {job_dir}")

    routing_stats = load_json_if_exists(job_dir / "switchyard" / "routing_stats_final.json")
    per_model_metrics = switchyard_model_metrics(routing_stats)
    if per_model_metrics:
        experiment = replace(
            experiment,
            metadata={
                **experiment.metadata,
                **{
                    metric: json.dumps(values, separators=(",", ":"), sort_keys=True)
                    for metric, values in per_model_metrics.items()
                },
            },
        )

    evaluation_id = build_experiment_name(experiment.benchmark, experiment.run_key)
    payloads = trial_payloads(
        job_dir,
        target.workspace,
        target.app,
        target.source,
        evaluation_run_id=evaluation_run_id,
        evaluation_id=evaluation_id,
        test_case_id=test_case_id,
        routing_stats=routing_stats,
    )
    ensured_id = ensure_experiment(target.base_url, target.workspace, experiment, timeout)
    if ensured_id != evaluation_id:
        raise IntakeError(
            f"Intake Experiment identity changed while preparing upload: {evaluation_id!r} != {ensured_id!r}"
        )
    for item in payloads:
        post_atif_payload(target.base_url, target.workspace, item, timeout)
    return UploadResult(
        uploaded=len(payloads),
        job_dir=job_dir,
        experiment_ref=evaluation_id,
        run_refs=tuple(item.external_id for item in payloads),
    )


def upload_job_atif_warn(
    job_dir: Path,
    target: IntakeTarget,
    *,
    evaluation_run_id: str,
    experiment: ExperimentRequest,
    test_case_id: str | None = None,
    timeout: float = 30.0,
    fail_on_error: bool = False,
) -> str | None:
    """Create the Experiment + upload ATIF; return a human note or ``None`` on success/no-op.

    Logs warnings on failure unless ``fail_on_error`` is set (then re-raises).
    """
    experiment_ref = build_experiment_name(experiment.benchmark, experiment.run_key)
    try:
        result = upload_job_atif(
            job_dir,
            target,
            evaluation_run_id=evaluation_run_id,
            experiment=experiment,
            test_case_id=test_case_id,
            timeout=timeout,
        )
    except IntakeError as exc:
        detail = redact_secret_text(str(exc))[:2000]
        match = _ATTEMPT_COUNT_RE.search(detail)
        attempts = int(next(group for group in match.groups() if group)) if match else None
        _write_upload_diagnostic(
            job_dir,
            status="failed",
            uploaded=0,
            error_type=type(exc).__name__,
            error=detail,
            attempts=attempts,
            experiment_ref=experiment_ref,
        )
        if fail_on_error:
            raise
        LOG.warning("Intake ATIF upload failed for %s", job_dir, exc_info=True)
        return f"intake ATIF upload failed: {detail}"
    if result.uploaded == 0:
        _write_upload_diagnostic(
            job_dir,
            status="no_records",
            uploaded=0,
            experiment_ref=result.experiment_ref,
            run_refs=result.run_refs,
        )
        return "intake: no ATIF task records found in job dir"
    _write_upload_diagnostic(
        job_dir,
        status="succeeded",
        uploaded=result.uploaded,
        experiment_ref=result.experiment_ref,
        run_refs=result.run_refs,
    )
    return f"intake: uploaded {result.uploaded} ATIF task records"


def _write_upload_diagnostic(
    job_dir: Path,
    *,
    status: str,
    uploaded: int,
    error_type: str | None = None,
    error: str | None = None,
    attempts: int | None = None,
    experiment_ref: str | None = None,
    run_refs: tuple[str, ...] = (),
) -> None:
    diagnostic = {
        "schema_version": "scaled-evals-intake-upload-v1",
        "recorded_at": datetime.now(tz=UTC).isoformat(),
        "status": status,
        "uploaded": uploaded,
        "attempts": attempts,
        "error_type": error_type,
        "error": error,
        "experiment_ref": experiment_ref,
        "run_refs": list(run_refs),
    }
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "intake-upload.json").write_text(
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        LOG.warning("failed to persist Intake upload diagnostic for %s", job_dir, exc_info=True)
