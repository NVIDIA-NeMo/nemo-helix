# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import JobsClient
from nhx.core.jobs.config import config

_MAX_LOG_ENTRIES = 20
_MAX_ERROR_STACK_CHARS = 2048


class JobDiagnosticTarget(Protocol):
    workspace: str
    job: str
    name: str


@dataclass(frozen=True)
class _JobDiagnosticRef:
    workspace: str
    job: str
    name: str


def _trim_error_stack(value: str | None) -> str | None:
    if value is None or len(value) <= _MAX_ERROR_STACK_CHARS:
        return value
    return value[-_MAX_ERROR_STACK_CHARS:]


def _trim_error_details(value: Any) -> Any:
    if isinstance(value, str):
        return _trim_error_stack(value)
    if isinstance(value, dict):
        return {key: _trim_error_details(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_trim_error_details(item) for item in value]
    return value


def _task_dict(task: Any) -> dict[str, Any]:
    return {
        "name": task.name,
        "status": task.status,
        "status_details": task.status_details,
        "error_details": _trim_error_details(task.error_details),
        "error_stack": _trim_error_stack(task.error_stack),
    }


def _step_dict(step: Any) -> dict[str, Any]:
    return {
        "name": step.name,
        "status": step.status,
        "status_details": step.status_details,
        "error_details": _trim_error_details(step.error_details),
    }


def _job_dict(job: Any) -> dict[str, Any]:
    return {
        "name": job.name,
        "status": job.status,
        "status_details": job.status_details,
        "error_details": _trim_error_details(job.error_details),
    }


def collect_job_diagnostics(
    sdk: NeMoHelix,
    step: JobDiagnosticTarget | None = None,
    *,
    workspace: str | None = None,
    job_name: str | None = None,
    step_name: str | None = None,
    context: str,
) -> dict[str, Any]:
    step_ref: JobDiagnosticTarget
    if step is None:
        if workspace is None or job_name is None or step_name is None:
            raise ValueError("Either step or workspace/job_name/step_name must be provided")
        step_ref = _JobDiagnosticRef(workspace=workspace, job=job_name, name=step_name)
    else:
        step_ref = step

    diagnostics: dict[str, Any] = {
        "diagnostic_context": context,
        "workspace": step_ref.workspace,
        "job_name": step_ref.job,
        "step_name": step_ref.name,
    }

    jobs = client_from_platform(sdk, JobsClient)

    try:
        job = jobs.get_job(name=step_ref.job, workspace=step_ref.workspace).data()
        diagnostics["job"] = _job_dict(job)
    except Exception as exc:
        diagnostics["job_error"] = str(exc)

    try:
        status = jobs.get_job_status(name=step_ref.job, workspace=step_ref.workspace).data()
        diagnostics["status_api"] = {
            "status": status.status,
            "status_details": status.status_details,
            "error_details": _trim_error_details(status.error_details),
            "steps": [
                {
                    **_step_dict(status_step),
                    "tasks": [_task_dict(task) for task in status_step.tasks],
                }
                for status_step in status.steps
            ],
        }
    except Exception as exc:
        diagnostics["status_api_error"] = str(exc)

    try:
        refreshed_step = jobs.get_job_step(name=step_ref.name, job=step_ref.job, workspace=step_ref.workspace).data()
        diagnostics["step"] = _step_dict(refreshed_step)
    except Exception as exc:
        diagnostics["step_error"] = str(exc)

    try:
        tasks = jobs.list_job_step_tasks(name=step_ref.name, job=step_ref.job, workspace=step_ref.workspace).data()
        diagnostics["tasks_api"] = [_task_dict(task) for task in tasks.data]
    except Exception as exc:
        diagnostics["tasks_api_error"] = str(exc)

    try:
        if config.include_job_logs_in_diagnostics:
            logs = jobs.list_job_logs(workspace=step_ref.workspace, name=step_ref.job).page()
            diagnostics["job_logs"] = [entry.message for entry in logs.items[-_MAX_LOG_ENTRIES:]]
    except Exception as exc:
        diagnostics["job_logs_error"] = str(exc)

    return diagnostics


def log_job_diagnostics_if_debug(
    sdk: NeMoHelix,
    step: JobDiagnosticTarget | None = None,
    *,
    logger: logging.Logger,
    workspace: str | None = None,
    job_name: str | None = None,
    step_name: str | None = None,
    context: str,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    step_ref: JobDiagnosticTarget
    if step is None:
        if workspace is None or job_name is None or step_name is None:
            raise ValueError("Either step or workspace/job_name/step_name must be provided")
        step_ref = _JobDiagnosticRef(workspace=workspace, job=job_name, name=step_name)
    else:
        step_ref = step

    logger.debug(
        "Job diagnostics snapshot",
        extra={
            "diagnostic_context": context,
            "workspace": step_ref.workspace,
            "job_name": step_ref.job,
            "step_name": step_ref.name,
            "job_diagnostics": collect_job_diagnostics(sdk, step_ref, context=context),
        },
    )
