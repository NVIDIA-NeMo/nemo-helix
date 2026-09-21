# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nemo_platform_plugin.jobs.types import (
    PlatformJobStepWithContext,
    PlatformJobTaskResponse,
    PlatformJobTaskUpdate,
)
from nmp.common.auth import AuthContext
from nmp.common.config import get_platform_config
from nmp.common.jobs.constants import (
    CONFIG_TASK_STORAGE_PATH_ENVVAR,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_SECRETS_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_NAME,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.app.providers import SubprocessExecutionProvider
from nmp.core.jobs.app.schemas import BaseExecutionProfile
from nmp.core.jobs.controllers.backends.base import (
    JobBackend,
    JobExecutionProfileConfig,
    JobUpdate,
    get_logs_endpoint_config_from_fileset,
)
from nmp.core.jobs.controllers.backends.subprocess_runtime import (
    SubprocessOtelLogger,
    create_otel_logger,
    inject_secret_env_vars,
    start_log_capture,
)
from pydantic import Field

logger = logging.getLogger(__name__)

SUBPROCESS_WORKDIR_STATUS_KEY = "subprocess_work_dir"
SUBPROCESS_PID_STATUS_KEY = "pid"
SUBPROCESS_PGID_STATUS_KEY = "pgid"
SUBPROCESS_PERSISTENT_STORAGE_STATUS_KEY = "subprocess_persistent_storage_path"
_MISSING_METADATA_PENDING_GRACE_SECONDS = 5
_ORPHAN_TERMINATE_TIMEOUT_SECONDS = 2.0
_ORPHAN_TERMINATE_POLL_SECONDS = 0.1
_ORPHANED_MESSAGE = "Job was cancelled: the platform stopped while this job was running"
SUBPROCESS_INHERITED_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "VIRTUAL_ENV",
        # A path on the build host, so operator-set like the rest of this list.
        "NEMO_AGENTS_WHEEL",
    }
)

_ERR_COMMAND_REQUIRED = "subprocess requires command to be set"


class SubprocessJobExecutionProfileConfig(JobExecutionProfileConfig):
    working_directory: str = Field(
        default="/tmp/nmp-subprocess-jobs",
        description="Root directory for subprocess job state, config, storage, and logs.",
    )
    graceful_shutdown_timeout_seconds: int = Field(
        default=10,
        description="How long to wait after SIGTERM before force killing the process group.",
    )
    orphan_recovery_grace_seconds: int = Field(
        default=60,
        description=(
            "How long a step may go without updates before a restarted controller treats its subprocess "
            "as orphaned (platform stopped mid-run) and cancels the step."
        ),
    )
    cleanup_completed_jobs_immediately: bool = Field(
        default=False,
        description="Keep subprocess working directories by default so runs remain inspectable.",
    )


class SubprocessJobExecutionProfile(BaseExecutionProfile):
    provider: Literal["subprocess"] = "subprocess"
    backend: Literal["subprocess"] = "subprocess"
    config: SubprocessJobExecutionProfileConfig = Field(
        default_factory=SubprocessJobExecutionProfileConfig,
        description="Additional configuration for the subprocess executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        return True


@dataclass(frozen=True)
class SubprocessProcessKey:
    workspace: str
    job: str
    attempt_id: str
    step: str


@dataclass
class SubprocessProcessMetadata:
    task_id: str
    process: subprocess.Popen[str]
    work_dir: Path
    log_path: Path
    persistent_dir: Path
    otel_logger: SubprocessOtelLogger | None = None
    closed_logs: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    log_threads: list[threading.Thread] = field(default_factory=list)


class SubprocessProcessRegistry:
    def __init__(self) -> None:
        self._processes: dict[SubprocessProcessKey, SubprocessProcessMetadata] = {}
        self._lock = threading.Lock()

    def values(self) -> list[SubprocessProcessMetadata]:
        with self._lock:
            return list(self._processes.values())

    def items(self) -> list[tuple[SubprocessProcessKey, SubprocessProcessMetadata]]:
        with self._lock:
            return list(self._processes.items())

    def get(self, key: SubprocessProcessKey) -> SubprocessProcessMetadata | None:
        with self._lock:
            return self._processes.get(key)

    def set(self, key: SubprocessProcessKey, metadata: SubprocessProcessMetadata) -> None:
        with self._lock:
            self._processes[key] = metadata

    def pop(self, key: SubprocessProcessKey) -> None:
        with self._lock:
            self._processes.pop(key, None)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._processes


class SubprocessJobBackend(JobBackend[SubprocessExecutionProvider, SubprocessJobExecutionProfileConfig]):
    BACKEND_NAME = "subprocess"

    def init(self) -> None:
        self._root_dir = Path(self._execution_profile_config.working_directory)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._process_registry = SubprocessProcessRegistry()
        # Subprocess execution state lives in this process's memory, so anything the jobs
        # database still reports as running from before this timestamp was left behind by a
        # previous controller (see _orphaned_step_update).
        self._started_at = datetime.datetime.now(datetime.timezone.utc)

    def shutdown(self) -> None:
        for metadata in self._process_registry.values():
            self._terminate_process(metadata, force=True)
            self._finish_logs(metadata)

    def schedule(self, executor_config: SubprocessExecutionProvider, step: PlatformJobStepWithContext) -> JobUpdate:
        if not executor_config.command:
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details={"message": _ERR_COMMAND_REQUIRED},
                error_details={"message": _ERR_COMMAND_REQUIRED},
            )

        key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
        existing = self._process_registry.get(key)
        if existing is not None and existing.process.poll() is None:
            return JobUpdate(
                status=PlatformJobStatus.PENDING,
                status_details={
                    "message": "Subprocess already running",
                    **self._task_status_details(existing),
                },
            )

        env, task_id, work_dir, log_path, persistent_dir = self._prepare_runtime(step)
        command = self._build_command(executor_config, env.get("VIRTUAL_ENV"))
        log_extra = {
            "command": command,
            "path": env.get("PATH"),
            "virtual_env": env.get("VIRTUAL_ENV"),
        }
        try:
            logger.debug("Starting subprocess job task", extra=log_extra)
            process = subprocess.Popen(
                command,
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            self._cleanup_failed_startup_dirs(work_dir, persistent_dir)
            message = f"Failed to start subprocess: executable not found: {command[0]}"
            logger.exception(message, extra=log_extra)
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details={"message": message},
                error_details={"message": message, "error": str(exc)},
            )
        except Exception as exc:
            self._cleanup_failed_startup_dirs(work_dir, persistent_dir)
            message = f"Failed to start subprocess: {exc}"
            logger.exception(message)
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details={"message": message},
                error_details={"message": message},
            )

        try:
            otel_logger = create_otel_logger(
                env=env,
                workspace=step.workspace,
                job=step.job,
                attempt_id=str(step.attempt_id),
                step=step.name,
                task_id=task_id,
            )
        except Exception as exc:
            self._terminate_process_group(process, force=True)
            self._cleanup_failed_startup_dirs(work_dir, persistent_dir)
            message = f"Failed to initialize subprocess runtime: {exc}"
            logger.exception(message)
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details={"message": message},
                error_details={"message": message},
            )

        metadata = SubprocessProcessMetadata(
            task_id=task_id,
            process=process,
            work_dir=work_dir,
            log_path=log_path,
            persistent_dir=persistent_dir,
            otel_logger=otel_logger,
        )
        self._process_registry.set(key, metadata)

        self._start_log_capture(step, metadata, "stdout")
        self._start_log_capture(step, metadata, "stderr")

        status_details = {"message": "Subprocess scheduled", **self._task_status_details(metadata)}
        self._jobs.update_job_step_task(
            name=metadata.task_id,
            workspace=step.workspace,
            job=step.job,
            step=step.name,
            body=PlatformJobTaskUpdate(
                status=PlatformJobStatus.PENDING,
                status_details=status_details,
                error_details={},
            ),
        )

        return JobUpdate(status=PlatformJobStatus.PENDING, status_details=status_details)

    def sync(self, step: PlatformJobStepWithContext) -> JobUpdate:
        key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
        metadata = self._process_registry.get(key)

        if metadata is None:
            if step.status == PlatformJobStatus.CANCELLING:
                return JobUpdate(
                    status=PlatformJobStatus.CANCELLED,
                    status_details={"message": "Subprocess not found, job cancelled"},
                )
            if step.status == PlatformJobStatus.PAUSING:
                return JobUpdate(
                    status=PlatformJobStatus.PAUSED,
                    status_details={"message": "Subprocess not found, job paused"},
                )
            tasks = self._list_tasks_safe(step)
            orphaned = self._orphaned_step_update(step, tasks)
            if orphaned is not None:
                return orphaned
            task_fallback = self._task_fallback_update(tasks)
            if task_fallback is not None:
                return task_fallback
            if step.status == PlatformJobStatus.PENDING and not self._pending_step_missing_metadata_is_stale(step):
                # Stopgap only: the subprocess backend keeps execution metadata in controller
                # memory, while step/task state is persisted in the jobs database. Those two
                # sources of truth are not fully synchronized today because subprocess was not
                # originally designed around durable jobs-backed execution state. That means a
                # step can already be visible in the database as pending before this backend has
                # registered local subprocess metadata for it. Keep the step pending briefly
                # instead of failing it. The real fix is to move subprocess onto properly
                # serialized, jobs-backed state so reconciliation does not depend on process-local
                # memory.
                return JobUpdate(
                    status=PlatformJobStatus.PENDING,
                    status_details=step.status_details or {"message": "Awaiting subprocess metadata"},
                    error_details=step.error_details or {},
                )
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                error_details={"message": "Local subprocess metadata not found"},
            )

        if step.status == PlatformJobStatus.CANCELLING and metadata.process.poll() is None:
            self._terminate_process(metadata)

        if step.status == PlatformJobStatus.PAUSING and metadata.process.poll() is None:
            self._terminate_process(metadata)

        ttl_seconds = (
            self._execution_profile_config.ttl_seconds_active
            if step.status == PlatformJobStatus.ACTIVE
            else self._execution_profile_config.ttl_seconds_before_active
        )
        if self.check_step_ttl(step, ttl_seconds):
            message = f"Job timed out after reaching max TTL of {ttl_seconds} seconds"
            self._terminate_process(metadata, force=True)
            self._update_task(
                step,
                metadata,
                PlatformJobStatus.ERROR,
                {"message": message, **self._task_status_details(metadata)},
                {"message": message},
                self._tail_log_file(metadata.log_path),
            )
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details={"message": message},
                error_details={"message": message},
            )

        return self._create_step_update(step, metadata)

    def cleanup_steps(self) -> None:
        for key, metadata in self._process_registry.items():
            if metadata.process.poll() is None:
                continue

            step = self.get_step_safe(job=key.job, step_name=key.step, workspace=key.workspace)
            should_cleanup = step is None
            if step is not None and step.status in ("cancelled", "error", "completed"):
                should_cleanup = self._execution_profile_config.cleanup_completed_jobs_immediately
                if not should_cleanup and step.updated_at is not None:
                    updated_at = step.updated_at
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=datetime.timezone.utc)
                    expires_at = updated_at + datetime.timedelta(
                        seconds=self._execution_profile_config.ttl_seconds_after_finished
                    )
                    should_cleanup = expires_at < datetime.datetime.now(datetime.timezone.utc)

            if should_cleanup:
                shutil.rmtree(metadata.work_dir, ignore_errors=True)
                self._process_registry.pop(key)

    @classmethod
    def _task_fallback_update(cls, tasks: list[PlatformJobTaskResponse] | None) -> JobUpdate | None:
        latest_task = cls._latest_task(tasks) if tasks else None
        if latest_task is None:
            return None

        return JobUpdate(
            status=latest_task.status,
            status_details=latest_task.status_details,
            error_details=latest_task.error_details or {},
        )

    def _orphaned_step_update(
        self,
        step: PlatformJobStepWithContext,
        tasks: list[PlatformJobTaskResponse] | None,
    ) -> JobUpdate | None:
        """Cancel a step whose subprocess was left behind by a previous controller.

        Subprocess execution state is process-local, so a controller restart (platform
        stopped mid-run) leaves steps running in the database with nothing tracking them.
        Anything that stopped being updated before this controller started, and has stayed
        quiet past the grace period, is such a leftover: kill whatever is still running and
        cancel the step so the job reaches a terminal state.
        """
        if tasks is None:
            return None

        latest_task = self._latest_task(tasks)
        if latest_task is not None and PlatformJobStatus(latest_task.status).is_terminal():
            return None

        task_anchor = self._task_timestamp(latest_task) if latest_task is not None else None
        anchor = task_anchor or step.updated_at or step.created_at
        if not self._is_orphaned_anchor(anchor):
            return None

        logger.warning(
            "Cancelling job step orphaned by a previous jobs controller",
            extra={"job": step.job, "step": step.name, "workspace": step.workspace, "status": step.status},
        )
        if latest_task is not None:
            self._terminate_orphaned_process(latest_task.status_details or {})
            self._cancel_orphaned_task(step, latest_task)
        return JobUpdate(
            status=PlatformJobStatus.CANCELLED,
            status_details={"message": _ORPHANED_MESSAGE},
            error_details={},
        )

    def _list_tasks_safe(self, step: PlatformJobStepWithContext) -> list[PlatformJobTaskResponse] | None:
        try:
            tasks = self._jobs.list_job_step_tasks(
                name=step.name,
                job=step.job,
                workspace=step.workspace,
            ).data()
        except Exception:
            logger.warning(
                "Failed to fetch tasks for subprocess metadata fallback",
                extra={"job": step.job, "step": step.name, "workspace": step.workspace},
            )
            return None
        return list(tasks.data)

    @classmethod
    def _latest_task(cls, tasks: list[PlatformJobTaskResponse]) -> PlatformJobTaskResponse | None:
        if not tasks:
            return None
        return max(
            tasks,
            key=lambda task: cls._task_timestamp(task) or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
        )

    @staticmethod
    def _task_timestamp(task: PlatformJobTaskResponse) -> datetime.datetime | None:
        return task.updated_at or task.created_at

    def _is_orphaned_anchor(self, anchor: datetime.datetime | None) -> bool:
        if anchor is None:
            return False
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=datetime.timezone.utc)
        if anchor >= self._started_at:
            return False
        grace = datetime.timedelta(seconds=self._execution_profile_config.orphan_recovery_grace_seconds)
        return (anchor + grace) < datetime.datetime.now(datetime.timezone.utc)

    def _cancel_orphaned_task(self, step: PlatformJobStepWithContext, task: PlatformJobTaskResponse) -> None:
        task_name = task.name
        if not task_name:
            return
        try:
            self._jobs.update_job_step_task(
                name=task_name,
                workspace=step.workspace,
                job=step.job,
                step=step.name,
                body=PlatformJobTaskUpdate(
                    status=PlatformJobStatus.CANCELLED,
                    status_details={**(task.status_details or {}), "message": _ORPHANED_MESSAGE},
                    error_details={},
                ),
            )
        except Exception:
            logger.warning(
                "Failed to cancel orphaned subprocess task",
                extra={"job": step.job, "step": step.name, "workspace": step.workspace, "task": task_name},
            )

    @staticmethod
    def _terminate_orphaned_process(status_details: dict) -> None:
        """Best-effort kill of a process group left behind by a previous controller."""
        pid = status_details.get(SUBPROCESS_PID_STATUS_KEY)
        pgid = status_details.get(SUBPROCESS_PGID_STATUS_KEY)
        if not isinstance(pid, int) or not isinstance(pgid, int):
            return
        try:
            # Guard against PID reuse: a recycled PID is very unlikely to also lead the same group.
            if os.getpgid(pid) != pgid:
                return
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        except Exception:
            logger.exception("Failed to signal orphaned subprocess group", extra={"pid": pid, "pgid": pgid})
            return

        deadline = time.monotonic() + _ORPHAN_TERMINATE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_ORPHAN_TERMINATE_POLL_SECONDS)
            try:
                os.getpgid(pid)
            except ProcessLookupError:
                return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        except Exception:
            logger.exception("Failed to kill orphaned subprocess group", extra={"pid": pid, "pgid": pgid})

    @staticmethod
    def _pending_step_missing_metadata_is_stale(step: PlatformJobStepWithContext) -> bool:
        anchor = step.updated_at or step.created_at
        if anchor is None:
            return True
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=datetime.timezone.utc)
        return (anchor + datetime.timedelta(seconds=_MISSING_METADATA_PENDING_GRACE_SECONDS)) < datetime.datetime.now(
            datetime.timezone.utc
        )

    @staticmethod
    def _cleanup_failed_startup_dirs(work_dir: Path, persistent_dir: Path) -> None:
        shutil.rmtree(work_dir, ignore_errors=True)
        for path in (
            work_dir.parent,
            persistent_dir,
            persistent_dir.parent,
            persistent_dir.parent.parent,
            persistent_dir.parent.parent.parent,
        ):
            try:
                path.rmdir()
            except OSError:
                pass

    def _prepare_runtime(self, step: PlatformJobStepWithContext) -> tuple[dict[str, str], str, Path, Path, Path]:
        task_id = f"task-{uuid.uuid4().hex}"
        job_attempt_dir = self._root_dir / step.workspace / step.job / str(step.attempt_id)
        work_dir = job_attempt_dir / step.name / task_id
        config_dir = work_dir / "config"
        ephemeral_dir = work_dir / "scratch"
        persistent_dir = job_attempt_dir / "job-storage"
        config_dir.mkdir(parents=True, exist_ok=True)
        ephemeral_dir.mkdir(parents=True, exist_ok=True)
        persistent_dir.mkdir(parents=True, exist_ok=True)

        spec = step.step_spec
        config_path = config_dir / NEMO_JOB_STEP_CONFIG_FILE_NAME
        config_path.write_text(json.dumps(spec.config if spec else {}), encoding="utf-8")
        log_path = work_dir / "task.log.jsonl"
        log_path.touch()

        platform_config = get_platform_config()
        otlp_logs_endpoint = get_logs_endpoint_config_from_fileset(
            platform_config,
            step.workspace,
            step.fileset,
            loopback_address="localhost",
            artifact_base_path=step.artifact_base_path,
        )
        env = {name: value for name, value in os.environ.items() if name in SUBPROCESS_INHERITED_ENV_ALLOWLIST}
        env.update(self._execution_profile_config.env)
        env.update(
            {
                NEMO_JOB_ID_ENVVAR: step.job,
                NEMO_JOB_ATTEMPT_ID_ENVVAR: str(step.attempt_id),
                NEMO_JOB_STEP_ENVVAR: step.name,
                NEMO_JOB_TASK_ENVVAR: task_id,
                NEMO_JOB_WORKSPACE_ENVVAR: step.workspace,
                NEMO_JOB_FILESET_ENVVAR: step.fileset,
                EPHEMERAL_TASK_STORAGE_PATH_ENVVAR: str(ephemeral_dir),
                CONFIG_TASK_STORAGE_PATH_ENVVAR: str(config_dir),
                PERSISTENT_JOB_STORAGE_PATH_ENVVAR: str(persistent_dir),
                NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR: str(config_path),
                NEMO_JOB_SECRETS_ENVVAR: self.get_secrets_environment_variable_for_injection(step),
            }
        )
        env.update(otlp_logs_endpoint.to_env())

        if spec and spec.environment:
            for envvar in spec.environment:
                if envvar.value is None:
                    continue
                if envvar.name == PERSISTENT_JOB_STORAGE_PATH_ENVVAR:
                    logger.debug(
                        "Overriding subprocess job storage env var",
                        extra={"env_var": envvar.name, "replacement_path": str(persistent_dir)},
                    )
                    env[envvar.name] = str(persistent_dir)
                elif envvar.name == EPHEMERAL_TASK_STORAGE_PATH_ENVVAR:
                    logger.debug(
                        "Overriding subprocess job storage env var",
                        extra={"env_var": envvar.name, "replacement_path": str(ephemeral_dir)},
                    )
                    env[envvar.name] = str(ephemeral_dir)
                else:
                    env[envvar.name] = envvar.value

        env.update(platform_config.to_shared_envvars(loopback_address="localhost"))

        if step.auth_context:
            auth_context = AuthContext.model_validate(step.auth_context.model_dump(mode="python", exclude_none=True))
            principal = auth_context.to_principal()
            env.update(principal.get_env_var())

        inject_secret_env_vars(env)
        return env, task_id, work_dir, log_path, persistent_dir

    @staticmethod
    def _build_command(executor_config: SubprocessExecutionProvider, virtual_env: str | None) -> list[str]:
        command = executor_config.command
        if not command:
            raise ValueError(_ERR_COMMAND_REQUIRED)
        if command[0] in {"python", "python3"}:
            if virtual_env:
                venv_python = Path(virtual_env) / "bin" / "python"
                if os.access(venv_python, os.X_OK):
                    return [str(venv_python), *command[1:]]
            return [sys.executable, *command[1:]]
        return command

    def _start_log_capture(
        self,
        step: PlatformJobStepWithContext,
        metadata: SubprocessProcessMetadata,
        stream_name: str,
    ) -> None:
        thread = start_log_capture(
            metadata.process.stdout if stream_name == "stdout" else metadata.process.stderr,
            log_path=metadata.log_path,
            log_lock=metadata.lock,
            otel_logger=metadata.otel_logger,
            stream_name=stream_name,
            job=step.job,
            step=step.name,
            task_id=metadata.task_id,
        )
        if thread is not None:
            metadata.log_threads.append(thread)

    def _create_step_update(self, step: PlatformJobStepWithContext, metadata: SubprocessProcessMetadata) -> JobUpdate:
        status, status_details, error_details, error_stack = self._map_process_status(step, metadata)
        self._update_task(step, metadata, status, status_details, error_details, error_stack)
        return JobUpdate(status=status, status_details=status_details, error_details=error_details)

    def _map_process_status(
        self, step: PlatformJobStepWithContext, metadata: SubprocessProcessMetadata
    ) -> tuple[PlatformJobStatus, dict, dict, str]:
        exit_code = metadata.process.poll()
        status_details = self._task_status_details(metadata)
        error_details: dict[str, str] = {}
        error_stack = ""

        if exit_code is None:
            if step.status == PlatformJobStatus.CANCELLING:
                return PlatformJobStatus.CANCELLING, {"message": "Job is cancelling", **status_details}, {}, ""
            if step.status == PlatformJobStatus.PAUSING:
                return PlatformJobStatus.PAUSING, {"message": "Job is pausing", **status_details}, {}, ""
            return PlatformJobStatus.ACTIVE, {"message": "Job is running", **status_details}, {}, ""

        self._finish_logs(metadata)
        status_details["exit_code"] = exit_code
        # Check pausing/cancelling before treating a non-zero exit as an error.
        # The process was killed as part of pause/cancel — the exit code is expected.
        if step.status == PlatformJobStatus.PAUSING:
            return (
                PlatformJobStatus.PAUSED,
                {"message": f"Job paused with exit code {exit_code}", **status_details},
                {},
                "",
            )
        if step.status == PlatformJobStatus.CANCELLING:
            return (
                PlatformJobStatus.CANCELLED,
                {"message": f"Job was cancelled with exit code {exit_code}", **status_details},
                {},
                "",
            )
        if exit_code == 0:
            return (
                PlatformJobStatus.COMPLETED,
                {"message": f"Job completed successfully with exit code {exit_code}", **status_details},
                {},
                "",
            )

        error_stack = self._tail_log_file(metadata.log_path)
        error_details = {"message": f"Job exited with code {exit_code}"}
        return (
            PlatformJobStatus.ERROR,
            {"message": f"Job exited with code {exit_code}", **status_details},
            error_details,
            error_stack,
        )

    def _update_task(
        self,
        step: PlatformJobStepWithContext,
        metadata: SubprocessProcessMetadata,
        status: PlatformJobStatus,
        status_details: dict,
        error_details: dict,
        error_stack: str = "",
    ) -> None:
        self._jobs.update_job_step_task(
            name=metadata.task_id,
            workspace=step.workspace,
            job=step.job,
            step=step.name,
            body=PlatformJobTaskUpdate(
                status=status,
                status_details=status_details,
                error_details=error_details,
                error_stack=error_stack,
            ),
        )

    @staticmethod
    def _task_status_details(metadata: SubprocessProcessMetadata) -> dict[str, str | int]:
        details: dict[str, str | int] = {
            SUBPROCESS_WORKDIR_STATUS_KEY: str(metadata.work_dir),
            SUBPROCESS_PERSISTENT_STORAGE_STATUS_KEY: str(metadata.persistent_dir),
            SUBPROCESS_PID_STATUS_KEY: metadata.process.pid,
        }
        try:
            details[SUBPROCESS_PGID_STATUS_KEY] = os.getpgid(metadata.process.pid)
        except ProcessLookupError:
            pass
        return details

    def _terminate_process(self, metadata: SubprocessProcessMetadata, force: bool = False) -> None:
        if metadata.process.poll() is not None:
            return
        self._terminate_process_group(metadata.process, force=force)
        if force:
            try:
                metadata.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.exception("Subprocess did not exit after SIGKILL", extra={"pid": metadata.process.pid})
            return
        try:
            metadata.process.wait(timeout=self._execution_profile_config.graceful_shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._terminate_process(metadata, force=True)

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str], force: bool = False) -> None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError:
            return
        except Exception:
            logger.exception("Failed to signal subprocess group", extra={"pid": process.pid})
            try:
                process.send_signal(sig)
            except Exception:
                logger.exception("Failed to signal subprocess", extra={"pid": process.pid})

    def _finish_logs(self, metadata: SubprocessProcessMetadata) -> None:
        for thread in metadata.log_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        with metadata.lock:
            if metadata.closed_logs:
                return
            if metadata.otel_logger is not None:
                metadata.otel_logger.close()
            metadata.closed_logs = True

    @staticmethod
    def _tail_log_file(log_path: Path, max_chars: int = 2048) -> str:
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]
