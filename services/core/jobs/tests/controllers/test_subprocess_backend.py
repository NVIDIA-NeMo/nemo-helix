# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from nmp.common.config import PlatformConfig
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.app.providers import SubprocessExecutionProvider
from nmp.core.jobs.controllers.backends.base import (
    NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
    NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR,
    NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR,
)
from nmp.core.jobs.controllers.backends.subprocess import (
    SubprocessJobBackend,
    SubprocessJobExecutionProfileConfig,
    SubprocessProcessKey,
)

from services.core.jobs.tests.controllers.client_mocks import data_response


def _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config) -> SubprocessJobBackend:
    with patch("nmp.core.jobs.controllers.backends.subprocess.get_platform_config", return_value=mock_platform_config):
        return SubprocessJobBackend(
            mock_nmp_client,
            SubprocessJobExecutionProfileConfig(working_directory=str(tmp_path)),
            profile_name="subprocess",
        )


def _step_with_command(step, command: list[str]):
    updated_step = step.model_copy(deep=True)
    updated_step.step_spec.executor = SubprocessExecutionProvider(
        provider="subprocess", profile="default", command=command
    )
    return updated_step


def _step_with_unvalidated_command(step, command: list[str]):
    updated_step = step.model_copy(deep=True)
    step_spec = updated_step.step_spec
    assert step_spec is not None
    updated_step.step_spec = step_spec.model_copy(
        update={
            "executor": SubprocessExecutionProvider.model_construct(
                provider="subprocess", profile="default", command=command
            )
        }
    )
    return updated_step


def _step_named(step, name: str):
    updated_step = step.model_copy(deep=True)
    updated_step.id = f"{step.id}-{name}"
    updated_step.name = name
    updated_step.step_spec.name = name
    return updated_step


def _schedule_without_otel_export(backend: SubprocessJobBackend, step):
    with patch("nmp.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None):
        return backend.schedule(step.step_spec.executor, step)


def test_schedule_starts_process_and_stages_environment(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "printf 'hello local\\n'"])

    update = _schedule_without_otel_export(backend, step)

    assert update.status == PlatformJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0
    assert metadata.work_dir.is_relative_to(tmp_path)
    assert metadata.persistent_dir.is_relative_to(tmp_path)
    mock_jobs_client.update_job_step_task.assert_called()


def test_created_step_does_not_ttl_before_backend_acceptance(
    mock_nmp_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    ttl_seconds = backend._execution_profile_config.ttl_seconds_before_active
    old_timestamp = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds + 300)
    step.created_at = old_timestamp
    step.updated_at = old_timestamp
    step.status = PlatformJobStatus.CREATED

    update = _schedule_without_otel_export(backend, step)

    assert update.status == PlatformJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0


def test_subprocess_persistent_storage_is_shared_across_job_attempt(
    mock_nmp_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    first_step = _step_with_command(_step_named(test_step_pending, "first-step"), ["/bin/sh", "-c", "true"])
    second_step = _step_with_command(_step_named(test_step_pending, "second-step"), ["/bin/sh", "-c", "true"])

    _schedule_without_otel_export(backend, first_step)
    _schedule_without_otel_export(backend, second_step)

    first_metadata = backend._process_registry.get(
        SubprocessProcessKey(first_step.workspace, first_step.job, str(first_step.attempt_id), first_step.name)
    )
    second_metadata = backend._process_registry.get(
        SubprocessProcessKey(second_step.workspace, second_step.job, str(second_step.attempt_id), second_step.name)
    )
    assert first_metadata is not None
    assert second_metadata is not None
    assert first_metadata.process.wait(timeout=5) == 0
    assert second_metadata.process.wait(timeout=5) == 0
    assert first_metadata.work_dir != second_metadata.work_dir
    assert first_metadata.persistent_dir == second_metadata.persistent_dir
    assert (
        first_metadata.persistent_dir
        == tmp_path / first_step.workspace / first_step.job / str(first_step.attempt_id) / "job-storage"
    )


def test_schedule_uses_allowlisted_host_environment(mock_nmp_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(
        test_step_pending,
        [
            "/bin/sh",
            "-c",
            (
                'test "$PATH" = "/bin" && '
                'test "$VIRTUAL_ENV" = "/venv" && '
                'test -z "${HOME+x}" && '
                'test -z "${SECRET_TOKEN+x}"'
            ),
        ],
    )

    with (
        patch.dict(
            os.environ,
            {"HOME": "/home/test", "PATH": "/bin", "VIRTUAL_ENV": "/venv", "SECRET_TOKEN": "do-not-leak"},
            clear=True,
        ),
        patch("nmp.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == PlatformJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0


def test_schedule_passes_the_agent_wheel_through(mock_nmp_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(
        test_step_pending,
        [
            "/bin/sh",
            "-c",
            'test "$NEMO_AGENTS_WHEEL" = "/dist/nemo_platform.whl" && test -z "${SECRET_TOKEN+x}"',
        ],
    )

    with (
        patch.dict(
            os.environ,
            {"PATH": "/bin", "NEMO_AGENTS_WHEEL": "/dist/nemo_platform.whl", "SECRET_TOKEN": "do-not-leak"},
            clear=True,
        ),
        patch("nmp.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == PlatformJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0


def test_schedule_preserves_uds_otlp_metadata_in_runtime_env(mock_nmp_client, tmp_path, test_step_pending):
    platform_config = PlatformConfig(  # type: ignore[abstract]
        service_discovery={"files": "unix:///tmp/nemo-platform.sock"},
        loopback_address=None,
    )
    backend = _subprocess_backend(mock_nmp_client, tmp_path, platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    captured_env = {}

    def fake_create_otel_logger(*, env, **_kwargs):
        captured_env.update(env)
        return None

    with (
        patch("nmp.core.jobs.controllers.backends.subprocess.get_platform_config", return_value=platform_config),
        patch("nmp.core.jobs.controllers.backends.subprocess.create_otel_logger", side_effect=fake_create_otel_logger),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == PlatformJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0
    assert captured_env[NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR] == (
        "http://nemo-platform.local/apis/files/v2/workspaces/default/filesets/test-logs-fileset/otlp/v1/logs"
    )
    assert captured_env[NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR] == "uds"
    assert captured_env[NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR] == "/tmp/nemo-platform.sock"


def test_schedule_terminates_process_when_post_popen_setup_fails(
    mock_nmp_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])

    with (
        patch(
            "nmp.core.jobs.controllers.backends.subprocess.create_otel_logger", side_effect=RuntimeError("otel boom")
        ),
        patch.object(backend, "_terminate_process_group", wraps=backend._terminate_process_group) as mock_terminate,
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == PlatformJobStatus.ERROR
    assert "initialize subprocess runtime" in update.error_details["message"]
    mock_terminate.assert_called_once()
    assert backend._process_registry.is_empty()
    assert not any(tmp_path.iterdir())


def test_sync_completed_closes_logs(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "printf 'hello logs\\n'"])

    update = _schedule_without_otel_export(backend, step)
    assert update.status == PlatformJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    metadata.process.wait(timeout=5)

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.COMPLETED
    assert metadata.closed_logs is True
    assert "hello logs" in metadata.log_path.read_text(encoding="utf-8")
    last_call = mock_jobs_client.update_job_step_task.call_args
    assert last_call.kwargs["body"].status == PlatformJobStatus.COMPLETED


def test_shutdown_finishes_logs(mock_nmp_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])
    _schedule_without_otel_export(backend, step)
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None

    backend.shutdown()

    assert metadata.process.poll() is not None
    assert metadata.closed_logs is True


def test_sync_nonzero_exit_sets_error(mock_nmp_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "printf 'bad\\n' >&2; exit 7"])

    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    metadata.process.wait(timeout=5)

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.ERROR
    assert update.error_details == {"message": "Job exited with code 7"}


def test_missing_command_fails_without_process(mock_nmp_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_unvalidated_command(test_step_pending, [])

    update = backend.schedule(step.step_spec.executor, step)

    assert update.status == PlatformJobStatus.ERROR
    assert update.error_details is not None
    assert "subprocess requires" in update.error_details["message"]
    assert backend._process_registry.is_empty()


def test_build_command_uses_current_interpreter_for_python_module_commands() -> None:
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["python", "-m", "nemo_evaluator.tasks.evaluate"],
    )

    assert SubprocessJobBackend._build_command(executor, None) == [
        sys.executable,
        "-m",
        "nemo_evaluator.tasks.evaluate",
    ]


def test_build_command_uses_current_interpreter_for_python3_commands() -> None:
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["python3", "-m", "nemo_evaluator.tasks.evaluate"],
    )

    assert SubprocessJobBackend._build_command(executor, None) == [
        sys.executable,
        "-m",
        "nemo_evaluator.tasks.evaluate",
    ]


def test_build_command_prefers_virtual_env_python(tmp_path) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o755)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["python", "-m", "nemo_evaluator.tasks.evaluate"],
    )

    assert SubprocessJobBackend._build_command(executor, str(tmp_path / "venv")) == [
        str(venv_python),
        "-m",
        "nemo_evaluator.tasks.evaluate",
    ]


def test_schedule_python_command_does_not_depend_on_runtime_path(
    mock_nmp_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    step = _step_with_command(
        test_step_pending,
        ["python", "-c", f"import os; assert os.environ['PATH'] == {str(empty_path)!r}"],
    )

    with (
        patch.dict(os.environ, {"PATH": str(empty_path)}, clear=True),
        patch("nmp.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == PlatformJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.args[0] == sys.executable
    assert metadata.process.wait(timeout=5) == 0


def test_cancelling_terminates_running_process(mock_nmp_client, tmp_path, mock_platform_config, test_step_cancelling):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_cancelling, ["/bin/sh", "-c", "sleep 10"])

    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.poll() is None

    update = backend.sync(step)

    assert update.status in {PlatformJobStatus.CANCELLING, PlatformJobStatus.CANCELLED}
    for _ in range(20):
        if metadata.process.poll() is not None:
            break
        time.sleep(0.05)
    assert metadata.process.poll() is not None


def test_sync_uses_persisted_task_when_local_metadata_is_missing(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 10"])

    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    metadata.process.terminate()
    assert metadata.process.wait(timeout=5) is not None
    backend._process_registry.pop(key)
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    status=PlatformJobStatus.ACTIVE.value,
                    status_details={"message": "Job is running"},
                    error_details={},
                    created_at=step.created_at,
                    updated_at=step.updated_at,
                )
            ]
        )
    )

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.ACTIVE
    assert update.status_details == {"message": "Job is running"}
    assert update.error_details == {}


def test_task_fallback_update_handles_missing_task_timestamps(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "true"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    status=PlatformJobStatus.PENDING.value,
                    status_details={"message": "missing timestamps"},
                    error_details={},
                    created_at=None,
                    updated_at=None,
                ),
                SimpleNamespace(
                    status=PlatformJobStatus.ACTIVE.value,
                    status_details={"message": "latest"},
                    error_details={},
                    created_at=step.created_at,
                    updated_at=step.updated_at,
                ),
            ]
        )
    )

    update = backend._task_fallback_update(backend._list_tasks_safe(step))

    assert update is not None
    assert update.status == PlatformJobStatus.ACTIVE
    assert update.status_details == {"message": "latest"}
    assert update.error_details == {}


def test_sync_keeps_recent_pending_step_pending_when_local_metadata_is_missing(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.PENDING
    assert update.error_details == {}


def _orphaned_task(status: PlatformJobStatus, age_seconds: int, **overrides):
    timestamp = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    fields = {
        "name": "task-orphan",
        "status": status.value,
        "status_details": {"message": "Job is running"},
        "error_details": {},
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_sync_cancels_active_step_orphaned_by_previous_controller(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 10"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(data=[_orphaned_task(PlatformJobStatus.ACTIVE, age_seconds=3600)])
    )

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.CANCELLED
    assert update.status_details == {"message": "Job was cancelled: the platform stopped while this job was running"}
    last_call = mock_jobs_client.update_job_step_task.call_args
    assert last_call.kwargs["name"] == "task-orphan"
    assert last_call.kwargs["body"].status == PlatformJobStatus.CANCELLED


def test_sync_cancels_orphaned_step_without_tasks(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 10"])
    step.updated_at = datetime.now(timezone.utc) - timedelta(seconds=3600)
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.CANCELLED
    mock_jobs_client.update_job_step_task.assert_not_called()


def test_sync_does_not_cancel_step_updated_within_grace_period(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 10"])
    grace = backend._execution_profile_config.orphan_recovery_grace_seconds
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(data=[_orphaned_task(PlatformJobStatus.ACTIVE, age_seconds=grace // 2)])
    )

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.ACTIVE


def test_sync_does_not_cancel_step_owned_by_a_live_controller(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    backend._started_at = datetime.now(timezone.utc) - timedelta(days=1)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 10"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(data=[_orphaned_task(PlatformJobStatus.ACTIVE, age_seconds=3600)])
    )

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.ACTIVE


def test_sync_keeps_terminal_task_status_for_missing_metadata(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "true"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                _orphaned_task(
                    PlatformJobStatus.COMPLETED,
                    age_seconds=3600,
                    status_details={"message": "Job completed successfully with exit code 0"},
                )
            ]
        )
    )

    update = backend.sync(step)

    assert update.status == PlatformJobStatus.COMPLETED


def test_orphan_recovery_kills_leftover_process_group(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 30"])
    leftover = subprocess.Popen(["/bin/sh", "-c", "sleep 30"], start_new_session=True)
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                _orphaned_task(
                    PlatformJobStatus.ACTIVE,
                    age_seconds=3600,
                    status_details={"pid": leftover.pid, "pgid": os.getpgid(leftover.pid)},
                )
            ]
        )
    )

    try:
        update = backend.sync(step)

        assert update.status == PlatformJobStatus.CANCELLED
        assert leftover.wait(timeout=5) is not None
    finally:
        if leftover.poll() is None:
            leftover.kill()
            leftover.wait(timeout=5)


def test_orphan_recovery_skips_kill_when_recorded_pid_is_not_group_leader(
    mock_nmp_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nmp_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "sleep 30"])
    unrelated = subprocess.Popen(["/bin/sh", "-c", "sleep 5"], start_new_session=True)
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                _orphaned_task(
                    PlatformJobStatus.ACTIVE,
                    age_seconds=3600,
                    status_details={"pid": unrelated.pid, "pgid": os.getpgid(unrelated.pid) + 1},
                )
            ]
        )
    )

    try:
        update = backend.sync(step)

        assert update.status == PlatformJobStatus.CANCELLED
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_pending_step_missing_metadata_stale_check_accepts_typed_timestamp(test_step_pending):
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    step.updated_at = datetime.now(timezone.utc)
    step.created_at = None

    assert SubprocessJobBackend._pending_step_missing_metadata_is_stale(step) is False
