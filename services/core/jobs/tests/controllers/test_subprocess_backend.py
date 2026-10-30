# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nhx.common.config import HelixConfig
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.providers import SubprocessExecutionProvider
from nhx.core.jobs.app.schemas import StepLifecycle
from nhx.core.jobs.controllers.backends.base import (
    NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
    NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR,
    NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR,
)
from nhx.core.jobs.controllers.backends.subprocess import (
    SubprocessJobBackend,
    SubprocessJobExecutionProfileConfig,
    SubprocessProcessKey,
)

from services.core.jobs.tests.controllers.client_mocks import data_response


def _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config) -> SubprocessJobBackend:
    with patch("nhx.core.jobs.controllers.backends.subprocess.get_platform_config", return_value=mock_platform_config):
        return SubprocessJobBackend(
            mock_nemo_client,
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
    with patch("nhx.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None):
        return backend.schedule(step.step_spec.executor, step)


def test_schedule_starts_process_and_stages_environment(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "printf 'hello local\\n'"])

    update = _schedule_without_otel_export(backend, step)

    assert update.status == HelixJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0
    assert metadata.work_dir.is_relative_to(tmp_path)
    assert metadata.persistent_dir.is_relative_to(tmp_path)
    mock_jobs_client.update_attempt_step_task.assert_called()


def test_created_step_does_not_ttl_before_backend_acceptance(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    ttl_seconds = backend._execution_profile_config.ttl_seconds_before_active
    old_timestamp = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds + 300)
    step.created_at = old_timestamp
    step.updated_at = old_timestamp
    step.status = HelixJobStatus.CREATED

    update = _schedule_without_otel_export(backend, step)

    assert update.status == HelixJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0


def test_subprocess_persistent_storage_is_shared_across_job_attempt(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
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
    assert first_metadata.persistent_dir == tmp_path / first_step.workspace / first_step.job / "job-storage"


def test_schedule_uses_allowlisted_host_environment(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
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
        patch("nhx.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == HelixJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0


def test_schedule_passes_the_agent_wheel_through(mock_nemo_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(
        test_step_pending,
        [
            "/bin/sh",
            "-c",
            'test "$NEMO_AGENTS_WHEEL" = "/dist/nemo_helix.whl" && test -z "${SECRET_TOKEN+x}"',
        ],
    )

    with (
        patch.dict(
            os.environ,
            {"PATH": "/bin", "NEMO_AGENTS_WHEEL": "/dist/nemo_helix.whl", "SECRET_TOKEN": "do-not-leak"},
            clear=True,
        ),
        patch("nhx.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == HelixJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0


def test_schedule_preserves_uds_otlp_metadata_in_runtime_env(mock_nemo_client, tmp_path, test_step_pending):
    platform_config = HelixConfig(  # type: ignore[abstract]
        service_discovery={"files": "unix:///tmp/nemo-helix.sock"},
        loopback_address=None,
    )
    backend = _subprocess_backend(mock_nemo_client, tmp_path, platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    captured_env = {}

    def fake_create_otel_logger(*, env, **_kwargs):
        captured_env.update(env)
        return None

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess.get_platform_config", return_value=platform_config),
        patch("nhx.core.jobs.controllers.backends.subprocess.create_otel_logger", side_effect=fake_create_otel_logger),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == HelixJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0
    assert captured_env[NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR] == (
        "http://nemo-helix.local/apis/files/v2/workspaces/default/filesets/test-logs-fileset/otlp/v1/logs"
    )
    assert captured_env[NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR] == "uds"
    assert captured_env[NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR] == "/tmp/nemo-helix.sock"
    assert captured_env["NEMO_JOB_UID"] == "platform-job-test-uid"


def test_schedule_terminates_process_when_post_popen_setup_fails(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])

    with (
        patch(
            "nhx.core.jobs.controllers.backends.subprocess.create_otel_logger", side_effect=RuntimeError("otel boom")
        ),
        patch.object(backend, "_terminate_process_group", wraps=backend._terminate_process_group) as mock_terminate,
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == HelixJobStatus.ERROR
    assert "initialize subprocess runtime" in update.error_details["message"]
    mock_terminate.assert_called_once()
    assert backend._process_registry.is_empty()
    assert not any(tmp_path.iterdir())


def test_sync_completed_closes_logs(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "printf 'hello logs\\n'"])

    update = _schedule_without_otel_export(backend, step)
    assert update.status == HelixJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    metadata.process.wait(timeout=5)

    update = backend.sync(step)

    assert update.status == HelixJobStatus.COMPLETED
    assert metadata.closed_logs is True
    assert "hello logs" in metadata.log_path.read_text(encoding="utf-8")
    last_call = mock_jobs_client.update_attempt_step_task.call_args
    assert last_call.kwargs["body"].status == HelixJobStatus.COMPLETED


def test_shutdown_finishes_logs(mock_nemo_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])
    _schedule_without_otel_export(backend, step)
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None

    backend.shutdown()

    assert metadata.process.poll() is not None
    assert metadata.closed_logs is True


def test_sync_nonzero_exit_sets_error(mock_nemo_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "printf 'bad\\n' >&2; exit 7"])

    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    metadata.process.wait(timeout=5)

    update = backend.sync(step)

    assert update.status == HelixJobStatus.ERROR
    assert update.error_details == {"message": "Job exited with code 7"}


def test_missing_command_fails_without_process(mock_nemo_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_unvalidated_command(test_step_pending, [])

    update = backend.schedule(step.step_spec.executor, step)

    assert update.status == HelixJobStatus.ERROR
    assert update.error_details is not None
    assert "subprocess requires" in update.error_details["message"]
    assert backend._process_registry.is_empty()


def test_build_command_uses_current_interpreter_for_python_module_commands() -> None:
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["python", "-m", "nemo_evals.tasks.evaluate"],
    )

    assert SubprocessJobBackend._build_command(executor, None) == [
        sys.executable,
        "-m",
        "nemo_evals.tasks.evaluate",
    ]


def test_build_command_uses_current_interpreter_for_python3_commands() -> None:
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["python3", "-m", "nemo_evals.tasks.evaluate"],
    )

    assert SubprocessJobBackend._build_command(executor, None) == [
        sys.executable,
        "-m",
        "nemo_evals.tasks.evaluate",
    ]


def test_build_command_prefers_virtual_env_python(tmp_path) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o755)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["python", "-m", "nemo_evals.tasks.evaluate"],
    )

    assert SubprocessJobBackend._build_command(executor, str(tmp_path / "venv")) == [
        str(venv_python),
        "-m",
        "nemo_evals.tasks.evaluate",
    ]


def test_build_command_resolves_console_script_against_virtual_env_bin(tmp_path) -> None:
    venv_script = tmp_path / "venv" / "bin" / "nemo-helix"
    venv_script.parent.mkdir(parents=True)
    venv_script.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_script.chmod(0o755)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["nemo-helix", "run", "task", "--task", "nhx.hello_world.tasks.hello_world"],
    )

    assert SubprocessJobBackend._build_command(executor, str(tmp_path / "venv")) == [
        str(venv_script),
        "run",
        "task",
        "--task",
        "nhx.hello_world.tasks.hello_world",
    ]


def test_build_command_leaves_console_script_unchanged_when_absent_from_virtual_env_bin(tmp_path) -> None:
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["nemo-helix", "run", "task", "--task", "nhx.hello_world.tasks.hello_world"],
    )

    assert SubprocessJobBackend._build_command(executor, str(tmp_path / "venv")) == [
        "nemo-helix",
        "run",
        "task",
        "--task",
        "nhx.hello_world.tasks.hello_world",
    ]


def test_build_command_resolves_console_script_via_sys_executable_when_virtual_env_unset(tmp_path) -> None:
    # Direct-binary launch (e.g. `.venv/bin/nemo-helix` under Jenkins/systemd) leaves
    # VIRTUAL_ENV unset, but the controller still runs from the venv -- sys.executable
    # points at its bin. Resolve the console script against that dir so the fix holds
    # whether or not the environment was activated.
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    venv_script = bin_dir / "nemo-helix"
    venv_script.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_script.chmod(0o755)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["nemo-helix", "run", "task", "--task", "nhx.hello_world.tasks.hello_world"],
    )

    with patch.object(sys, "executable", str(bin_dir / "python")):
        result = SubprocessJobBackend._build_command(executor, None)

    assert result == [
        str(venv_script),
        "run",
        "task",
        "--task",
        "nhx.hello_world.tasks.hello_world",
    ]


def test_build_command_leaves_console_script_unchanged_when_absent_from_sys_executable_bin(tmp_path) -> None:
    # No VIRTUAL_ENV and the script is not in the interpreter's bin dir -> fall through
    # to the normal PATH lookup unchanged.
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["nemo-helix", "run", "task", "--task", "nhx.hello_world.tasks.hello_world"],
    )

    with patch.object(sys, "executable", str(bin_dir / "python")):
        result = SubprocessJobBackend._build_command(executor, None)

    assert result == [
        "nemo-helix",
        "run",
        "task",
        "--task",
        "nhx.hello_world.tasks.hello_world",
    ]


def test_build_command_does_not_resolve_a_path_bearing_command_against_virtual_env_bin(tmp_path) -> None:
    # Only a bare script name is resolved against venv/bin; a command carrying a path
    # separator is left to the normal lookup rather than joined under bin/. The bin/
    # entry below is executable, so without the separator guard the relative command
    # would be rewritten to it — this asserts it is not.
    venv_script = tmp_path / "venv" / "bin" / "nemo-helix"
    venv_script.parent.mkdir(parents=True)
    venv_script.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_script.chmod(0o755)
    executor = SubprocessExecutionProvider(
        provider="subprocess",
        profile="default",
        command=["./nemo-helix", "run", "task"],
    )

    assert SubprocessJobBackend._build_command(executor, str(tmp_path / "venv")) == [
        "./nemo-helix",
        "run",
        "task",
    ]


def test_schedule_python_command_does_not_depend_on_runtime_path(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    step = _step_with_command(
        test_step_pending,
        ["python", "-c", f"import os; assert os.environ['PATH'] == {str(empty_path)!r}"],
    )

    with (
        patch.dict(os.environ, {"PATH": str(empty_path)}, clear=True),
        patch("nhx.core.jobs.controllers.backends.subprocess.create_otel_logger", return_value=None),
    ):
        update = backend.schedule(step.step_spec.executor, step)

    assert update.status == HelixJobStatus.PENDING
    metadata = backend._process_registry.get(
        SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    )
    assert metadata is not None
    assert metadata.process.args[0] == sys.executable
    assert metadata.process.wait(timeout=5) == 0


def test_cancelling_terminates_running_process(mock_nemo_client, tmp_path, mock_platform_config, test_step_cancelling):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_cancelling, ["/bin/sh", "-c", "sleep 10"])

    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.poll() is None

    update = backend.sync(step)

    assert update.status in {HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED}
    for _ in range(20):
        if metadata.process.poll() is not None:
            break
        time.sleep(0.05)
    assert metadata.process.poll() is not None


def test_sync_uses_persisted_task_when_local_metadata_is_missing(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
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
                    status=HelixJobStatus.ACTIVE.value,
                    status_details={"message": "Job is running"},
                    error_details={},
                    created_at=step.created_at,
                    updated_at=step.updated_at,
                )
            ]
        )
    )

    update = backend.sync(step)

    assert update.status == HelixJobStatus.ACTIVE
    assert update.status_details == {"message": "Job is running"}
    assert update.error_details == {}


def test_get_task_fallback_update_handles_missing_task_timestamps(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_active
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_active, ["/bin/sh", "-c", "true"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    status=HelixJobStatus.PENDING.value,
                    status_details={"message": "missing timestamps"},
                    error_details={},
                    created_at=None,
                    updated_at=None,
                ),
                SimpleNamespace(
                    status=HelixJobStatus.ACTIVE.value,
                    status_details={"message": "latest"},
                    error_details={},
                    created_at=step.created_at,
                    updated_at=step.updated_at,
                ),
            ]
        )
    )

    update = backend._get_task_fallback_update(step)

    assert update is not None
    assert update.status == HelixJobStatus.ACTIVE
    assert update.status_details == {"message": "latest"}
    assert update.error_details == {}


def test_sync_keeps_recent_pending_step_pending_when_local_metadata_is_missing(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))

    update = backend.sync(step)

    assert update.status == HelixJobStatus.PENDING
    assert update.error_details == {}


def test_pending_step_missing_metadata_stale_check_accepts_typed_timestamp(test_step_pending):
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    step.updated_at = datetime.now(timezone.utc)
    step.created_at = None

    assert SubprocessJobBackend._pending_step_missing_metadata_is_stale(step) is False


def _mark_pausing(step, *, ago_seconds: int = 0, deadline_seconds: int = 3600):
    step.status = HelixJobStatus.PAUSING
    requested_at = datetime.now(timezone.utc) - timedelta(seconds=ago_seconds)
    step.status_details = {"pause_requested_at": requested_at.isoformat()}
    step.step_spec.lifecycle = StepLifecycle(pause_deadline_seconds=deadline_seconds)
    return step


def test_sync_pausing_does_not_terminate_a_running_process(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _mark_pausing(_step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"]))
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))

    update = _schedule_without_otel_export(backend, step)
    assert update.status == HelixJobStatus.PENDING
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    try:
        with patch.object(backend, "_terminate_process", wraps=backend._terminate_process) as terminate:
            synced = backend.sync(step)
        assert synced.status == HelixJobStatus.PAUSING
        terminate.assert_not_called()
        assert metadata.process.poll() is None
    finally:
        metadata.process.kill()
        metadata.process.wait(timeout=5)


def test_sync_pausing_nonzero_exit_is_error(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    """The task fails with the step and keeps the log tail, as any other failed exit does."""
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "echo out of memory; exit 1"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))
    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) != 0

    _mark_pausing(step)
    update = backend.sync(step)

    assert update.status == HelixJobStatus.ERROR
    task_update = mock_jobs_client.update_attempt_step_task.call_args.kwargs["body"]
    assert task_update.status == HelixJobStatus.ERROR
    assert "out of memory" in task_update.error_stack


def test_sync_pausing_exit_75_is_paused(mock_nemo_client, tmp_path, mock_platform_config, test_step_pending):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "exit 75"])
    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 75

    _mark_pausing(step)
    update = backend.sync(step)

    assert update.status == HelixJobStatus.PAUSED


def test_sync_pausing_exit_75_seen_after_the_deadline_is_paused(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    """A process that already exited stopped on its own, so its exit code decides, even past the deadline."""
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "exit 75"])
    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 75

    _mark_pausing(step, ago_seconds=120, deadline_seconds=60)
    update = backend.sync(step)

    assert update.status == HelixJobStatus.PAUSED


def test_sync_pausing_deadline_kills_the_process(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))
    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    try:
        _mark_pausing(step, ago_seconds=120, deadline_seconds=60)
        update = backend.sync(step)

        assert update.status == HelixJobStatus.ERROR
        assert metadata.process.wait(timeout=5) is not None
        assert metadata.closed_logs
        assert mock_jobs_client.update_attempt_step_task.call_args.kwargs["body"].status == HelixJobStatus.ERROR
    finally:
        if metadata.process.poll() is None:
            metadata.process.kill()
            metadata.process.wait(timeout=5)


@pytest.mark.parametrize("ago_seconds", [0, 120])
def test_sync_pausing_without_a_process(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending, ago_seconds
):
    """With no process left to stop, a pausing step is paused, even after the pause deadline."""
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _mark_pausing(test_step_pending, ago_seconds=ago_seconds, deadline_seconds=60)

    update = backend.sync(step)

    assert update.status == HelixJobStatus.PAUSED


def test_sync_pausing_step_older_than_the_ttl_is_not_timed_out(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    """A long training step is bounded by its pause deadline while pausing, not by the TTLs."""
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))
    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    ttl_seconds = backend._execution_profile_config.ttl_seconds_active
    step.created_at = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds + 300)
    _mark_pausing(step)
    try:
        update = backend.sync(step)
        assert update.status == HelixJobStatus.PAUSING
        assert metadata.process.poll() is None
    finally:
        metadata.process.kill()
        metadata.process.wait(timeout=5)


def test_sync_cancelling_step_older_than_the_ttl_is_cancelled(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    """Cancel terminates the process, so an old step ends cancelled rather than timed out."""
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])
    mock_jobs_client.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))
    _schedule_without_otel_export(backend, step)
    ttl_seconds = backend._execution_profile_config.ttl_seconds_active
    step.created_at = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds + 300)
    step.status = HelixJobStatus.CANCELLING

    update = backend.sync(step)

    assert update.status == HelixJobStatus.CANCELLED


def test_reclaim_removes_job_storage_and_tolerates_a_missing_directory(
    mock_nemo_client, tmp_path, mock_platform_config
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    job_dir = backend._root_dir / "default" / "train-llama"
    (job_dir / "checkpoints").mkdir(parents=True)

    assert backend.reclaim_persistent_storage("default", "train-llama") is True
    assert not job_dir.exists()
    assert backend.reclaim_persistent_storage("default", "train-llama") is True


def test_reclaim_raises_when_storage_cannot_be_removed(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess.shutil.rmtree", side_effect=PermissionError),
        pytest.raises(PermissionError),
    ):
        backend.reclaim_persistent_storage("default", "train-llama")


def test_cleanup_reads_step_from_registered_attempt(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])
    _schedule_without_otel_export(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    assert metadata.process.wait(timeout=5) == 0
    with patch.object(backend, "get_step_safe", return_value=None) as get_step_safe:
        backend.cleanup_steps()

    get_step_safe.assert_called_once_with(
        job=step.job,
        step_name=step.name,
        workspace=step.workspace,
        attempt_id=str(step.attempt_id),
    )


def _auth_metadata(tmp_path, *, exited: bool = False, refreshed_ago: float = 0.0):
    process = SimpleNamespace(poll=lambda: 0 if exited else None)
    return SimpleNamespace(
        process=process,
        task_auth_env={"NHX_PRINCIPAL": '{"id": "creator"}'},
        task_auth_token_file=tmp_path / ".nhx-access-token",
        task_auth_refreshed_at=time.monotonic() - refreshed_ago,
    )


@pytest.mark.parametrize(
    ("exited", "refreshed_ago", "expect_refresh"),
    [
        pytest.param(False, 10.0, False, id="token-still-fresh"),
        pytest.param(False, 200.0, True, id="half-ttl-elapsed"),
        pytest.param(True, 200.0, False, id="process-exited"),
    ],
)
def test_refresh_task_auth_rotates_token_only_for_running_processes_past_half_ttl(
    mock_nemo_client, tmp_path, mock_platform_config, exited, refreshed_ago, expect_refresh
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = _auth_metadata(tmp_path, exited=exited, refreshed_ago=refreshed_ago)

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess.task_access_token_refresh_interval", return_value=150),
        patch(
            "nhx.core.jobs.controllers.backends.subprocess.refresh_subprocess_task_auth", return_value=True
        ) as refresh,
    ):
        backend._refresh_task_auth(metadata)

    assert refresh.called is expect_refresh
    if expect_refresh:
        refresh.assert_called_once_with(metadata.task_auth_env, metadata.task_auth_token_file)
        assert time.monotonic() - metadata.task_auth_refreshed_at < 5


def test_refresh_task_auth_keeps_old_token_and_retries_after_a_failure(
    mock_nemo_client, tmp_path, mock_platform_config
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = _auth_metadata(tmp_path, refreshed_ago=200.0)
    stale_marker = metadata.task_auth_refreshed_at

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess.task_access_token_refresh_interval", return_value=150),
        patch(
            "nhx.core.jobs.controllers.backends.subprocess.refresh_subprocess_task_auth",
            side_effect=RuntimeError("signing key unavailable"),
        ),
    ):
        backend._refresh_task_auth(metadata)

    assert metadata.task_auth_refreshed_at == stale_marker


def test_refresh_task_auth_retries_when_no_token_could_be_minted(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = _auth_metadata(tmp_path, refreshed_ago=200.0)
    stale_marker = metadata.task_auth_refreshed_at

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess.task_access_token_refresh_interval", return_value=150),
        patch("nhx.core.jobs.controllers.backends.subprocess.refresh_subprocess_task_auth", return_value=False),
    ):
        backend._refresh_task_auth(metadata)

    assert metadata.task_auth_refreshed_at == stale_marker


def test_refresh_task_auth_is_a_no_op_without_a_token_file(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: None), task_auth_env=None, task_auth_token_file=None
    )

    with patch("nhx.core.jobs.controllers.backends.subprocess.refresh_subprocess_task_auth") as refresh:
        backend._refresh_task_auth(metadata)

    refresh.assert_not_called()


def test_exited_process_has_its_task_token_file_removed(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = _auth_metadata(tmp_path, exited=True, refreshed_ago=200.0)
    token_file = metadata.task_auth_token_file
    token_file.write_text("last-token", encoding="utf-8")

    with patch("nhx.core.jobs.controllers.backends.subprocess.refresh_subprocess_task_auth") as refresh:
        backend._refresh_task_auth(metadata)

    refresh.assert_not_called()
    assert not token_file.exists()
    assert metadata.task_auth_token_file is None


def test_shutdown_removes_the_task_token_file_of_a_running_process(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = _auth_metadata(tmp_path)
    token_file = metadata.task_auth_token_file
    token_file.write_text("last-token", encoding="utf-8")
    backend._process_registry.set(SubprocessProcessKey("ws", "job", "1", "step"), metadata)

    with (
        patch.object(backend, "_terminate_process") as terminate,
        patch.object(backend, "_finish_logs"),
    ):
        backend.shutdown()

    terminate.assert_called_once_with(metadata, force=True)
    assert not token_file.exists()
    assert metadata.task_auth_token_file is None


def test_init_removes_task_token_files_left_by_a_previous_run(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    config_dir = tmp_path / "ws" / "job" / "1" / "step" / "task-abc" / "config"
    config_dir.mkdir(parents=True)
    token_file = config_dir / ".nhx-access-token"
    token_file.write_text("stale", encoding="utf-8")
    kept = config_dir / "step-config.json"
    kept.write_text("{}", encoding="utf-8")

    backend.init()

    assert not token_file.exists()
    assert kept.exists()


def test_failed_secret_injection_leaves_no_task_token_file(
    mock_nemo_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "true"])

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess.inject_secret_env_vars", side_effect=RuntimeError("boom")),
        patch("nhx.core.jobs.controllers.backends.subprocess.inject_subprocess_task_auth") as inject_auth,
        pytest.raises(RuntimeError),
    ):
        backend._prepare_runtime(step)

    inject_auth.assert_not_called()


def test_task_token_file_removal_is_retried_after_a_failure(mock_nemo_client, tmp_path, mock_platform_config):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    metadata = _auth_metadata(tmp_path, exited=True)
    token_file = metadata.task_auth_token_file

    with patch("pathlib.Path.unlink", side_effect=PermissionError("denied")):
        backend._refresh_task_auth(metadata)

    assert metadata.task_auth_token_file == token_file


def _schedule_with_delegated_token(backend: SubprocessJobBackend, step, token_file_name: str = ".nhx-access-token"):
    def _inject(env: dict[str, str], token_file) -> dict[str, str]:
        assert token_file.name == token_file_name
        token_file.write_text("minted-token", encoding="utf-8")
        env["NHX_ACCESS_TOKEN_FILE"] = str(token_file)
        env["NHX_PRINCIPAL"] = '{"id": "creator"}'
        return env

    with patch("nhx.core.jobs.controllers.backends.subprocess.inject_subprocess_task_auth", side_effect=_inject):
        return _schedule_without_otel_export(backend, step)


def test_schedule_registers_the_delegated_token_for_rotation(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "exit 0"])

    _schedule_with_delegated_token(backend, step)

    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None
    metadata.process.wait(timeout=5)
    assert metadata.task_auth_token_file is not None
    assert metadata.task_auth_token_file.read_text(encoding="utf-8") == "minted-token"
    assert metadata.task_auth_env == {"NHX_PRINCIPAL": '{"id": "creator"}'}


def test_sync_rotates_the_token_of_a_running_process(
    mock_nemo_client, mock_jobs_client, tmp_path, mock_platform_config, test_step_pending
):
    backend = _subprocess_backend(mock_nemo_client, tmp_path, mock_platform_config)
    step = _step_with_command(test_step_pending, ["/bin/sh", "-c", "sleep 30"])
    _schedule_with_delegated_token(backend, step)
    key = SubprocessProcessKey(step.workspace, step.job, str(step.attempt_id), step.name)
    metadata = backend._process_registry.get(key)
    assert metadata is not None

    try:
        with (
            patch("nhx.core.jobs.controllers.backends.subprocess.task_access_token_refresh_interval", return_value=0),
            patch(
                "nhx.core.jobs.controllers.backends.subprocess.refresh_subprocess_task_auth", return_value=True
            ) as refresh,
        ):
            backend.sync(step)
    finally:
        metadata.process.kill()
        metadata.process.wait(timeout=5)

    refresh.assert_called_once_with(metadata.task_auth_env, metadata.task_auth_token_file)
