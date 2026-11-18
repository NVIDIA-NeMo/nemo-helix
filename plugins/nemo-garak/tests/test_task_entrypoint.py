# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the audit task container entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nemo_garak.service import GarakPluginService
from nemo_garak.tasks import audit as audit_task


def test_task_authenticates_as_the_registered_garak_service() -> None:
    # The authorization service admits a service principal only when it is a registered service
    # name. A name that is not registered fails every platform call the task makes.
    assert audit_task.TASK_SERVICE_NAME == GarakPluginService.name


def test_main_builds_both_clients_as_the_task_service_and_runs_the_job() -> None:
    client = MagicMock()
    async_client = MagicMock()
    job = MagicMock()
    job.run.return_value = {"status": "completed"}

    with (
        patch.object(audit_task, "configure_task_logging"),
        patch.object(audit_task, "signal"),
        patch.object(audit_task, "get_task_nemo_client", return_value=client) as get_client,
        patch.object(audit_task, "get_async_task_nemo_client", return_value=async_client) as get_async_client,
        patch.object(audit_task, "build_ctx_from_env", return_value="ctx") as build_ctx,
        patch.object(audit_task, "read_step_config", return_value={"config": "c", "target": "t"}),
        patch.object(audit_task, "AuditJob", return_value=job),
    ):
        exit_code = audit_task.main()

    assert exit_code == 0
    get_client.assert_called_once_with("garak")
    get_async_client.assert_called_once_with("garak")
    build_ctx.assert_called_once_with(client)
    job.run.assert_called_once_with({"config": "c", "target": "t"}, ctx="ctx", sdk=client, async_sdk=async_client)


def test_main_reports_a_failed_job_through_its_exit_code() -> None:
    job = MagicMock()
    job.run.return_value = {"status": "failed", "error": "Publication to Intake failed"}

    with (
        patch.object(audit_task, "configure_task_logging"),
        patch.object(audit_task, "signal"),
        patch.object(audit_task, "get_task_nemo_client"),
        patch.object(audit_task, "get_async_task_nemo_client"),
        patch.object(audit_task, "build_ctx_from_env"),
        patch.object(audit_task, "read_step_config", return_value={}),
        patch.object(audit_task, "AuditJob", return_value=job),
    ):
        assert audit_task.main() == 1
