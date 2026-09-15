# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from nemo_platform_ext.cli.commands.api.inference.deployments import create_deployments
from nemo_platform_ext.cli.commands.jobs import create_jobs, watch_platform_job
from nemo_platform_ext.cli.core.job_watch_renderer import JobWatchRenderResult

PLATFORM_SPEC_JSON = '{"steps":[{"name":"step-one","executor":{"provider":"cpu","container":{"image":"x"}}}]}'


class _CreatedJob:
    def __init__(self, *, name: str = "created-job", workspace: str = "result-workspace") -> None:
        self.name = name
        self.workspace = workspace

    def model_dump(self, *, mode: str = "json") -> dict[str, str]:
        assert mode == "json"
        return {"name": self.name, "workspace": self.workspace}


class _Response:
    def __init__(self, body: Any) -> None:
        self._body = body

    def data(self) -> Any:
        return self._body


def _ctx(client: object) -> SimpleNamespace:
    state = MagicMock()
    state.agent_mode = False
    state.get_client.return_value = client
    state.get_output_format.return_value = None
    state.get_no_truncate.return_value = False
    state.get_timestamp_format.return_value = None
    return SimpleNamespace(obj=state)


def _assert_created_job_request(jobs_client: MagicMock, *, workspace: str = "test-workspace") -> None:
    jobs_client.create_job.assert_called_once()
    call_kwargs = jobs_client.create_job.call_args.kwargs
    assert call_kwargs["workspace"] == workspace
    body = call_kwargs["body"]
    assert body.name == "input-job"
    assert body.source == "test-source"
    assert body.spec == {}
    assert body.platform_spec.steps[0].name == "step-one"
    assert body.platform_spec.steps[0].executor.provider == "cpu"


def test_jobs_create_watch_uses_sdk_watcher_and_outputs_created_job() -> None:
    created_job = _CreatedJob()
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    ctx = _ctx(client)
    events = object()
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(created_job)
    jobs_client.watch_job.return_value = events

    with (
        patch(
            "nemo_platform_ext.cli.commands.jobs.handle_code_generation",
            return_value=False,
        ) as handle_code_generation,
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch("nemo_platform_ext.cli.commands.jobs.format_output") as format_output,
        patch(
            "nemo_platform_ext.cli.commands.jobs.render_job_watch_events",
            return_value=JobWatchRenderResult.SUCCEEDED,
        ) as render_events,
    ):
        create_jobs(
            ctx,
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=False,
            watch=True,
            timeout=42,
            poll_interval=7,
        )

    expected_kwargs = {
        "workspace": "test-workspace",
        "platform_spec": {
            "steps": [{"name": "step-one", "executor": {"provider": "cpu", "container": {"image": "x"}}}]
        },
        "source": "test-source",
        "spec": {},
        "name": "input-job",
    }
    handle_code_generation.assert_called_once_with(
        ["jobs"],
        "create",
        expected_kwargs,
        None,
        ctx.obj,
        watch_config={"type": "platform_job", "resource_label": "job"},
        watch_options={"timeout": 42, "poll_interval": 7},
        wait_config=None,
        wait_options=None,
    )
    _assert_created_job_request(jobs_client)
    format_output.assert_called_once_with(
        created_job,
        is_list=False,
        output_format=None,
        no_truncate=False,
        timestamp_format=None,
    )
    jobs_client.watch_job.assert_called_once_with(
        "created-job",
        workspace="result-workspace",
        timeout=42,
        poll_interval=7,
    )
    render_events.assert_called_once_with(events, resource_label="job")


def test_jobs_create_watch_exits_when_renderer_reports_failure() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(_CreatedJob())
    jobs_client.watch_job.return_value = object()

    with (
        patch("nemo_platform_ext.cli.commands.jobs.handle_code_generation", return_value=False),
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch(
            "nemo_platform_ext.cli.commands.jobs.render_job_watch_events",
            return_value=JobWatchRenderResult.FAILED,
        ),
        pytest.raises(typer.Exit) as exc_info,
    ):
        create_jobs(
            _ctx(client),
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=False,
            watch=True,
        )

    assert exc_info.value.exit_code == 1


def test_jobs_create_watch_exits_130_when_renderer_reports_interrupted() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(_CreatedJob())
    jobs_client.watch_job.return_value = object()

    with (
        patch("nemo_platform_ext.cli.commands.jobs.handle_code_generation", return_value=False),
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch(
            "nemo_platform_ext.cli.commands.jobs.render_job_watch_events",
            return_value=JobWatchRenderResult.INTERRUPTED,
        ),
        pytest.raises(typer.Exit) as exc_info,
    ):
        create_jobs(
            _ctx(client),
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=False,
            watch=True,
        )

    assert exc_info.value.exit_code == 130


def test_jobs_create_watch_has_no_default_timeout() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    ctx = _ctx(client)
    events = object()
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(_CreatedJob())
    jobs_client.watch_job.return_value = events

    with (
        patch(
            "nemo_platform_ext.cli.commands.jobs.handle_code_generation",
            return_value=False,
        ) as handle_code_generation,
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch(
            "nemo_platform_ext.cli.commands.jobs.render_job_watch_events",
            return_value=JobWatchRenderResult.SUCCEEDED,
        ),
    ):
        create_jobs(
            ctx,
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=False,
            watch=True,
        )

    handle_code_generation.assert_called_once()
    assert handle_code_generation.call_args.kwargs["watch_options"] == {"timeout": None, "poll_interval": 3}
    jobs_client.watch_job.assert_called_once()
    assert jobs_client.watch_job.call_args.kwargs["timeout"] is None


def test_jobs_create_wait_uses_quiet_waiter_and_outputs_created_job() -> None:
    created_job = _CreatedJob()
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    ctx = _ctx(client)
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(created_job)

    with (
        patch(
            "nemo_platform_ext.cli.commands.jobs.handle_code_generation",
            return_value=False,
        ) as handle_code_generation,
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch("nemo_platform_ext.cli.commands.jobs.format_output") as format_output,
        patch("nemo_platform_ext.cli.commands.jobs.wait_for_platform_job", return_value=True) as wait_for_job,
        patch("nemo_platform_ext.cli.commands.jobs.render_job_watch_events") as render_events,
    ):
        create_jobs(
            ctx,
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=True,
            watch=False,
            timeout=42,
            poll_interval=7,
        )

    expected_kwargs = {
        "workspace": "test-workspace",
        "platform_spec": {
            "steps": [{"name": "step-one", "executor": {"provider": "cpu", "container": {"image": "x"}}}]
        },
        "source": "test-source",
        "spec": {},
        "name": "input-job",
    }
    handle_code_generation.assert_called_once_with(
        ["jobs"],
        "create",
        expected_kwargs,
        None,
        ctx.obj,
        watch_config=None,
        watch_options=None,
        wait_config={"type": "platform_job", "resource_label": "job"},
        wait_options={"timeout": 42, "poll_interval": 7},
    )
    _assert_created_job_request(jobs_client)
    wait_for_job.assert_called_once_with(
        jobs_client,
        "created-job",
        workspace="result-workspace",
        resource_label="job",
        timeout=42,
        poll_interval=7,
    )
    format_output.assert_called_once_with(
        created_job,
        is_list=False,
        output_format=None,
        no_truncate=False,
        timestamp_format=None,
    )
    jobs_client.watch_job.assert_not_called()
    render_events.assert_not_called()


def test_jobs_create_wait_uses_waiter_default_timeout() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(_CreatedJob())

    with (
        patch(
            "nemo_platform_ext.cli.commands.jobs.handle_code_generation",
            return_value=False,
        ) as handle_code_generation,
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch("nemo_platform_ext.cli.commands.jobs.wait_for_platform_job", return_value=True) as wait_for_job,
    ):
        create_jobs(
            _ctx(client),
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=True,
            watch=False,
        )

    assert handle_code_generation.call_args.kwargs["wait_options"] == {"timeout": 1200, "poll_interval": 3}
    wait_for_job.assert_called_once()
    assert wait_for_job.call_args.kwargs["timeout"] == 1200


def test_jobs_create_wait_exits_when_waiter_reports_failure() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    jobs_client = MagicMock()
    jobs_client.create_job.return_value = _Response(_CreatedJob())

    with (
        patch("nemo_platform_ext.cli.commands.jobs.handle_code_generation", return_value=False),
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch("nemo_platform_ext.cli.commands.jobs.wait_for_platform_job", return_value=False),
        pytest.raises(typer.Exit) as exc_info,
    ):
        create_jobs(
            _ctx(client),
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=True,
            watch=False,
        )

    assert exc_info.value.exit_code == 1


def test_jobs_create_rejects_wait_and_watch_together() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))

    with pytest.raises(SystemExit) as exc_info:
        create_jobs(
            _ctx(client),
            name="input-job",
            workspace="test-workspace",
            platform_spec=PLATFORM_SPEC_JSON,
            source="test-source",
            spec="{}",
            wait=True,
            watch=True,
        )

    assert exc_info.value.code == 2


def test_jobs_watch_command_uses_sdk_watcher() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    events = object()
    jobs_client = MagicMock()
    jobs_client.watch_job.return_value = events

    with (
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch(
            "nemo_platform_ext.cli.commands.jobs.render_job_watch_events",
            return_value=JobWatchRenderResult.SUCCEEDED,
        ) as render_events,
    ):
        watch_platform_job(
            _ctx(client),
            name="job-a",
            workspace=None,
            attempt_id=1,
            step_id="step-1",
            task_id="task-1",
            limit=25,
            timeout=42,
            poll_interval=7,
            include_history=False,
        )

    jobs_client.watch_job.assert_called_once_with(
        "job-a",
        workspace="default",
        attempt_id=1,
        step_id="step-1",
        task_id="task-1",
        limit=25,
        timeout=42,
        poll_interval=7,
        include_history=False,
    )
    render_events.assert_called_once_with(events, resource_label="job")


def test_jobs_watch_command_exits_130_when_renderer_reports_interrupted() -> None:
    client = SimpleNamespace(_get_workspace_path_param=MagicMock(return_value="default"))
    jobs_client = MagicMock()
    jobs_client.watch_job.return_value = object()

    with (
        patch("nemo_platform_ext.cli.commands.jobs.client_from_platform", return_value=jobs_client),
        patch(
            "nemo_platform_ext.cli.commands.jobs.render_job_watch_events",
            return_value=JobWatchRenderResult.INTERRUPTED,
        ),
        pytest.raises(typer.Exit) as exc_info,
    ):
        watch_platform_job(
            _ctx(client),
            name="job-a",
            workspace="default",
            attempt_id=None,
            step_id=None,
            task_id=None,
            limit=None,
            timeout=None,
            poll_interval=3,
            include_history=True,
        )

    assert exc_info.value.exit_code == 130


def test_inference_deployment_create_exits_when_wait_fails() -> None:
    deployments = MagicMock()
    deployments.create.return_value = SimpleNamespace(name="deployment-a")
    client = SimpleNamespace(inference=SimpleNamespace(deployments=deployments))

    with (
        patch("nemo_platform_ext.cli.commands.api.inference.deployments.handle_code_generation", return_value=False),
        patch("nemo_platform_ext.cli.commands.api.inference.deployments.format_output"),
        patch(
            "nemo_platform_ext.cli.commands.api.inference.deployments.wait_for_inference_deployment",
            return_value=False,
        ) as wait_for_inference_deployment,
        pytest.raises(typer.Exit) as exc_info,
    ):
        create_deployments(
            _ctx(client),
            name="deployment-a",
            workspace="test-workspace",
            config="deployment-config",
            wait=True,
            watch=False,
            timeout=90,
            poll_interval=10,
        )

    assert exc_info.value.exit_code == 1
    wait_for_inference_deployment.assert_called_once_with(
        client,
        "deployment-a",
        workspace="test-workspace",
        timeout=90,
        poll_interval=10,
    )


def test_inference_deployment_create_watch_uses_waiter() -> None:
    deployments = MagicMock()
    deployments.create.return_value = SimpleNamespace(name="deployment-a")
    client = SimpleNamespace(inference=SimpleNamespace(deployments=deployments))
    ctx = _ctx(client)

    with (
        patch(
            "nemo_platform_ext.cli.commands.api.inference.deployments.handle_code_generation",
            return_value=False,
        ) as handle_code_generation,
        patch("nemo_platform_ext.cli.commands.api.inference.deployments.format_output") as format_output,
        patch(
            "nemo_platform_ext.cli.commands.api.inference.deployments.wait_for_inference_deployment",
            return_value=True,
        ) as wait_for_inference_deployment,
    ):
        create_deployments(
            ctx,
            name="deployment-a",
            workspace="test-workspace",
            config="deployment-config",
            wait=False,
            watch=True,
            timeout=90,
            poll_interval=10,
        )

    expected_kwargs = {
        "workspace": "test-workspace",
        "config": "deployment-config",
        "name": "deployment-a",
    }
    handle_code_generation.assert_called_once_with(
        ["inference", "deployments"],
        "create",
        expected_kwargs,
        None,
        ctx.obj,
        watch_config={"type": "inference_deployment", "resource_label": "deployment"},
        watch_options={"timeout": 90, "poll_interval": 10},
        wait_config=None,
        wait_options=None,
    )
    deployments.create.assert_called_once_with(**expected_kwargs)
    format_output.assert_called_once()
    wait_for_inference_deployment.assert_called_once_with(
        client,
        "deployment-a",
        workspace="test-workspace",
        timeout=90,
        poll_interval=10,
    )


def test_inference_deployment_create_rejects_wait_and_watch_together() -> None:
    deployments = MagicMock()
    client = SimpleNamespace(inference=SimpleNamespace(deployments=deployments))

    with pytest.raises(SystemExit) as exc_info:
        create_deployments(
            _ctx(client),
            name="deployment-a",
            workspace="test-workspace",
            config="deployment-config",
            wait=True,
            watch=True,
        )

    assert exc_info.value.code == 2
    deployments.create.assert_not_called()
