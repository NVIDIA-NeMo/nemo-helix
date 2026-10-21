# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers that do not call a platform."""

from types import SimpleNamespace

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.watch_types import JobLogEvent, JobStatusEvent, JobWarningEvent
from support import (
    accept_report,
    chat_target,
    default_task_image,
    report_path,
    task_image_from_launcher,
    wait_for_deployment,
    wait_for_job,
)


def _client() -> NemoClient:
    """A client that is never called: each test replaces the service client built from it."""
    return NemoClient(base_url="http://platform.test", workspace="default")


def test_chat_target_uses_the_model_workspace() -> None:
    workspace, model = chat_target("system/meta-llama-3-1-8b-instruct", "check-1")
    assert workspace == "system"
    assert model == "system/meta-llama-3-1-8b-instruct"


def test_chat_target_keeps_a_bare_name_in_the_check_workspace() -> None:
    workspace, model = chat_target("my-model", "check-1")
    assert workspace == "check-1"
    assert model == "my-model"


def test_task_image_replaces_the_launcher_name() -> None:
    image = task_image_from_launcher("nvcr.io/example/nemo-helix-dev/nhx-api:abc123")
    assert image == "nvcr.io/example/nemo-helix-dev/nhx-tasks:abc123"


def test_task_image_ignores_other_launcher_images() -> None:
    assert task_image_from_launcher("nvcr.io/example/jobs-launcher:latest") is None


def test_default_task_image_does_not_borrow_another_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = [
        SimpleNamespace(profile="default", config=SimpleNamespace(default_task_image=None)),
        SimpleNamespace(profile="gpu", config=SimpleNamespace(default_task_image="registry.example/nhx-tasks:gpu")),
    ]

    class Jobs:
        def list_execution_profiles(self) -> list[SimpleNamespace]:
            return profiles

    monkeypatch.setattr("support.JobsClient.from_client", lambda client: Jobs())
    assert default_task_image(_client(), "default") is None
    assert default_task_image(_client(), "missing") is None
    assert default_task_image(_client(), None) == "registry.example/nhx-tasks:gpu"


def test_report_path_is_unique_to_the_run() -> None:
    assert report_path("job", "abc") == "job-abc.json"


def test_accept_report_rejects_a_stale_fileset_object() -> None:
    stale = {"report_id": "old", "mode": "anonymous"}
    current = {"report_id": "new", "mode": "on_behalf_of"}
    assert accept_report(stale, "new") is None
    assert accept_report(current, "new") == current
    assert accept_report(None, "new") is None


def test_wait_for_job_prints_watch_events(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    job = SimpleNamespace(name="job-1", status="completed")
    events = [
        JobStatusEvent(
            kind="status",
            job_name="job-1",
            status="pending",
            status_details={},
            terminal=False,
            successful=None,
        ),
        JobLogEvent(kind="log", job_name="job-1", timestamp=None, step_id=None, task_id=None, message="probe line"),
        JobWarningEvent(kind="warning", job_name="job-1", message="log page retried"),
        JobStatusEvent(
            kind="status",
            job_name="job-1",
            status="completed",
            status_details={},
            terminal=True,
            successful=True,
        ),
    ]

    class Jobs:
        def get_job(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(data=lambda: job)

        def watch_job(self, name: str, **kwargs: object) -> list[JobStatusEvent | JobLogEvent | JobWarningEvent]:
            assert name == "job-1"
            assert kwargs["workspace"] == "ws"
            return events

    monkeypatch.setattr("support.JobsClient.from_client", lambda client: Jobs())
    assert wait_for_job(_client(), "ws", "job-1", 5) is job
    output = capsys.readouterr().out
    assert "watching job job-1" in output
    assert "job job-1: pending" in output
    assert "job job-1 log: probe line" in output
    assert "job job-1 warning: log page retried" in output
    assert "job job-1: completed" in output


def test_wait_for_deployment_prints_status_history(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deployment = SimpleNamespace(
        status="SUCCEEDED",
        status_history=[
            SimpleNamespace(status="PENDING", status_message="scheduled"),
            SimpleNamespace(status="SUCCEEDED", status_message="Container exited successfully"),
        ],
    )

    class Deployments:
        def get_deployment(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(data=lambda: deployment)

    monkeypatch.setattr("support.DeploymentsClient.from_client", lambda client: Deployments())
    assert wait_for_deployment(_client(), "ws", "dep-1", 5) is deployment
    output = capsys.readouterr().out
    assert "watching deployment dep-1" in output
    assert "deployment dep-1: PENDING scheduled" in output
    assert "deployment dep-1: SUCCEEDED Container exited successfully" in output
