# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model_entity task reports its own phase, so the job does not keep showing the upload's."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.customization_common.service.context import NHXJobContext

# The package re-exports the ``run`` function under the module's own name.
model_entity_run = importlib.import_module("nhx.customization_common.tasks.model_entity.run")


class _RecordingReporter:
    def __init__(self) -> None:
        self.reports: list[tuple[HelixJobStatus, Any]] = []

    def update_progress(self, status: HelixJobStatus, status_details: dict[str, Any] | None = None, **_: Any) -> None:
        self.reports.append((status, (status_details or {}).get("phase")))


@pytest.fixture
def reporter(monkeypatch: pytest.MonkeyPatch) -> _RecordingReporter:
    recording = _RecordingReporter()
    monkeypatch.setattr(
        model_entity_run.JobsServiceProgressReporter,
        "create_progress_reporter",
        staticmethod(lambda jobs, job_ctx: recording),
    )
    monkeypatch.setattr(
        model_entity_run,
        "get_config",
        lambda path: SimpleNamespace(
            name="m", workspace="default", fileset=SimpleNamespace(workspace=None, name="f"), deployment_config=None
        ),
    )
    return recording


def _run(runner: MagicMock, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(model_entity_run, "ModelEntityRunner", lambda **kwargs: runner)
    job_ctx = NHXJobContext(
        workspace="default",
        job_id="job-1",
        attempt_id="attempt-1",
        step="step",
        task="task",
        jobs_url=None,
        files_url=None,
        storage_path=Path("/storage"),
        config_path=Path("/config.json"),
    )
    return model_entity_run.run(MagicMock(), job_ctx, service_name="customizer")


def test_reports_registering_then_completed(reporter: _RecordingReporter, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = MagicMock()
    runner.create_model_entity.return_value = ({}, MagicMock())

    assert _run(runner, monkeypatch) == 0
    assert reporter.reports == [
        (HelixJobStatus.ACTIVE, "registering_model"),
        (HelixJobStatus.COMPLETED, "completed"),
    ]


def test_a_failed_registration_never_reports_completed(
    reporter: _RecordingReporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = MagicMock()
    runner.create_model_entity.side_effect = RuntimeError("models service unreachable")

    assert _run(runner, monkeypatch) == 1
    assert reporter.reports == [(HelixJobStatus.ACTIVE, "registering_model")]
