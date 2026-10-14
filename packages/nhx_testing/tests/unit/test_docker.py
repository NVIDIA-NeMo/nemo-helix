# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Docker testing helpers."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nhx.testing.docker import (
    MODELS_CONTROLLER_MANAGED_LABEL,
    cleanup_model_deployment_containers,
    remove_intake_clickhouse,
)


def _container(name: str, labels: dict[str, str]) -> MagicMock:
    container = MagicMock()
    container.name = name
    container.labels = labels
    return container


def test_cleanup_model_deployment_containers_filters_by_owner_labels():
    """Cleanup removes only managed containers matching the supplied owner labels."""
    owner_labels = {
        "nhx.nvidia.com/test-run": "run-1",
        "nhx.nvidia.com/test-worker": "gw0",
    }
    matching_container = _container(
        "matching",
        {
            MODELS_CONTROLLER_MANAGED_LABEL: "models-controller",
            **owner_labels,
        },
    )
    other_worker_container = _container(
        "other-worker",
        {
            MODELS_CONTROLLER_MANAGED_LABEL: "models-controller",
            "nhx.nvidia.com/test-run": "run-1",
            "nhx.nvidia.com/test-worker": "gw1",
        },
    )

    docker_client = MagicMock()
    docker_client.containers.list.return_value = [matching_container, other_worker_container]

    removed = cleanup_model_deployment_containers(docker_client, labels=owner_labels)

    assert removed == 1
    docker_client.containers.list.assert_called_once_with(
        all=True,
        filters={"label": MODELS_CONTROLLER_MANAGED_LABEL},
        ignore_removed=True,
    )
    matching_container.stop.assert_called_once()
    matching_container.remove.assert_called_once_with(force=True)
    other_worker_container.stop.assert_not_called()
    other_worker_container.remove.assert_not_called()


def test_remove_intake_clickhouse_removes_the_platform_data_dir_container(monkeypatch):
    """Removal targets Intake's ClickHouse subdir and runs with the platform's NHX_DATA_DIR."""
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(subprocess, "run", run)
    data_dir = Path("/tmp/platform/data")

    remove_intake_clickhouse(data_dir, env={"NHX_BASE_URL": "http://localhost:8090"})

    command = run.call_args.args[0]
    assert command == [
        sys.executable,
        "-m",
        "nhx.intake.local_clickhouse",
        "--remove",
        "--data-dir",
        str(data_dir / "intake-clickhouse"),
    ]
    assert run.call_args.kwargs["env"] == {"NHX_BASE_URL": "http://localhost:8090", "NHX_DATA_DIR": str(data_dir)}


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.CompletedProcess([], 1, stdout="", stderr="Docker daemon is unavailable."),
        subprocess.TimeoutExpired(["python"], 120),
        OSError("no python"),
    ],
    ids=["nonzero-exit", "timeout", "spawn-error"],
)
def test_remove_intake_clickhouse_failures_warn_without_raising(monkeypatch, capsys, outcome):
    """Teardown cleanup is best effort, so a failed removal cannot mask the test's own result."""

    def run(*args, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(subprocess, "run", run)

    remove_intake_clickhouse(Path("/tmp/platform/data"), env={})

    assert "Could not remove Intake ClickHouse for /tmp/platform/data" in capsys.readouterr().out
