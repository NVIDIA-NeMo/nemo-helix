# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_deployments_plugin.entities import Deployment, Volume
from nemo_deployments_plugin.types import Endpoint
from nhx.core.models.controllers.backends.backends import DeploymentStatusUpdate
from nhx.core.models.controllers.backends.deployments_plugin.status import (
    aggregate_status,
    apply_deleting_timeout,
    apply_pending_timeout,
    build_deleting_timeout_error,
    build_pending_timeout_error,
    map_status,
    project_host_url,
)


@pytest.mark.parametrize(
    ("plugin_status", "model_status"),
    [
        ("PENDING", "PENDING"),
        ("STARTING", "PENDING"),
        ("READY", "READY"),
        ("FAILED", "ERROR"),
        ("LOST", "LOST"),
        ("UNKNOWN", "UNKNOWN"),
        ("DELETING", "DELETING"),
        ("SUCCEEDED", "PENDING"),
    ],
)
def test_map_status(plugin_status: str, model_status: str) -> None:
    assert map_status(plugin_status) == model_status


def test_map_status_defaults_to_unknown() -> None:
    assert map_status("SOMETHING_NEW") == "UNKNOWN"


def test_ready_projects_http_endpoint() -> None:
    server = Deployment(
        name="server",
        workspace="default",
        deployment_config="server",
        status="READY",
        endpoints=[Endpoint(name="http", url="https://server", protocol="https")],
    )
    assert project_host_url(server.endpoints) == "https://server"
    assert aggregate_status(None, None, server).host_url == "https://server"


def test_missing_ready_server_is_lost_and_failed_puller_is_error() -> None:
    assert aggregate_status(None, None, None, previously_ready=True).status == "LOST"
    puller = Deployment(name="puller", workspace="default", deployment_config="puller", status="FAILED")
    assert aggregate_status(None, puller, None).status == "ERROR"


def test_no_substrate_yet_is_pending() -> None:
    result = aggregate_status(None, None, None)
    assert result.status == "PENDING"


def test_failed_volume_is_error() -> None:
    volume = Volume(name="volume", workspace="default", size="1Gi", status="FAILED")
    assert aggregate_status(volume, None, None).status == "ERROR"


def test_lost_puller_is_error() -> None:
    puller = Deployment(name="puller", workspace="default", deployment_config="puller", status="LOST")
    assert aggregate_status(None, puller, None).status == "ERROR"


def test_unknown_puller_surfaces_unknown() -> None:
    puller = Deployment(name="puller", workspace="default", deployment_config="puller", status="UNKNOWN")
    assert aggregate_status(None, puller, None).status == "UNKNOWN"


def test_apply_pending_timeout_escalates_pending_only() -> None:
    pending = aggregate_status(None, None, None)
    assert (
        apply_pending_timeout(
            pending,
            elapsed_seconds=30,
            timeout_seconds=60,
            deployment_name="my-dep",
        ).status
        == "PENDING"
    )
    timed_out = apply_pending_timeout(
        pending,
        elapsed_seconds=90,
        timeout_seconds=60,
        deployment_name="my-dep",
    )
    assert timed_out.status == "ERROR"
    assert timed_out.error_details is not None
    assert timed_out.error_details["reason"] == "pending_timeout"


def test_build_pending_timeout_error_includes_substrate() -> None:
    substrate = {"server": {"status": "STARTING"}}
    result = build_pending_timeout_error(
        deployment_name="my-dep",
        elapsed_seconds=120,
        timeout_seconds=60,
        substrate=substrate,
    )
    assert result.status == "ERROR"
    assert result.error_details is not None
    assert result.error_details["substrate"] == substrate


def test_apply_deleting_timeout_escalates_deleting_only() -> None:
    deleting = DeploymentStatusUpdate(status="DELETING", status_message="waiting")
    assert (
        apply_deleting_timeout(
            deleting,
            elapsed_seconds=30,
            timeout_seconds=60,
            deployment_name="my-dep",
        ).status
        == "DELETING"
    )
    timed_out = apply_deleting_timeout(
        deleting,
        elapsed_seconds=90,
        timeout_seconds=60,
        deployment_name="my-dep",
    )
    assert timed_out.status == "ERROR"
    assert timed_out.error_details is not None
    assert timed_out.error_details["reason"] == "deleting_timeout"


def test_build_deleting_timeout_error_message() -> None:
    result = build_deleting_timeout_error(
        deployment_name="my-dep",
        elapsed_seconds=120,
        timeout_seconds=60,
    )
    assert result.status == "ERROR"
    assert "my-dep" in result.status_message
    assert result.error_details is not None
    assert result.error_details["reason"] == "deleting_timeout"
