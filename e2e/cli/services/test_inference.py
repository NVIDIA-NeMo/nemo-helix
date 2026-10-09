# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for inference CLI commands."""

from __future__ import annotations

import json
import os
import uuid

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    CreateModelProviderRequest,
    ModelProviderStatus,
    UpdateModelProviderStatusRequest,
)
from nhx.core.inference_gateway.api.mock_provider import MOCK_SERVED_MODELS_HEADER
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_inference_providers_lifecycle(client: NemoClient, workspace: str, nemo_run: NemoRun) -> None:
    """Full inference providers lifecycle via CLI: create, list, get, update, delete.
    Also verifies deployments and deployment-configs list, wait inference provider,
    and wait inference deployment --status DELETED."""
    result = nemo_run("inference", "deployments", "list", workspace=workspace)
    assert_exit_0(result, "deployments list failed")
    assert "data" in json.loads(result.stdout)

    result = nemo_run("inference", "deployment-configs", "list", workspace=workspace)
    assert_exit_0(result, "deployment-configs list failed")
    assert "data" in json.loads(result.stdout)

    name = f"e2e-cli-provider-{uuid.uuid4().hex[:8]}"

    result = nemo_run(
        "inference",
        "providers",
        "create",
        name,
        "--host-url",
        "http://localhost:8000",
        workspace=workspace,
    )
    assert_exit_0(result, "providers create failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("inference", "providers", "list", workspace=workspace)
    assert_exit_0(result, "providers list failed")
    assert any(p["name"] == name for p in json.loads(result.stdout).get("data", []))

    result = nemo_run("inference", "providers", "get", name, workspace=workspace)
    assert_exit_0(result, "providers get failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run(
        "inference",
        "providers",
        "update",
        name,
        "--host-url",
        "http://localhost:9000",
        "--description",
        "updated description",
        workspace=workspace,
    )
    assert_exit_0(result, "providers update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    result = nemo_run("inference", "providers", "delete", name, workspace=workspace)
    assert_exit_0(result, "providers delete failed")

    result = nemo_run("inference", "providers", "list", workspace=workspace)
    assert_exit_0(result, "providers list failed")
    assert all(p["name"] != name for p in json.loads(result.stdout).get("data", []))

    # wait inference provider only needs the gateway provider cache, not model or VirtualModel routing.
    model_name = f"e2e-wait-{uuid.uuid4().hex[:8]}"
    mock_provider_name = f"{os.environ.get('NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX', 'igw-mock-')}{model_name}"
    models = ModelsClient.from_client(client)
    models.create_provider(
        workspace=workspace,
        body=CreateModelProviderRequest(
            name=mock_provider_name,
            host_url="http://mock.local",
            default_extra_headers={MOCK_SERVED_MODELS_HEADER: json.dumps([model_name])},
        ),
    )
    models.update_provider_status(
        workspace=workspace,
        name=mock_provider_name,
        body=UpdateModelProviderStatusRequest(status=ModelProviderStatus.READY),
    )
    try:
        result = nemo_run(
            "wait",
            "inference",
            "provider",
            mock_provider_name,
            "--workspace",
            workspace,
            "--timeout",
            "60",
            workspace=workspace,
            timeout=90,
        )
        assert_exit_0(result, "wait inference provider failed")
    finally:
        try:
            models.delete_provider(workspace=workspace, name=mock_provider_name)
        except Exception:
            pass

    # wait inference deployment --status DELETED: NotFoundError is treated as success
    result = nemo_run(
        "wait",
        "inference",
        "deployment",
        f"nonexistent-{uuid.uuid4().hex[:8]}",
        "--status",
        "DELETED",
        "--timeout",
        "10",
        workspace=workspace,
        timeout=30,
    )
    assert_exit_0(result, "wait inference deployment --status DELETED failed")
