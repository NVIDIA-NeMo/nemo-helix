# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for models CLI commands."""

from __future__ import annotations

import json
import uuid

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nhx.testing import MockProviderResponse, NemoRun, add_mock_provider, assert_exit_0

from e2e.external.helpers import wait_for_deployment_status

pytestmark = [pytest.mark.timeout(600), pytest.mark.feature("gpu"), pytest.mark.platform("docker")]


def test_models_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full models lifecycle via CLI: create, list, get, update, delete."""
    name = f"e2e-cli-model-{uuid.uuid4().hex[:8]}"

    result = nemo_run("models", "create", name, workspace=workspace)
    assert_exit_0(result, "models create failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("models", "list", workspace=workspace)
    assert_exit_0(result, "models list failed")
    assert "data" in json.loads(result.stdout)
    assert any(m["name"] == name for m in json.loads(result.stdout).get("data", []))

    result = nemo_run("models", "get", name, workspace=workspace)
    assert_exit_0(result, "models get failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run(
        "models",
        "update",
        name,
        "--description",
        "updated description",
        workspace=workspace,
    )
    assert_exit_0(result, "models update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    result = nemo_run("models", "delete", name, workspace=workspace)
    assert_exit_0(result, "models delete failed")

    result = nemo_run("models", "list", workspace=workspace)
    assert_exit_0(result, "models list failed")
    assert all(m["name"] != name for m in json.loads(result.stdout).get("data", []))


def test_model_deployment_lifecycle(client: NemoClient, workspace: str, nemo_run: NemoRun, mock_nim_image: str) -> None:
    """Deploy a mock NIM model via CLI: create deployment config, deploy, simulate READY, verify, delete."""
    run_id = uuid.uuid4().hex[:8]
    model_name = f"e2e-cli-model-{run_id}"
    config_name = f"e2e-cli-dc-{run_id}"
    deploy_name = f"e2e-cli-deploy-{run_id}"
    provider_name = f"nim-provider-{run_id}"

    result = nemo_run("models", "create", model_name, workspace=workspace)
    assert_exit_0(result, "models create failed")
    assert json.loads(result.stdout).get("name") == model_name

    # Create deployment config pointing at mock NIM
    image_name, image_tag = mock_nim_image.rsplit(":", 1)
    nim_config = json.dumps(
        {
            "gpu": 0,
            "model_name": model_name,
            "model_namespace": workspace,
            "image_name": image_name,
            "image_tag": image_tag,
        }
    )
    result = nemo_run(
        "inference",
        "deployment-configs",
        "create",
        config_name,
        "--nim-deployment",
        nim_config,
        "--model-entity-id",
        model_name,
        workspace=workspace,
    )
    assert_exit_0(result, "deployment-configs create failed")
    assert json.loads(result.stdout).get("name") == config_name

    # Create deployment
    result = nemo_run(
        "inference",
        "deployments",
        "create",
        deploy_name,
        "--config",
        config_name,
        workspace=workspace,
    )
    assert_exit_0(result, "deployments create failed")
    assert json.loads(result.stdout).get("name") == deploy_name

    # Verify deployment appears in list and get
    result = nemo_run("inference", "deployments", "list", workspace=workspace)
    assert_exit_0(result, "deployments list failed")
    assert any(d["name"] == deploy_name for d in json.loads(result.stdout).get("data", []))

    result = nemo_run("inference", "deployments", "get", deploy_name, workspace=workspace)
    assert_exit_0(result, "deployments get failed")
    assert json.loads(result.stdout).get("name") == deploy_name

    # Simulate NIM serving the model via mock provider
    mock_response = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model_name,
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    provider = add_mock_provider(
        client,
        workspace=workspace,
        name=provider_name,
        mock_response_body_by_model={
            model_name: [MockProviderResponse(response_body=mock_response)],
        },
        served_models={model_name: model_name},
    )

    # Wait for the controller to health-check the container and set READY before
    # associating the mock provider — otherwise the controller's PENDING transition
    # (container starting) clears model_provider_id.
    wait_for_deployment_status(client, deploy_name, workspace, expected_status="READY", timeout=180.0)

    # Associate mock provider with the deployment via CLI
    result = nemo_run(
        "inference",
        "deployments",
        "update-status",
        deploy_name,
        "--status",
        "READY",
        "--model-provider-id",
        f"{workspace}/{provider.name}",
        workspace=workspace,
    )
    assert_exit_0(result, "deployments update-status failed")
    assert json.loads(result.stdout).get("status") == "READY"

    # Verify READY status via get
    result = nemo_run("inference", "deployments", "get", deploy_name, workspace=workspace)
    assert_exit_0(result, "deployments get failed")
    assert json.loads(result.stdout).get("status") == "READY"

    # Verify model is discoverable through inference gateway
    result = nemo_run("inference", "models", "list", workspace=workspace)
    assert_exit_0(result, "inference models list failed")
    model_ids = [m.get("id", "") for m in json.loads(result.stdout).get("data", [])]
    assert any(model_name in mid for mid in model_ids)

    # Retry until the config delete succeeds. The models controller may race to set ERROR
    # after our DELETED update (including an SDK-level retry with a 0.4s delay). Once the
    # controller finishes, ERROR is terminal and a subsequent DELETED + delete will succeed.
    for _ in range(10):
        nemo_run(
            "inference",
            "deployments",
            "update-status",
            deploy_name,
            "--status",
            "DELETED",
            workspace=workspace,
        )
        result = nemo_run("inference", "deployment-configs", "delete", config_name, workspace=workspace)
        if result.returncode == 0:
            break

    assert_exit_0(result, "deployment-configs delete failed")

    result = nemo_run("models", "delete", model_name, workspace=workspace)
    assert_exit_0(result, "models delete failed")
