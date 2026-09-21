# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo inference`` against the in-process Models service."""

import json
import uuid

from nemo_platform_ext.cli.app import app

from ..utils import assert_exit_code


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def test_providers_lifecycle(runner, random_workspace: str) -> None:
    ws = random_workspace
    name = _name("prov")

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "create",
            name,
            "--host-url",
            "https://integrate.api.nvidia.com/v1",
            "--description",
            "demo",
            "--enabled-models",
            "meta/llama",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["workspace"] == ws
    assert created["description"] == "demo"
    assert created["enabled_models"] == ["meta/llama"]

    result = runner.invoke(app, ["inference", "providers", "get", name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["host_url"] == "https://integrate.api.nvidia.com/v1"

    result = runner.invoke(app, ["inference", "providers", "list", "--workspace", ws])
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = runner.invoke(
        app,
        ["inference", "providers", "list", "--workspace", ws, "--filter.name", "does-not-match"],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"] == []

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "update",
            name,
            "--host-url",
            "https://other.example.com/v1",
            "--description",
            "changed",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    updated = json.loads(result.stdout)
    assert updated["host_url"] == "https://other.example.com/v1"
    assert updated["description"] == "changed"

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "update-status",
            name,
            "--status",
            "READY",
            "--status-message",
            "up",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["status"] == "READY"

    result = runner.invoke(app, ["inference", "get-url", "--provider", name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert result.stdout.strip().endswith(f"/apis/inference-gateway/v2/workspaces/{ws}/provider/{name}/-")

    result = runner.invoke(app, ["inference", "providers", "delete", name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["inference", "providers", "get", name, "--workspace", ws])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr


def test_providers_create_exist_ok_returns_existing(runner, random_workspace: str) -> None:
    ws = random_workspace
    name = _name("prov")
    args = ["inference", "providers", "create", name, "--host-url", "https://h.example.com/v1", "--workspace", ws]

    assert_exit_code(runner.invoke(app, args), 0)

    result = runner.invoke(app, args)
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr

    result = runner.invoke(app, [*args, "--exist-ok"])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["name"] == name


def test_providers_list_all_pages(runner, random_workspace: str) -> None:
    ws = random_workspace
    names = sorted(_name("prov") for _ in range(3))
    for name in names:
        assert_exit_code(
            runner.invoke(
                app,
                ["inference", "providers", "create", name, "--host-url", "https://h.example.com/v1", "--workspace", ws],
            ),
            0,
        )

    result = runner.invoke(app, ["inference", "providers", "list", "--workspace", ws, "--page-size", "2"])
    assert_exit_code(result, 0)
    first_page = json.loads(result.stdout)
    assert len(first_page["data"]) == 2
    assert first_page["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr

    result = runner.invoke(
        app, ["inference", "providers", "list", "--workspace", ws, "--page-size", "2", "--all-pages", "--sort", "name"]
    )
    assert_exit_code(result, 0)
    all_pages = json.loads(result.stdout)
    assert [item["name"] for item in all_pages["data"]] == names
    assert all_pages["pagination"]["total_results"] == 3
    assert "More pages" not in result.stderr


def test_deployment_configs_and_deployments_lifecycle(runner, random_workspace: str) -> None:
    ws = random_workspace
    config_name = _name("cfg")
    deployment_name = _name("dep")

    result = runner.invoke(
        app,
        [
            "inference",
            "deployment-configs",
            "create",
            config_name,
            "--engine",
            "vllm",
            "--model-spec",
            '{"model_name": "meta/llama-3.2-1b-instruct"}',
            "--executor-config",
            '{"gpu": 1}',
            "--description",
            "demo config",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    config = json.loads(result.stdout)
    assert config["name"] == config_name
    assert config["entity_version"] == 1
    assert config["engine"] == "vllm"
    assert config["executor_config"]["gpu"] == 1

    result = runner.invoke(app, ["inference", "deployment-configs", "get", config_name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "demo config"

    result = runner.invoke(app, ["inference", "deployment-configs", "list", "--workspace", ws])
    assert_exit_code(result, 0)
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == [config_name]

    # Updating creates a new immutable version.
    result = runner.invoke(
        app,
        [
            "inference",
            "deployment-configs",
            "update",
            config_name,
            "--engine",
            "vllm",
            "--model-spec",
            '{"model_name": "meta/llama-3.2-1b-instruct"}',
            "--executor-config",
            '{"gpu": 2}',
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["entity_version"] == 2

    result = runner.invoke(app, ["inference", "deployment-configs", "versions", "list", config_name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert sorted(item["entity_version"] for item in json.loads(result.stdout)) == [1, 2]

    result = runner.invoke(
        app,
        ["inference", "deployment-configs", "versions", "get", "1", "--config", config_name, "--workspace", ws],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["executor_config"]["gpu"] == 1

    # Deployments referencing the config.
    result = runner.invoke(
        app,
        ["inference", "deployments", "create", deployment_name, "--config", config_name, "--workspace", ws],
    )
    assert_exit_code(result, 0)
    deployment = json.loads(result.stdout)
    assert deployment["name"] == deployment_name
    assert deployment["config"] == config_name
    assert deployment["entity_version"] == 1

    result = runner.invoke(app, ["inference", "deployments", "get", deployment_name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["config_version"] == 2

    result = runner.invoke(app, ["inference", "deployments", "list", "--workspace", ws])
    assert_exit_code(result, 0)
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == [deployment_name]

    result = runner.invoke(
        app,
        [
            "inference",
            "deployments",
            "update",
            deployment_name,
            "--config",
            config_name,
            "--config-version",
            "1",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    updated = json.loads(result.stdout)
    assert updated["entity_version"] == 2
    assert updated["config_version"] == 1

    result = runner.invoke(app, ["inference", "deployments", "list", "--workspace", ws, "--all-versions"])
    assert_exit_code(result, 0)
    assert sorted(item["entity_version"] for item in json.loads(result.stdout)["data"]) == [1, 2]

    result = runner.invoke(app, ["inference", "deployments", "versions", "list", deployment_name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert sorted(item["entity_version"] for item in json.loads(result.stdout)) == [1, 2]

    result = runner.invoke(
        app,
        ["inference", "deployments", "versions", "get", "1", "--deployment", deployment_name, "--workspace", ws],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["entity_version"] == 1

    result = runner.invoke(
        app,
        [
            "inference",
            "deployments",
            "update-status",
            deployment_name,
            "--status",
            "READY",
            "--status-message",
            "serving",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    status = json.loads(result.stdout)
    assert status["status"] == "READY"
    assert status["status_history"][-1]["status_message"] == "serving"

    result = runner.invoke(app, ["inference", "deployments", "list-models", deployment_name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["name"] == deployment_name

    # Config deletion is blocked while a live deployment references it.
    result = runner.invoke(app, ["inference", "deployment-configs", "delete", config_name, "--workspace", ws])
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr

    result = runner.invoke(app, ["inference", "deployments", "delete", deployment_name, "--workspace", ws])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout


def test_deployments_create_requires_existing_config(runner, random_workspace: str) -> None:
    result = runner.invoke(
        app,
        [
            "inference",
            "deployments",
            "create",
            _name("dep"),
            "--config",
            "missing-config",
            "--workspace",
            random_workspace,
        ],
    )
    assert_exit_code(result, 3)


def test_prompts_lifecycle(runner, random_workspace: str) -> None:
    ws = random_workspace
    name = _name("prompt")

    result = runner.invoke(
        app,
        [
            "inference",
            "prompts",
            "create",
            name,
            "--messages",
            '[{"role": "system", "content": "Hello {{name}}"}]',
            "--input-variables",
            "name",
            "--tags",
            "demo",
            "--workspace",
            ws,
        ],
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["messages"][0]["role"] == "system"
    assert created["input_variables"] == ["name"]

    result = runner.invoke(app, ["inference", "prompts", "get", name, "--workspace", ws])
    assert_exit_code(result, 0)

    result = runner.invoke(app, ["inference", "prompts", "list", "--workspace", ws])
    assert_exit_code(result, 0)
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == [name]

    result = runner.invoke(app, ["inference", "prompts", "update", name, "--description", "changed", "--workspace", ws])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "changed"

    result = runner.invoke(app, ["inference", "prompts", "delete", name, "--workspace", ws])
    assert_exit_code(result, 0)

    result = runner.invoke(app, ["inference", "prompts", "get", name, "--workspace", ws])
    assert_exit_code(result, 3)


def test_inference_code_output_sends_nothing(runner, random_workspace: str) -> None:
    ws = random_workspace
    result = runner.invoke(
        app,
        ["inference", "providers", "create", "code-only", "--host-url", "https://h", "--workspace", ws, "-f", "code"],
    )
    assert_exit_code(result, 0)
    assert "from nemo_platform_plugin.models.client import ModelsClient" in result.stdout
    assert "client.create_provider(" in result.stdout
    assert "NeMoPlatform" not in result.stdout

    result = runner.invoke(app, ["inference", "providers", "list", "--workspace", ws])
    assert json.loads(result.stdout)["data"] == []


def test_get_url_base(runner, random_workspace: str) -> None:
    result = runner.invoke(app, ["inference", "get-url", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert result.stdout.strip().endswith(f"/apis/inference-gateway/v2/workspaces/{random_workspace}/openai/-/v1")
