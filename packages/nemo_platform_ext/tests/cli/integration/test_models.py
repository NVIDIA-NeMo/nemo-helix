# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo models`` and ``nemo adapters`` against the in-process Models service."""

import json
import uuid

import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest

from ..utils import assert_exit_code


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def adapter_fileset(files_client: FilesClient, random_workspace: str) -> str:
    """Create a non-empty fileset the Models service accepts when validating adapter requests."""
    fileset = files_client.create_fileset(
        body=CreateFilesetRequest(name=_name("fs")), workspace=random_workspace
    ).data()
    files_client.upload_file(workspace=random_workspace, name=fileset.name, path="adapter.bin", content=b"weights")
    return f"{random_workspace}/{fileset.name}"


def test_models_lifecycle(runner, random_workspace: str) -> None:
    name = _name("model")

    result = runner.invoke(
        app,
        [
            "models",
            "create",
            name,
            "--description",
            "demo",
            "--custom-fields",
            '{"team": "nemo"}',
            "--workspace",
            random_workspace,
        ],
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["workspace"] == random_workspace
    assert created["description"] == "demo"
    assert created["custom_fields"] == {"team": "nemo"}

    result = runner.invoke(app, ["models", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["name"] == name

    result = runner.invoke(app, ["models", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = runner.invoke(app, ["models", "update", name, "--description", "changed", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "changed"

    result = runner.invoke(app, ["models", "delete", name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["models", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr


def test_models_create_from_stdin(runner, random_workspace: str) -> None:
    name = _name("model")
    result = runner.invoke(
        app,
        ["models", "create", "--input-file", "-", "--workspace", random_workspace],
        input=json.dumps({"name": name, "description": "piped"}),
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "piped"

    result = runner.invoke(app, ["models", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "piped"


def test_models_create_exist_ok(runner, random_workspace: str) -> None:
    name = _name("model")
    assert_exit_code(runner.invoke(app, ["models", "create", name, "--workspace", random_workspace]), 0)

    result = runner.invoke(app, ["models", "create", name, "--workspace", random_workspace])
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr

    result = runner.invoke(app, ["models", "create", name, "--exist-ok", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["name"] == name


def test_models_list_filter_and_all_pages(runner, random_workspace: str) -> None:
    names = sorted(_name("model") for _ in range(3))
    for name in names:
        assert_exit_code(runner.invoke(app, ["models", "create", name, "--workspace", random_workspace]), 0)

    result = runner.invoke(app, ["models", "list", "--workspace", random_workspace, "--page-size", "2"])
    assert_exit_code(result, 0)
    first_page = json.loads(result.stdout)
    assert len(first_page["data"]) == 2
    assert first_page["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr

    result = runner.invoke(app, ["models", "list", "--workspace", random_workspace, "--page-size", "2", "--all-pages"])
    assert_exit_code(result, 0)
    all_pages = json.loads(result.stdout)
    assert sorted(item["name"] for item in all_pages["data"]) == names
    assert all_pages["pagination"]["total_results"] == 3
    assert "More pages" not in result.stderr

    result = runner.invoke(app, ["models", "list", "--workspace", random_workspace, "--filter.name", names[0]])
    assert_exit_code(result, 0)
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == [names[0]]


def test_models_adapters_nested_lifecycle(runner, random_workspace: str, adapter_fileset: str) -> None:
    model_name = _name("model")
    adapter_name = _name("adapter")
    assert_exit_code(runner.invoke(app, ["models", "create", model_name, "--workspace", random_workspace]), 0)

    result = runner.invoke(
        app,
        [
            "models",
            "adapters",
            "create",
            model_name,
            adapter_name,
            "--fileset",
            adapter_fileset,
            "--finetuning-type",
            "lora",
            "--workspace",
            random_workspace,
        ],
    )
    assert_exit_code(result, 0)
    adapter = json.loads(result.stdout)
    assert adapter["name"] == adapter_name
    assert adapter["fileset"] == adapter_fileset
    assert adapter["finetuning_type"] == "lora"

    result = runner.invoke(app, ["models", "get", model_name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert [a["name"] for a in json.loads(result.stdout)["adapters"]] == [adapter_name]

    result = runner.invoke(
        app,
        [
            "models",
            "adapters",
            "update",
            adapter_name,
            "--model-name",
            model_name,
            "--description",
            "tuned",
            "--workspace",
            random_workspace,
        ],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "tuned"

    result = runner.invoke(
        app,
        ["models", "adapters", "delete", adapter_name, "--model-name", model_name, "--workspace", random_workspace],
    )
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["adapters", "get", adapter_name, "--workspace", random_workspace])
    assert_exit_code(result, 3)


def test_adapters_top_level_lifecycle(runner, random_workspace: str, adapter_fileset: str) -> None:
    model_name = _name("model")
    adapter_name = _name("adapter")
    assert_exit_code(runner.invoke(app, ["models", "create", model_name, "--workspace", random_workspace]), 0)

    result = runner.invoke(
        app,
        [
            "adapters",
            "create",
            adapter_name,
            "--fileset",
            adapter_fileset,
            "--finetuning-type",
            "lora",
            "--model",
            model_name,
            "--workspace",
            random_workspace,
        ],
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == adapter_name
    assert created["model"] == f"{random_workspace}/{model_name}"

    result = runner.invoke(app, ["adapters", "get", adapter_name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["name"] == adapter_name

    result = runner.invoke(app, ["adapters", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [adapter_name]

    result = runner.invoke(
        app, ["adapters", "patch", adapter_name, "--description", "patched", "--workspace", random_workspace]
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "patched"

    result = runner.invoke(app, ["adapters", "delete", adapter_name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["adapters", "get", adapter_name, "--workspace", random_workspace])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr


def test_models_code_output_sends_nothing(runner, random_workspace: str) -> None:
    result = runner.invoke(
        app, ["models", "create", "code-model", "--description", "d", "--workspace", random_workspace, "-f", "code"]
    )
    assert_exit_code(result, 0)
    assert "from nemo_platform_plugin.models.client import ModelsClient" in result.stdout
    assert "client.create_model(" in result.stdout
    assert "NeMoPlatform" not in result.stdout

    result = runner.invoke(app, ["models", "list", "--workspace", random_workspace])
    assert json.loads(result.stdout)["data"] == []


def test_models_missing_workspace_is_usage_error(runner) -> None:
    # The injected client has no default workspace and none was passed.
    result = runner.invoke(app, ["models", "get", "x"], obj=None)
    assert result.exit_code == 2
