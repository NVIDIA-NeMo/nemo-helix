# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo inference`` against a scripted typed client.

Each test drives the real Typer command with a ``NemoClient`` whose transport
records the HTTP requests, so the assertions pin the exact method, path, query
string, and JSON body every sub-command sends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import patch

import httpx
import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from typer.testing import CliRunner

MODELS = "/apis/models/v2/workspaces"
GATEWAY = "/apis/inference-gateway/v2/workspaces"

TIMESTAMPS = {"created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}
PROVIDER = {
    "id": "p1",
    "name": "nvidia",
    "workspace": "default",
    "host_url": "https://integrate.api.nvidia.com/v1",
    "description": "NVIDIA",
    "status": "READY",
    **TIMESTAMPS,
}
DEPLOYMENT = {
    "id": "d1",
    "name": "dep-a",
    "workspace": "default",
    "entity_version": 1,
    "config": "cfg",
    "config_version": 1,
    "status": "CREATED",
    **TIMESTAMPS,
}
DEPLOYMENT_CONFIG = {
    "id": "c1",
    "name": "cfg",
    "workspace": "default",
    "entity_version": 1,
    "engine": "vllm",
    "model_spec": {"model_name": "meta/llama"},
    "executor_config": {"gpu": 1},
    "description": "cfg",
    **TIMESTAMPS,
}
PROMPT = {"id": "pr1", "name": "greet", "workspace": "default", "description": "hi", **TIMESTAMPS}
VIRTUAL_MODEL = {"id": "vm1", "name": "router", "workspace": "default", **TIMESTAMPS}
MODEL_ENTITY = {"id": "m1", "name": "llama", "workspace": "default", **TIMESTAMPS}
OPENAI_MODEL = {"id": "default/router", "object": "model", "created": 1, "owned_by": "nemo"}


@dataclass
class Recorder:
    """Records requests and answers each with the next scripted response."""

    responses: list[httpx.Response]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def make_runner(recorder: Recorder, *, workspace: str | None = "default") -> tuple[CliRunner, CLIContext]:
    client = NemoClient(
        base_url="http://test",
        workspace=workspace,
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )
    state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=client)
    return CliRunner(), state


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def _page(items: list[dict], page: int, total_pages: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": items,
            "pagination": {
                "page": page,
                "page_size": 1,
                "current_page_size": len(items),
                "total_pages": total_pages,
                "total_results": total_pages,
            },
        },
    )


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


def test_providers_get() -> None:
    recorder = Recorder([httpx.Response(200, json=PROVIDER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "get", "nvidia"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{MODELS}/default/providers/nvidia"
    assert json.loads(result.stdout)["name"] == "nvidia"


def test_providers_get_explicit_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json={**PROVIDER, "workspace": "other"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "get", "nvidia", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/other/providers/nvidia"


def test_providers_get_without_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["inference", "providers", "get", "nvidia"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_providers_get_not_found_maps_to_remote_error() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Provider 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Provider 'nope' not found" in result.stderr
    assert "nemo inference providers list" in result.stderr


def test_providers_create_sends_only_provided_fields() -> None:
    recorder = Recorder([httpx.Response(201, json=PROVIDER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "create",
            "nvidia",
            "--host-url",
            "https://integrate.api.nvidia.com/v1",
            "--api-key-secret-name",
            "nvidia-key",
            "--enabled-models",
            "a",
            "--enabled-models",
            "b",
            "--default-extra-headers",
            '{"X-Test": "1"}',
            "--status",
            "READY",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{MODELS}/default/providers"
    assert _body(recorder.last) == {
        "name": "nvidia",
        "host_url": "https://integrate.api.nvidia.com/v1",
        "api_key_secret_name": "nvidia-key",
        "enabled_models": ["a", "b"],
        "default_extra_headers": {"X-Test": "1"},
        "status": "READY",
    }
    assert json.loads(result.stdout)["name"] == "nvidia"


def test_providers_create_from_input_data_with_workspace_key() -> None:
    recorder = Recorder([httpx.Response(201, json=PROVIDER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "create",
            "--input-data",
            '{"name": "nvidia", "host_url": "https://h", "workspace": "other", "description": "d"}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/other/providers"
    assert _body(recorder.last) == {"name": "nvidia", "host_url": "https://h", "description": "d"}


def test_providers_create_requires_host_url() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "create", "nvidia"], obj=state)

    assert result.exit_code == 2
    assert "--host-url" in result.stderr
    assert recorder.requests == []


def test_providers_create_exist_ok_returns_existing_on_conflict() -> None:
    recorder = Recorder(
        [httpx.Response(409, json={"detail": "exists"}), httpx.Response(200, json=PROVIDER)],
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["inference", "providers", "create", "nvidia", "--host-url", "https://h", "--exist-ok"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [
        ("POST", f"{MODELS}/default/providers"),
        ("GET", f"{MODELS}/default/providers/nvidia"),
    ]
    assert json.loads(result.stdout)["name"] == "nvidia"


def test_providers_create_conflict_without_exist_ok_fails() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "create", "nvidia", "--host-url", "https://h"], obj=state)

    assert result.exit_code == 3
    assert "Conflict" in result.stderr


def test_providers_update_is_put_upsert() -> None:
    recorder = Recorder([httpx.Response(200, json=PROVIDER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "providers", "update", "nvidia", "--host-url", "https://h2", "--description", "new"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PUT"
    assert recorder.last.url.path == f"{MODELS}/default/providers/nvidia"
    assert _body(recorder.last) == {"host_url": "https://h2", "description": "new"}


def test_providers_update_requires_host_url() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "update", "nvidia", "--description", "x"], obj=state)

    assert result.exit_code == 2
    assert "--host-url" in result.stderr


def test_providers_update_status() -> None:
    recorder = Recorder([httpx.Response(200, json=PROVIDER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "update-status",
            "nvidia",
            "--status",
            "READY",
            "--status-message",
            "up",
            "--served-models",
            '[{"model_entity_id": "default/llama", "served_model_name": "llama"}]',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PUT"
    assert recorder.last.url.path == f"{MODELS}/default/providers/nvidia/status"
    assert _body(recorder.last) == {
        "status": "READY",
        "status_message": "up",
        "served_models": [{"model_entity_id": "default/llama", "served_model_name": "llama"}],
    }


def test_providers_delete() -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "delete", "nvidia"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{MODELS}/default/providers/nvidia"
    assert "Deleted successfully" in result.stdout


def test_providers_list_query_params_and_more_pages_warning() -> None:
    recorder = Recorder([_page([PROVIDER], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "providers",
            "list",
            "--page",
            "1",
            "--page-size",
            "1",
            "--sort",
            "-created_at",
            "--filter.status",
            "READY",
            "--filter.name",
            "nv",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/default/providers"
    params = dict(recorder.last.url.params)
    assert params["page"] == "1"
    assert params["page_size"] == "1"
    assert params["sort"] == "-created_at"
    assert json.loads(params["filter"]) == {"status": "READY", "name": "nv"}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["nvidia"]
    assert body["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_providers_list_text_filter_passes_through() -> None:
    recorder = Recorder([_page([PROVIDER], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "list", "--filter", 'name~"nv"'], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"filter": 'name~"nv"'}


def test_providers_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([PROVIDER], 1, 2), _page([{**PROVIDER, "name": "second"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["nvidia", "second"]
    assert body["pagination"]["total_results"] == 2
    assert "More pages" not in result.stderr


def test_providers_list_table_uses_default_columns() -> None:
    recorder = Recorder([_page([PROVIDER], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "description" in output and "created_at" in output
    assert "host_url" not in output


def test_providers_create_code_output_sends_nothing() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "providers", "create", "nvidia", "--host-url", "https://h", "--exist-ok", "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.models.client import ModelsClient" in result.stdout
    assert "from nemo_platform_plugin.models.types import CreateModelProviderRequest" in result.stdout
    assert 'client = ModelsClient(base_url="http://test/")' in result.stdout
    assert "client.create_provider(" in result.stdout
    assert 'CreateModelProviderRequest(name="nvidia", host_url="https://h")' in result.stdout
    assert "exist_ok=True" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_providers_list_code_output() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "providers", "list", "--page", "2", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'client.list_providers(query_params={"page": 2})' in result.stdout
    assert "for item in response.page().items:" in result.stdout


# ---------------------------------------------------------------------------
# deployments
# ---------------------------------------------------------------------------


def test_deployments_create() -> None:
    recorder = Recorder([httpx.Response(201, json=DEPLOYMENT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "deployments", "create", "dep-a", "--config", "cfg", "--config-version", "2", "--project", "p"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{MODELS}/default/deployments"
    assert _body(recorder.last) == {"name": "dep-a", "config": "cfg", "config_version": 2, "project": "p"}
    assert json.loads(result.stdout)["name"] == "dep-a"


def test_deployments_create_requires_config() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "create", "dep-a"], obj=state)

    assert result.exit_code == 2
    assert "--config" in result.stderr
    assert recorder.requests == []


def test_deployments_create_rejects_wait_and_watch() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["inference", "deployments", "create", "dep-a", "--config", "cfg", "--wait", "--watch"], obj=state
    )

    assert result.exit_code == 2
    assert "Cannot combine --wait and --watch" in result.stderr
    assert recorder.requests == []


def test_deployments_create_wait_calls_waiter_with_client() -> None:
    recorder = Recorder([httpx.Response(201, json=DEPLOYMENT)])
    runner, state = make_runner(recorder)

    with patch(
        "nemo_platform_ext.cli.commands.inference.deployments.wait_for_inference_deployment", return_value=True
    ) as waiter:
        result = runner.invoke(
            app,
            [
                "inference",
                "deployments",
                "create",
                "dep-a",
                "--config",
                "cfg",
                "--workspace",
                "ws",
                "--wait",
                "--timeout",
                "90",
                "--poll-interval",
                "10",
            ],
            obj=state,
        )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/ws/deployments"
    waiter.assert_called_once_with(state.get_client(), "dep-a", workspace="ws", timeout=90, poll_interval=10)


def test_deployments_create_wait_failure_exits_1() -> None:
    recorder = Recorder([httpx.Response(201, json=DEPLOYMENT)])
    runner, state = make_runner(recorder)

    with patch(
        "nemo_platform_ext.cli.commands.inference.deployments.wait_for_inference_deployment", return_value=False
    ):
        result = runner.invoke(
            app, ["inference", "deployments", "create", "dep-a", "--config", "cfg", "--watch"], obj=state
        )

    assert result.exit_code == 1


def test_deployments_create_wait_code_output_renders_lifecycle() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["inference", "deployments", "create", "dep-a", "--config", "cfg", "--wait", "-f", "code"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'CreateModelDeploymentRequest(name="dep-a", config="cfg")' in result.stdout
    assert "deadline = time.monotonic() + 1200" in result.stdout
    assert "gateway.provider_ready(" in result.stdout
    assert "--wait" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_deployments_get_update_delete() -> None:
    recorder = Recorder(
        [httpx.Response(200, json=DEPLOYMENT), httpx.Response(200, json=DEPLOYMENT), httpx.Response(202)]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "get", "dep-a"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{MODELS}/default/deployments/dep-a")

    result = runner.invoke(
        app, ["inference", "deployments", "update", "dep-a", "--config", "cfg2", "--config-version", "3"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("POST", f"{MODELS}/default/deployments/dep-a")
    assert _body(recorder.last) == {"config": "cfg2", "config_version": 3}

    result = runner.invoke(app, ["inference", "deployments", "delete", "dep-a"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("DELETE", f"{MODELS}/default/deployments/dep-a")
    assert "Deleted successfully" in result.stdout


def test_deployments_update_status_with_version_query() -> None:
    recorder = Recorder([httpx.Response(200, json=DEPLOYMENT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "deployments",
            "update-status",
            "dep-a",
            "--status",
            "READY",
            "--version",
            "2",
            "--model-provider-id",
            "default/dep-a",
            "--status-message",
            "ok",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{MODELS}/default/deployments/dep-a/status"
    assert dict(recorder.last.url.params) == {"version": "2"}
    assert _body(recorder.last) == {"status": "READY", "model_provider_id": "default/dep-a", "status_message": "ok"}


def test_deployments_update_status_requires_status() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "update-status", "dep-a"], obj=state)

    assert result.exit_code == 2
    assert "--status" in result.stderr


def test_deployments_list_with_all_versions_and_filters() -> None:
    recorder = Recorder([_page([DEPLOYMENT], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "deployments", "list", "--all-versions", "--filter.config", "cfg", "--page-size", "5"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/default/deployments"
    params = dict(recorder.last.url.params)
    assert params["all_versions"] == "true"
    assert params["page_size"] == "5"
    assert json.loads(params["filter"]) == {"config": "cfg"}
    assert "More pages" not in result.stderr


def test_deployments_list_all_pages() -> None:
    recorder = Recorder([_page([DEPLOYMENT], 1, 2), _page([{**DEPLOYMENT, "name": "dep-b"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "list", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == ["dep-a", "dep-b"]


def test_deployments_list_table_default_columns() -> None:
    recorder = Recorder([_page([DEPLOYMENT], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "status" in output and "created_at" in output
    assert "config_version" not in output


def test_deployments_list_models() -> None:
    recorder = Recorder([httpx.Response(200, json={"workspace": "default", "name": "dep-a", "models": []})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "list-models", "dep-a"], obj=state)

    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{MODELS}/default/deployments/dep-a/models")
    assert json.loads(result.stdout) == {"workspace": "default", "name": "dep-a", "models": []}


def test_deployments_versions() -> None:
    recorder = Recorder(
        [
            httpx.Response(200, json=[DEPLOYMENT, {**DEPLOYMENT, "entity_version": 2}]),
            httpx.Response(200, json={**DEPLOYMENT, "entity_version": 2}),
            httpx.Response(204),
        ]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "versions", "list", "dep-a"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{MODELS}/default/deployments/dep-a/versions")
    assert [item["entity_version"] for item in json.loads(result.stdout)] == [1, 2]

    result = runner.invoke(
        app, ["inference", "deployments", "versions", "get", "2", "--deployment", "dep-a"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{MODELS}/default/deployments/dep-a/versions/2")
    assert json.loads(result.stdout)["entity_version"] == 2

    result = runner.invoke(
        app, ["inference", "deployments", "versions", "delete", "2", "--deployment", "dep-a"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == (
        "DELETE",
        f"{MODELS}/default/deployments/dep-a/versions/2",
    )
    assert "Deleted successfully" in result.stdout


def test_deployments_versions_get_requires_deployment_option() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployments", "versions", "get", "2"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# deployment-configs
# ---------------------------------------------------------------------------


def test_deployment_configs_create() -> None:
    recorder = Recorder([httpx.Response(201, json=DEPLOYMENT_CONFIG)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "deployment-configs",
            "create",
            "cfg",
            "--engine",
            "vllm",
            "--model-spec",
            '{"model_name": "meta/llama"}',
            "--executor-config",
            '{"gpu": 1}',
            "--description",
            "cfg",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("POST", f"{MODELS}/default/deployment-configs")
    assert _body(recorder.last) == {
        "name": "cfg",
        "engine": "vllm",
        "model_spec": {"model_name": "meta/llama"},
        "executor_config": {"gpu": 1},
        "description": "cfg",
    }


def test_deployment_configs_create_requires_fields() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployment-configs", "create", "cfg", "--engine", "vllm"], obj=state)

    assert result.exit_code == 2
    assert "--executor-config" in result.stderr and "--model-spec" in result.stderr


def test_deployment_configs_create_from_input_file(tmp_path) -> None:
    payload = tmp_path / "cfg.json"
    payload.write_text(
        json.dumps({"engine": "nim", "model_spec": {"model_name": "m"}, "executor_config": {"gpu": 2}, "name": "cfg"})
    )
    recorder = Recorder([httpx.Response(201, json=DEPLOYMENT_CONFIG)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "deployment-configs", "create", "--input-file", str(payload), "--description", "override"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert _body(recorder.last) == {
        "engine": "nim",
        "model_spec": {"model_name": "m"},
        "executor_config": {"gpu": 2},
        "name": "cfg",
        "description": "override",
    }


def test_deployment_configs_get_update_delete() -> None:
    recorder = Recorder(
        [httpx.Response(200, json=DEPLOYMENT_CONFIG), httpx.Response(200, json=DEPLOYMENT_CONFIG), httpx.Response(204)]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployment-configs", "get", "cfg"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{MODELS}/default/deployment-configs/cfg")

    result = runner.invoke(
        app,
        [
            "inference",
            "deployment-configs",
            "update",
            "cfg",
            "--engine",
            "generic",
            "--model-spec",
            "{}",
            "--executor-config",
            '{"gpu": 0}',
        ],
        obj=state,
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("POST", f"{MODELS}/default/deployment-configs/cfg")
    assert _body(recorder.last) == {"engine": "generic", "model_spec": {}, "executor_config": {"gpu": 0}}

    result = runner.invoke(app, ["inference", "deployment-configs", "delete", "cfg"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("DELETE", f"{MODELS}/default/deployment-configs/cfg")
    assert "Deleted successfully" in result.stdout


def test_deployment_configs_list_and_all_pages() -> None:
    recorder = Recorder([_page([DEPLOYMENT_CONFIG], 1, 2), _page([{**DEPLOYMENT_CONFIG, "name": "cfg2"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["inference", "deployment-configs", "list", "--filter.name", "cfg", "--all-pages"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests[0].url.path == f"{MODELS}/default/deployment-configs"
    assert json.loads(dict(recorder.requests[0].url.params)["filter"]) == {"name": "cfg"}
    assert dict(recorder.requests[1].url.params)["page"] == "2"
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == ["cfg", "cfg2"]


def test_deployment_configs_list_single_page_warns() -> None:
    recorder = Recorder([_page([DEPLOYMENT_CONFIG], 1, 3)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployment-configs", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    assert "More pages" in result.stderr
    output = result.stdout.lower()
    assert "name" in output and "description" in output


def test_deployment_configs_versions() -> None:
    recorder = Recorder(
        [
            httpx.Response(200, json=[DEPLOYMENT_CONFIG]),
            httpx.Response(200, json=DEPLOYMENT_CONFIG),
            httpx.Response(204),
        ]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "deployment-configs", "versions", "list", "cfg"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == (
        "GET",
        f"{MODELS}/default/deployment-configs/cfg/versions",
    )
    assert [item["name"] for item in json.loads(result.stdout)] == ["cfg"]

    result = runner.invoke(
        app, ["inference", "deployment-configs", "versions", "get", "1", "--config", "cfg"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == (
        "GET",
        f"{MODELS}/default/deployment-configs/cfg/versions/1",
    )

    result = runner.invoke(
        app, ["inference", "deployment-configs", "versions", "delete", "1", "--config", "cfg"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == (
        "DELETE",
        f"{MODELS}/default/deployment-configs/cfg/versions/1",
    )
    assert "Deleted successfully" in result.stdout


def test_deployment_configs_code_output() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "deployment-configs",
            "create",
            "cfg",
            "--engine",
            "vllm",
            "--model-spec",
            "{}",
            "--executor-config",
            '{"gpu": 1}',
            "-f",
            "code",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.create_deployment_config(" in result.stdout
    assert "Engine.VLLM" in result.stdout
    assert "ContainerExecutorConfig(gpu=1)" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# virtual-models
# ---------------------------------------------------------------------------


def test_virtual_models_create() -> None:
    recorder = Recorder([httpx.Response(201, json=VIRTUAL_MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "virtual-models",
            "create",
            "router",
            "--default-model-entity",
            "default/llama",
            "--request-middleware",
            '[{"name": "guardrails", "config_type": "guardrail_config", "config_id": "default/g"}]',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("POST", f"{GATEWAY}/default/virtual-models")
    assert _body(recorder.last) == {
        "name": "router",
        "default_model_entity": "default/llama",
        "request_middleware": [{"name": "guardrails", "config_type": "guardrail_config", "config_id": "default/g"}],
    }


def test_virtual_models_create_exist_ok_replays_get_on_conflict() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"}), httpx.Response(200, json=VIRTUAL_MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "virtual-models", "create", "router", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [
        ("POST", f"{GATEWAY}/default/virtual-models"),
        ("GET", f"{GATEWAY}/default/virtual-models/router"),
    ]


def test_virtual_models_create_requires_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "virtual-models", "create"], obj=state)

    assert result.exit_code == 2
    assert "--name" in result.stderr


def test_virtual_models_get_patch_delete() -> None:
    recorder = Recorder(
        [httpx.Response(200, json=VIRTUAL_MODEL), httpx.Response(200, json=VIRTUAL_MODEL), httpx.Response(204)]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "virtual-models", "get", "router"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{GATEWAY}/default/virtual-models/router")

    result = runner.invoke(
        app, ["inference", "virtual-models", "patch", "router", "--override-proxy", "plug.proxy"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("PATCH", f"{GATEWAY}/default/virtual-models/router")
    assert _body(recorder.last) == {"override_proxy": "plug.proxy"}

    result = runner.invoke(
        app, ["inference", "virtual-models", "delete", "router", "--expected-db-version", "4"], obj=state
    )
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("DELETE", f"{GATEWAY}/default/virtual-models/router")
    assert dict(recorder.last.url.params) == {"expected_db_version": "4"}
    assert "Deleted successfully" in result.stdout


def test_virtual_models_delete_without_version_has_no_query() -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "virtual-models", "delete", "router"], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {}


def test_virtual_models_list() -> None:
    recorder = Recorder([_page([VIRTUAL_MODEL], 1, 2), _page([{**VIRTUAL_MODEL, "name": "other"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "virtual-models", "list", "--exclude-autoprovisioned", "--filter.name", "r", "--all-pages"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    first = dict(recorder.requests[0].url.params)
    assert recorder.requests[0].url.path == f"{GATEWAY}/default/virtual-models"
    assert first["exclude_autoprovisioned"] == "true"
    assert json.loads(first["filter"]) == {"name": "r"}
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == ["router", "other"]


def test_virtual_models_code_output() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "virtual-models", "list", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert "from nemo_platform_plugin.virtual_models.client import VirtualModelsClient" in result.stdout
    assert "client.list_virtual_models()" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_prompts_create() -> None:
    recorder = Recorder([httpx.Response(201, json=PROMPT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "prompts",
            "create",
            "greet",
            "--messages",
            '[{"role": "system", "content": "Hi {{name}}"}]',
            "--input-variables",
            "name",
            "--tags",
            "a",
            "--tags",
            "b",
            "--inference-params",
            '{"temperature": 0.2}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("POST", f"{MODELS}/default/prompts")
    assert _body(recorder.last) == {
        "name": "greet",
        "messages": [{"role": "system", "content": "Hi {{name}}"}],
        "input_variables": ["name"],
        "tags": ["a", "b"],
        "inference_params": {"temperature": 0.2},
    }


def test_prompts_create_exist_ok_on_conflict() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"}), httpx.Response(200, json=PROMPT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "prompts", "create", "greet", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert [r.method for r in recorder.requests] == ["POST", "GET"]
    assert recorder.last.url.path == f"{MODELS}/default/prompts/greet"


def test_prompts_get_update_delete() -> None:
    recorder = Recorder([httpx.Response(200, json=PROMPT), httpx.Response(200, json=PROMPT), httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "prompts", "get", "greet"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{MODELS}/default/prompts/greet")

    result = runner.invoke(app, ["inference", "prompts", "update", "greet", "--description", "new"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("PUT", f"{MODELS}/default/prompts/greet")
    assert _body(recorder.last) == {"description": "new"}

    result = runner.invoke(app, ["inference", "prompts", "delete", "greet"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("DELETE", f"{MODELS}/default/prompts/greet")
    assert "Deleted successfully" in result.stdout


def test_prompts_list() -> None:
    recorder = Recorder([_page([PROMPT], 1, 2), _page([{**PROMPT, "name": "bye"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["inference", "prompts", "list", "--sort", "name", "--filter.project", "p", "--all-pages"], obj=state
    )

    assert result.exit_code == 0, result.output
    first = dict(recorder.requests[0].url.params)
    assert recorder.requests[0].url.path == f"{MODELS}/default/prompts"
    assert first["sort"] == "name"
    assert json.loads(first["filter"]) == {"project": "p"}
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == ["greet", "bye"]


def test_prompts_list_warns_on_more_pages() -> None:
    recorder = Recorder([_page([PROMPT], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "prompts", "list"], obj=state)

    assert result.exit_code == 0, result.output
    assert "More pages" in result.stderr


def test_prompts_code_output() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["inference", "prompts", "create", "greet", "--messages", '[{"role":"user","content":"x"}]', "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.create_prompt(" in result.stdout
    assert "PromptMessage(role=PromptMessageRole.USER" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# gateway
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "path_segment"),
    [("provider", "provider"), ("model", "model")],
)
def test_gateway_get_and_delete(group: str, path_segment: str) -> None:
    recorder = Recorder([httpx.Response(200, json={"object": "list", "data": []}), httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "gateway", group, "get", "v1/models", "--name", "nvidia"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == (
        "GET",
        f"{GATEWAY}/default/{path_segment}/nvidia/-/v1/models",
    )
    assert json.loads(result.stdout) == {"object": "list", "data": []}

    result = runner.invoke(app, ["inference", "gateway", group, "delete", "v1/files/f1", "--name", "nvidia"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == (
        "DELETE",
        f"{GATEWAY}/default/{path_segment}/nvidia/-/v1/files/f1",
    )
    assert "Deleted successfully" in result.stdout


@pytest.mark.parametrize("group", ["provider", "model"])
def test_gateway_get_requires_name_option(group: str) -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "gateway", group, "get", "v1/models"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


@pytest.mark.parametrize(
    ("group", "verb"), [("provider", "post"), ("provider", "put"), ("model", "post"), ("model", "put")]
)
def test_gateway_post_put_positional_name_and_json_body(group: str, verb: str) -> None:
    recorder = Recorder([httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "gateway",
            group,
            verb,
            "v1/chat/completions",
            "nvidia",
            "--body",
            '{"model": "x", "messages": [{"role": "user", "content": "hi"}]}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == verb.upper()
    assert recorder.last.url.path == f"{GATEWAY}/default/{group}/nvidia/-/v1/chat/completions"
    assert recorder.last.headers["content-type"] == "application/json"
    assert _body(recorder.last) == {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert json.loads(result.stdout)["id"] == "chatcmpl-1"


@pytest.mark.parametrize("group", ["provider", "model"])
def test_gateway_patch_uses_name_option(group: str) -> None:
    recorder = Recorder([httpx.Response(200, json={"ok": True})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "gateway",
            group,
            "patch",
            "v1/thing",
            "--name",
            "nvidia",
            "--body",
            '{"a": 1}',
            "--workspace",
            "ws",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("PATCH", f"{GATEWAY}/ws/{group}/nvidia/-/v1/thing")
    assert _body(recorder.last) == {"a": 1}


def test_gateway_post_without_body_sends_empty_object() -> None:
    recorder = Recorder([httpx.Response(200, json={})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "gateway", "provider", "post", "v1/ping", "nvidia"], obj=state)

    assert result.exit_code == 0, result.output
    assert _body(recorder.last) == {}


def test_gateway_post_requires_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "gateway", "provider", "post", "v1/chat/completions"], obj=state)

    assert result.exit_code == 2
    assert "--name" in result.stderr
    assert recorder.requests == []


def test_gateway_post_body_from_input_data() -> None:
    recorder = Recorder([httpx.Response(200, json={})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "gateway",
            "model",
            "post",
            "v1/completions",
            "--input-data",
            '{"name": "llama", "body": {"prompt": "hi"}}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{GATEWAY}/default/model/llama/-/v1/completions"
    assert _body(recorder.last) == {"prompt": "hi"}


def test_gateway_post_code_output_renders_json_body() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "gateway",
            "provider",
            "post",
            "v1/chat/completions",
            "nvidia",
            "--body",
            '{"model": "x"}',
            "-f",
            "code",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient" in result.stdout
    assert 'body=JsonBody({"model": "x"})' in result.stdout
    assert "client.provider_post(" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_gateway_provider_ready() -> None:
    recorder = Recorder([httpx.Response(200, json={"workspace": "default", "name": "nvidia"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "gateway", "provider", "ready", "nvidia"], obj=state)

    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{GATEWAY}/default/provider/nvidia/ready")
    assert json.loads(result.stdout)["name"] == "nvidia"


def test_gateway_provider_ready_not_found() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "not registered"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "gateway", "provider", "ready", "nvidia"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) not registered" in result.stderr


def test_gateway_post_rejects_unknown_input_keys() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "inference",
            "gateway",
            "model",
            "post",
            "v1/completions",
            "--input-data",
            '{"name": "llama", "prompt": "hi"}',
        ],
        obj=state,
    )

    assert result.exit_code == 2
    assert "prompt" in result.stderr and "body" in result.stderr
    assert recorder.requests == []


@pytest.mark.parametrize("prefix", [["inference", "models"], ["inference", "gateway", "openai", "v1", "models"]])
def test_openai_models_list_table_uses_openai_columns(prefix: list[str]) -> None:
    recorder = Recorder([httpx.Response(200, json={"object": "list", "data": [OPENAI_MODEL]})])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, [*prefix, "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "id" in output and "owned_by" in output and "created" in output
    assert "default/router" in output


@pytest.mark.parametrize("prefix", [["inference", "models"], ["inference", "gateway", "openai", "v1", "models"]])
def test_openai_models_list_and_get(prefix: list[str]) -> None:
    recorder = Recorder(
        [
            httpx.Response(200, json={"object": "list", "data": [OPENAI_MODEL]}),
            httpx.Response(200, json=OPENAI_MODEL),
        ]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, [*prefix, "list"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{GATEWAY}/default/openai/-/v1/models")
    assert [item["id"] for item in json.loads(result.stdout)["data"]] == ["default/router"]

    result = runner.invoke(app, [*prefix, "get", "router", "--workspace", "ws"], obj=state)
    assert result.exit_code == 0, result.output
    assert (recorder.last.method, recorder.last.url.path) == ("GET", f"{GATEWAY}/ws/openai/-/v1/models/router")
    assert json.loads(result.stdout)["id"] == "default/router"


def test_models_list_table_uses_openai_columns() -> None:
    recorder = Recorder([httpx.Response(200, json={"object": "list", "data": [OPENAI_MODEL]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "models", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "owned_by" in output and "default/router" in output


def test_models_list_code_output() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "models", "list", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert "client.list_openai_models()" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# get-url
# ---------------------------------------------------------------------------


def test_get_url_default_is_workspace_openai_base() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "get-url"], obj=state)

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "http://test/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert recorder.requests == []


def test_get_url_explicit_workspace() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "get-url", "--workspace", "ws"], obj=state)

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "http://test/apis/inference-gateway/v2/workspaces/ws/openai/-/v1"


def test_get_url_without_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["inference", "get-url"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr


def test_get_url_provider_route_strips_v1() -> None:
    recorder = Recorder([httpx.Response(200, json=PROVIDER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "get-url", "--provider", "nvidia"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/default/providers/nvidia"
    assert result.stdout.strip() == "http://test/apis/inference-gateway/v2/workspaces/default/provider/nvidia/-"


def test_get_url_virtual_model_route() -> None:
    recorder = Recorder([httpx.Response(200, json=MODEL_ENTITY)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "get-url", "--virtual-model", "llama"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS}/default/models/llama"
    assert result.stdout.strip() == "http://test/apis/inference-gateway/v2/workspaces/default/model/llama/-"


def test_get_url_rejects_provider_and_virtual_model_together() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["inference", "get-url", "--provider", "a", "--virtual-model", "b"], obj=state)

    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    assert recorder.requests == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False), (None, None)],
)
def test_pop_exist_ok_parses_string_booleans(raw: object, expected: bool | None) -> None:
    from nemo_platform_ext.cli.commands.inference._common import pop_exist_ok

    payload = {"name": "x"} if raw is None else {"name": "x", "exist_ok": raw}

    assert pop_exist_ok(payload) is expected
    assert "exist_ok" not in payload
