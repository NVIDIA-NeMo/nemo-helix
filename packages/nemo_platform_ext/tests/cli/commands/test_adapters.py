# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for the hidden ``nemo adapters`` group against a scripted typed client.

Each test drives the real Typer command with a ``NemoClient`` whose transport
records the HTTP requests, so the assertions pin the exact method, path, query
string, and JSON body the command sends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from typer.testing import CliRunner

ADAPTERS_BASE = "/apis/models/v2/workspaces/default/adapters"

ADAPTER = {
    "name": "lora-1",
    "workspace": "default",
    "description": "an adapter",
    "fileset": "default/adapter-files",
    "finetuning_type": "lora",
    "enabled": True,
    "model": "llama",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


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


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)


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


def test_get_uses_client_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "get", "lora-1"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{ADAPTERS_BASE}/lora-1"
    assert json.loads(result.stdout)["name"] == "lora-1"


def test_get_explicit_workspace_overrides_default() -> None:
    recorder = Recorder([httpx.Response(200, json={**ADAPTER, "workspace": "other"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "get", "lora-1", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/models/v2/workspaces/other/adapters/lora-1"


def test_get_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["adapters", "get", "lora-1"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_get_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Adapter 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Adapter 'nope' not found" in result.stderr
    assert "nemo adapters list" in result.stderr


def test_create_sends_only_provided_fields() -> None:
    recorder = Recorder([httpx.Response(201, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "adapters",
            "create",
            "lora-1",
            "--fileset",
            "default/adapter-files",
            "--finetuning-type",
            "lora",
            "--model",
            "llama",
            "--description",
            "an adapter",
            "--lora-config",
            '{"rank": 8}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == ADAPTERS_BASE
    assert json.loads(recorder.last.content) == {
        "name": "lora-1",
        "fileset": "default/adapter-files",
        "finetuning_type": "lora",
        "model": "llama",
        "description": "an adapter",
        "lora_config": {"rank": 8},
    }
    assert json.loads(result.stdout)["name"] == "lora-1"


def test_create_from_input_data_with_flag_override() -> None:
    recorder = Recorder([httpx.Response(201, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "adapters",
            "create",
            "--input-data",
            '{"name": "lora-1", "fileset": "default/adapter-files", "finetuning_type": "lora", "model": "old", "workspace": "ws2"}',
            "--model",
            "llama",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/models/v2/workspaces/ws2/adapters"
    assert json.loads(recorder.last.content) == {
        "name": "lora-1",
        "fileset": "default/adapter-files",
        "finetuning_type": "lora",
        "model": "llama",
    }


def test_create_requires_all_required_fields() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "create", "lora-1", "--fileset", "ws/fs"], obj=state)

    assert result.exit_code == 2
    assert "--finetuning-type" in result.stderr
    assert "--model" in result.stderr
    assert "--fileset" not in result.stderr.split("Missing required options:")[-1]
    assert recorder.requests == []


def test_create_rejects_invalid_model_ref_locally() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["adapters", "create", "lora-1", "--fileset", "ws/fs", "--finetuning-type", "lora", "--model", "a/b/c"],
        obj=state,
    )

    assert result.exit_code == 2
    assert "Invalid input" in result.stderr
    assert recorder.requests == []


def test_patch_only_sends_provided_fields() -> None:
    recorder = Recorder([httpx.Response(200, json={**ADAPTER, "enabled": False})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["adapters", "patch", "lora-1", "--description", "new", "--fileset", "ws/fs2"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PATCH"
    assert recorder.last.url.path == f"{ADAPTERS_BASE}/lora-1"
    assert json.loads(recorder.last.content) == {"description": "new", "fileset": "ws/fs2"}


def test_patch_enabled_flag() -> None:
    recorder = Recorder([httpx.Response(200, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "patch", "lora-1", "--enabled"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"enabled": True}


def test_patch_from_input_data() -> None:
    recorder = Recorder([httpx.Response(200, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["adapters", "patch", "lora-1", "--input-data", '{"enabled": false, "workspace": "ws2"}'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/models/v2/workspaces/ws2/adapters/lora-1"
    assert json.loads(recorder.last.content) == {"enabled": False}


def test_delete() -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "delete", "lora-1"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{ADAPTERS_BASE}/lora-1"
    assert "Deleted successfully" in result.stdout


def test_list_passes_pagination_query_params_and_warns() -> None:
    recorder = Recorder([_page([ADAPTER], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["adapters", "list", "--page", "1", "--page-size", "1", "--sort", "-created_at"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == ADAPTERS_BASE
    assert dict(recorder.last.url.params) == {"page": "1", "page_size": "1", "sort": "-created_at"}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["lora-1"]
    assert body["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_list_filter_fields_are_sent_as_json_filter() -> None:
    recorder = Recorder([_page([ADAPTER], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["adapters", "list", "--filter.model", "llama", "--filter.enabled", "--filter.finetuning-type", "lora"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    params = dict(recorder.last.url.params)
    assert set(params) == {"filter"}
    assert json.loads(params["filter"]) == {"model": "llama", "enabled": True, "finetuning_type": "lora"}


def test_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([ADAPTER], 1, 2), _page([{**ADAPTER, "name": "second"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["lora-1", "second"]
    assert body["pagination"]["total_results"] == 2
    assert "More pages" not in result.stderr


def test_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([_page([ADAPTER], 1, 1)])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["adapters", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "workspace" in output and "created_at" in output
    assert "description" not in output


def test_code_output_does_not_send_request() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "adapters",
            "create",
            "lora-1",
            "--fileset",
            "ws/fs",
            "--finetuning-type",
            "lora",
            "--model",
            "llama",
            "-f",
            "code",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.models.client import ModelsClient" in result.stdout
    assert 'client = ModelsClient(base_url="http://test/")' in result.stdout
    assert "client.create_adapter(" in result.stdout
    assert (
        'CreateAdapterRequest(name="lora-1", fileset="ws/fs", finetuning_type=FinetuningType.LORA, model="llama")'
        in (result.stdout)
    )
    assert "NeMoPlatform" not in result.stdout


def test_code_output_for_patch() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["adapters", "patch", "lora-1", "--enabled", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'client.update_adapter(name="lora-1", body=UpdateAdapterRequest(enabled=True))' in result.stdout
    assert "NeMoPlatform" not in result.stdout
