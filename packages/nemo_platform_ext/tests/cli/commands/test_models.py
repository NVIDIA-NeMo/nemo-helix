# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo models`` against a scripted typed client.

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

MODELS_BASE = "/apis/models/v2/workspaces/default"

MODEL = {
    "id": "model-1",
    "name": "llama",
    "workspace": "default",
    "description": "base model",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

ADAPTER = {
    "name": "lora-1",
    "workspace": "default",
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


# ---------------------------------------------------------------------------
# models get
# ---------------------------------------------------------------------------


def test_get_uses_client_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "get", "llama"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(result.stdout)["name"] == "llama"


def test_get_explicit_workspace_and_verbose() -> None:
    recorder = Recorder([httpx.Response(200, json={**MODEL, "workspace": "other"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "get", "llama", "--workspace", "other", "--verbose"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/models/v2/workspaces/other/models/llama"
    assert dict(recorder.last.url.params) == {"verbose": "true"}


def test_get_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["models", "get", "llama"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_get_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Model 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Model 'nope' not found" in result.stderr
    assert "nemo models list" in result.stderr


# ---------------------------------------------------------------------------
# models create
# ---------------------------------------------------------------------------


def test_create_sends_only_provided_fields() -> None:
    recorder = Recorder([httpx.Response(201, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "models",
            "create",
            "llama",
            "--description",
            "base model",
            "--finetuning-type",
            "lora",
            "--backend-format",
            "OPENAI_CHAT",
            "--model-providers",
            "default/p1",
            "--model-providers",
            "default/p2",
            "--trust-remote-code",
            "--custom-fields",
            '{"team": "nemo"}',
            "--api-endpoint",
            '{"url": "https://api.example.com/v1", "model_id": "gpt"}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{MODELS_BASE}/models"
    assert json.loads(recorder.last.content) == {
        "name": "llama",
        "description": "base model",
        "finetuning_type": "lora",
        "backend_format": "OPENAI_CHAT",
        "model_providers": ["default/p1", "default/p2"],
        "trust_remote_code": True,
        "custom_fields": {"team": "nemo"},
        "api_endpoint": {"url": "https://api.example.com/v1", "model_id": "gpt"},
    }
    assert json.loads(result.stdout)["name"] == "llama"


def test_create_minimal_body_is_just_name() -> None:
    recorder = Recorder([httpx.Response(201, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "create", "llama"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "llama"}


def test_create_merges_input_data_with_flags_taking_precedence() -> None:
    recorder = Recorder([httpx.Response(201, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "models",
            "create",
            "--input-data",
            '{"name": "from-json", "description": "json desc", "workspace": "json-ws", "base_model": "root"}',
            "--description",
            "flag desc",
            "--workspace",
            "flag-ws",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/models/v2/workspaces/flag-ws/models"
    assert json.loads(recorder.last.content) == {"name": "from-json", "description": "flag desc", "base_model": "root"}


def test_create_reads_input_from_stdin() -> None:
    recorder = Recorder([httpx.Response(201, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "create", "--input-file", "-"],
        obj=state,
        input='{"name": "llama", "description": "piped"}',
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "llama", "description": "piped"}


def test_create_requires_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "create", "--description", "d"], obj=state)

    assert result.exit_code == 2
    assert "name" in result.stderr
    assert recorder.requests == []


def test_create_exist_ok_returns_existing_on_conflict() -> None:
    recorder = Recorder(
        [
            httpx.Response(409, json={"detail": "already exists"}),
            httpx.Response(200, json=MODEL),
        ]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "create", "llama", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [
        ("POST", f"{MODELS_BASE}/models"),
        ("GET", f"{MODELS_BASE}/models/llama"),
    ]
    assert json.loads(result.stdout)["name"] == "llama"


def test_create_conflict_without_exist_ok_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "already exists"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "create", "llama"], obj=state)

    assert result.exit_code == 3
    assert "Conflict" in result.stderr
    assert len(recorder.requests) == 1


def test_create_rejects_invalid_name_locally() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "create", "bad name!"], obj=state)

    assert result.exit_code == 2
    assert "Invalid input" in result.stderr
    assert "name" in result.stderr
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# models update
# ---------------------------------------------------------------------------


def test_update_only_sends_provided_fields_and_verbose_query() -> None:
    recorder = Recorder([httpx.Response(200, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "update", "llama", "--description", "new", "--model-providers", "default/p1", "--verbose"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PATCH"
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama"
    assert dict(recorder.last.url.params) == {"verbose": "true"}
    assert json.loads(recorder.last.content) == {"description": "new", "model_providers": ["default/p1"]}


def test_update_with_no_fields_sends_empty_body() -> None:
    recorder = Recorder([httpx.Response(200, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "update", "llama"], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {}


def test_update_from_input_data() -> None:
    recorder = Recorder([httpx.Response(200, json=MODEL)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "update", "llama", "--input-data", '{"description": "d", "trust_remote_code": true}'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"description": "d", "trust_remote_code": True}


# ---------------------------------------------------------------------------
# models delete
# ---------------------------------------------------------------------------


def test_delete() -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "delete", "llama"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama"
    assert "Deleted successfully" in result.stdout


def test_delete_not_found() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Model 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "delete", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found" in result.stderr


# ---------------------------------------------------------------------------
# models list
# ---------------------------------------------------------------------------


def test_list_passes_pagination_query_params_and_warns() -> None:
    recorder = Recorder([_page([MODEL], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["models", "list", "--page", "1", "--page-size", "1", "--sort", "-created_at", "--verbose"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{MODELS_BASE}/models"
    assert dict(recorder.last.url.params) == {"page": "1", "page_size": "1", "sort": "-created_at", "verbose": "true"}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["llama"]
    assert body["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_list_filter_fields_are_sent_as_json_filter() -> None:
    recorder = Recorder([_page([MODEL], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "list", "--filter.name", "llama", "--filter.lora-enabled", "--filter.base-model", "root"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    params = dict(recorder.last.url.params)
    assert set(params) == {"filter"}
    assert json.loads(params["filter"]) == {"name": "llama", "lora_enabled": True, "base_model": "root"}


def test_list_json_filter_is_merged_with_field_options() -> None:
    recorder = Recorder([_page([MODEL], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "list", "--filter", '{"created_at": {"gte": "2026-01-01"}, "name": "old"}', "--filter.name", "new"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(dict(recorder.last.url.params)["filter"]) == {"created_at": {"gte": "2026-01-01"}, "name": "new"}


def test_list_text_filter_is_forwarded_verbatim() -> None:
    recorder = Recorder([_page([MODEL], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "list", "--filter", 'name~"llama"'], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"filter": 'name~"llama"'}


def test_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([MODEL], 1, 2), _page([{**MODEL, "name": "second"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["llama", "second"]
    assert body["pagination"]["total_results"] == 2
    assert "More pages" not in result.stderr


def test_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([_page([MODEL], 1, 1)])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["models", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "description" in output
    assert "created_at" not in output


def test_list_stream_requires_json_or_raw() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "list", "-f", "table", "--stream"], obj=state)

    assert result.exit_code == 2
    assert "--stream requires --output json or --output raw" in result.stderr
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# models adapters (nested)
# ---------------------------------------------------------------------------


def test_adapters_create_posts_under_model() -> None:
    recorder = Recorder([httpx.Response(201, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "models",
            "adapters",
            "create",
            "llama",
            "lora-1",
            "--fileset",
            "default/adapter-files",
            "--finetuning-type",
            "lora",
            "--lora-config",
            '{"rank": 8, "alpha": 16}',
            "--enabled",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama/adapters"
    assert json.loads(recorder.last.content) == {
        "name": "lora-1",
        "fileset": "default/adapter-files",
        "finetuning_type": "lora",
        "lora_config": {"rank": 8, "alpha": 16},
        "enabled": True,
    }
    assert json.loads(result.stdout)["name"] == "lora-1"


def test_adapters_create_reads_required_fields_from_input_data() -> None:
    recorder = Recorder([httpx.Response(201, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "models",
            "adapters",
            "create",
            "llama",
            "--input-data",
            '{"name": "lora-1", "fileset": "default/adapter-files", "finetuning_type": "lora", "workspace": "ws2"}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/models/v2/workspaces/ws2/models/llama/adapters"
    assert json.loads(recorder.last.content) == {
        "name": "lora-1",
        "fileset": "default/adapter-files",
        "finetuning_type": "lora",
    }


def test_adapters_create_requires_fileset_finetuning_type_and_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "adapters", "create", "llama"], obj=state)

    assert result.exit_code == 2
    assert "--fileset" in result.stderr
    assert "--finetuning-type" in result.stderr
    assert "--name" in result.stderr
    assert recorder.requests == []


def test_adapters_update_patches_under_model() -> None:
    recorder = Recorder([httpx.Response(200, json={**ADAPTER, "enabled": False})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "adapters", "update", "lora-1", "--model-name", "llama", "--description", "d", "--enabled"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PATCH"
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama/adapters/lora-1"
    assert json.loads(recorder.last.content) == {"description": "d", "enabled": True}


def test_adapters_update_requires_model_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "adapters", "update", "lora-1", "--description", "d"], obj=state)

    assert result.exit_code == 2
    assert "--model-name" in result.stderr
    assert recorder.requests == []


def test_adapters_update_model_name_from_input_data() -> None:
    recorder = Recorder([httpx.Response(200, json=ADAPTER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "adapters", "update", "lora-1", "--input-data", '{"model_name": "llama", "fileset": "ws/fs2"}'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama/adapters/lora-1"
    assert json.loads(recorder.last.content) == {"fileset": "ws/fs2"}


def test_adapters_delete() -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "adapters", "delete", "lora-1", "--model-name", "llama"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{MODELS_BASE}/models/llama/adapters/lora-1"
    assert "Deleted successfully" in result.stdout


def test_adapters_delete_requires_model_name_option() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["models", "adapters", "delete", "lora-1"], obj=state)

    assert result.exit_code == 2
    assert "Missing required argument" in result.stderr
    assert "MODEL_NAME" in result.stderr
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# -f code
# ---------------------------------------------------------------------------


def test_code_output_does_not_send_request() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["models", "create", "llama", "--description", "d", "--finetuning-type", "lora", "--exist-ok", "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.models.client import ModelsClient" in result.stdout
    assert 'client = ModelsClient(base_url="http://test/")' in result.stdout
    assert "client.create_model(" in result.stdout
    assert (
        'CreateModelEntityRequest(name="llama", description="d", finetuning_type=FinetuningType.LORA)' in result.stdout
    )
    assert "exist_ok=True" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_code_output_for_list_uses_query_params() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["models", "list", "--page-size", "5", "--filter.name", "llama", "-f", "code"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.list_models(" in result.stdout
    assert '"page_size": 5' in result.stdout
    assert "for item in response.page().items:" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_code_output_for_nested_adapter_update() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["models", "adapters", "update", "lora-1", "--model-name", "llama", "--enabled", "-f", "code"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert (
        'client.update_model_adapter(model_name="llama", adapter="lora-1", body=UpdateAdapterRequest(enabled=True))'
        in (result.stdout)
    )
    assert "NeMoPlatform" not in result.stdout
