# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo guardrail`` against a scripted typed client."""

from __future__ import annotations

import json

import httpx
from nemo_guardrails_plugin.cli import GuardrailCLI

CONFIGS = "/apis/guardrails/v2/workspaces/default/configs"
CHECKS = "/apis/guardrails/v2/workspaces/default/checks"

CONFIG = {
    "id": "guardrail_config-abc123",
    "name": "safety",
    "workspace": "default",
    "description": "Content safety rails",
    "data": {"models": [], "rails": {"input": {"flows": ["content safety check input $model=content_safety"]}}},
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

CHECK_RESPONSE = {
    "status": "success",
    "rails_status": {"content safety check input": {"status": "success"}},
    "guardrails_data": {"config_ids": ["default/safety"]},
}

MESSAGES = '[{"role": "user", "content": "hello"}]'


def test_plugin_metadata() -> None:
    assert GuardrailCLI.name == "guardrail"
    assert GuardrailCLI.description == "Manage guardrails."
    app = GuardrailCLI().get_cli()
    assert app.info.help == GuardrailCLI.description
    assert {cmd.name for cmd in app.registered_commands} == {"check"}
    (configs_group,) = app.registered_groups
    assert configs_group.name == "configs"
    assert configs_group.typer_instance is not None
    assert {cmd.name for cmd in configs_group.typer_instance.registered_commands} == {
        "create",
        "delete",
        "list",
        "get",
        "update",
    }


# ---------------------------------------------------------------------------
# guardrail check
# ---------------------------------------------------------------------------


def test_check_posts_messages_and_model(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["check", "--messages", MESSAGES, "--model", "meta/llama-3.1-8b-instruct"],
        [httpx.Response(200, json=CHECK_RESPONSE)],
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == CHECKS
    assert json.loads(recorder.last.content) == {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert json.loads(result.stdout)["status"] == "success"


def test_check_forwards_optional_sampling_and_guardrails_fields(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        [
            "check",
            "--messages",
            MESSAGES,
            "--model",
            "m",
            "--guardrails",
            '{"config_id": "default/safety"}',
            "--temperature",
            "0.2",
            "--max-tokens",
            "16",
            "--seed",
            "7",
            "--stop",
            '["\\n"]',
            "--logprobs",
            "--user",
            "u-1",
        ],
        [httpx.Response(200, json=CHECK_RESPONSE)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "guardrails": {"config_id": "default/safety"},
        "temperature": 0.2,
        "max_tokens": 16,
        "seed": 7,
        "stop": ["\n"],
        "logprobs": True,
        "user": "u-1",
    }


def test_check_explicit_workspace_overrides_default(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["check", "--messages", MESSAGES, "--model", "m", "--workspace", "other"],
        [httpx.Response(200, json=CHECK_RESPONSE)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/checks"
    assert "workspace" not in json.loads(recorder.last.content)


def test_check_from_input_data_with_flag_override(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["check", "--input-data", '{"messages": [{"role": "user", "content": "x"}], "model": "old"}', "--model", "new"],
        [httpx.Response(200, json=CHECK_RESPONSE)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"messages": [{"role": "user", "content": "x"}], "model": "new"}


def test_check_input_workspace_key_selects_path_workspace(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["check", "--input-data", '{"messages": [], "model": "m", "workspace": "other"}'],
        [httpx.Response(200, json=CHECK_RESPONSE)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/checks"
    assert json.loads(recorder.last.content) == {"messages": [], "model": "m"}


def test_check_from_stdin(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["check", "--input-file", "-"],
        [httpx.Response(200, json=CHECK_RESPONSE)],
        input='{"messages": [{"role": "user", "content": "piped"}], "model": "m"}\n',
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content)["messages"] == [{"role": "user", "content": "piped"}]


def test_check_requires_messages_and_model(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["check", "--temperature", "0.1"])

    assert result.exit_code == 2
    assert "--messages" in result.stderr
    assert "--model" in result.stderr
    assert recorder.requests == []


def test_check_without_any_workspace_is_usage_error(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["check", "--messages", MESSAGES, "--model", "m"], workspace=None)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_check_not_found_maps_to_remote_error(guardrail_cli) -> None:
    result, _ = guardrail_cli.run(
        ["check", "--messages", MESSAGES, "--model", "m", "--guardrails", '{"config_id": "default/nope"}'],
        [httpx.Response(404, json={"detail": "Guardrail config 'default/nope' not found."})],
    )

    assert result.exit_code == 3
    assert "Not found: (404) Guardrail config 'default/nope' not found." in result.stderr


def test_check_code_output_renders_typed_client_without_request(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["check", "--messages", MESSAGES, "--model", "m", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.guardrail.client import GuardrailClient" in result.stdout
    assert "from nemo_platform_plugin.guardrail.types import GuardrailCheckRequest" in result.stdout
    assert 'client = GuardrailClient(base_url="http://test/")' in result.stdout
    assert "response = client.check_guardrail(" in result.stdout
    assert 'GuardrailCheckRequest(model="m", messages=[{"role": "user", "content": "hello"}])' in result.stdout
    assert "print(response.data())" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# guardrail configs get
# ---------------------------------------------------------------------------


def test_configs_get_uses_client_default_workspace(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "get", "safety"], [httpx.Response(200, json=CONFIG)])

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{CONFIGS}/safety"
    assert json.loads(result.stdout)["name"] == "safety"


def test_configs_get_explicit_workspace_overrides_default(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "get", "safety", "--workspace", "other"],
        [httpx.Response(200, json={**CONFIG, "workspace": "other"})],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/configs/safety"


def test_configs_get_without_any_workspace_is_usage_error(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "get", "safety"], workspace=None)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_configs_get_not_found_maps_to_remote_error(guardrail_cli) -> None:
    result, _ = guardrail_cli.run(
        ["configs", "get", "nope"], [httpx.Response(404, json={"detail": "Guardrail config not found."})]
    )

    assert result.exit_code == 3
    assert "Not found: (404) Guardrail config not found." in result.stderr
    assert "nemo configs list" in result.stderr


def test_configs_get_code_output(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "get", "safety", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'response = client.get_guardrail_config(name="safety")' in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# guardrail configs create
# ---------------------------------------------------------------------------


def test_configs_create_sends_only_provided_fields(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        [
            "configs",
            "create",
            "safety",
            "--description",
            "Content safety rails",
            "--data",
            '{"rails": {"input": {"flows": ["self check input"]}}}',
        ],
        [httpx.Response(201, json=CONFIG)],
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == CONFIGS
    assert json.loads(recorder.last.content) == {
        "name": "safety",
        "description": "Content safety rails",
        "data": {"rails": {"input": {"flows": ["self check input"]}}},
    }
    assert json.loads(result.stdout)["name"] == "safety"


def test_configs_create_name_only_omits_optional_fields(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "create", "safety"], [httpx.Response(201, json=CONFIG)])

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "safety"}


def test_configs_create_from_input_data_with_flag_override(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "create", "--input-data", '{"name": "from-file", "data": {"a": 1}}', "--description", "flag"],
        [httpx.Response(201, json=CONFIG)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "from-file", "data": {"a": 1}, "description": "flag"}


def test_configs_create_input_workspace_key_selects_path_workspace(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "create", "--input-data", '{"name": "safety", "workspace": "other"}'],
        [httpx.Response(201, json=CONFIG)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/configs"
    assert json.loads(recorder.last.content) == {"name": "safety"}


def test_configs_create_from_stdin(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "create", "--input-file", "-"], [httpx.Response(201, json=CONFIG)], input='{"name": "piped"}\n'
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "piped"}


def test_configs_create_requires_name(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "create", "--description", "x"])

    assert result.exit_code == 2
    assert "--name" in result.stderr
    assert recorder.requests == []


def test_configs_create_exist_ok_returns_existing_on_conflict(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "create", "safety", "--exist-ok"],
        [httpx.Response(409, json={"detail": "Config 'safety' already exists."}), httpx.Response(200, json=CONFIG)],
    )

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [("POST", CONFIGS), ("GET", f"{CONFIGS}/safety")]
    assert json.loads(result.stdout)["name"] == "safety"


def test_configs_create_conflict_without_exist_ok_fails(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "create", "safety"], [httpx.Response(409, json={"detail": "Config 'safety' already exists."})]
    )

    assert result.exit_code == 3
    assert "Conflict: (409) Config 'safety' already exists." in result.stderr
    assert len(recorder.requests) == 1


def test_configs_create_code_output_renders_typed_client_without_request(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "create", "safety", "--description", "demo", "--exist-ok", "-f", "code"]
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.guardrail.client import GuardrailClient" in result.stdout
    assert "from nemo_platform_plugin.guardrail.types import CreateGuardrailConfigRequest" in result.stdout
    assert 'client = GuardrailClient(base_url="http://test/")' in result.stdout
    assert 'body=CreateGuardrailConfigRequest(name="safety", description="demo")' in result.stdout
    assert "exist_ok=True" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# guardrail configs update
# ---------------------------------------------------------------------------


def test_configs_update_patches_only_provided_fields(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "update", "safety", "--description", "changed"], [httpx.Response(200, json=CONFIG)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PATCH"
    assert recorder.last.url.path == f"{CONFIGS}/safety"
    assert json.loads(recorder.last.content) == {"description": "changed"}


def test_configs_update_data_from_flag(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "update", "safety", "--data", '{"rails": {}}'], [httpx.Response(200, json=CONFIG)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"data": {"rails": {}}}


def test_configs_update_from_input_data_with_flag_override_and_workspace(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        [
            "configs",
            "update",
            "safety",
            "--input-data",
            '{"description": "file", "data": {"k": "v"}}',
            "--description",
            "flag",
            "--workspace",
            "other",
        ],
        [httpx.Response(200, json=CONFIG)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/configs/safety"
    assert json.loads(recorder.last.content) == {"description": "flag", "data": {"k": "v"}}


def test_configs_update_without_fields_sends_empty_body(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "update", "safety"], [httpx.Response(200, json=CONFIG)])

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {}


def test_configs_update_code_output(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "update", "safety", "--description", "d", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.update_guardrail_config(" in result.stdout
    assert 'body=UpdateGuardrailConfigRequest(description="d")' in result.stdout


# ---------------------------------------------------------------------------
# guardrail configs delete
# ---------------------------------------------------------------------------


def test_configs_delete(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "delete", "safety"], [httpx.Response(200, json={"id": CONFIG["id"], "deleted": True})]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{CONFIGS}/safety"
    assert "Deleted successfully" in result.stdout


def test_configs_delete_explicit_workspace(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "delete", "safety", "--workspace", "other"],
        [httpx.Response(200, json={"id": CONFIG["id"], "deleted": True})],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/configs/safety"


def test_configs_delete_in_use_conflict_maps_to_remote_error(guardrail_cli) -> None:
    result, _ = guardrail_cli.run(
        ["configs", "delete", "safety"],
        [httpx.Response(409, json={"detail": "Guardrail config 'default/safety' is applied by default/vm."})],
    )

    assert result.exit_code == 3
    assert "Conflict: (409) Guardrail config 'default/safety' is applied by default/vm." in result.stderr


# ---------------------------------------------------------------------------
# guardrail configs list
# ---------------------------------------------------------------------------


def test_configs_list_passes_filter_sort_and_pagination_and_warns(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        [
            "configs",
            "list",
            "--page",
            "1",
            "--page-size",
            "1",
            "--sort",
            "-name",
            "--filter.name",
            "safety",
            "--filter.project",
            "p",
        ],
        [guardrail_cli.page([CONFIG], 1, 2)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == CONFIGS
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"name": "safety", "project": "p"}
    assert params == {"page": "1", "page_size": "1", "sort": "-name"}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["safety"]
    assert body["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_configs_list_without_options_sends_no_query(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "list"], [guardrail_cli.page([CONFIG], 1, 1)])

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {}
    assert "More pages" not in result.stderr


def test_configs_list_json_filter_merges_with_field_options(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        [
            "configs",
            "list",
            "--filter",
            '{"created_at": {"gte": "2026-01-01"}, "name": "old"}',
            "--filter.name",
            "new",
            "--filter.description",
            "d",
        ],
        [guardrail_cli.page([CONFIG], 1, 1)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(dict(recorder.last.url.params)["filter"]) == {
        "created_at": {"gte": "2026-01-01"},
        "name": "new",
        "description": "d",
    }


def test_configs_list_text_filter_is_forwarded_verbatim(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "list", "--filter", 'name~"safe"'], [guardrail_cli.page([CONFIG], 1, 1)]
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"filter": 'name~"safe"'}


def test_configs_list_all_pages_follows_every_page(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "list", "--page-size", "1", "--all-pages"],
        [guardrail_cli.page([CONFIG], 1, 2), guardrail_cli.page([{**CONFIG, "name": "second"}], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["safety", "second"]
    assert body["pagination"]["total_results"] == 2
    assert "More pages" not in result.stderr


def test_configs_list_table_output_uses_default_columns(guardrail_cli) -> None:
    result, _ = guardrail_cli.run(["configs", "list", "-f", "table"], [guardrail_cli.page([CONFIG], 1, 1)])

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "description" in output and "created_at" in output
    assert "workspace" not in output


def test_configs_list_explicit_workspace(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(
        ["configs", "list", "--workspace", "other"], [guardrail_cli.page([CONFIG], 1, 1)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/guardrails/v2/workspaces/other/configs"


def test_configs_list_code_output(guardrail_cli) -> None:
    result, recorder = guardrail_cli.run(["configs", "list", "--sort", "name", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'response = client.list_guardrail_configs(query_params={"sort": "name"})' in result.stdout
    assert "for item in response.page().items:" in result.stdout
    assert "NeMoPlatform" not in result.stdout
