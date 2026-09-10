# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo experiments`` against a scripted typed client."""

from __future__ import annotations

import json

import httpx
from nmp.intake.cli import ExperimentsCLI

BASE = "/apis/intake/v2/workspaces/default/experiments"

EXPERIMENT = {
    "id": "exp-id-1",
    "name": "exp-1",
    "workspace": "default",
    "description": "demo",
    "default_sort": "-created_at",
    "pareto": {"x_metric": "cost_usd", "y_metric": "latency_ms"},
    "is_favorite": False,
    "show_evaluations_over_time": False,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "evaluation_count": 0,
}


def test_plugin_metadata() -> None:
    assert ExperimentsCLI.name == "experiments"
    assert ExperimentsCLI.description == "Manage experiments."
    assert {cmd.name for cmd in ExperimentsCLI().get_cli().registered_commands} == {
        "create",
        "delete",
        "list",
        "get",
        "update",
    }


def test_get_uses_client_default_workspace(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["get", "exp-1"], [httpx.Response(200, json=EXPERIMENT)])

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{BASE}/exp-1"
    assert json.loads(result.stdout)["name"] == "exp-1"


def test_get_explicit_workspace_overrides_default(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["get", "exp-1", "--workspace", "other"], [httpx.Response(200, json={**EXPERIMENT, "workspace": "other"})]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/intake/v2/workspaces/other/experiments/exp-1"


def test_get_without_any_workspace_is_usage_error(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["get", "exp-1"], workspace=None)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_get_not_found_maps_to_remote_error(experiments_cli) -> None:
    result, _ = experiments_cli.run(
        ["get", "nope"], [httpx.Response(404, json={"detail": "Experiment 'default/nope' not found."})]
    )

    assert result.exit_code == 3
    assert "Not found: (404) Experiment 'default/nope' not found." in result.stderr


def test_create_sends_only_provided_fields(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        [
            "create",
            "exp-1",
            "--description",
            "demo",
            "--is-favorite",
            "--metadata",
            '{"team": "core"}',
            "--pareto",
            '{"x_metric": "cost_usd", "y_metric": "evaluators.reward"}',
        ],
        [httpx.Response(201, json=EXPERIMENT)],
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == BASE
    assert json.loads(recorder.last.content) == {
        "name": "exp-1",
        "description": "demo",
        "is_favorite": True,
        "metadata": {"team": "core"},
        "pareto": {"x_metric": "cost_usd", "y_metric": "evaluators.reward"},
    }
    assert json.loads(result.stdout)["name"] == "exp-1"


def test_create_from_input_data_with_flag_override(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["create", "--input-data", '{"name": "from-file", "summary": "s"}', "--description", "flag"],
        [httpx.Response(201, json=EXPERIMENT)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "from-file", "summary": "s", "description": "flag"}


def test_create_input_workspace_key_selects_path_workspace(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["create", "--input-data", '{"name": "exp-1", "workspace": "other"}'], [httpx.Response(201, json=EXPERIMENT)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/intake/v2/workspaces/other/experiments"
    assert json.loads(recorder.last.content) == {"name": "exp-1"}


def test_create_from_stdin(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["create", "--input-file", "-"], [httpx.Response(201, json=EXPERIMENT)], input='{"name": "piped"}\n'
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "piped"}


def test_create_requires_name(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["create", "--description", "x"])

    assert result.exit_code == 2
    assert "--name" in result.stderr
    assert recorder.requests == []


def test_create_exist_ok_returns_existing_on_conflict(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["create", "exp-1", "--exist-ok"],
        [httpx.Response(409, json={"detail": "exists"}), httpx.Response(200, json=EXPERIMENT)],
    )

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [("POST", BASE), ("GET", f"{BASE}/exp-1")]
    assert json.loads(result.stdout)["name"] == "exp-1"


def test_create_conflict_without_exist_ok_fails(experiments_cli) -> None:
    result, _ = experiments_cli.run(
        ["create", "exp-1"], [httpx.Response(409, json={"detail": "Experiment 'default/exp-1' already exists."})]
    )

    assert result.exit_code == 3
    assert "Conflict: (409)" in result.stderr


def test_update_puts_body_name_and_provided_fields(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["update", "exp-1", "--body-name", "exp-1", "--summary", "done", "--baseline-evaluation-name", "eval-a"],
        [httpx.Response(200, json=EXPERIMENT)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PUT"
    assert recorder.last.url.path == f"{BASE}/exp-1"
    assert json.loads(recorder.last.content) == {
        "name": "exp-1",
        "summary": "done",
        "baseline_evaluation_name": "eval-a",
    }


def test_update_body_name_flag_wins_over_input_file_name(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["update", "exp-1", "--body-name", "exp-1-renamed", "--input-data", '{"name": "from-file", "summary": "s"}'],
        [httpx.Response(200, json=EXPERIMENT)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "exp-1-renamed", "summary": "s"}


def test_update_requires_body_name(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["update", "exp-1", "--summary", "x"])

    assert result.exit_code == 2
    assert "--body-name" in result.stderr
    assert recorder.requests == []


def test_delete(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["delete", "exp-1"], [httpx.Response(204)])

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{BASE}/exp-1"
    assert "Deleted successfully" in result.stdout


def test_list_passes_filter_sort_and_pagination_and_warns(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["list", "--page", "1", "--page-size", "1", "--sort", "-name", "--filter.is-favorite", "--filter.name", "exp"],
        [experiments_cli.page([EXPERIMENT], 1, 2)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == BASE
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"is_favorite": True, "name": "exp"}
    assert params == {"page": "1", "page_size": "1", "sort": "-name"}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["exp-1"]
    assert body["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_list_json_filter_merges_with_field_options(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["list", "--filter", '{"metadata": {"team": "core"}}', "--filter.is-deleted"],
        [experiments_cli.page([EXPERIMENT], 1, 1)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(dict(recorder.last.url.params)["filter"]) == {"metadata": {"team": "core"}, "is_deleted": True}


def test_list_all_pages_follows_every_page(experiments_cli) -> None:
    result, recorder = experiments_cli.run(
        ["list", "--page-size", "1", "--all-pages"],
        [experiments_cli.page([EXPERIMENT], 1, 2), experiments_cli.page([{**EXPERIMENT, "name": "exp-2"}], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["exp-1", "exp-2"]
    assert body["pagination"]["total_results"] == 2
    assert "More pages" not in result.stderr


def test_list_table_output_uses_default_columns(experiments_cli) -> None:
    result, _ = experiments_cli.run(["list", "-f", "table"], [experiments_cli.page([EXPERIMENT], 1, 1)])

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "workspace" in output and "created_at" in output
    assert "description" not in output


def test_create_code_output_renders_typed_client_without_request(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["create", "exp-1", "--description", "demo", "--exist-ok", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.intake.client import IntakeClient" in result.stdout
    assert 'client = IntakeClient(base_url="http://test/")' in result.stdout
    assert 'body=ExperimentCreateRequest(name="exp-1", description="demo")' in result.stdout
    assert "exist_ok=True" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_list_code_output(experiments_cli) -> None:
    result, recorder = experiments_cli.run(["list", "--sort", "name", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'response = client.list_experiments(query_params={"sort": "name"})' in result.stdout
    assert "for item in response.page().items:" in result.stdout
