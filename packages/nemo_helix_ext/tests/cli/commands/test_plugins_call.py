# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for discovering and calling registered external plugins from the CLI.

Command tests drive the real Typer app over a recorded ``httpx.MockTransport``, so they
pin the exact method, path, query, and body sent to the gateway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import click
import httpx
import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.commands.plugins_external import (
    endpoint_name,
    endpoint_rows,
    fill_path,
    parse_body,
    parse_pairs,
    scalar_columns,
    select_endpoint,
)
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.community_plugins.registry import ExternalPlugin
from typer.testing import CliRunner


def _endpoint(method: str, name: str, access: str) -> dict:
    return {
        "method": method,
        "path": f"/v2/workspaces/{{workspace}}/{name}",
        "access": access,
        "upstream_url": f"http://127.0.0.1:6007/v1/workspaces/{{workspace}}/{name}",
    }


PHOENIX = {
    "name": "phoenix",
    "title": "Phoenix traces",
    "enabled": True,
    "endpoints": [
        _endpoint("GET", "projects", "read"),
        _endpoint("GET", "spans", "read"),
        _endpoint("POST", "spans", "write"),
    ],
}
CHRISTMAS = {
    "name": "is-it-christmas",
    "title": "Is it Christmas?",
    "enabled": True,
    "endpoints": [_endpoint("GET", "is-it-christmas", "read")],
}
REGISTRY = {"schema_version": 1, "plugins": [PHOENIX, CHRISTMAS]}


@dataclass
class Recorder:
    """Records requests; answers the registry lookup, then the scripted response."""

    response: httpx.Response
    registry: dict = field(default_factory=lambda: REGISTRY)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/apis/plugins/-/installed":
            return httpx.Response(200, json=self.registry)
        return self.response

    @property
    def plugin_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path != "/apis/plugins/-/installed"]


def make_runner(recorder: Recorder) -> tuple[CliRunner, CLIContext]:
    client = NemoClient(base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(recorder)))
    state = CLIContext(overrides={"base_url": "http://test", "output_format": "table"}, _client=client)
    return CliRunner(), state


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NHX_ACCESS_TOKEN", raising=False)


def _plugin(raw: dict) -> ExternalPlugin:
    return ExternalPlugin.model_validate(raw)


# --- pure helpers ---------------------------------------------------------------


def test_endpoint_name_is_the_last_non_template_segment() -> None:
    assert endpoint_name("/v2/workspaces/{workspace}/spans") == "spans"
    assert endpoint_name("/v2/workspaces/{workspace}/is-it-christmas") == "is-it-christmas"
    assert endpoint_name("/v2/workspaces/{workspace}/items/{id}") == "items"


def test_single_endpoint_plugin_needs_no_name() -> None:
    assert select_endpoint(_plugin(CHRISTMAS), None, None).path.endswith("/is-it-christmas")


def test_name_selects_and_get_is_preferred_when_ambiguous() -> None:
    plugin = _plugin(PHOENIX)
    assert select_endpoint(plugin, "spans", None).method == "GET"
    assert select_endpoint(plugin, "spans", "post").method == "POST"


@pytest.mark.parametrize(
    ("name", "method", "message"),
    [
        (None, None, "several endpoints"),
        ("nope", None, "no endpoint 'nope'"),
        ("projects", "POST", "no POST endpoint"),
    ],
)
def test_unselectable_endpoints_explain_the_choices(name: str | None, method: str | None, message: str) -> None:
    with pytest.raises(click.UsageError, match=message):
        select_endpoint(_plugin(PHOENIX), name, method)


def test_fill_path_quotes_values_as_single_segments_and_demands_missing_ones() -> None:
    path = "/v2/workspaces/{workspace}/items/{id}"
    assert fill_path(path, workspace="my ws", params={"id": "a/b"}) == "/v2/workspaces/my%20ws/items/a%2Fb"
    with pytest.raises(click.UsageError, match="--param id=VALUE"):
        fill_path(path, workspace="default", params={})


def test_parse_pairs_and_body_reject_malformed_input(tmp_path) -> None:
    assert parse_pairs(["a=1", "b=x=y"], option="-q") == {"a": "1", "b": "x=y"}
    with pytest.raises(click.UsageError, match="KEY=VALUE"):
        parse_pairs(["oops"], option="-q")
    body_file = tmp_path / "body.json"
    body_file.write_text('{"spans": []}')
    assert json.loads(parse_body(f"@{body_file}", method="POST") or b"") == {"spans": []}
    with pytest.raises(click.UsageError, match="not valid JSON"):
        parse_body("{nope", method="POST")
    with pytest.raises(click.UsageError, match="cannot be sent with GET"):
        parse_body("{}", method="GET")


def test_endpoint_rows_skip_disabled_plugins_and_give_pasteable_examples() -> None:
    rows = {
        (r.plugin, r.endpoint, r.method): r.example
        for r in endpoint_rows([_plugin(PHOENIX), _plugin({**CHRISTMAS, "enabled": False})])
    }
    assert rows == {
        ("phoenix", "projects", "GET"): "nemo plugins call phoenix projects",
        ("phoenix", "spans", "GET"): "nemo plugins call phoenix spans",
        ("phoenix", "spans", "POST"): "nemo plugins call phoenix spans -X POST --data '{...}'",
    }


def test_scalar_columns_skip_nested_values_and_cap_the_width() -> None:
    assert scalar_columns([{"a": 1, "nested": {"x": 1}, "b": "t"}], limit=5) == ["a", "b"]
    assert len(scalar_columns([{f"k{i}": i for i in range(20)}])) == 6


# --- commands over the wire -----------------------------------------------------


def test_endpoints_lists_every_endpoint_with_an_example() -> None:
    runner, state = make_runner(Recorder(httpx.Response(500)))

    result = runner.invoke(app, ["plugins", "endpoints", "-f", "json"], obj=state)

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert {(r["plugin"], r["endpoint"], r["method"]) for r in rows} == {
        ("phoenix", "projects", "GET"),
        ("phoenix", "spans", "GET"),
        ("phoenix", "spans", "POST"),
        ("is-it-christmas", "is-it-christmas", "GET"),
    }


def test_endpoints_for_an_unknown_or_disabled_plugin_is_an_error() -> None:
    disabled = {"schema_version": 1, "plugins": [{**CHRISTMAS, "enabled": False}]}
    runner, state = make_runner(Recorder(httpx.Response(500), registry=disabled))

    assert runner.invoke(app, ["plugins", "endpoints", "nope"], obj=state).exit_code != 0
    result = runner.invoke(app, ["plugins", "endpoints", "is-it-christmas"], obj=state)
    assert result.exit_code != 0
    assert "disabled" in result.output


def test_call_sends_a_get_through_the_gateway_with_workspace_and_query() -> None:
    recorder = Recorder(httpx.Response(200, json={"data": []}))
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["plugins", "call", "phoenix", "spans", "-w", "my-ws", "-q", "project=demo", "-q", "limit=5", "-f", "json"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    [call] = recorder.plugin_calls
    assert (call.method, call.url.path) == ("GET", "/apis/plugins/phoenix/v2/workspaces/my-ws/spans")
    assert dict(call.url.params) == {"project": "demo", "limit": "5"}


def test_call_posts_a_json_body_for_a_write_endpoint() -> None:
    recorder = Recorder(httpx.Response(200, json={"accepted": 1}))
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "plugins",
            "call",
            "phoenix",
            "spans",
            "-X",
            "POST",
            "-q",
            "project=demo",
            "-d",
            '{"spans": []}',
            "-f",
            "json",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    [call] = recorder.plugin_calls
    assert (call.method, call.url.path) == ("POST", "/apis/plugins/phoenix/v2/workspaces/default/spans")
    assert json.loads(call.content) == {"spans": []}
    assert call.headers["content-type"] == "application/json"


def test_call_shows_a_single_object_as_aligned_key_values() -> None:
    answer = {"is_christmas": False, "days_until": 36, "message": "No. 36 days to go."}
    runner, state = make_runner(Recorder(httpx.Response(200, json=answer)))

    result = runner.invoke(app, ["plugins", "call", "is-it-christmas", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    assert "is_christmas  false" in result.output
    assert "message       No. 36 days to go." in result.output


def test_call_shows_a_paged_list_as_a_table() -> None:
    page = {
        "data": [{"name": "support_agent.run", "status_code": "OK"}, {"name": "lookup_order", "status_code": "ERROR"}]
    }
    runner, state = make_runner(Recorder(httpx.Response(200, json=page)))

    result = runner.invoke(app, ["plugins", "call", "phoenix", "spans", "-f", "table", "--no-truncate"], obj=state)

    assert result.exit_code == 0, result.output
    for expected in ("name", "status_code", "support_agent.run", "lookup_order", "ERROR"):
        assert expected in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["nope"], "No external plugin named 'nope'"),
        (["phoenix", "nope"], "no endpoint 'nope'"),
        (["phoenix"], "several endpoints"),
        (["phoenix", "projects", "-d", "{}"], "cannot be sent with GET"),
    ],
)
def test_call_refuses_bad_requests_before_contacting_the_plugin(args: list[str], message: str) -> None:
    recorder = Recorder(httpx.Response(200, json={}))
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "call", *args], obj=state)

    assert result.exit_code != 0
    assert message in " ".join(result.output.split())
    assert recorder.plugin_calls == []


def test_call_reports_an_upstream_failure() -> None:
    runner, state = make_runner(Recorder(httpx.Response(412, json={"detail": "Secret missing"})))

    result = runner.invoke(app, ["plugins", "call", "phoenix", "projects"], obj=state)

    assert result.exit_code != 0
