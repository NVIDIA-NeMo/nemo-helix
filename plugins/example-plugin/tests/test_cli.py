# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo example middleware-configs`` — wire-level CLI tests.

The command runs against a stand-in ``nemo`` CLI state whose typed client
talks to a recording ``httpx.MockTransport``, so each test pins the request
the command sent and what it printed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import pytest
import typer
from click.testing import Result
from nemo_example_plugin.cli import ExampleCLI
from nemo_helix_plugin.client.client import NemoClient
from typer.testing import CliRunner

BASE_URL = "http://nhx.test"
CONFIGS = "/apis/example/v2/workspaces/team-a/middleware-configs"

ClientT = TypeVar("ClientT", bound=NemoClient)

CONFIG = {
    "name": "no-secrets",
    "workspace": "team-a",
    "blocked_keywords": ["password"],
    "block_message": "Nope.",
    "id": "cfg-1",
    "created_at": "2026-09-01T12:00:00Z",
    "updated_at": "2026-09-02T12:00:00Z",
}


@dataclass
class _Api:
    responses: list[httpx.Response] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


@dataclass
class _State:
    """The slice of the ``nemo`` CLI state (``CLIState``) these commands use."""

    api: _Api
    output_format: str = "json"

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        http_client = httpx.Client(transport=httpx.MockTransport(self.api))
        return client_cls.from_client(NemoClient(base_url=BASE_URL, http_client=http_client))

    def get_workspace(self) -> str | None:
        return "team-a"

    def get_base_url(self, default: str | None = None) -> str | None:
        return BASE_URL

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or self.output_format

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


@pytest.fixture
def api() -> _Api:
    return _Api()


@pytest.fixture
def app() -> typer.Typer:
    return ExampleCLI().get_cli()


def _invoke(app: typer.Typer, api: _Api, *args: str, output_format: str = "json") -> Result:
    return CliRunner().invoke(app, ["middleware-configs", *args], obj=_State(api, output_format))


def _body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def test_create_posts_only_the_given_fields(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(201, json=CONFIG))

    result = _invoke(app, api, "create", "no-secrets", "--blocked-keywords", "password, token")

    assert result.exit_code == 0, result.output
    (request,) = api.requests
    assert (request.method, request.url.path) == ("POST", CONFIGS)
    # Unset fields stay off the wire, so the server applies its own defaults.
    assert _body(request) == {"name": "no-secrets", "blocked_keywords": ["password", "token"]}
    printed = json.loads(result.stdout)
    assert printed["id"] == "cfg-1"
    assert printed["created_at"] == "2026-09-01T12:00:00Z"


def test_list_sends_the_query_and_renders_a_table(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(200, json=[CONFIG]))

    result = _invoke(app, api, "list", "--page-size", "5", "--sort", "name", output_format="table")

    assert result.exit_code == 0, result.output
    assert dict(api.requests[0].url.params) == {"page_size": "5", "sort": "name"}
    assert "no-secrets" in result.stdout
    assert "blocked_keywords" in result.stdout


def test_list_json(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(200, json=[CONFIG]))

    result = _invoke(app, api, "list", "-f", "json")

    assert result.exit_code == 0, result.output
    assert [config["name"] for config in json.loads(result.stdout)] == ["no-secrets"]


def test_get_supports_yaml(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(200, json=CONFIG))

    result = _invoke(app, api, "get", "no-secrets", "--output-format", "yaml")

    assert result.exit_code == 0, result.output
    assert api.requests[0].url.path == f"{CONFIGS}/no-secrets"
    assert "name: no-secrets" in result.stdout


def test_update_patches_only_the_given_fields(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(200, json=CONFIG))

    result = _invoke(app, api, "update", "no-secrets", "--block-message", "Nope.")

    assert result.exit_code == 0, result.output
    (request,) = api.requests
    assert request.method == "PATCH"
    assert _body(request) == {"block_message": "Nope."}


def test_delete(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(204))

    result = _invoke(app, api, "delete", "no-secrets", "--yes")

    assert result.exit_code == 0, result.output
    assert (api.requests[0].method, api.requests[0].url.path) == ("DELETE", f"{CONFIGS}/no-secrets")
    assert "Deleted 'team-a/no-secrets'." in result.stdout


def test_missing_config_is_a_not_found_error(app: typer.Typer, api: _Api) -> None:
    api.responses.append(httpx.Response(404, json={"detail": "Middleware config 'nope' not found."}))

    result = _invoke(app, api, "get", "nope")

    assert result.exit_code == 3
    assert "not found" in result.output.lower()


@pytest.mark.parametrize(
    ("args", "call"),
    [
        (["create", "no-secrets"], "client.create_middleware_config("),
        (["list"], "client.list_middleware_configs("),
        (["get", "no-secrets"], "client.get_middleware_config("),
        (["update", "no-secrets", "--block-message", "x"], "client.update_middleware_config("),
    ],
)
def test_code_output_prints_the_typed_client_call(app: typer.Typer, api: _Api, args: list[str], call: str) -> None:
    result = _invoke(app, api, *args, "-f", "code")

    assert result.exit_code == 0, result.output
    assert api.requests == []
    assert "from nemo_example_plugin.client import ExampleClient" in result.stdout
    assert call in result.stdout


def test_platform_comes_from_the_cli_context(app: typer.Typer, api: _Api) -> None:
    result = _invoke(app, api, "list", "--base-url", "http://elsewhere")

    assert result.exit_code == 2
