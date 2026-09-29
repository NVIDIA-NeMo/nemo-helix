# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditor plugin CLI's CRUD subcommands.

Commands run against a stand-in ``nemo`` CLI state whose ``AuditorClient``
talks to a recording ``httpx.MockTransport``, so each test pins the request
the command sent and what it printed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
import httpx
import pytest
import typer
from click.testing import Result
from nemo_auditor.cli import AuditorPluginCLI
from nemo_auditor.jobs.audit import AuditJob
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.commands import add_job_commands
from typer.testing import CliRunner

BASE_URL = "http://nhx.test"
CONFIGS = "/apis/auditor/v2/workspaces/default/configs"
TARGET_BODY = '{"type": "nim", "model": "meta/llama-3.1-8b-instruct"}'

ClientT = TypeVar("ClientT", bound=NemoClient)
Handler = Callable[[httpx.Request], httpx.Response]


class _State:
    """The slice of the ``nemo`` CLI state (``CLIState``) these commands use."""

    def __init__(self, handler: Handler, *, workspace: str | None = None, output_format: str = "json") -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler
        self._workspace = workspace
        self._output_format = output_format

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        http_client = httpx.Client(transport=httpx.MockTransport(self._record))
        return client_cls.from_client(NemoClient(base_url=BASE_URL, http_client=http_client))

    def get_workspace(self) -> str | None:
        return self._workspace

    def get_base_url(self, default: str | None = None) -> str | None:
        return BASE_URL

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or self._output_format

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


def _ok(body: Any = None, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, json=body if body is not None else {"name": "x"})


def _page(items: list[dict[str, Any]], *, page: int = 1, total_pages: int = 1) -> dict[str, Any]:
    return {
        "data": items,
        "pagination": {
            "page": page,
            "page_size": 10,
            "current_page_size": len(items),
            "total_pages": total_pages,
            "total_results": total_pages * len(items),
        },
    }


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)


@pytest.fixture
def app() -> typer.Typer:
    return AuditorPluginCLI().get_cli()


def _invoke(app: typer.Typer, state: _State, *args: str) -> Result:
    return CliRunner().invoke(app, list(args), obj=state)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_config_with_data_file_posts_to_plugin_route(self, app: typer.Typer, tmp_path: Path) -> None:
        body = {"description": "test", "system": {"lite": True}}
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps(body))
        state = _State(_ok({"name": "cfg-1", "workspace": "default", **body}, 201))

        result = _invoke(app, state, "configs", "create", "cfg-1", "--data-file", str(f))

        assert result.exit_code == 0, result.output
        (request,) = state.requests
        assert (request.method, request.url.path) == ("POST", CONFIGS)
        assert json.loads(request.content) == {"name": "cfg-1", **body}
        assert json.loads(result.stdout)["name"] == "cfg-1"

    def test_create_target_with_inline_data(self, app: typer.Typer) -> None:
        state = _State(_ok({"name": "tgt-1"}, 201))

        result = _invoke(app, state, "targets", "create", "tgt-1", "-d", TARGET_BODY, "--workspace", "prod")

        assert result.exit_code == 0, result.output
        request = state.requests[0]
        assert request.url.path == "/apis/auditor/v2/workspaces/prod/targets"
        assert json.loads(request.content) == {
            "name": "tgt-1",
            "type": "nim",
            "model": "meta/llama-3.1-8b-instruct",
        }

    def test_create_without_data_exits_2(self, app: typer.Typer) -> None:
        result = _invoke(app, _State(_ok()), "configs", "create", "cfg-1")
        assert result.exit_code == 2
        assert "--data-file" in result.stderr or "--data" in result.stderr

    def test_create_with_both_data_sources_exits_2(self, app: typer.Typer, tmp_path: Path) -> None:
        f = tmp_path / "cfg.json"
        f.write_text("{}")
        result = _invoke(app, _State(_ok()), "configs", "create", "cfg-1", "--data-file", str(f), "--data", "{}")
        assert result.exit_code == 2
        assert "not both" in result.stderr

    def test_create_with_invalid_json_exits_2(self, app: typer.Typer) -> None:
        result = _invoke(app, _State(_ok()), "configs", "create", "cfg-1", "--data", "{not-json")
        assert result.exit_code == 2
        assert "invalid JSON" in result.stderr

    def test_create_with_non_object_json_exits_2(self, app: typer.Typer) -> None:
        result = _invoke(app, _State(_ok()), "configs", "create", "cfg-1", "--data", "[1,2,3]")
        assert result.exit_code == 2
        assert "JSON object" in result.stderr

    def test_unknown_fields_are_rejected_before_sending(self, app: typer.Typer) -> None:
        state = _State(_ok())

        result = _invoke(app, state, "configs", "create", "cfg-1", "--data", '{"descripton": "typo"}')

        assert result.exit_code == 2
        assert "descripton" in result.output
        assert state.requests == []

    def test_create_surfaces_server_validation_error(self, app: typer.Typer) -> None:
        detail = {"detail": [{"type": "value_error", "loc": ["body", "model"], "msg": "Unknown model"}]}
        state = _State(_ok(detail, 422))

        result = _invoke(app, state, "targets", "create", "bad", "--data", TARGET_BODY)

        assert result.exit_code == 3
        assert "422" in result.output
        assert "model" in result.output


# ---------------------------------------------------------------------------
# list / get
# ---------------------------------------------------------------------------


class TestRead:
    def test_list_hits_collection_endpoint(self, app: typer.Typer) -> None:
        state = _State(_ok(_page([{"name": "cfg-1", "description": "d"}])))

        result = _invoke(app, state, "configs", "list")

        assert result.exit_code == 0, result.output
        assert (state.requests[0].method, state.requests[0].url.path) == ("GET", CONFIGS)
        assert [config["name"] for config in json.loads(result.stdout)["data"]] == ["cfg-1"]

    def test_list_renders_a_table(self, app: typer.Typer) -> None:
        state = _State(_ok(_page([{"name": "tgt-1", "type": "nim", "model": "m"}])), output_format="table")

        result = _invoke(app, state, "targets", "list")

        assert result.exit_code == 0, result.output
        header = result.stdout.splitlines()[1]
        for column in ("name", "type", "model", "created_at"):
            assert column in header

    def test_list_all_pages(self, app: typer.Typer) -> None:
        pages = [_page([{"name": "a"}], page=1, total_pages=2), _page([{"name": "b"}], page=2, total_pages=2)]
        state = _State(lambda request: httpx.Response(200, json=pages[int(request.url.params.get("page", 1)) - 1]))

        result = _invoke(app, state, "configs", "list", "--all-pages")

        assert result.exit_code == 0, result.output
        assert [config["name"] for config in json.loads(result.stdout)["data"]] == ["a", "b"]

    def test_list_warns_when_more_pages_exist(self, app: typer.Typer) -> None:
        state = _State(_ok(_page([{"name": "a"}], total_pages=2)))

        result = _invoke(app, state, "configs", "list")

        assert result.exit_code == 0, result.output
        assert "--all-pages" in result.stderr

    def test_get_hits_named_endpoint(self, app: typer.Typer) -> None:
        state = _State(_ok({"name": "cfg-1"}))

        result = _invoke(app, state, "configs", "get", "cfg-1", "-f", "yaml")

        assert result.exit_code == 0, result.output
        assert state.requests[0].url.path == f"{CONFIGS}/cfg-1"
        assert "name: cfg-1" in result.stdout

    def test_get_404_is_a_not_found_error(self, app: typer.Typer) -> None:
        state = _State(_ok({"detail": "Entity not found"}, 404))

        result = _invoke(app, state, "configs", "get", "missing")

        assert result.exit_code == 3
        assert "Entity not found" in result.output


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_sends_put_with_only_the_given_fields(self, app: typer.Typer) -> None:
        state = _State(_ok({"name": "cfg-1", "description": "new"}))

        result = _invoke(app, state, "configs", "update", "cfg-1", "--data", '{"description":"new"}')

        assert result.exit_code == 0, result.output
        request = state.requests[0]
        assert (request.method, request.url.path) == ("PUT", f"{CONFIGS}/cfg-1")
        assert json.loads(request.content) == {"description": "new"}

    def test_update_without_data_exits_2(self, app: typer.Typer) -> None:
        result = _invoke(app, _State(_ok()), "configs", "update", "cfg-1")
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_hits_named_endpoint_and_prints_confirmation(self, app: typer.Typer) -> None:
        state = _State(lambda request: httpx.Response(204))

        result = _invoke(app, state, "targets", "delete", "tgt-1", "--workspace", "prod")

        assert result.exit_code == 0, result.output
        request = state.requests[0]
        assert (request.method, request.url.path) == ("DELETE", "/apis/auditor/v2/workspaces/prod/targets/tgt-1")
        assert "Target 'tgt-1' deleted." in result.stdout

    def test_delete_404_is_a_not_found_error(self, app: typer.Typer) -> None:
        state = _State(_ok({"detail": "Entity not found"}, 404))

        result = _invoke(app, state, "configs", "delete", "missing")

        assert result.exit_code == 3
        assert "Entity not found" in result.output


# ---------------------------------------------------------------------------
# output formats and routing
# ---------------------------------------------------------------------------


class TestOutput:
    @pytest.mark.parametrize(
        ("args", "call"),
        [
            (["configs", "create", "cfg-1", "--data", "{}"], "client.create_audit_config("),
            (["configs", "list"], "client.list_audit_configs("),
            (["targets", "get", "tgt-1"], "client.get_audit_target("),
            (["targets", "update", "tgt-1", "--data", TARGET_BODY], "client.update_audit_target("),
        ],
    )
    def test_code_output_prints_the_typed_client_call(self, app: typer.Typer, args: list[str], call: str) -> None:
        state = _State(_ok())

        result = _invoke(app, state, *args, "-f", "code")

        assert result.exit_code == 0, result.output
        assert state.requests == []
        assert "from nemo_helix_plugin.auditor.client import AuditorClient" in result.stdout
        assert call in result.stdout

    def test_data_file_no_longer_has_a_short_flag(self, app: typer.Typer, tmp_path: Path) -> None:
        """``-f`` is the standard output-format flag."""
        f = tmp_path / "cfg.json"
        f.write_text("{}")

        result = _invoke(app, _State(_ok()), "configs", "create", "cfg-1", "-f", str(f))

        assert result.exit_code == 2

    def test_platform_comes_from_the_cli_context(self, app: typer.Typer) -> None:
        result = _invoke(app, _State(_ok()), "configs", "list", "--base-url", "http://custom:9999")

        assert result.exit_code == 2
        assert "No such option" in result.output

    def test_audit_job_uses_flat_command_shape(self) -> None:
        cli = AuditorPluginCLI()
        app = cli.get_cli()
        add_job_commands(app, {"auditor.audit": AuditJob}, cli=cli)
        audit_command = typer.main.get_command(app).commands["audit"]
        assert isinstance(audit_command, click.Group)
        assert "explain" in audit_command.commands
        assert "run" not in audit_command.commands
        assert "submit" not in audit_command.commands

        result = CliRunner().invoke(app, ["audit", "--help"])

        assert result.exit_code == 0, result.output
        assert "--spec" in result.stdout

        legacy_result = CliRunner().invoke(app, ["audit", "submit"])
        assert legacy_result.exit_code != 0

    def test_help_lists_both_subgroups(self, app: typer.Typer) -> None:
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "configs" in result.stdout
        assert "targets" in result.stdout


# ---------------------------------------------------------------------------
# workspace resolution
# ---------------------------------------------------------------------------


def _collect_workspace_params(app: typer.Typer) -> dict[str, click.Parameter]:
    """Map ``"<command path>"`` -> the ``--workspace`` param of every subcommand."""
    found: dict[str, click.Parameter] = {}

    def _walk(command: click.Command, path: str) -> None:
        for param in command.params:
            if "--workspace" in getattr(param, "opts", []):
                found[path or command.name or "<root>"] = param
        if isinstance(command, click.Group):
            for name, sub in command.commands.items():
                _walk(sub, f"{path} {name}".strip())

    _walk(typer.main.get_command(app), "")
    return found


class TestWorkspaceResolution:
    """The auditor CRUD verbs must not fall back to a hardcoded 'default'.

    Regression guard: ``--workspace`` used to be declared with a literal
    ``"default"`` Typer default, so the command body could never tell an
    omitted flag from an explicit one and the workspace selected via
    ``nemo config use-context`` (or ``$NHX_WORKSPACE``) was silently
    discarded.
    """

    def test_explicit_flag_wins_over_context(self, app: typer.Typer) -> None:
        state = _State(_ok(_page([])), workspace="my-team-ws")

        result = _invoke(app, state, "configs", "list", "--workspace", "team-alpha")

        assert result.exit_code == 0, result.output
        assert state.requests[0].url.path == "/apis/auditor/v2/workspaces/team-alpha/configs"

    def test_a_context_without_a_workspace_falls_back_to_default(self, app: typer.Typer) -> None:
        state = _State(_ok(_page([])))

        result = _invoke(app, state, "configs", "list")

        assert result.exit_code == 0, result.output
        assert state.requests[0].url.path == CONFIGS

    def test_nhx_workspace_env_applies_when_the_context_has_none(
        self, app: typer.Typer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
        state = _State(_ok(_page([])))

        result = _invoke(app, state, "configs", "list")

        assert result.exit_code == 0, result.output
        assert state.requests[0].url.path == "/apis/auditor/v2/workspaces/env-ws/configs"

    @pytest.mark.parametrize(
        ("argv", "method", "expected_path"),
        [
            (["configs", "create", "cfg-1", "--data", "{}"], "POST", "/configs"),
            (["configs", "list"], "GET", "/configs"),
            (["configs", "get", "cfg-1"], "GET", "/configs/cfg-1"),
            (["configs", "update", "cfg-1", "--data", "{}"], "PUT", "/configs/cfg-1"),
            (["configs", "delete", "cfg-1"], "DELETE", "/configs/cfg-1"),
            (["targets", "create", "tgt-1", "--data", TARGET_BODY], "POST", "/targets"),
            (["targets", "list"], "GET", "/targets"),
            (["targets", "get", "tgt-1"], "GET", "/targets/tgt-1"),
            (["targets", "update", "tgt-1", "--data", TARGET_BODY], "PUT", "/targets/tgt-1"),
            (["targets", "delete", "tgt-1"], "DELETE", "/targets/tgt-1"),
        ],
    )
    def test_every_crud_verb_honors_context_workspace(
        self, app: typer.Typer, argv: list[str], method: str, expected_path: str
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE":
                return httpx.Response(204)
            if request.method == "GET" and request.url.path.endswith(("/configs", "/targets")):
                return httpx.Response(200, json=_page([]))
            return httpx.Response(200, json={"name": "x"})

        state = _State(handler, workspace="my-team-ws")

        result = _invoke(app, state, *argv)

        assert result.exit_code == 0, result.output
        assert state.requests[0].method == method
        assert state.requests[0].url.path == f"/apis/auditor/v2/workspaces/my-team-ws{expected_path}"

    def test_no_workspace_option_declares_a_literal_default(self, app: typer.Typer) -> None:
        """A literal Typer default makes the active context unable to win."""
        params = _collect_workspace_params(app)
        assert params, "no --workspace option found; this guard would be vacuous"
        assert len(params) == 10, f"expected 10 --workspace options, found {sorted(params)}"
        offenders = {path: param.default for path, param in params.items() if param.default is not None}
        assert not offenders, f"--workspace must default to None so the CLI context can win: {offenders}"
