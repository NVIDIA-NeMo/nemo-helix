# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditor plugin CLI's CRUD subcommands."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import click
import httpx
import pytest
import typer
from nemo_auditor.cli import AuditorPluginCLI
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app():
    return AuditorPluginCLI().get_cli()


def _install_mock_transport(
    handler,
) -> tuple[list[httpx.Request], AbstractContextManager[Any]]:
    """Patch httpx.Client so every request is routed to ``handler``.

    Returns ``(captured_requests, patch_context_manager)``. Tests use the
    captured list after invoking the CLI to assert on the requests that
    were issued, and use the context manager in a ``with`` block to scope
    the patch.
    """
    captured: list[httpx.Request] = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(_wrapped)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    return captured, patch("nemo_auditor.cli.httpx.Client", _factory)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_config_with_data_file_posts_to_plugin_route(self, runner, app, tmp_path: Path) -> None:
        body = {"description": "test", "system": {"lite": True}}
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps(body))

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                201,
                json={"name": "cfg-1", "workspace": "default", **body},
            )

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "create", "cfg-1", "--data-file", str(f)])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert len(captured) == 1
        req = captured[0]
        assert req.method == "POST"
        assert req.url.path == "/apis/auditor/v2/workspaces/default/configs"
        assert json.loads(req.content) == {"name": "cfg-1", **body}
        assert "cfg-1" in result.stdout

    def test_create_target_with_inline_data(self, runner, app) -> None:
        inline = '{"type":"nim","model":"meta/llama-3.1-8b-instruct"}'

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"name": "tgt-1"})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(
                app,
                ["targets", "create", "tgt-1", "--data", inline, "--workspace", "prod"],
            )

        assert result.exit_code == 0, result.stdout + result.stderr
        req = captured[0]
        assert req.url.path == "/apis/auditor/v2/workspaces/prod/targets"
        assert json.loads(req.content) == {
            "name": "tgt-1",
            "type": "nim",
            "model": "meta/llama-3.1-8b-instruct",
        }

    def test_create_without_data_exits_2(self, runner, app) -> None:
        result = runner.invoke(app, ["configs", "create", "cfg-1"])
        assert result.exit_code == 2
        assert "--data-file" in result.stderr or "--data" in result.stderr

    def test_create_with_both_data_sources_exits_2(self, runner, app, tmp_path: Path) -> None:
        f = tmp_path / "cfg.json"
        f.write_text("{}")
        result = runner.invoke(app, ["configs", "create", "cfg-1", "--data-file", str(f), "--data", "{}"])
        assert result.exit_code == 2
        assert "not both" in result.stderr

    def test_create_with_invalid_json_exits_2(self, runner, app) -> None:
        result = runner.invoke(app, ["configs", "create", "cfg-1", "--data", "{not-json"])
        assert result.exit_code == 2
        assert "invalid JSON" in result.stderr

    def test_create_with_non_object_json_exits_2(self, runner, app) -> None:
        result = runner.invoke(app, ["configs", "create", "cfg-1", "--data", "[1,2,3]"])
        assert result.exit_code == 2
        assert "JSON object" in result.stderr

    def test_create_surfaces_server_validation_error(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422,
                json={
                    "detail": [
                        {"type": "missing", "loc": ["body", "model"], "msg": "Field required"},
                    ]
                },
            )

        _, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["targets", "create", "bad", "--data", '{"type":"nim"}'])

        assert result.exit_code == 1
        assert "422" in result.stderr
        assert "model" in result.stderr


# ---------------------------------------------------------------------------
# list / get
# ---------------------------------------------------------------------------


class TestRead:
    def test_list_hits_collection_endpoint(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [], "pagination": {"total_results": 0}})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "list"])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0].method == "GET"
        assert captured[0].url.path == "/apis/auditor/v2/workspaces/default/configs"

    def test_get_hits_named_endpoint(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"name": "cfg-1"})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "get", "cfg-1"])

        assert result.exit_code == 0
        assert captured[0].url.path == "/apis/auditor/v2/workspaces/default/configs/cfg-1"
        assert "cfg-1" in result.stdout

    def test_get_404_exits_1(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "Entity not found"})

        _, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "get", "missing"])

        assert result.exit_code == 1
        assert "404" in result.stderr
        assert "Entity not found" in result.stderr
        assert "Request: GET http://localhost:8080/apis/auditor/v2/workspaces/default/configs/missing" in result.stderr
        assert "Target:" not in result.stderr
        assert "Check the resource name and workspace" in result.stderr


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_sends_put_with_flat_body(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"name": "cfg-1", "description": "new"})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(
                app,
                ["configs", "update", "cfg-1", "--data", '{"description":"new"}'],
            )

        assert result.exit_code == 0, result.stdout + result.stderr
        req = captured[0]
        assert req.method == "PUT"
        assert req.url.path == "/apis/auditor/v2/workspaces/default/configs/cfg-1"
        assert json.loads(req.content) == {"description": "new"}

    def test_update_without_data_exits_2(self, runner, app) -> None:
        result = runner.invoke(app, ["configs", "update", "cfg-1"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_hits_named_endpoint_and_prints_confirmation(self, runner, app) -> None:
        # Plugin route returns 204 with no body for delete.
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(204)

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["targets", "delete", "tgt-1", "--workspace", "prod"])

        assert result.exit_code == 0, result.stdout + result.stderr
        req = captured[0]
        assert req.method == "DELETE"
        assert req.url.path == "/apis/auditor/v2/workspaces/prod/targets/tgt-1"
        assert "tgt-1" in result.stdout
        assert "deleted" in result.stdout.lower()

    def test_delete_404_exits_1(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "Entity not found"})

        _, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "delete", "missing"])

        assert result.exit_code == 1
        assert "404" in result.stderr
        assert "Entity not found" in result.stderr
        assert (
            "Request: DELETE http://localhost:8080/apis/auditor/v2/workspaces/default/configs/missing" in result.stderr
        )
        assert "Target:" not in result.stderr


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


class TestStructure:
    def test_help_lists_both_subgroups(self, runner, app) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "configs" in result.stdout
        assert "targets" in result.stdout

    def test_connection_error_exits_1(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=req)

        _, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "list"])

        assert result.exit_code == 1
        assert "Error: GET auditor API failed: connection refused" in result.stderr
        assert "Request: GET http://localhost:8080/apis/auditor/v2/workspaces/default/configs" in result.stderr
        assert "Target: auditor API route /apis/auditor/v2/workspaces/default/configs" in result.stderr
        assert "nemo config view" in result.stderr

    def test_base_url_override_is_used(self, runner, app) -> None:
        captured: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return httpx.Response(200, json={"data": []})

        _, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(app, ["configs", "list", "--base-url", "http://custom:9999"])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert str(captured[0].url).startswith("http://custom:9999/")


# ---------------------------------------------------------------------------
# workspace resolution
# ---------------------------------------------------------------------------


class _ContextState:
    """Minimal stand-in for ``CLIContext`` with a non-'default' workspace."""

    def __init__(self, workspace: str | None = "my-team-ws") -> None:
        self._workspace = workspace

    def get_workspace(self) -> str | None:
        return self._workspace


def _app_with_state(app: typer.Typer, state: object | None) -> typer.Typer:
    """Wrap *app* in a parent group whose callback seeds ``ctx.obj`` with *state*.

    Mirrors how the top-level ``nemo`` CLI installs its state object, which is
    what ``resolve_cli_workspace`` reads off the ``typer.Context`` the command
    receives.
    """
    parent = typer.Typer()

    @parent.callback()
    def _root(ctx: typer.Context) -> None:
        ctx.obj = state

    parent.add_typer(app, name="auditor")
    return parent


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

    def test_list_uses_context_workspace(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(_app_with_state(app, _ContextState()), ["auditor", "configs", "list"])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0].url.path == "/apis/auditor/v2/workspaces/my-team-ws/configs"

    def test_explicit_flag_wins_over_context(self, runner, app) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(
                _app_with_state(app, _ContextState()),
                ["auditor", "configs", "list", "--workspace", "team-alpha"],
            )

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0].url.path == "/apis/auditor/v2/workspaces/team-alpha/configs"

    def test_falls_back_to_default_without_state(self, runner, app, monkeypatch) -> None:
        monkeypatch.delenv("NHX_WORKSPACE", raising=False)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(_app_with_state(app, None), ["auditor", "configs", "list"])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0].url.path == "/apis/auditor/v2/workspaces/default/configs"

    def test_falls_back_to_env_without_state(self, runner, app, monkeypatch) -> None:
        monkeypatch.setenv("NHX_WORKSPACE", "env-ws")

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(_app_with_state(app, None), ["auditor", "configs", "list"])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0].url.path == "/apis/auditor/v2/workspaces/env-ws/configs"

    @pytest.mark.parametrize(
        ("argv", "method", "expected_path"),
        [
            (
                ["auditor", "configs", "create", "cfg-1", "--data", "{}"],
                "POST",
                "/apis/auditor/v2/workspaces/my-team-ws/configs",
            ),
            (["auditor", "configs", "list"], "GET", "/apis/auditor/v2/workspaces/my-team-ws/configs"),
            (["auditor", "configs", "get", "cfg-1"], "GET", "/apis/auditor/v2/workspaces/my-team-ws/configs/cfg-1"),
            (
                ["auditor", "configs", "update", "cfg-1", "--data", "{}"],
                "PUT",
                "/apis/auditor/v2/workspaces/my-team-ws/configs/cfg-1",
            ),
            (
                ["auditor", "configs", "delete", "cfg-1"],
                "DELETE",
                "/apis/auditor/v2/workspaces/my-team-ws/configs/cfg-1",
            ),
            (
                ["auditor", "targets", "create", "tgt-1", "--data", "{}"],
                "POST",
                "/apis/auditor/v2/workspaces/my-team-ws/targets",
            ),
            (["auditor", "targets", "list"], "GET", "/apis/auditor/v2/workspaces/my-team-ws/targets"),
            (["auditor", "targets", "get", "tgt-1"], "GET", "/apis/auditor/v2/workspaces/my-team-ws/targets/tgt-1"),
            (
                ["auditor", "targets", "update", "tgt-1", "--data", "{}"],
                "PUT",
                "/apis/auditor/v2/workspaces/my-team-ws/targets/tgt-1",
            ),
            (
                ["auditor", "targets", "delete", "tgt-1"],
                "DELETE",
                "/apis/auditor/v2/workspaces/my-team-ws/targets/tgt-1",
            ),
        ],
    )
    def test_every_crud_verb_honors_context_workspace(
        self, runner, app, argv: list[str], method: str, expected_path: str
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        captured, ctx = _install_mock_transport(handler)
        with ctx:
            result = runner.invoke(_app_with_state(app, _ContextState()), argv)

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0].method == method
        assert captured[0].url.path == expected_path

    def test_no_workspace_option_declares_a_literal_default(self, app) -> None:
        """A literal Typer default makes the active context unable to win."""
        params = _collect_workspace_params(app)
        assert params, "no --workspace option found; this guard would be vacuous"
        assert len(params) == 10, f"expected 10 --workspace options, found {sorted(params)}"
        offenders = {path: param.default for path, param in params.items() if param.default is not None}
        assert not offenders, f"--workspace must default to None so the CLI context can win: {offenders}"
