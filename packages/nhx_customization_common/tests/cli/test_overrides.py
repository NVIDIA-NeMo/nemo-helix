# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared customization submit override.

Focus: the hand-written ``submit`` verb installed by
``apply_job_cli_overrides`` must resolve ``--workspace`` the same way the
auto-generated verbs do — explicit flag > ``$NHX_WORKSPACE`` > the active
CLI context's configured workspace > ``"default"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from nhx.customization_common.cli.overrides import apply_job_cli_overrides
from typer.testing import CliRunner


class _ContextState:
    """Minimal stand-in for ``CLIContext`` with a non-'default' workspace."""

    def __init__(self, workspace: str | None = "my-team-ws") -> None:
        self._workspace = workspace

    def get_workspace(self) -> str | None:
        return self._workspace


def _build_app(calls: list[dict[str, object]]) -> typer.Typer:
    """Generated-shaped app (stub ``run``/``submit``) with the overrides applied."""
    app = typer.Typer(no_args_is_help=True)

    @app.command("run")
    def generated_run() -> None:  # pragma: no cover - should be dropped
        raise AssertionError("overrides should remove generated run commands")

    @app.command("submit")
    def generated_submit(typer_ctx: typer.Context, **kwargs: object) -> None:
        calls.append(dict(kwargs))

    apply_job_cli_overrides(
        app,
        backend="plugin",
        load_job_json=lambda path: json.dumps(json.loads(Path(path).read_text())),
        job_json_help="Path to the job JSON.",
    )
    return app


def _app_with_state(app: typer.Typer, state: object | None) -> typer.Typer:
    """Wrap *app* in a parent group whose callback seeds ``ctx.obj`` with *state*."""
    parent = typer.Typer()

    @parent.callback()
    def _root(ctx: typer.Context) -> None:
        ctx.obj = state

    parent.add_typer(app, name="plugin")
    return parent


@pytest.fixture
def job_file(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"model": {"name": "m"}}))
    return path


class TestSubmitWorkspaceResolution:
    def test_omitted_flag_uses_active_context_workspace(self, job_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NHX_WORKSPACE", raising=False)
        calls: list[dict[str, object]] = []
        parent = _app_with_state(_build_app(calls), _ContextState("my-team-ws"))

        result = CliRunner().invoke(parent, ["plugin", "submit", str(job_file)])

        assert result.exit_code == 0, result.output
        assert calls[0]["workspace"] == "my-team-ws"

    def test_explicit_flag_beats_context(self, job_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NHX_WORKSPACE", raising=False)
        calls: list[dict[str, object]] = []
        parent = _app_with_state(_build_app(calls), _ContextState("my-team-ws"))

        result = CliRunner().invoke(parent, ["plugin", "submit", str(job_file), "--workspace", "acme-corp"])

        assert result.exit_code == 0, result.output
        assert calls[0]["workspace"] == "acme-corp"

    def test_explicit_default_beats_context(self, job_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--workspace default`` is honored, not silently replaced by the context."""
        monkeypatch.delenv("NHX_WORKSPACE", raising=False)
        calls: list[dict[str, object]] = []
        parent = _app_with_state(_build_app(calls), _ContextState("my-team-ws"))

        result = CliRunner().invoke(parent, ["plugin", "submit", str(job_file), "-w", "default"])

        assert result.exit_code == 0, result.output
        assert calls[0]["workspace"] == "default"

    def test_env_var_used_when_no_state(self, job_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
        calls: list[dict[str, object]] = []
        parent = _app_with_state(_build_app(calls), None)

        result = CliRunner().invoke(parent, ["plugin", "submit", str(job_file)])

        assert result.exit_code == 0, result.output
        assert calls[0]["workspace"] == "env-ws"

    def test_falls_back_to_default_without_cli_state(self, job_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NHX_WORKSPACE", raising=False)
        calls: list[dict[str, object]] = []
        parent = _app_with_state(_build_app(calls), None)

        result = CliRunner().invoke(parent, ["plugin", "submit", str(job_file)])

        assert result.exit_code == 0, result.output
        assert calls[0]["workspace"] == "default"
