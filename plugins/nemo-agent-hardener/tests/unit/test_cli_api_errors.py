# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commands report a backend or pre-check failure as its message and a non-zero exit, never a traceback."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_agent_hardener_plugin.cli import _shared, war_game
from nemo_agent_hardener_plugin.cli import main as cli_main
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_MANIFEST, AgentHardenerRunError
from nemo_helix_plugin.cli_error_handling import REMOTE_ERROR_EXIT_CODE
from nemo_helix_plugin.client.errors import (
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    UnprocessableEntityError,
)
from typer.testing import CliRunner

_JOBS_URL = "http://localhost:8080/apis/jobs/v2/workspaces/default/jobs"


def _response(status: int, detail: str) -> httpx.Response:
    return httpx.Response(status, json={"detail": detail}, request=httpx.Request("POST", _JOBS_URL))


def _raise(exc: Exception) -> Any:
    def _call(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return _call


def _app(monkeypatch: pytest.MonkeyPatch, **sdk_methods: Any) -> Any:
    fake_sdk = SimpleNamespace(agent_hardener=SimpleNamespace(**sdk_methods))
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c: None)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: fake_sdk)
    monkeypatch.setattr(_shared, "base_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(
        _shared.AgentHardenerConfig,
        "get",
        classmethod(lambda _cls: SimpleNamespace(default_workspace="default", operator_env_file=Path(".env"))),
    )
    return cli_main.AgentHardenerCLI().get_cli()


def test_sanity_check_shows_the_submit_gate_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detail = "Secrets not found: default/no-such-key (attack model key). Create them, then retry."
    rejected = httpx.HTTPStatusError("422", request=httpx.Request("POST", _JOBS_URL), response=_response(422, detail))
    app = _app(monkeypatch, sanity_check=_raise(rejected))
    monkeypatch.setattr(war_game, "wait_for_job", lambda *_a: pytest.fail("must not wait on a rejected submit"))
    mitigations = tmp_path / "mitigations.json"
    mitigations.write_text(json.dumps({"defenses": []}), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["sanity-check", "--manifest-id", "m1", "--mitigations", str(mitigations), "--replay-hitlog", "d/h"]
    )

    assert result.exit_code == REMOTE_ERROR_EXIT_CODE, result.output
    assert detail in result.output
    assert "sanity-check failed" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("error_cls", "status", "label"),
    [(NotFoundError, 404, "Not found"), (AuthenticationError, 401, "Authentication error")],
)
def test_synth_benign_shows_typed_client_errors(
    error_cls: type[Exception], status: int, label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = f"backend says {status}"
    app = _app(monkeypatch, synth_benign=_raise(error_cls(_response(status, detail))))

    result = CliRunner().invoke(app, ["synth-benign", "--manifest-id", "gone", "--no-interactive"])

    assert result.exit_code == REMOTE_ERROR_EXIT_CODE, result.output
    assert label in result.output
    assert detail in result.output
    assert "Traceback" not in result.output


def test_run_shows_a_pre_check_failure_with_its_remediation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "agent-hardener.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")
    unreadable = AgentHardenerRunError(CATEGORY_MANIFEST, "Manifest is not valid YAML.", remediation="Fix line 3.")
    app = _app(monkeypatch, run=lambda **_kwargs: pytest.fail("must not launch"))
    monkeypatch.setattr(war_game, "missing_secrets", _raise(unreadable))

    result = CliRunner().invoke(app, ["run", "--config", str(config)])

    assert result.exit_code == 1, result.output
    assert "Error: Manifest is not valid YAML." in result.output
    assert "Fix: Fix line 3." in result.output


@pytest.mark.parametrize(
    ("argv", "method", "error_cls", "status"),
    [
        (["manifest", "show", "gone"], "get", NotFoundError, 404),
        (["manifest", "set", "m1", "--rounds", "2"], "update", UnprocessableEntityError, 422),
        (["refresh", "--manifest-id", "m1"], "refresh", InternalServerError, 500),
    ],
)
def test_manifest_commands_show_backend_errors(
    argv: list[str], method: str, error_cls: type[Exception], status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = f"backend says {status}"
    manifests = SimpleNamespace(**{method: _raise(error_cls(_response(status, detail)))})
    app = _app(monkeypatch, manifests=manifests)

    result = CliRunner().invoke(app, argv)

    assert result.exit_code == REMOTE_ERROR_EXIT_CODE, result.output
    assert detail in result.output
    assert "Traceback" not in result.output
