# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nemo agent-hardener init --project-dir`, focused on the BYO (custom image) launch mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_agent_hardener_plugin import sdk as sdk_module
from nemo_agent_hardener_plugin.cli import _shared, lifecycle
from nemo_agent_hardener_plugin.sdk import AgentHardenerPluginResource
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import NemoClient


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "deploy").mkdir(parents=True)
    (project / "deploy" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    return project


def _patch_cli(
    cli_main: Any, monkeypatch: pytest.MonkeyPatch, captured: list[list[str]], created: dict, project: Path
) -> Any:
    """Neutralize preflight/SDK/upload and capture both the init argv and the create body."""

    def fake_create(*, workspace: str, **body):
        created.update(body)
        return {"name": body["name"], "launch_mode": body.get("launch_mode"), "dockerfile": "deploy/Dockerfile"}

    def fake_inspect_project(project_fileset: str, *, dockerfile: str | None = None, workspace: str = "default"):
        from nemo_agent_hardener_plugin.project_resolver import inspect_project

        return inspect_project(project, dockerfile=dockerfile)

    fake_sdk = SimpleNamespace(
        agent_hardener=SimpleNamespace(
            manifests=SimpleNamespace(create=fake_create, inspect_project=fake_inspect_project)
        )
    )
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c: None)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: fake_sdk)
    monkeypatch.setattr(_shared, "base_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(
        _shared.AgentHardenerConfig,
        "get",
        classmethod(
            lambda _cls: SimpleNamespace(
                default_workspace="default",
                operator_env_file=Path(".env"),
                agent_hardener_bin=Path("/bin/agent-hardener"),
            )
        ),
    )
    monkeypatch.setattr(lifecycle, "upload_project_dir", lambda *_a, **_k: "default/proj-bundle")

    def fake_subprocess(cmd, _action, *, cwd=None, timeout=None, env=None):
        captured.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("agent:\n  name: lab\n  project_dir: .\n", encoding="utf-8")

    monkeypatch.setattr(lifecycle.provisioning, "run_subprocess", fake_subprocess)
    return cli_main.AgentHardenerCLI().get_cli()


def test_init_project_dir_creates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`init --project-dir` uploads the project, derives what it can, and creates the manifest."""
    from nemo_agent_hardener_plugin.cli import main as cli_main
    from typer.testing import CliRunner

    project = _project(tmp_path)
    captured: list[list[str]] = []
    created: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured, created, project)

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--start-command",
            "/app/run.sh",
            "--harness",
            "langgraph",
            "--relay-confirmed",
            "--output",
            str(tmp_path / "agent-hardener.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert created["source_type"] == "project"
    assert created["project_fileset"] == "default/proj-bundle"
    assert created["start_command"] == "/app/run.sh"
    assert created["harness"] == "langgraph"
    assert created["relay_integration_confirmed"] is True


def test_project_init_uses_typed_client_without_sdk_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"dockerfile": "deploy/Dockerfile", "unresolved": []})

    monkeypatch.setattr(sdk_module, "client_from_platform", client_from_platform)
    monkeypatch.setattr(lifecycle, "upload_project_dir", lambda *_args, **_kwargs: "default/proj-bundle")
    with NemoClient(
        base_url="http://platform.test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        context = SimpleNamespace(sdk=client, agent_hardener=AgentHardenerPluginResource(client), workspace="default")
        body = lifecycle._project_init_body(
            context,
            project_dir=str(_project(tmp_path)),
            name="byo",
            dockerfile=None,
            start_command="/app/run.sh",
            binaries=[],
            harness="langgraph",
            relay_confirmed=True,
        )
    assert body["project_fileset"] == "default/proj-bundle"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/apis/agent-hardener/v2/workspaces/default/manifests/inspect-project"
