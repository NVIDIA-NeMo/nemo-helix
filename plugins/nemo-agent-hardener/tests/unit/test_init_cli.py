# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nemo agent-hardener init --project-dir`, focused on the BYO (custom image) launch mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agent_hardener_plugin.cli import _shared, lifecycle
from typer.testing import CliRunner


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "deploy").mkdir(parents=True)
    (project / "deploy" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    return project


def _patch_cli(cli_main: Any, monkeypatch: pytest.MonkeyPatch, captured: list[list[str]], created: dict) -> Any:
    """Neutralize preflight/SDK/upload and capture both the init argv and the create body."""

    def fake_create(*, workspace: str, **body):
        created.update(body)
        return {"name": body["name"], "launch_mode": body.get("launch_mode"), "dockerfile": "deploy/Dockerfile"}

    fake_sdk = SimpleNamespace(agent_hardener=SimpleNamespace(manifests=SimpleNamespace(create=fake_create)))
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


def test_init_forwards_dockerfile_and_binary_to_agent_hardener(tmp_path, monkeypatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: list[list[str]] = []
    created: dict = {}
    app = _patch_cli(cli_main, monkeypatch, captured, created)

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--project-dir",
            str(_project(tmp_path)),
            "--name",
            "byo",
            "--yes",
            "--dockerfile",
            "deploy/Dockerfile",
            "--binary",
            "/app/.venv/bin/**",
            "--workflow",
            "agents/lab/workflow.yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    argv = captured[0]
    assert argv[argv.index("--dockerfile") + 1] == "deploy/Dockerfile"
    assert [argv[i + 1] for i, tok in enumerate(argv) if tok == "--binary"] == ["/app/.venv/bin/**"]
    # --workflow is kept: the image is how the environment is built, not what gets served.
    assert argv[argv.index("--workflow") + 1] == "agents/lab/workflow.yaml"
    assert created["launch_mode"] == "byo"


def test_init_without_a_dockerfile_stays_workflow_mode(tmp_path, monkeypatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: list[list[str]] = []
    created: dict = {}
    app = _patch_cli(cli_main, monkeypatch, captured, created)

    result = CliRunner().invoke(app, ["init", "--project-dir", str(_project(tmp_path)), "--name", "plain", "--yes"])

    assert result.exit_code == 0, result.output
    assert "--dockerfile" not in captured[0]
    assert created["launch_mode"] == "workflow"


def test_init_rejects_a_dockerfile_without_a_binary(tmp_path, monkeypatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    app = _patch_cli(cli_main, monkeypatch, [], {})

    result = CliRunner().invoke(
        app,
        ["init", "--project-dir", str(_project(tmp_path)), "--yes", "--dockerfile", "deploy/Dockerfile"],
    )

    assert result.exit_code == 1
    assert "--binary" in result.output


def test_init_rejects_an_absolute_dockerfile(tmp_path, monkeypatch) -> None:
    """An absolute path resolves here but not on the host that re-materializes the manifest."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    project = _project(tmp_path)
    app = _patch_cli(cli_main, monkeypatch, [], {})

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--yes",
            "--dockerfile",
            str(project / "deploy" / "Dockerfile"),
            "--binary",
            "/app/**",
        ],
    )

    assert result.exit_code == 1
    assert "relative" in result.output


def test_init_rejects_a_missing_dockerfile(tmp_path, monkeypatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    app = _patch_cli(cli_main, monkeypatch, [], {})

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--project-dir",
            str(_project(tmp_path)),
            "--yes",
            "--dockerfile",
            "deploy/Nope",
            "--binary",
            "/app/**",
        ],
    )

    assert result.exit_code == 1
    assert "not found" in result.output


def test_init_rejects_dockerfile_with_an_agent_source(monkeypatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    app = _patch_cli(cli_main, monkeypatch, [], {})

    result = CliRunner().invoke(
        app, ["init", "--agent", "clockbot", "--dockerfile", "deploy/Dockerfile", "--binary", "/app/**"]
    )

    assert result.exit_code == 1
    assert "--project-dir" in result.output
