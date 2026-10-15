# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nemo agent-hardener init --project-dir`, focused on the BYO (custom image) launch mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml
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


def test_init_passes_repeated_backends_to_the_create_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main
    from typer.testing import CliRunner

    created: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, [], created, _project(tmp_path))

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--agent",
            "clockbot",
            "--backend",
            "remote-mcp:9100,9101",
            "--backend",
            "db:5432",
            "--output",
            str(tmp_path / "agent-hardener.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert created["backends"] == ["remote-mcp:9100,9101", "db:5432"]


def test_init_omits_backends_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main
    from typer.testing import CliRunner

    created: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, [], created, _project(tmp_path))

    result = CliRunner().invoke(app, ["init", "--agent", "clockbot", "--output", str(tmp_path / "out.yaml")])

    assert result.exit_code == 0, result.output
    assert "backends" not in created


# --------------------------------------------------------------------------- init -o
def _stored_manifest(project_dir: str = ".agent-hardener-agents/x") -> dict[str, Any]:
    manifest_yaml = yaml.safe_dump({"agent": {"name": "x", "project_dir": project_dir, "port": 8000}})
    return {"agent_fileset": "default/fs-1", "manifest_yaml": manifest_yaml}


def _project_dir_of(manifest_yaml: str) -> str:
    return yaml.safe_load(manifest_yaml)["agent"]["project_dir"]


def test_output_yaml_points_project_dir_at_the_downloaded_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "out" / "hardener.yaml"
    out_path.parent.mkdir()

    def fake_download(_sdk: Any, fileset: str, dest: Path) -> Path:
        assert fileset == "default/fs-1"
        (dest / "project").mkdir(parents=True)
        return dest / "project"

    monkeypatch.setattr(lifecycle, "download_and_extract_project", fake_download)

    rendered = lifecycle._with_local_project(SimpleNamespace(sdk=object()), _stored_manifest(), out_path)

    assert _project_dir_of(rendered) == "hardener-victim/project"


def test_output_yaml_is_unchanged_when_the_bundle_cannot_be_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_download(*_a: Any, **_k: Any) -> Path:
        raise RuntimeError("fileset gone")

    monkeypatch.setattr(lifecycle, "download_and_extract_project", failing_download)
    manifest = _stored_manifest()

    rendered = lifecycle._with_local_project(SimpleNamespace(sdk=object()), manifest, tmp_path / "hardener.yaml")

    assert rendered == manifest["manifest_yaml"]


def test_output_yaml_is_unchanged_when_the_stored_yaml_is_malformed(tmp_path: Path) -> None:
    manifest = {"agent_fileset": "default/fs-1", "manifest_yaml": "agent: [unclosed"}

    rendered = lifecycle._with_local_project(SimpleNamespace(sdk=object()), manifest, tmp_path / "hardener.yaml")

    assert rendered == "agent: [unclosed"


def test_an_unrelated_directory_with_the_victim_name_is_never_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "hardener-victim"
    workdir.mkdir()
    precious = workdir / "notes.txt"
    precious.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(
        lifecycle, "download_and_extract_project", lambda *_a, **_k: pytest.fail("must not download over it")
    )
    manifest = _stored_manifest()

    rendered = lifecycle._with_local_project(SimpleNamespace(sdk=object()), manifest, tmp_path / "hardener.yaml")

    assert rendered == manifest["manifest_yaml"]
    assert precious.read_text(encoding="utf-8") == "keep me"
