# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentHardenerManifest routes (list/get/delete + the `init` create path).

TestClient + dependency_overrides mock the entity client; the platform SDK and the (network-bound)
agent resolver are monkeypatched so `init` exercises the request/response flow without a live agent.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.agent_resolver import ResolvedManifest
from nemo_agent_hardener_plugin.api.v2 import manifests as manifests_module
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError, NemoPaginationInfo, get_entity_client

NOW = datetime.now(timezone.utc)
PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"


def _resolved() -> ResolvedManifest:
    return ResolvedManifest(
        manifest={"agent": {"name": "clockbot", "port": 8000}, "backends": []},
        workflow_path=Path("/tmp/workflow.yaml"),
        project_dir=Path("/tmp/proj"),
        workspace="default",
        agent_name="clockbot",
        port=8000,
        secrets=["INFERENCE_API_KEY"],
        warnings=["no running deployment; defaulting port to 8000."],
    )


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(manifests_module, "get_platform_sdk", lambda **_: MagicMock())
    monkeypatch.setattr(manifests_module, "resolve_agent_to_manifest", lambda *_a, **_k: _resolved())
    # Resolution now writes a scaffold that gets frozen as a fileset; the real upload needs a real dir.
    monkeypatch.setattr(manifests_module, "upload_project_dir", lambda _sdk, _dir, *, workspace: "default/agent-fs-1")
    monkeypatch.setattr(manifests_module, "delete_fileset", lambda _sdk, _ref: None)
    app = FastAPI()
    app.include_router(manifests_module.router, prefix=PREFIX)
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


def test_init_from_agent_creates_manifest(client, mock_entity_client) -> None:
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "clockbot-hardening", "source_type": "agent", "agent": "clockbot"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "clockbot-hardening"
    assert body["agent"] == "default/clockbot"
    assert body["port"] == 8000
    assert "clockbot" in body["manifest_yaml"]
    call = mock_entity_client.create.await_args
    assert call is not None
    created = call.args[0]
    assert isinstance(created, AgentHardenerManifest)


def test_inspect_agent_returns_derived_defaults(client, monkeypatch) -> None:
    monkeypatch.setattr(
        manifests_module,
        "inspect_agent",
        lambda *_a, **_k: ("default/clockbot", 9123, ["INFERENCE_API_KEY"], ["heads up"]),
    )
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests/inspect-agent",
        json={"agent": "clockbot"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "agent": "default/clockbot",
        "port": 9123,
        "secrets": ["INFERENCE_API_KEY"],
        "warnings": ["heads up"],
    }


def test_inspect_agent_reports_resolution_error(client, monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise manifests_module.AgentResolutionError("agent 'ghost' not found")

    monkeypatch.setattr(manifests_module, "inspect_agent", _boom)
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests/inspect-agent",
        json={"agent": "ghost"},
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


def test_init_agent_source_requires_agent(client) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "no-agent", "source_type": "agent"},
    )
    assert resp.status_code == 422


def test_init_project_source_requires_fileset(client) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "from-project", "source_type": "project"},
    )
    assert resp.status_code == 422
    assert "project_fileset" in resp.json()["detail"]


_INSPECT_JSON = (
    '{"project_dir": "myproject", "workflows": ["agents/research/workflow.yaml"], "dockerfiles": [], '
    '"suggested_launch_mode": "workflow", "default_agent_name": "research", "default_port": 8000, '
    '"secrets_file": ".env", "secret_names": ["INFERENCE_API_KEY"], "egress": ["inference-api.nvidia.com"]}'
)


def _stub_project_subprocess(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int, stdout: str = "", stderr: str = ""
) -> None:
    """Stub the fileset download + agent-hardener subprocess for the project manifest paths."""
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/proj"))
    monkeypatch.setattr(
        manifests_module.AgentHardenerConfig,
        "get",
        classmethod(lambda cls: MagicMock(agent_hardener_bin=Path("/bin/agent-hardener"))),
    )

    def fake_run(cmd, **_kwargs):
        # `init` writes to the -o path; emulate it so _init can read the manifest back.
        if returncode == 0 and "-o" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text("agent:\n  name: research\n  project_dir: /tmp/proj\n  port: 8000\n", encoding="utf-8")
        return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(manifests_module.subprocess, "run", fake_run)


def test_inspect_project_returns_detection(client, monkeypatch) -> None:
    _stub_project_subprocess(monkeypatch, returncode=0, stdout=_INSPECT_JSON)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests/inspect",
        json={"project_fileset": "default/proj-bundle"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["workflows"] == ["agents/research/workflow.yaml"]
    assert body["default_agent_name"] == "research"
    assert body["secret_names"] == ["INFERENCE_API_KEY"]


def test_inspect_project_reports_subprocess_failure(client, monkeypatch) -> None:
    _stub_project_subprocess(monkeypatch, returncode=1, stderr="no workflow found")

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests/inspect",
        json={"project_fileset": "default/proj-bundle"},
    )

    assert resp.status_code == 400
    assert "no workflow found" in resp.json()["detail"]


def _stub_hanging_subprocess(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Make the agent-hardener subprocess time out, recording the timeout it was given."""
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/proj"))
    monkeypatch.setattr(
        manifests_module.AgentHardenerConfig,
        "get",
        classmethod(lambda cls: MagicMock(agent_hardener_bin=Path("/bin/agent-hardener"))),
    )
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout") or 0)

    monkeypatch.setattr(manifests_module.subprocess, "run", fake_run)
    return seen


def test_inspect_project_bounds_a_hanging_subprocess(client, monkeypatch) -> None:
    """Unbounded, a wedged `agent-hardener inspect` pins its threadpool worker for the process's life."""
    seen = _stub_hanging_subprocess(monkeypatch)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests/inspect",
        json={"project_fileset": "default/proj-bundle"},
    )

    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]
    assert seen["timeout"] == manifests_module._SUBPROCESS_TIMEOUT_SECONDS


def test_create_project_manifest_bounds_a_hanging_subprocess(client, monkeypatch) -> None:
    seen = _stub_hanging_subprocess(monkeypatch)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "project", "project_fileset": "default/proj-bundle"},
    )

    assert resp.status_code == 504
    assert seen["timeout"] == manifests_module._SUBPROCESS_TIMEOUT_SECONDS


def test_create_project_manifest(client, mock_entity_client, monkeypatch) -> None:
    _stub_project_subprocess(monkeypatch, returncode=0)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "research-hardening",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "workflow": "agents/research/workflow.yaml",
            "secrets": ["INFERENCE_API_KEY"],
        },
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["source_type"] == "project"
    assert body["project_fileset"] == "default/proj-bundle"
    assert body["workflow"] == "agents/research/workflow.yaml"
    # The persisted manifest can't hold the temp path; project_dir is normalized to '.'.
    assert "project_dir: ." in body["manifest_yaml"]


def test_create_project_manifest_accepts_a_prebuilt_yaml(client, mock_entity_client, monkeypatch) -> None:
    """A supplied manifest is stored as-is: rebuilding it with `init --yes` would discard the
    answers the operator gave agent-hardener at their terminal."""
    ran: list[list[str]] = []
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/proj"))
    monkeypatch.setattr(manifests_module.subprocess, "run", lambda cmd, **_k: ran.append(cmd))
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "from-cli",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "manifest_yaml": (
                "agent:\n  name: research\n  project_dir: /local/path\n  workflow: wf.yaml\n"
                "  port: 9100\n  secrets:\n  - INFERENCE_API_KEY\n  egress:\n  - en.wikipedia.org\n"
            ),
        },
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert ran == []  # agent-hardener was never re-run
    assert "project_dir: ." in body["manifest_yaml"]  # still normalized off the operator's local path
    # Entity fields describe the manifest, so the two can't disagree when the client omits them.
    assert body["workflow"] == "wf.yaml"
    assert body["port"] == 9100
    assert body["secrets"] == ["INFERENCE_API_KEY"]
    assert body["egress"] == ["en.wikipedia.org"]


def test_create_project_manifest_rejects_a_yaml_without_an_agent_section(
    client, mock_entity_client, monkeypatch
) -> None:
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/proj"))
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "junk",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "manifest_yaml": "not: a manifest\n",
        },
    )

    assert resp.status_code == 422
    assert "agent" in resp.json()["detail"]


def test_create_project_manifest_forwards_egress(client, mock_entity_client, monkeypatch) -> None:
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/proj"))
    monkeypatch.setattr(
        manifests_module.AgentHardenerConfig,
        "get",
        classmethod(lambda cls: MagicMock(agent_hardener_bin=Path("/bin/agent-hardener"))),
    )
    captured: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("agent:\n  name: research\n  project_dir: .\n", encoding="utf-8")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(manifests_module.subprocess, "run", fake_run)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "research-hardening",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "workflow": "agents/research/workflow.yaml",
            "egress": ["host.docker.internal:8086", "inference-api.nvidia.com"],
        },
    )

    assert resp.status_code == 201, resp.text
    argv = captured[0]
    egress_flags = [argv[i + 1] for i, tok in enumerate(argv) if tok == "--egress"]
    assert egress_flags == ["host.docker.internal:8086", "inference-api.nvidia.com"]


def test_create_project_manifest_forwards_backends(client, mock_entity_client, monkeypatch) -> None:
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/proj"))
    monkeypatch.setattr(
        manifests_module.AgentHardenerConfig,
        "get",
        classmethod(lambda cls: MagicMock(agent_hardener_bin=Path("/bin/agent-hardener"))),
    )
    captured: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("agent:\n  name: research\n  project_dir: .\n", encoding="utf-8")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(manifests_module.subprocess, "run", fake_run)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "finance",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "workflow": "agents_lab/agents/finance/workflow.yaml",
            "backends": ["finance:8086", "cache:6379,6380"],
        },
    )

    assert resp.status_code == 201, resp.text
    argv = captured[0]
    backend_flags = [argv[i + 1] for i, tok in enumerate(argv) if tok == "--backend"]
    assert backend_flags == ["finance:8086", "cache:6379,6380"]


def _stub_byo_project(monkeypatch, project_dir: Path, captured: list[list[str]], agent_yaml: str) -> None:
    """Stub the project paths against a real directory, so the dockerfile containment check can run."""
    (project_dir / "deploy").mkdir(parents=True, exist_ok=True)
    (project_dir / "deploy" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: project_dir)
    monkeypatch.setattr(
        manifests_module.AgentHardenerConfig,
        "get",
        classmethod(lambda cls: MagicMock(agent_hardener_bin=Path("/bin/agent-hardener"))),
    )

    def fake_run(cmd, **_kwargs):
        captured.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text(agent_yaml, encoding="utf-8")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(manifests_module.subprocess, "run", fake_run)


_BYO_YAML = (
    "agent:\n"
    "  name: lab\n"
    "  project_dir: .\n"
    "  workflow: agents/lab/workflow.yaml\n"
    "  dockerfile: deploy/Dockerfile\n"
    "  binaries:\n"
    "  - /app/.venv/bin/**\n"
)


def test_create_byo_manifest_forwards_dockerfile_and_binaries(client, mock_entity_client, monkeypatch, tmp_path):
    captured: list[list[str]] = []
    _stub_byo_project(monkeypatch, tmp_path, captured, _BYO_YAML)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "byo",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "launch_mode": "byo",
            "dockerfile": "deploy/Dockerfile",
            "binaries": ["/app/.venv/bin/**"],
            "workflow": "agents/lab/workflow.yaml",
        },
    )

    assert resp.status_code == 201, resp.text
    argv = captured[0]
    # Relative, not resolved: the run re-materializes the manifest elsewhere.
    assert argv[argv.index("--dockerfile") + 1] == "deploy/Dockerfile"
    assert [argv[i + 1] for i, tok in enumerate(argv) if tok == "--binary"] == ["/app/.venv/bin/**"]
    created_call = mock_entity_client.create.await_args
    assert created_call is not None
    stored = created_call.args[0]
    assert stored.launch_mode == "byo"
    assert stored.workflow == "agents/lab/workflow.yaml"  # the image does not displace the workflow
    assert stored.dockerfile == "deploy/Dockerfile"


def test_byo_manifest_yaml_is_labelled_byo_without_an_explicit_launch_mode(
    client, mock_entity_client, monkeypatch, tmp_path
):
    """The CLI sends a pre-built manifest and no launch_mode; it must not be labelled 'workflow'."""
    _stub_byo_project(monkeypatch, tmp_path, [], _BYO_YAML)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "byo-cli",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "manifest_yaml": _BYO_YAML,
        },
    )

    assert resp.status_code == 201, resp.text
    created_call = mock_entity_client.create.await_args
    assert created_call is not None
    stored = created_call.args[0]
    assert stored.launch_mode == "byo"
    assert stored.dockerfile == "deploy/Dockerfile"


def test_byo_without_a_workflow_is_rejected(client, mock_entity_client, monkeypatch, tmp_path) -> None:
    """agent-hardener can't launch a BYO victim without a workflow, and the platform never sets start_command."""
    no_workflow = "agent:\n  name: lab\n  project_dir: .\n  dockerfile: deploy/Dockerfile\n  binaries:\n  - /app/**\n"
    _stub_byo_project(monkeypatch, tmp_path, [], no_workflow)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "byo-noflow",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "manifest_yaml": no_workflow,
        },
    )

    assert resp.status_code == 422
    assert "workflow" in resp.json()["detail"]


def test_create_byo_manifest_rejects_a_dockerfile_outside_the_project(client, monkeypatch, tmp_path) -> None:
    _stub_byo_project(monkeypatch, tmp_path, [], _BYO_YAML)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "escape",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "dockerfile": "../../../etc/passwd",
            "binaries": ["/app/.venv/bin/**"],
        },
    )

    assert resp.status_code == 400
    assert "inside the uploaded project" in resp.json()["detail"]


def test_create_byo_manifest_requires_binaries(client) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "no-binaries",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "dockerfile": "deploy/Dockerfile",
        },
    )

    assert resp.status_code == 422
    assert "binaries" in resp.json()["detail"]


def test_create_manifest_rejects_an_unknown_launch_mode(client) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "weird",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "launch_mode": "kubernetes",
        },
    )

    assert resp.status_code == 422
    assert "launch_mode" in resp.json()["detail"]


def test_byo_without_a_dockerfile_anywhere_is_rejected(client) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "byo-empty",
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "launch_mode": "byo",
        },
    )

    assert resp.status_code == 422
    assert "dockerfile" in resp.json()["detail"]


def test_delete_missing_manifest_returns_404(client, mock_entity_client) -> None:
    mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))

    resp = client.delete("/apis/agent-hardener/v2/workspaces/default/manifests/ghost")

    assert resp.status_code == 404


def test_patch_updates_benign_suite_and_port(client, mock_entity_client) -> None:
    existing = AgentHardenerManifest(name="m1", workspace="default", agent="default/clockbot", port=8000)
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.patch(
        "/apis/agent-hardener/v2/workspaces/default/manifests/m1",
        json={"benign_suite": [{"tool": "clock", "payload": "what time", "label": "benign"}], "port": 9000},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["port"] == 9000
    assert body["benign_suite"][0]["tool"] == "clock"
    call = mock_entity_client.update.await_args
    assert call is not None
    updated = call.args[0]
    assert updated.port == 9000
    assert updated.benign_suite[0]["payload"] == "what time"


def test_patch_updates_egress(client, mock_entity_client) -> None:
    """Egress is a persisted setting now, so it has to be editable after creation."""
    existing = AgentHardenerManifest(name="m1", workspace="default", agent="default/clockbot", port=8000)
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.patch(
        "/apis/agent-hardener/v2/workspaces/default/manifests/m1",
        json={"egress": ["en.wikipedia.org", "api.example.com:443"]},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["egress"] == ["en.wikipedia.org", "api.example.com:443"]


def test_create_agent_manifest_persists_egress(client, mock_entity_client) -> None:
    """Passed to the resolver *and* stored: the run re-resolves and would otherwise drop it."""
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot", "egress": ["en.wikipedia.org"]},
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["egress"] == ["en.wikipedia.org"]


def test_patch_missing_manifest_returns_404(client, mock_entity_client) -> None:
    mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))

    resp = client.patch("/apis/agent-hardener/v2/workspaces/default/manifests/ghost", json={"port": 9000})

    assert resp.status_code == 404


def test_list_returns_envelope(client, mock_entity_client) -> None:
    manifest = AgentHardenerManifest(name="m1", workspace="default", agent="default/clockbot")
    page = MagicMock()
    page.data = [manifest]
    page.pagination = NemoPaginationInfo(page=1, page_size=20, current_page_size=1, total_pages=1, total_results=1)
    mock_entity_client.list = AsyncMock(return_value=page)

    resp = client.get("/apis/agent-hardener/v2/workspaces/default/manifests")

    assert resp.status_code == 200, resp.text
    assert [m["name"] for m in resp.json()["data"]] == ["m1"]


# ── frozen targets ───────────────────────────────────────────────────────────


def test_create_agent_manifest_freezes_the_scaffold(client, mock_entity_client) -> None:
    """Resolution writes an installable project; storing it is what makes the target reproducible."""
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot"},
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["agent_fileset"] == "default/agent-fs-1"


def test_refresh_rebuilds_the_scaffold_but_keeps_operator_settings(client, mock_entity_client) -> None:
    """Refresh is the deliberate way to take agent changes — it must not discard what the user chose."""
    existing = AgentHardenerManifest(
        name="m1",
        workspace="default",
        agent="default/clockbot",
        agent_fileset="default/old-fs",
        egress=["en.wikipedia.org"],
        defenders=["guardrails"],
        benign_suite=[{"tool": "t", "payload": "p", "label": "benign", "rationale": "r", "persona": "x"}],
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/m1/refresh")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent_fileset"] == "default/agent-fs-1"  # re-frozen
    assert body["egress"] == ["en.wikipedia.org"]
    assert body["defenders"] == ["guardrails"]
    assert len(body["benign_suite"]) == 1


def test_refresh_rejects_a_project_manifest(client, mock_entity_client) -> None:
    """There is no agent to re-resolve from; the fix is re-uploading, so say so instead of 500ing."""
    mock_entity_client.get = AsyncMock(
        return_value=AgentHardenerManifest(
            name="p1", workspace="default", source_type="project", project_fileset="default/proj"
        )
    )

    resp = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/p1/refresh")

    assert resp.status_code == 422
    assert "re-upload" in resp.json()["detail"]


def test_delete_manifest_removes_its_bundle(client, mock_entity_client, monkeypatch) -> None:
    """A bundle outliving its manifest is unreachable storage nobody will ever clean up."""
    deleted: list[str] = []
    monkeypatch.setattr(manifests_module, "delete_fileset", lambda _sdk, ref: deleted.append(ref))
    mock_entity_client.get = AsyncMock(
        return_value=AgentHardenerManifest(name="m1", workspace="default", agent_fileset="default/agent-fs-1")
    )
    mock_entity_client.delete = AsyncMock(return_value=None)

    resp = client.delete("/apis/agent-hardener/v2/workspaces/default/manifests/m1")

    assert resp.status_code == 204
    assert deleted == ["default/agent-fs-1"]


def test_delete_manifest_keeps_a_caller_supplied_project_bundle(client, mock_entity_client, monkeypatch) -> None:
    """Nothing stops two manifests naming one uploaded bundle, so deleting it would break the other."""
    deleted: list[str] = []
    monkeypatch.setattr(manifests_module, "delete_fileset", lambda _sdk, ref: deleted.append(ref))
    mock_entity_client.get = AsyncMock(
        return_value=AgentHardenerManifest(
            name="p1", workspace="default", source_type="project", project_fileset="default/proj-fs-1"
        )
    )
    mock_entity_client.delete = AsyncMock(return_value=None)

    resp = client.delete("/apis/agent-hardener/v2/workspaces/default/manifests/p1")

    assert resp.status_code == 204
    assert deleted == []


def test_env_is_persisted_and_survives_refresh(client, mock_entity_client) -> None:
    """env is operator intent like egress — rebuilding the scaffold must not discard it."""
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot", "env": {"DEMO": "1"}},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["env"] == {"DEMO": "1"}

    existing = AgentHardenerManifest(
        name="m1", workspace="default", agent="default/clockbot", agent_fileset="default/old", env={"DEMO": "1"}
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    refreshed = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/m1/refresh")

    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["env"] == {"DEMO": "1"}


def test_patch_updates_env(client, mock_entity_client) -> None:
    mock_entity_client.get = AsyncMock(
        return_value=AgentHardenerManifest(name="m1", workspace="default", env={"OLD": "1"})
    )
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.patch("/apis/agent-hardener/v2/workspaces/default/manifests/m1", json={"env": {"NEW": "2"}})

    assert resp.status_code == 200, resp.text
    assert resp.json()["env"] == {"NEW": "2"}


def test_validate_model_uses_the_provisioned_key_when_no_secret_is_named(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null api_key_secret means the provisioned agent-hardener key, not "no key".

    Probing unauthenticated reported 401 for every model a run would in fact reach, so this endpoint —
    and Studio's "Test connection" — rejected valid choices.
    """
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        manifests_module, "resolve_model_key", lambda _sdk, secret, *, workspace: seen.setdefault("secret", secret)
    )
    monkeypatch.setattr(
        manifests_module,
        "validate_choice",
        lambda model, base_url, key: seen.update(key=key) or MagicMock(ok=True, reason="", available=[], detail=""),
    )

    response = client.post(
        f"{PREFIX}/model-config/validate",
        json={"model": "some/model", "base_url": "https://x/v1"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert seen["secret"] is None  # the resolver is asked, and decides the fallback


def test_validate_model_passes_a_named_secret_through(client, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        manifests_module, "resolve_model_key", lambda _sdk, secret, *, workspace: seen.setdefault("secret", secret)
    )
    monkeypatch.setattr(
        manifests_module, "validate_choice", lambda *_a: MagicMock(ok=True, reason="", available=[], detail="")
    )

    response = client.post(
        f"{PREFIX}/model-config/validate",
        json={"model": "m", "base_url": "https://x/v1", "api_key_secret": "my-key"},
    )

    assert response.status_code == 200
    assert seen["secret"] == "my-key"
