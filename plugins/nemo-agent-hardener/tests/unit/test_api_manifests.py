# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentHardenerManifest routes (list/get/delete + the `init` create path).

TestClient + dependency_overrides mock the entity client; the platform SDK and the (network-bound)
agent resolver are monkeypatched so `init` exercises the request/response flow without a live agent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.agent_resolver import ResolvedManifest
from nemo_agent_hardener_plugin.api.v2 import manifests as manifests_module
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_agent_hardener_plugin.jobs import manifest as job_manifest
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, NemoPaginationInfo, get_entity_client

NOW = datetime.now(timezone.utc)
PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"


def _resolved(
    egress: list[str] | None = None, agent_updated_at: str = "", deployed_secret_refs: dict[str, str] | None = None
) -> ResolvedManifest:
    """The real resolver returns the *effective* egress — the flag plus what the config declares."""
    return ResolvedManifest(
        manifest={"agent": {"name": "clockbot", "port": 8000}, "backends": []},
        agent_config_path=Path("/tmp/agent.yaml"),
        project_dir=Path("/tmp/proj"),
        workspace="default",
        agent_name="clockbot",
        port=8000,
        secrets=["INFERENCE_API_KEY"],
        egress=list(egress or []),
        warnings=["no running deployment; defaulting port to 8000."],
        agent_updated_at=agent_updated_at,
        deployed_secret_refs=deployed_secret_refs or {},
    )


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(manifests_module, "get_nemo_client", lambda **_: MagicMock())
    monkeypatch.setattr(manifests_module, "resolve_agent_to_manifest", lambda *_a, **kw: _resolved(kw.get("egress")))
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
        lambda *_a, **_k: (
            "default/clockbot",
            9123,
            ["INFERENCE_API_KEY"],
            ["https://inference-api.nvidia.com/v1"],
            ["heads up"],
        ),
    )
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests/inspect-agent",
        json={"agent": "clockbot"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Egress is pre-filled rather than left blank: an empty box reads as "no egress", which invites
    # typing the hosts in by hand — and an explicit value then wins over the derived one.
    assert body == {
        "agent": "default/clockbot",
        "port": 9123,
        "secrets": ["INFERENCE_API_KEY"],
        "egress": ["https://inference-api.nvidia.com/v1"],
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


@pytest.mark.parametrize("command", ["python app.py --port 8000", 'python app.py --port "$PORT"'])
@pytest.mark.parametrize("yaml_port", [8000, "8000", 9100])
def test_patch_saved_default_synchronizes_listener_and_next_run(
    client, mock_entity_client, monkeypatch, tmp_path, command, yaml_port
) -> None:
    original_yaml = yaml.safe_dump(
        {
            "agent": {
                "port": yaml_port,
                "start_command": command,
                "env": {"PORT": "8000", "BASE": "keep"},
            }
        }
    )
    existing = AgentHardenerManifest(
        name="m1",
        workspace="default",
        port=8000,
        manifest_yaml=original_yaml,
        env={"PORT": "8000", "OVERRIDE": "keep"},
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    # Studio's Save as default uses this same PATCH, followed by a run with no override.
    response = client.patch(
        "/apis/agent-hardener/v2/workspaces/default/manifests/m1",
        json={"port": 9100},
    )
    assert response.status_code == 200, response.text
    stored = response.json()
    agent = yaml.safe_load(stored["manifest_yaml"])["agent"]
    expected_command = command if "$PORT" in command else command.replace("8000", "9100")
    assert agent["start_command"] == expected_command
    assert agent["port"] == stored["port"] == 9100
    assert agent["env"] == {"PORT": "9100", "BASE": "keep", "OVERRIDE": "keep"}
    assert stored["env"] == {"PORT": "9100", "OVERRIDE": "keep"}
    assert existing.manifest_yaml == original_yaml  # request normalization does not mutate the fetched object
    monkeypatch.setattr(job_manifest, "download_and_extract_project", lambda *_args: tmp_path)
    monkeypatch.setattr(job_manifest, "gateway_backend", lambda _url: None)
    restored = job_manifest._materialize_from_bundle(None, "m1", stored, tmp_path, "default/bundle")
    assert restored["agent"]["start_command"] == expected_command
    assert restored["agent"]["port"] == 9100
    assert restored["agent"]["env"]["PORT"] == "9100"


@pytest.mark.parametrize(
    "command",
    [
        "python app.py --port $PORT",
        "python app.py --port ${PORT:-8000}",
        'sh -c "cd app && python main.py --port 8000"',
    ],
)
def test_unchanged_opaque_port_survives_create_patch_refresh_and_run(
    client, mock_entity_client, monkeypatch, tmp_path, command
) -> None:
    resolved = _resolved()
    resolved.manifest["agent"]["start_command"] = command
    monkeypatch.setattr(manifests_module, "resolve_agent_to_manifest", lambda *_args, **_kwargs: resolved)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)
    response = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "agent": "clockbot", "port": 8000},
    )
    assert response.status_code == 201, response.text
    stored = response.json()
    assert yaml.safe_load(stored["manifest_yaml"])["agent"]["start_command"] == command

    for method, suffix, kwargs in [
        (client.patch, "", {"json": {"port": 8000}}),
        (client.post, "/refresh", {}),
    ]:
        mock_entity_client.get = AsyncMock(return_value=AgentHardenerManifest.model_validate(stored))
        response = method(f"/apis/agent-hardener/v2/workspaces/default/manifests/m1{suffix}", **kwargs)
        assert response.status_code == 200, response.text
        stored = response.json()
        assert yaml.safe_load(stored["manifest_yaml"])["agent"]["start_command"] == command
        assert stored["port"] == 8000

    monkeypatch.setattr(job_manifest, "download_and_extract_project", lambda *_args: tmp_path)
    monkeypatch.setattr(job_manifest, "gateway_backend", lambda _url: None)
    restored = job_manifest._materialize_from_bundle(None, "m1", stored, tmp_path, "default/bundle")
    assert restored["agent"]["start_command"] == command
    assert restored["agent"]["port"] == 8000


def test_patch_unsupported_listener_does_not_persist_partial_changes(client, mock_entity_client) -> None:
    existing = AgentHardenerManifest(
        name="m1",
        workspace="default",
        port=8000,
        manifest_yaml=yaml.safe_dump({"agent": {"port": 8000, "start_command": "custom-launcher"}}),
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    response = client.patch(
        "/apis/agent-hardener/v2/workspaces/default/manifests/m1",
        json={"port": 9100, "env": {"NEW": "value"}, "defenders": ["guardrails"]},
    )
    assert response.status_code == 422, response.text
    mock_entity_client.update.assert_not_awaited()
    assert existing.port == 8000
    assert existing.env == {}
    assert existing.defenders == []


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
    """Passed to the resolver *and* stored as what it resolved to: the run re-resolves otherwise."""
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


# ── backends ─────────────────────────────────────────────────────────────────

GATEWAY = {"name": "nemo-inference-gateway", "ports": [8080]}
REMOTE_MCP = {"name": "remote-mcp", "ports": [9100, 9101]}


@pytest.fixture
def resolver_backends(monkeypatch: pytest.MonkeyPatch) -> list[list[dict] | None]:
    """Backends each resolver call received."""
    seen: list[list[dict] | None] = []

    def _resolve(*_a, **kw):
        seen.append(kw.get("backends"))
        return _resolved(kw.get("egress"))

    monkeypatch.setattr(manifests_module, "resolve_agent_to_manifest", _resolve)
    return seen


def test_create_agent_manifest_passes_backends_to_the_resolver(client, mock_entity_client, resolver_backends) -> None:
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot", "backends": ["remote-mcp:9100,9101"]},
    )

    assert resp.status_code == 201, resp.text
    assert resolver_backends == [[REMOTE_MCP]]


@pytest.mark.parametrize("spec", ["remote-mcp", "remote-mcp:abc", "nemo-inference-gateway:8080"])
def test_create_manifest_rejects_a_bad_backend_with_422(client, resolver_backends, spec: str) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot", "backends": [spec]},
    )

    assert resp.status_code == 422, resp.text
    assert "backend" in resp.text
    assert resolver_backends == []


def test_refresh_carries_stored_backends_forward_without_the_gateway(
    client, mock_entity_client, resolver_backends
) -> None:
    stored_yaml = yaml.safe_dump({"agent": {"name": "clockbot"}, "backends": [GATEWAY, REMOTE_MCP]})
    existing = AgentHardenerManifest(
        name="m1", workspace="default", agent="default/clockbot", agent_fileset="default/old", manifest_yaml=stored_yaml
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/m1/refresh")

    assert resp.status_code == 200, resp.text
    assert resolver_backends == [[REMOTE_MCP]]


def test_refresh_tolerates_a_manifest_without_backends(client, mock_entity_client, resolver_backends) -> None:
    existing = AgentHardenerManifest(
        name="m1", workspace="default", agent="default/clockbot", agent_fileset="default/old", manifest_yaml=""
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/m1/refresh")

    assert resp.status_code == 200, resp.text
    assert resolver_backends == [[]]


def test_create_project_manifest_stores_backends(client, mock_entity_client, monkeypatch) -> None:
    monkeypatch.setattr(manifests_module, "download_and_extract_project", lambda *_a, **_k: Path("/tmp/p"))
    monkeypatch.setattr(
        manifests_module,
        "inspect_project",
        lambda *_a, **_k: {
            "dockerfile": "Dockerfile",
            "start_command": "run",
            "binaries": ["/app/**"],
            "port": 8000,
            "secrets": [],
            "egress": [],
            "env": {},
            "warnings": [],
        },
    )
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={
            "name": "p1",
            "source_type": "project",
            "project_fileset": "default/proj",
            "backends": ["remote-mcp:9100,9101"],
        },
    )

    assert resp.status_code == 201, resp.text
    assert yaml.safe_load(resp.json()["manifest_yaml"])["backends"] == [REMOTE_MCP]


# ── agent staleness marker ───────────────────────────────────────────────────


def test_create_records_the_agents_timestamp(client, mock_entity_client, monkeypatch) -> None:
    monkeypatch.setattr(
        manifests_module,
        "resolve_agent_to_manifest",
        lambda *_a, **kw: _resolved(kw.get("egress"), agent_updated_at="2026-10-01T10:00:00Z"),
    )
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot"},
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["agent_updated_at"] == "2026-10-01T10:00:00Z"


def test_refresh_advances_the_recorded_agent_timestamp(client, mock_entity_client, monkeypatch) -> None:
    monkeypatch.setattr(
        manifests_module,
        "resolve_agent_to_manifest",
        lambda *_a, **kw: _resolved(kw.get("egress"), agent_updated_at="2026-10-05T10:00:00Z"),
    )
    existing = AgentHardenerManifest(
        name="m1",
        workspace="default",
        agent="default/clockbot",
        agent_fileset="default/old",
        agent_updated_at="2026-10-01T10:00:00Z",
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/m1/refresh")

    assert resp.status_code == 200, resp.text
    assert resp.json()["agent_updated_at"] == "2026-10-05T10:00:00Z"


def test_create_splits_secret_specs_into_names_and_refs(client, mock_entity_client, monkeypatch) -> None:
    seen: list[list[str] | None] = []

    def _resolve(*_a, **kw):
        seen.append(kw.get("secrets"))
        return _resolved(kw.get("egress"))

    monkeypatch.setattr(manifests_module, "resolve_agent_to_manifest", _resolve)
    mock_entity_client.create = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot", "secrets": ["A", "B=default/b-secret"]},
    )

    assert resp.status_code == 201, resp.text
    assert seen == [["A", "B"]]
    assert resp.json()["secret_refs"] == {"B": "default/b-secret"}
    assert resp.json()["declared_secrets"] == ["A", "B"]


def test_create_rejects_a_bad_secret_spec_with_422(client, mock_entity_client) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/manifests",
        json={"name": "m1", "source_type": "agent", "agent": "clockbot", "secrets": ["B=Bad_Name"]},
    )

    assert resp.status_code == 422, resp.text
    assert "invalid secret" in resp.text
    mock_entity_client.create.assert_not_called()


def test_refresh_keeps_explicit_refs_and_swaps_the_deployments(client, mock_entity_client, monkeypatch) -> None:
    seen: list[list[str] | None] = []

    def _resolve(*_a, **kw):
        seen.append(kw.get("secrets"))
        return _resolved(kw.get("egress"), deployed_secret_refs={"NEW_DEPLOYED": "default/new"})

    monkeypatch.setattr(manifests_module, "resolve_agent_to_manifest", _resolve)
    existing = AgentHardenerManifest(
        name="m1",
        workspace="default",
        agent="default/clockbot",
        agent_fileset="default/old",
        secrets=["PLAIN", "EXPLICIT", "BARE", "OLD_DEPLOYED"],
        secret_refs={"EXPLICIT": "default/explicit"},
        declared_secrets=["EXPLICIT", "BARE"],
        deployed_secret_refs={"OLD_DEPLOYED": "default/old", "EXPLICIT": "default/other", "BARE": "default/bare"},
    )
    mock_entity_client.get = AsyncMock(return_value=existing)
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: entity)

    resp = client.post("/apis/agent-hardener/v2/workspaces/default/manifests/m1/refresh")

    assert resp.status_code == 200, resp.text
    # A name only the old deployment supplied is dropped; one the operator also passed is kept.
    assert seen == [["PLAIN", "EXPLICIT", "BARE"]]
    body = resp.json()
    assert body["secret_refs"] == {"EXPLICIT": "default/explicit"}
    assert body["deployed_secret_refs"] == {"NEW_DEPLOYED": "default/new"}


@pytest.mark.parametrize("manifest_yaml", ["backends: [unclosed", "backends: api", "- just a list"])
def test_stored_user_backends_ignores_unusable_yaml(manifest_yaml: str) -> None:
    assert manifests_module._stored_user_backends(manifest_yaml) == []


def test_stored_user_backends_drops_the_gateway_entry() -> None:
    manifest_yaml = yaml.safe_dump(
        {
            "backends": [
                {"name": manifests_module.GATEWAY_BACKEND_NAME, "ports": [8080]},
                {"name": "api", "ports": [9000]},
            ]
        }
    )
    assert manifests_module._stored_user_backends(manifest_yaml) == [{"name": "api", "ports": [9000]}]
