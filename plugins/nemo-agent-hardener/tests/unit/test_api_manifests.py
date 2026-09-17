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
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.agent_resolver import ResolvedManifest
from nemo_agent_hardener_plugin.api.v2 import manifests as manifests_module
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError, NemoPaginationInfo, get_entity_client

NOW = datetime.now(timezone.utc)
PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"


def _resolved(egress: list[str] | None = None) -> ResolvedManifest:
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
    )


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(manifests_module, "get_platform_sdk", lambda **_: MagicMock())
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
