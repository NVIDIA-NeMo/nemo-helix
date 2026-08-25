# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Agent Deployment route handlers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import deployments as deployments_router_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client, get_files_client
from nemo_agents_plugin.config import AgentsConfig
from nemo_agents_plugin.entities import (
    NAT_WORKFLOW_CONFIG_FORMAT,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentSandboxSpec,
    ComputeResources,
    DeploymentStatus,
)
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.client.errors import NotFoundError as PluginClientNotFoundError
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_platform_plugin.files.storage_config import GithubStorageConfig, LocalStorageConfig, StorageConfig

NOW = datetime.now(timezone.utc)


def _fabric_agent_config() -> dict[str, Any]:
    return {
        "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        "name": "fabric-agent",
        "description": "Fabric-backed agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
            },
        },
        "models": {
            "default": {
                "provider": "openai",
                "model": "openai/gpt-5.4",
            },
        },
    }


def _make_agent(
    *,
    name: str = "fabric-agent",
    workspace: str = "default",
    config: dict[str, Any] | None = None,
    config_format: str = NEMO_AGENTS_SPEC_CONFIG_FORMAT,
) -> Agent:
    agent = Agent(
        name=name,
        workspace=workspace,
        config=config or _fabric_agent_config(),
        config_format=config_format,
    )
    agent._id = f"agent-{name}-id"
    agent._created_at = NOW
    return agent


def _make_deployment(
    *,
    name: str = "fabric-dep",
    workspace: str = "default",
    agent: str = "fabric-agent",
    status: DeploymentStatus = "pending",
) -> AgentDeployment:
    deployment = AgentDeployment(name=name, workspace=workspace, agent=agent, status=status)
    deployment._id = f"deployment-{name}-id"
    deployment._created_at = NOW
    return deployment


def _files_client(storage: StorageConfig | None = None) -> AsyncMock:
    """A files client whose Ethos fileset is absent unless *storage* is given."""
    client = AsyncMock()
    if storage is None:
        client.get_fileset = AsyncMock(
            side_effect=PluginClientNotFoundError(httpx.Response(404, json={"detail": "not found"}))
        )
        return client

    response = AsyncMock()
    response.data = lambda: SimpleNamespace(storage=storage)
    client.get_fileset = AsyncMock(return_value=response)
    return client


def _github_storage(revision: str, original_revision: str) -> GithubStorageConfig:
    return GithubStorageConfig(owner="acme", repo="agents", revision=revision, original_revision=original_revision)


def _test_client(mock_entity_client: AsyncMock, mock_files_client: AsyncMock | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(
        deployments_router_module.router,
        prefix="/apis/agents/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    app.dependency_overrides[get_files_client] = lambda: mock_files_client or _files_client()
    return TestClient(app, raise_server_exceptions=False)


class TestSpecRevisionSnapshot:
    @staticmethod
    def _create(files_client: AsyncMock, *, agent: Agent | None = None) -> AgentDeployment:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=agent or _make_agent())
        mock_entity_client.create = AsyncMock(side_effect=lambda deployment: deployment)
        client = _test_client(mock_entity_client, files_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep"},
        )

        assert resp.status_code == 201
        return mock_entity_client.create.call_args[0][0]

    def test_records_the_revision_the_deployment_stages(self) -> None:
        files_client = _files_client(_github_storage(revision="abc123", original_revision="main"))

        deployment = self._create(files_client)

        assert deployment.spec_revision == "abc123"
        assert deployment.spec_tracked_revision == "main"
        assert files_client.get_fileset.call_args.kwargs["name"] == "fabric-agent-ethos"

    def test_a_fileset_pinned_to_a_commit_tracks_nothing(self) -> None:
        sha = "1" * 40
        deployment = self._create(_files_client(_github_storage(revision=sha, original_revision=sha)))

        assert deployment.spec_revision == sha
        assert deployment.spec_tracked_revision == ""

    def test_reads_no_fileset_for_an_agent_that_does_not_stage_one(self) -> None:
        files_client = _files_client(_github_storage(revision="abc123", original_revision="main"))
        nat_agent = _make_agent(
            name="nat-agent",
            config={"workflow": {"_type": "chat_completion"}},
            config_format=NAT_WORKFLOW_CONFIG_FORMAT,
        )

        deployment = self._create(files_client, agent=nat_agent)

        assert deployment.spec_revision == ""
        assert deployment.spec_tracked_revision == ""
        files_client.get_fileset.assert_not_awaited()

    def test_records_nothing_for_an_agent_with_no_fileset(self) -> None:
        deployment = self._create(_files_client())

        assert deployment.spec_revision == ""
        assert deployment.spec_tracked_revision == ""

    def test_records_nothing_for_a_backend_that_pins_no_revision(self) -> None:
        deployment = self._create(_files_client(LocalStorageConfig(path="/data")))

        assert deployment.spec_revision == ""
        assert deployment.spec_tracked_revision == ""

    def test_an_unreadable_fileset_does_not_fail_the_deployment(self) -> None:
        files_client = AsyncMock()
        files_client.get_fileset = AsyncMock(side_effect=RuntimeError("files service down"))

        deployment = self._create(files_client)

        assert deployment.spec_revision == ""
        assert deployment.status == "pending"


class TestCreateDeployment:
    def test_create_preserves_platform_agent_config(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep"},
        )

        assert resp.status_code == 201
        created_deployment: AgentDeployment = mock_entity_client.create.call_args[0][0]
        assert created_deployment.config["config_format"] == NEMO_AGENTS_SPEC_CONFIG_FORMAT
        assert created_deployment.config["environment"]["provider"] == "local"
        assert created_deployment.config["models"]["default"]["base_url"] == (
            "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )
        assert "functions" not in created_deployment.config
        assert "workflow" not in created_deployment.config

    def test_create_snapshots_image_entrypoint_mode(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={
                "agent": "fabric-agent",
                "name": "fabric-dep",
                "deployment_mode": "docker",
                "image": "hand-built-agent:latest",
                "use_image_entrypoint": True,
            },
        )

        assert resp.status_code == 201
        created_deployment: AgentDeployment = mock_entity_client.create.call_args[0][0]
        assert created_deployment.deployment_mode == "docker"
        assert created_deployment.image == "hand-built-agent:latest"
        assert created_deployment.use_image_entrypoint is True

    def test_create_rejects_a_mode_the_executor_will_not_honour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_deployments_plugin.config import DeploymentsConfig, ExecutorConfigEntry

        # A standalone config: DeploymentsConfig.get() is a cached singleton, and
        # assigning to it would outlive this test.
        deployments_cfg = DeploymentsConfig(executors=[ExecutorConfigEntry(name="default-exec", backend="docker")])
        monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: deployments_cfg))
        agents_cfg = AgentsConfig.get()
        monkeypatch.setattr(agents_cfg.deployments, "default_executor", "default-exec")
        monkeypatch.setattr(agents_cfg.deployments, "k8s_executor", None)
        monkeypatch.setattr(AgentsConfig, "get", classmethod(lambda cls: agents_cfg))
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={
                "agent": "fabric-agent",
                "name": "fabric-dep",
                "deployment_mode": "k8s",
                "image": "registry.example/agent:1.0",
            },
        )

        assert resp.status_code == 400
        assert "runs on 'docker'" in resp.json()["detail"]
        # The controller would have failed this on reconcile; nothing should persist.
        mock_entity_client.create.assert_not_called()

    def test_create_rejects_container_deployment_with_no_resolvable_image(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep", "deployment_mode": "docker"},
        )

        assert resp.status_code == 400
        assert "requires a container image" in resp.json()["detail"]
        assert "deployments.default_image" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_create_allows_container_deployment_without_image_when_a_default_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = AgentsConfig.get()
        monkeypatch.setattr(config.deployments, "default_image", "registry.example/agent:pinned")
        monkeypatch.setattr(AgentsConfig, "get", classmethod(lambda cls: config))

        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep", "deployment_mode": "docker"},
        )

        assert resp.status_code == 201
        created_deployment: AgentDeployment = mock_entity_client.create.call_args[0][0]
        assert created_deployment.image == ""

    def test_create_ignores_raw_auth_context_headers(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={
                "agent": "fabric-agent",
                "name": "fabric-dep",
                "deployment_mode": "docker",
                "image": "registry.example/agent:1.0",
            },
            headers={
                "X-NMP-Principal-Id": "user:alice",
                "X-NMP-Principal-Email": "alice@example.com",
                "X-NMP-Principal-Groups": "research,platform",
            },
        )

        assert resp.status_code == 201
        created_deployment: AgentDeployment = mock_entity_client.create.call_args[0][0]
        assert created_deployment.auth_context is None

    def test_create_captures_runtime_auth_context(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        with patch(
            "nemo_agents_plugin.api.v2.deployments.current_auth_context",
            return_value=AuthContext(
                principal_id="bearer@example.com",
                principal_email="bearer@example.com",
                principal_groups=["runtime"],
            ),
        ):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments",
                json={
                    "agent": "fabric-agent",
                    "name": "fabric-dep",
                    "deployment_mode": "docker",
                    "image": "registry.example/agent:1.0",
                },
            )

        assert resp.status_code == 201
        created_deployment: AgentDeployment = mock_entity_client.create.call_args[0][0]
        assert created_deployment.auth_context is not None
        assert created_deployment.auth_context.principal_id == "bearer@example.com"
        assert created_deployment.auth_context.principal_email == "bearer@example.com"
        assert created_deployment.auth_context.principal_groups == ["runtime"]

    def test_create_allows_subprocess_deployment_without_an_image(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep", "deployment_mode": "subprocess"},
        )

        assert resp.status_code == 201

    def test_create_rejects_image_entrypoint_for_subprocess(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent())
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep", "use_image_entrypoint": True},
        )

        assert resp.status_code == 400
        assert "use_image_entrypoint requires deployment_mode" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_create_with_environment_ref_snapshots_config_and_compute(self) -> None:
        agent = _make_agent()
        environment = AgentEnvironment(
            name="env1",
            workspace="default",
            environment_spec="default/espec",
            sandbox_spec="default/sspec",
            compute_spec="default/cspec",
        )
        espec = AgentEnvironmentSpec(
            name="espec",
            workspace="default",
            env={"CUSTOM": "from-spec"},
            secrets={"APP_TOKEN": "default/app-token"},
        )
        sspec = AgentSandboxSpec(
            name="sspec",
            workspace="default",
            provider="openshell",
            provider_config={"timeout": 30},
        )
        cspec = AgentComputeSpec(name="cspec", workspace="default", resources=ComputeResources(limits={"cpu": "2"}))

        mock_entity_client = AsyncMock()
        # get order: agent (route), then AgentEnvironment, env spec, sandbox spec, compute spec (resolver).
        mock_entity_client.get = AsyncMock(side_effect=[agent, environment, espec, sspec, cspec])

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep", "environment": "default/env1"},
        )

        assert resp.status_code == 201
        created: AgentDeployment = mock_entity_client.create.call_args[0][0]
        # Raw environment ref is snapshotted for provenance.
        assert created.environment == "default/env1"
        # Environment spec env merged into the resolved config.
        assert created.config["environment"]["env"]["CUSTOM"] == "from-spec"
        # Sandbox spec snapshotted onto the deployment.
        assert created.sandbox is not None
        assert created.sandbox.provider == "openshell"
        assert created.sandbox.provider_config == {"timeout": 30}
        # Compute spec snapshotted onto the deployment.
        assert created.compute is not None
        assert created.compute.resources.limits == {"cpu": "2"}
        # Secret env references snapshotted (never merged into config as plaintext).
        assert created.secrets == {"APP_TOKEN": "default/app-token"}
        assert "APP_TOKEN" not in created.config.get("environment", {}).get("env", {})

    def test_create_with_inline_environment(self) -> None:
        agent = _make_agent()
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=agent)

        async def _save_deployment(deployment: AgentDeployment) -> AgentDeployment:
            deployment._id = f"deployment-{deployment.name}-id"
            deployment._created_at = NOW
            return deployment

        mock_entity_client.create = AsyncMock(side_effect=_save_deployment)
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={
                "agent": "fabric-agent",
                "name": "fabric-dep",
                "environment": {
                    "environment_spec": {"env": {"INLINE": "yes"}},
                    "sandbox_spec": {"provider": "openshell", "provider_config": {"timeout": 30}},
                    "compute_spec": {"resources": {"requests": {"cpu": "1"}}},
                },
            },
        )

        assert resp.status_code == 201
        created: AgentDeployment = mock_entity_client.create.call_args[0][0]
        assert created.config["environment"]["env"]["INLINE"] == "yes"
        assert created.sandbox is not None
        assert created.sandbox.provider == "openshell"
        assert created.sandbox.provider_config == {"timeout": 30}
        assert created.compute is not None
        assert created.compute.resources.requests == {"cpu": "1"}
        # Only the agent lookup hit the entity store; inline specs need no deref.
        assert mock_entity_client.get.await_count == 1

    def test_create_rejects_missing_environment_ref(self) -> None:
        agent = _make_agent()
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(side_effect=[agent, NemoEntityNotFoundError("gone")])
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep", "environment": "default/missing"},
        )

        assert resp.status_code == 422
        assert "AgentEnvironment 'missing' not found" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_create_rejects_invalid_platform_agent_config(self) -> None:
        config = _fabric_agent_config()
        config["default_harness"] = "missing"
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_agent(config=config))
        client = _test_client(mock_entity_client)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments",
            json={"agent": "fabric-agent", "name": "fabric-dep"},
        )

        assert resp.status_code == 400
        assert "Invalid agent config" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()


class TestDeleteDeployment:
    def test_delete_marks_deployment_deleting(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="starting"))
        mock_entity_client.update = AsyncMock(return_value=None)
        client = _test_client(mock_entity_client)

        resp = client.delete("/apis/agents/v2/workspaces/default/deployments/fabric-dep")

        assert resp.status_code == 204
        updated: AgentDeployment = mock_entity_client.update.call_args[0][0]
        assert updated.status == "deleting"

    def test_delete_retries_concurrent_update_conflict(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(
            side_effect=[
                _make_deployment(status="pending"),
                _make_deployment(status="starting"),
            ]
        )
        mock_entity_client.update = AsyncMock(side_effect=[NemoEntityConflictError("conflict"), None])
        client = _test_client(mock_entity_client)

        resp = client.delete("/apis/agents/v2/workspaces/default/deployments/fabric-dep")

        assert resp.status_code == 204
        assert mock_entity_client.get.await_count == 2
        assert mock_entity_client.update.await_count == 2

    def test_delete_returns_success_when_entity_disappears_during_retry(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(
            side_effect=[
                _make_deployment(status="pending"),
                NemoEntityNotFoundError("gone"),
            ]
        )
        mock_entity_client.update = AsyncMock(side_effect=NemoEntityConflictError("conflict"))
        client = _test_client(mock_entity_client)

        resp = client.delete("/apis/agents/v2/workspaces/default/deployments/fabric-dep")

        assert resp.status_code == 204

    def test_delete_returns_409_when_conflicts_exhausted(self) -> None:
        mock_entity_client = AsyncMock()
        mock_entity_client.get = AsyncMock(
            side_effect=[
                _make_deployment(status="pending")
                for _ in range(deployments_router_module._DELETE_MARK_ATTEMPTS)  # noqa: SLF001
            ]
        )
        mock_entity_client.update = AsyncMock(side_effect=NemoEntityConflictError("conflict"))
        client = _test_client(mock_entity_client)

        resp = client.delete("/apis/agents/v2/workspaces/default/deployments/fabric-dep")

        assert resp.status_code == 409
        assert mock_entity_client.update.await_count == deployments_router_module._DELETE_MARK_ATTEMPTS  # noqa: SLF001
