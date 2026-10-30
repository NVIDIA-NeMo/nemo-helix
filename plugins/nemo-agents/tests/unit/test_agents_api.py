# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Agent CRUD route handlers.

Uses FastAPI's TestClient with dependency_overrides to mock EntityClient.
No network or real entity store required.

The test app mounts the agents router at the same prefix as the platform
would: ``/apis/agents``.  This means full URLs match production paths.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import agents as agents_router_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client, get_files_client
from nemo_agents_plugin.entities import (
    NAT_WORKFLOW_CONFIG_FORMAT,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    Agent,
    AgentDeployment,
    DeploymentStatus,
)
from nemo_helix_plugin.client.errors import InternalServerError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
    NemoPaginationInfo,
)
from nemo_helix_plugin.files.client import AsyncFilesClient

NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    name: str = "calc",
    workspace: str = "default",
    description: str = "",
    config: dict | None = None,
    config_format: str = NAT_WORKFLOW_CONFIG_FORMAT,
) -> Agent:
    """Return a populated Agent entity (simulates what the entity store returns)."""
    a = Agent(
        name=name,
        workspace=workspace,
        description=description,
        config=config or {},
        config_format=config_format,
    )
    # Simulate fields set by the entity store
    a._id = f"agent-{name}-id"
    a._created_at = NOW
    return a


def _list_response(items: list[Any]) -> MagicMock:
    """Mock the ListResponse returned by EntityClient.list()."""
    resp = MagicMock()
    resp.data = items
    resp.pagination = NemoPaginationInfo(
        page=1,
        page_size=20,
        current_page_size=len(items),
        total_pages=1,
        total_results=len(items),
    )
    return resp


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


# ---------------------------------------------------------------------------
# Fixture — TestClient with mocked EntityClient
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    """A fully mocked async EntityClient."""
    return AsyncMock()


@pytest.fixture
def mock_files_client() -> AsyncMock:
    client = AsyncMock(spec=AsyncFilesClient)
    client.get_fileset.return_value = MagicMock()
    return client


@pytest.fixture
def test_app(mock_entity_client: AsyncMock, mock_files_client: AsyncMock) -> FastAPI:
    """FastAPI app with the agents router mounted to match the production path.

    Production: platform mounts at /apis/agents, RouterSpec.prefix adds
    /v2/workspaces/{workspace}.  We replicate both here.
    """
    app = FastAPI()
    app.include_router(
        agents_router_module.router,
        prefix="/apis/agents/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    app.dependency_overrides[get_files_client] = lambda: mock_files_client
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /agents — register
# ---------------------------------------------------------------------------


class TestRegisterAgent:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"name": "calc", "config": None},
            {"name": "calc", "config": []},
            {"name": "calc", "config": {"config_format": None}},
            {"name": "calc", "reference": {"fileset": ""}},
            {"name": "calc", "reference": {"fileset": "ws/source/extra"}},
            {"name": "calc", "reference": {"fileset": "ws/source#agent.yaml"}},
            {"name": "calc", "reference": {"fileset": "source name"}},
            {"name": "calc", "reference": {"fileset": "source", "container": "alpine"}},
            {"name": "calc", "reference": {"container_image": ""}},
            {"name": "calc", "reference": {"container_image": " agent:1"}},
            {"name": "calc", "reference": {"container_image": "https://nvcr.io/org/agent:1"}},
            {"name": "calc", "reference": {"container_image": "org/Agent:1"}},
            {"name": "calc", "reference": {"container_image": "agent@sha256:abc"}},
            {
                "name": "calc",
                "config": {"config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT},
                "config_format": NAT_WORKFLOW_CONFIG_FORMAT,
            },
        ],
    )
    def test_register_invalid_request_returns_422(
        self, client: TestClient, mock_entity_client: AsyncMock, mock_files_client: AsyncMock, body: dict[str, Any]
    ) -> None:
        resp = client.post("/apis/agents/v2/workspaces/default/agents", json=body)

        assert resp.status_code == 422
        mock_entity_client.create.assert_not_called()
        mock_files_client.get_fileset.assert_not_called()

    def test_register_returns_201(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        saved = _make_agent("calc", description="calculator", config={"llms": {}})
        mock_entity_client.create = AsyncMock(return_value=saved)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "description": "calculator", "config": {"llms": {}}},
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "calc"
        assert body["workspace"] == "default"
        assert body["description"] == "calculator"
        assert body["config"] == {"llms": {}}
        assert body["config_format"] == "nat-workflow-v1"

    def test_register_response_includes_id_and_created_at(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """Agent entity includes id and created_at as computed fields."""
        saved = _make_agent("calc")
        mock_entity_client.create = AsyncMock(return_value=saved)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "config": {}},
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "agent-calc-id"
        assert body["created_at"] is not None

    def test_register_calls_entity_client_create(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        saved = _make_agent("calc")
        mock_entity_client.create = AsyncMock(return_value=saved)

        client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "config": {}},
        )

        mock_entity_client.create.assert_called_once()
        created_entity: Agent = mock_entity_client.create.call_args[0][0]
        assert created_entity.name == "calc"
        assert created_entity.workspace == "default"

    @pytest.mark.parametrize("config", [None, {"llms": {}}, _fabric_agent_config()])
    @pytest.mark.parametrize("fileset", ["agent-source", "other/agent-source"])
    def test_register_stores_references(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
        mock_files_client: AsyncMock,
        config: dict[str, Any] | None,
        fileset: str,
    ) -> None:
        mock_entity_client.create.side_effect = lambda agent: agent
        reference = {"fileset": fileset, "container_image": "nvcr.io/my-org/agent:1.0"}
        body: dict[str, Any] = {"name": "calc", "reference": reference}
        if config is not None:
            body["config"] = config

        resp = client.post("/apis/agents/v2/workspaces/default/agents", json=body)

        assert resp.status_code == 201
        fileset_workspace = "other" if "/" in fileset else "default"
        mock_files_client.get_fileset.assert_awaited_once_with(workspace=fileset_workspace, name="agent-source")
        expected = {**reference, "fileset": f"{fileset_workspace}/agent-source"}
        agent = mock_entity_client.create.call_args[0][0]
        assert agent.reference.model_dump() == expected
        assert "reference" not in agent.config
        assert resp.json()["reference"] == expected
        if config is None:
            assert agent.config == {}
            assert agent.config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT

    @pytest.mark.parametrize(
        ("error", "status_code"),
        [
            (NotFoundError(httpx.Response(404)), 400),
            (PermissionDeniedError(httpx.Response(403)), 403),
            (InternalServerError(httpx.Response(500)), 500),
        ],
    )
    def test_register_rejects_unavailable_fileset(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
        mock_files_client: AsyncMock,
        error: Exception,
        status_code: int,
    ) -> None:
        mock_files_client.get_fileset.side_effect = error

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "reference": {"fileset": "source"}},
        )

        assert resp.status_code == status_code
        mock_entity_client.create.assert_not_called()

    @pytest.mark.parametrize("explicit_format", [False, True])
    def test_register_validates_platform_agent_config(
        self, client: TestClient, mock_entity_client: AsyncMock, explicit_format: bool
    ) -> None:
        async def _save_agent(agent: Agent) -> Agent:
            agent._id = f"agent-{agent.name}-id"
            agent._created_at = NOW
            return agent

        mock_entity_client.create = AsyncMock(side_effect=_save_agent)
        config = _fabric_agent_config()
        body = {"name": "fabric-agent", "description": "Fabric-backed agent", "config": config}
        if explicit_format:
            body["config_format"] = NEMO_AGENTS_SPEC_CONFIG_FORMAT

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json=body,
        )

        assert resp.status_code == 201
        created_entity: Agent = mock_entity_client.create.call_args[0][0]
        assert created_entity.config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT
        assert created_entity.config["config_format"] == NEMO_AGENTS_SPEC_CONFIG_FORMAT
        assert created_entity.config["environment"]["provider"] == "local"
        assert resp.json()["config_format"] == NEMO_AGENTS_SPEC_CONFIG_FORMAT

    def test_register_invalid_platform_agent_config_returns_400(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        config = _fabric_agent_config()
        config["default_harness"] = "missing"

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={
                "name": "fabric-agent",
                "config": config,
            },
        )

        assert resp.status_code == 400
        assert "Invalid agent config" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_register_unsupported_config_format_returns_400(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "custom-agent", "config": {}, "config_format": "custom-v2"},
        )

        assert resp.status_code == 400
        assert "Unsupported config_format" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_register_conflict_returns_409(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.create = AsyncMock(side_effect=NemoEntityConflictError("already exists"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "config": {}},
        )

        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_register_validation_error_returns_422(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.create = AsyncMock(
            side_effect=NemoEntityValidationError("name: string does not match pattern")
        )

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "my agent", "config": {}},
        )

        assert resp.status_code == 422
        assert "pattern" in resp.json()["detail"]

    def test_register_server_error_returns_500(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.create = AsyncMock(side_effect=RuntimeError("db down"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "config": {}},
        )

        assert resp.status_code == 500

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"config": {}},
            {"config": {}, "config_format": NAT_WORKFLOW_CONFIG_FORMAT},
            {"reference": {"container_image": "alpine"}},
        ],
    )
    def test_register_name_only_returns_201(
        self, client: TestClient, mock_entity_client: AsyncMock, mock_files_client: AsyncMock, fields: dict[str, Any]
    ) -> None:
        config_format = fields.get("config_format", NEMO_AGENTS_SPEC_CONFIG_FORMAT)
        mock_entity_client.create.return_value = _make_agent("calc", config_format=config_format)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", **fields},
        )
        assert resp.status_code == 201
        agent = mock_entity_client.create.call_args[0][0]
        assert agent.config == {}
        assert agent.config_format == config_format
        assert resp.json()["config"] == {}
        assert resp.json()["config_format"] == config_format
        mock_files_client.get_fileset.assert_not_called()

    def test_register_preserves_workspace_from_path(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        saved = _make_agent("calc", workspace="prod")
        mock_entity_client.create = AsyncMock(return_value=saved)

        resp = client.post(
            "/apis/agents/v2/workspaces/prod/agents",
            json={"name": "calc", "config": {}},
        )

        assert resp.status_code == 201
        assert resp.json()["workspace"] == "prod"


# ---------------------------------------------------------------------------
# GET /agents — list (NemoListResponse envelope)
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_list_returns_200_with_envelope(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """List response is now a NemoListResponse envelope, not a bare array."""
        agents = [_make_agent("a"), _make_agent("b")]
        mock_entity_client.list = AsyncMock(return_value=_list_response(agents))

        resp = client.get("/apis/agents/v2/workspaces/default/agents")

        assert resp.status_code == 200
        body = resp.json()
        # NemoListResponse envelope
        assert set(body.keys()) == {"data", "pagination", "sort", "filter"}
        assert len(body["data"]) == 2
        assert {a["name"] for a in body["data"]} == {"a", "b"}

    def test_list_response_includes_pagination(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([_make_agent("a")]))

        resp = client.get("/apis/agents/v2/workspaces/default/agents")

        assert resp.status_code == 200
        assert resp.json()["pagination"]["total_results"] == 1

    def test_list_data_includes_id_and_created_at(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Each agent entity in data includes id and created_at as computed fields."""
        mock_entity_client.list = AsyncMock(return_value=_list_response([_make_agent("calc")]))

        resp = client.get("/apis/agents/v2/workspaces/default/agents")

        item = resp.json()["data"][0]
        assert item["id"] == "agent-calc-id"
        assert item["created_at"] is not None

    def test_list_empty_workspace_returns_empty_data(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get("/apis/agents/v2/workspaces/default/agents")

        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_pagination_params_forwarded(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        client.get("/apis/agents/v2/workspaces/staging/agents?page=2&page_size=5&sort=name")

        call_kwargs = mock_entity_client.list.call_args.kwargs
        assert call_kwargs["page"] == 2
        assert call_kwargs["page_size"] == 5
        assert call_kwargs["sort"] == "name"
        assert call_kwargs["workspace"] == "staging"

    def test_list_server_error_returns_500(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(side_effect=RuntimeError("store unavailable"))

        resp = client.get("/apis/agents/v2/workspaces/default/agents")

        assert resp.status_code == 500

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            (
                "filter[labels.helix-agents/managed-by]=insights",
                {"data.labels.helix-agents/managed-by": {"$eq": "insights"}},
            ),
            (
                "filter[labels][helix-agents/managed-by]=insights",
                {"data.labels.helix-agents/managed-by": {"$eq": "insights"}},
            ),
            ("filter[labels.team][$in]=a,b", {"data.labels.team": {"$in": ["a", "b"]}}),
            ("filter[labels][team][in]=a,b", {"data.labels.team": {"$in": ["a", "b"]}}),
            ("filter[config_format]=nemo-agents-spec-v1", {"data.config_format": {"$eq": "nemo-agents-spec-v1"}}),
        ],
    )
    def test_list_translates_filters_to_entity_paths(
        self, client: TestClient, mock_entity_client: AsyncMock, query: str, expected: dict[str, Any]
    ) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get(f"/apis/agents/v2/workspaces/default/agents?{query}")

        assert resp.status_code == 200
        assert mock_entity_client.list.call_args.kwargs["filter_operation"].to_dict() == expected

    def test_list_ands_multiple_label_filters(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get("/apis/agents/v2/workspaces/default/agents?filter[labels][a]=1&filter[labels][b]=2")

        assert resp.status_code == 200
        # Query parsing does not preserve parameter order, so compare the clauses as a set.
        clauses = mock_entity_client.list.call_args.kwargs["filter_operation"].to_dict()["$and"]
        assert sorted(clauses, key=str) == [{"data.labels.a": {"$eq": "1"}}, {"data.labels.b": {"$eq": "2"}}]

    @pytest.mark.parametrize(
        "query",
        [
            "filter[labels]=x",
            "filter[labels][$like]=x",
            "filter[labels][$hasKey]=team",
            "filter[labels][$exists]=true",
            'filter={"labels": {}}',
        ],
    )
    def test_list_rejects_bare_labels_filters(
        self, client: TestClient, mock_entity_client: AsyncMock, query: str
    ) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get(f"/apis/agents/v2/workspaces/default/agents?{query}")

        assert resp.status_code == 400
        mock_entity_client.list.assert_not_awaited()

    def test_list_rejects_unknown_filter_field(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get("/apis/agents/v2/workspaces/default/agents?filter[colour]=red")

        assert resp.status_code == 400
        mock_entity_client.list.assert_not_awaited()


class TestAgentLabels:
    def test_register_stores_labels(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        labels = {"helix-compass/target-agent": "default/support-bot", "team": "search"}
        saved = _make_agent("calc")
        saved.labels = labels
        mock_entity_client.create = AsyncMock(return_value=saved)

        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "calc", "labels": labels})

        assert resp.status_code == 201
        assert mock_entity_client.create.call_args.args[0].labels == labels
        assert resp.json()["labels"] == labels

    def test_register_defaults_labels_to_empty(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.create = AsyncMock(return_value=_make_agent("calc"))

        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "calc"})

        assert resp.status_code == 201
        assert mock_entity_client.create.call_args.args[0].labels == {}
        assert resp.json()["labels"] == {}

    @pytest.mark.parametrize(
        "labels",
        [
            # Dots would split the ``labels.<key>`` filter path.
            {"helix.agents.managed-by": "x"},
            {"helix.agents/managed-by": "x"},
            {"": "x"},
            {"/name": "x"},
            {"a/b/c": "x"},
            {"-leading": "x"},
            {"k" * 64: "x"},
            {"key": "-leading"},
            {"key": "v" * 254},
            {"key": "has space"},
        ],
    )
    def test_register_rejects_invalid_labels(
        self, client: TestClient, mock_entity_client: AsyncMock, labels: dict[str, str]
    ) -> None:
        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "calc", "labels": labels})

        assert resp.status_code == 422
        mock_entity_client.create.assert_not_awaited()

    @pytest.mark.parametrize(
        "labels",
        [
            {"k" * 63: "v" * 253},
            {f"{'p' * 63}/{'n' * 63}": "x"},
            {"key": ""},
            {"Key_1": "v1.2_beta"},
        ],
    )
    def test_register_accepts_boundary_labels(
        self, client: TestClient, mock_entity_client: AsyncMock, labels: dict[str, str]
    ) -> None:
        mock_entity_client.create = AsyncMock(return_value=_make_agent("calc"))

        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "calc", "labels": labels})

        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# GET /agents/{name} — get
# ---------------------------------------------------------------------------


class TestGetAgent:
    def test_get_existing_agent(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc", description="A calculator"))

        resp = client.get("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 200
        assert resp.json()["name"] == "calc"
        assert resp.json()["description"] == "A calculator"
        assert resp.json()["id"] == "agent-calc-id"

    def test_get_not_found_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.get("/apis/agents/v2/workspaces/default/agents/nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_server_error_returns_500(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(side_effect=Exception("connection timeout"))

        resp = client.get("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /agents/{name} — delete
# ---------------------------------------------------------------------------


def _make_deployment(
    name: str = "calc-dep",
    agent: str = "calc",
    workspace: str = "default",
    status: DeploymentStatus = "running",
) -> AgentDeployment:
    return AgentDeployment(name=name, workspace=workspace, agent=agent, status=status)


class TestDeleteAgent:
    def test_delete_existing_returns_204(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.delete = AsyncMock(return_value=None)

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 204

    def test_delete_calls_entity_client_delete(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.delete = AsyncMock(return_value=None)

        client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        mock_entity_client.delete.assert_called_once_with(
            Agent,
            name="calc",
            workspace="default",
        )

    def test_delete_not_found_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))
        mock_entity_client.delete = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/nonexistent")

        assert resp.status_code == 404

    def test_delete_conflict_returns_409(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))
        mock_entity_client.delete = AsyncMock(side_effect=NemoEntityConflictError("changed"))

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 409

    def test_delete_blocks_on_running_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """DELETE /agents/{name} returns 409 when a running deployment references the agent."""
        dep = _make_deployment(agent="calc", status="running")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 409
        assert "calc-dep" in resp.json()["detail"]

    def test_delete_blocks_on_pending_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(agent="calc", status="pending")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 409

    def test_delete_blocks_on_starting_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(agent="calc", status="starting")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 409

    def test_delete_allowed_when_only_failed_deployments(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """failed deployments do not block deletion — they are terminal."""
        dep = _make_deployment(agent="calc", status="failed")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.delete = AsyncMock(return_value=None)

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 204

    def test_delete_allowed_when_only_deleting_deployments(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """deleting deployments do not block — they are already being cleaned up."""
        dep = _make_deployment(agent="calc", status="deleting")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.delete = AsyncMock(return_value=None)

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 204

    def test_delete_only_checks_deployments_for_this_agent(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """A running deployment for a *different* agent must not block deletion."""
        other_dep = _make_deployment(name="other-dep", agent="other-agent", status="running")
        mock_entity_client.list = AsyncMock(return_value=_list_response([other_dep]))
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.delete = AsyncMock(return_value=None)

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 204

    def test_delete_server_error_on_delete_returns_500(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.delete = AsyncMock(side_effect=RuntimeError("db error"))

        resp = client.delete("/apis/agents/v2/workspaces/default/agents/calc")

        assert resp.status_code == 500


class TestAgentRunDeclarations:
    @staticmethod
    def _echo_create(mock_entity_client: AsyncMock) -> None:
        async def _save_agent(agent: Agent) -> Agent:
            agent._id = f"agent-{agent.name}-id"
            agent._created_at = NOW
            return agent

        mock_entity_client.create = AsyncMock(side_effect=_save_agent)

    def test_register_stores_the_run_declarations(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        self._echo_create(mock_entity_client)
        context_schema = {
            "type": "object",
            "properties": {"since": {"type": "string", "format": "date-time"}},
            "additionalProperties": False,
        }

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={
                "name": "analyst",
                "context_schema": context_schema,
                "default_input": "Analyze recent traces.",
                "post_run_actions": ["insights.analysis", "agents.noop"],
            },
        )

        assert resp.status_code == 201
        created: Agent = mock_entity_client.create.call_args.args[0]
        assert created.context_schema == context_schema
        assert created.default_input == "Analyze recent traces."
        assert created.post_run_actions == {"agents.noop", "insights.analysis"}
        # Serialized as a sorted array of unique strings.
        assert resp.json()["post_run_actions"] == ["agents.noop", "insights.analysis"]

    def test_register_defaults_post_run_actions_to_noop(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        self._echo_create(mock_entity_client)

        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "calc"})

        assert resp.status_code == 201
        assert resp.json()["post_run_actions"] == ["agents.noop"]
        assert resp.json()["context_schema"] is None
        assert resp.json()["default_input"] is None

    def test_register_rejects_an_uninstalled_post_run_action(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "post_run_actions": ["agents.noop", "acme.missing"]},
        )

        assert resp.status_code == 400
        assert "Unknown post-run action 'acme.missing'" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_register_rejects_an_empty_post_run_actions(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "calc", "post_run_actions": []})

        assert resp.status_code == 422
        mock_entity_client.create.assert_not_called()

    def test_register_rejects_an_invalid_context_schema(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={"name": "calc", "context_schema": {"type": "objekt"}},
        )

        assert resp.status_code == 400
        assert "context_schema is not a valid JSON Schema" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()

    def test_register_rejects_a_registration_block_inside_config(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """Nothing reads it from a stored config, so accepting it would silently drop the caller's intent."""
        config = {**_fabric_agent_config(), "helix_registration": {"default_input": "hello"}}

        resp = client.post("/apis/agents/v2/workspaces/default/agents", json={"name": "fabric-agent", "config": config})

        assert resp.status_code == 400
        assert "'helix_registration' belongs in agent.yaml only" in resp.json()["detail"]
        mock_entity_client.create.assert_not_called()
