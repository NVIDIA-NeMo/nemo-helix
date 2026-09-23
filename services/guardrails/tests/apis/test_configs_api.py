# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Guardrails config API endpoints."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.entities.base import ListResponse, PaginationInfo
from nemo_helix_plugin.inference_middleware import NemoInferenceMiddleware
from nhx.common.entities import EntityNotFoundError
from nhx.common.service.dependencies import get_entity_client
from nhx.core.inference_gateway.api.dependencies import global_middleware_registry
from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.service import GuardrailsService
from nhx.testing import create_test_client


def _empty_page() -> ListResponse:
    """An empty single-page list response, for mocked entity-store lookups."""
    return ListResponse(
        data=[],
        pagination=PaginationInfo(page=1, page_size=200, current_page_size=0, total_pages=0, total_results=0),
    )


class TestGuardrailConfigsAPI:
    """Tests for guardrail config CRUD operations."""

    def test_list_guardrail_configs(self, client: TestClient):
        """Test listing guardrail configs."""
        response = client.get("/apis/guardrails/v2/workspaces/default/configs")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "pagination" in data
        assert "sort" in data

    def test_list_sort_param_is_reflected(self, client: TestClient):
        """The sort query param value is included in the response."""
        response = client.get("/apis/guardrails/v2/workspaces/default/configs?sort=-created_at")
        assert response.status_code == 200
        assert response.json()["sort"] == "-created_at"

    def test_create_config(self, client: TestClient):
        """Test creating a guardrail config."""
        response = client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "test-config",
            },
        )
        assert response.status_code == 201
        json_response = response.json()
        assert "created_at" in json_response
        assert "updated_at" in json_response
        assert json_response["name"] == "test-config"
        assert json_response["workspace"] == "default"  # namespace comes from workspace in URL

    def test_get_guardrail_config(self, client: TestClient):
        """Test getting a guardrail config."""
        # First create a config
        client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "get-test-config",
            },
        )

        # Then get it
        response = client.get("/apis/guardrails/v2/workspaces/default/configs/get-test-config")
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["name"] == "get-test-config"
        assert json_response["workspace"] == "default"  # namespace comes from workspace in URL

    def test_update_config(self, client: TestClient):
        """Test updating a guardrail config."""
        # First create a config
        client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "update-test-config",
                "description": "Original description",
            },
        )

        # Then update it
        response = client.patch(
            "/apis/guardrails/v2/workspaces/default/configs/update-test-config",
            json={"description": "Updated description"},
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["description"] == "Updated description"

    def test_delete_config(self, client: TestClient):
        """Test deleting a guardrail config."""
        # First create a config
        client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "delete-test-config",
            },
        )

        # Then delete it
        response = client.delete("/apis/guardrails/v2/workspaces/default/configs/delete-test-config")
        assert response.status_code == 200
        assert response.json()["message"] == "Resource deleted successfully."

    def test_delete_config_not_found_during_delete(self, client: TestClient):
        """Test a config deleted after lookup returns 404."""
        entities_client = AsyncMock()
        entities_client.get.return_value = GuardrailConfig(name="delete-race-config", workspace="default")
        # Delete first scans for VirtualModels applying this config; nothing references it here.
        entities_client.list.return_value = _empty_page()
        entities_client.delete.side_effect = EntityNotFoundError("not found")
        dependency_overrides = getattr(client.app, "dependency_overrides")
        dependency_overrides[get_entity_client] = lambda: entities_client
        try:
            response = client.delete("/apis/guardrails/v2/workspaces/default/configs/delete-race-config")
        finally:
            dependency_overrides.pop(get_entity_client, None)

        assert response.status_code == 404
        assert response.json()["detail"] == "Guardrail config not found."

    def test_get_config_not_found(self, client: TestClient):
        """Test getting a non-existent config returns 404."""
        response = client.get("/apis/guardrails/v2/workspaces/default/configs/nonexistent-config")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_update_config_not_found(self, client: TestClient):
        """Test updating a non-existent config returns 404."""
        response = client.patch(
            "/apis/guardrails/v2/workspaces/default/configs/nonexistent-config",
            json={"description": "Updated description"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.skip(reason="Test requires custom workspace which is not auto-created in test harness")
    def test_create_config_with_custom_namespace(self, client: TestClient):
        """Test creating a config with a custom namespace via workspace in URL."""
        response = client.post(
            "/apis/guardrails/v2/workspaces/nvidia/configs",  # workspace determines namespace
            json={
                "name": "custom-ns-config",
                "description": "Config in custom namespace",
            },
        )
        assert response.status_code == 201
        json_response = response.json()
        assert json_response["name"] == "custom-ns-config"
        assert json_response["workspace"] == "nvidia"  # namespace comes from workspace in URL

    def test_create_config_with_data(self, client: TestClient):
        """Test creating a config with inline data."""
        response = client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "config-with-data",
                "description": "Config with data",
                "data": {
                    "models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.1-8b-instruct"}],
                    "instructions": [{"type": "general", "content": "You are a helpful AI assistant."}],
                },
            },
        )
        assert response.status_code == 201
        json_response = response.json()
        assert json_response["data"] is not None
        assert "models" in json_response["data"]

    def test_create_config_validation_error(self, client: TestClient):
        """Test that invalid config data returns 422 with a user-friendly error message."""
        response = client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "invalid-config",
                "data": {
                    "rails": {"input": {"flows": ["self check input"]}},
                },
            },
        )

        assert response.status_code == 422
        assert response.json() == {
            "detail": "Validation error at data: Missing a `self_check_input` prompt template, which is required for the `self check input` rail."
        }

    def test_update_config_validation_error(self, client: TestClient):
        """Test that updating a config with invalid data returns 422 with a user-friendly error message."""
        # First, create a valid config
        client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={
                "name": "config-to-update",
                "data": {
                    "rails": {"input": {"flows": ["self check input"]}},
                    "prompts": [
                        {
                            "task": "self_check_input",
                            "content": "Check if the input is safe.",
                        }
                    ],
                },
            },
        )

        # Update config with invalid data
        response = client.patch(
            "/apis/guardrails/v2/workspaces/default/configs/config-to-update",
            json={
                "data": {
                    "rails": {"input": {"flows": ["self check input"]}},
                },
            },
        )

        assert response.status_code == 422
        assert response.json() == {
            "detail": "Validation error at data: Missing a `self_check_input` prompt template, which is required for the `self check input` rail."
        }


class TestGuardrailConfigsFilter:
    """Tests for filtering the guardrail configs list endpoint."""

    @staticmethod
    def _seed_config(client: TestClient, name: str, description: str = "") -> None:
        response = client.post(
            "/apis/guardrails/v2/workspaces/default/configs",
            json={"name": name, "description": description} if description else {"name": name},
        )
        assert response.status_code == 201

    def test_filter_by_name_exact_match(self, client: TestClient):
        """filter[name]=<value> returns only configs with that exact name."""
        self._seed_config(client, "filter-alpha")
        self._seed_config(client, "filter-beta")

        response = client.get("/apis/guardrails/v2/workspaces/default/configs?filter[name]=filter-alpha")
        assert response.status_code == 200
        names = [c["name"] for c in response.json()["data"]]
        assert "filter-alpha" in names
        assert "filter-beta" not in names

    def test_filter_by_name_substring(self, client: TestClient):
        """filter[name][$like] returns configs whose name contains the substring."""
        self._seed_config(client, "substr-apple")
        self._seed_config(client, "substr-apricot")
        self._seed_config(client, "substr-banana")

        response = client.get("/apis/guardrails/v2/workspaces/default/configs?filter[name][%24like]=ap")
        assert response.status_code == 200
        names = {c["name"] for c in response.json()["data"]}
        assert {"substr-apple", "substr-apricot"}.issubset(names)
        assert "substr-banana" not in names

    def test_filter_by_description(self, client: TestClient):
        """filter[description]=<value> filters by description field."""
        self._seed_config(client, "desc-one", description="content-safety rails")
        self._seed_config(client, "desc-two", description="pii detection rails")

        response = client.get("/apis/guardrails/v2/workspaces/default/configs?filter[description]=content-safety rails")
        assert response.status_code == 200
        names = [c["name"] for c in response.json()["data"]]
        assert "desc-one" in names
        assert "desc-two" not in names

    def test_filter_rejects_unknown_field(self, client: TestClient):
        """Filter validation rejects fields not declared on GuardrailConfigFilter."""
        response = client.get("/apis/guardrails/v2/workspaces/default/configs?filter[unknown_field]=x")
        assert response.status_code == 400


CONFIGS = "/apis/guardrails/v2/workspaces/default/configs"
VMS = "/apis/inference-gateway/v2/workspaces/default/virtual-models"


class TestDeleteConfigInUse:
    """Deleting a guardrail config that a VirtualModel applies is refused.

    Runs Guardrails and the Inference Gateway against one in-process entity store, so the
    VirtualModels are created through IGW's real endpoint and the in-use scan reads exactly what
    production would.
    """

    @pytest.fixture
    def client(self) -> Iterator[TestClient]:
        """Guardrails and the Inference Gateway over one in-process entity store."""
        with create_test_client(
            GuardrailsService,
            InferenceGatewayService,
            client_type=TestClient,
            # A second workspace, so the cross-workspace reference case is reachable.
            workspaces=["default", "other"],
        ) as tc:
            # The guardrails middleware plugin isn't loaded in tests; stub it so IGW accepts
            # config_id references without resolving them through the real plugin.
            plugin = MagicMock(spec=NemoInferenceMiddleware)
            plugin.get_middleware_config = AsyncMock(return_value={"stored": True})
            plugin.validate_middleware_config = AsyncMock(side_effect=lambda _type, config: config)
            tc.app.dependency_overrides[global_middleware_registry] = lambda: MiddlewareRegistry(
                plugins={"nemo-guardrails": plugin}
            )
            yield tc

    @staticmethod
    def _rail(config_id: str) -> dict:
        """A middleware call applying the stored guardrail config ``config_id``."""
        return {"name": "nemo-guardrails", "config_type": "guardrail_config", "config_id": config_id}

    def _create_config(self, client: TestClient, name: str) -> None:
        """Create a guardrail config, asserting it was stored."""
        assert client.post(CONFIGS, json={"name": name}).status_code == 201

    def test_delete_refused_while_a_virtual_model_applies_the_config(self, client: TestClient):
        """A referenced config returns 409, names the VirtualModel, and survives."""
        self._create_config(client, "cs")
        assert (
            client.post(VMS, json={"name": "vm-guarded", "request_middleware": [self._rail("default/cs")]}).status_code
            == 201
        )

        resp = client.delete(f"{CONFIGS}/cs")
        assert resp.status_code == 409, resp.text
        assert "default/vm-guarded" in resp.json()["detail"]
        assert client.get(f"{CONFIGS}/cs").status_code == 200

    def test_delete_refused_for_output_rails_too(self, client: TestClient):
        """The scan covers response and post-response pipelines, not just request."""
        self._create_config(client, "cs-out")
        assert (
            client.post(VMS, json={"name": "vm-out", "response_middleware": [self._rail("default/cs-out")]}).status_code
            == 201
        )

        assert client.delete(f"{CONFIGS}/cs-out").status_code == 409

    def test_delete_allowed_when_no_virtual_model_references_it(self, client: TestClient):
        """An unreferenced config still deletes, even with other guarded VirtualModels around."""
        self._create_config(client, "unused")
        self._create_config(client, "used")
        assert (
            client.post(VMS, json={"name": "vm-other", "request_middleware": [self._rail("default/used")]}).status_code
            == 201
        )

        assert client.delete(f"{CONFIGS}/unused").status_code == 200
        assert client.get(f"{CONFIGS}/unused").status_code == 404

    def test_delete_allowed_after_rails_are_detached(self, client: TestClient):
        """Detaching the config releases the config for deletion."""
        self._create_config(client, "cs-detach")
        assert (
            client.post(
                VMS, json={"name": "vm-detach", "request_middleware": [self._rail("default/cs-detach")]}
            ).status_code
            == 201
        )
        assert client.delete(f"{CONFIGS}/cs-detach").status_code == 409

        assert client.patch(f"{VMS}/vm-detach", json={"request_middleware": []}).status_code == 200
        assert client.delete(f"{CONFIGS}/cs-detach").status_code == 200

    def test_delete_not_blocked_by_a_same_named_config_of_another_type(self, client: TestClient):
        """The in-use check compares config_type, so another plugin's config never blocks."""
        self._create_config(client, "shared-name")
        payload = {
            "name": "vm-switchyard",
            "request_middleware": [
                {"name": "nemo-guardrails", "config_type": "routellm_config", "config_id": "default/shared-name"}
            ],
        }
        assert client.post(VMS, json=payload).status_code == 201

        assert client.delete(f"{CONFIGS}/shared-name").status_code == 200

    def test_delete_refused_when_referenced_from_another_workspace(self, client: TestClient):
        """Config references are fully qualified, so the scan spans workspaces."""
        self._create_config(client, "cross")
        payload = {"name": "vm-cross", "request_middleware": [self._rail("default/cross")]}
        assert (
            client.post("/apis/inference-gateway/v2/workspaces/other/virtual-models", json=payload).status_code == 201
        )

        resp = client.delete(f"{CONFIGS}/cross")
        assert resp.status_code == 409, resp.text
        assert "other/vm-cross" in resp.json()["detail"]
