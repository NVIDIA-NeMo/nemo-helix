# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the guardrails service.

These tests verify:
- Guardrail config endpoints are properly registered in OpenAPI
- Code-defined default configs (default, content-safety, self-check) are loaded correctly on startup
- File-based configs from CONFIG_STORE_PATH are loaded correctly on startup
- Code-defined configs take precedence when a file-based config has the same name
- Basic CRUD operations work

Uses the create_test_client pattern for fast in-memory testing.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Generator, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.core.auth.service import AuthService
from nhx.core.files.service import FilesService
from nhx.core.models.service import ModelsService
from nhx.core.secrets.service import SecretsService
from nhx.guardrails.config import GuardrailsServiceConfig
from nhx.guardrails.service import GuardrailsService
from nhx.platform_seed.config import HelixSeedConfig
from nhx.platform_seed.tasks.seed import run_platform_seed
from nhx.testing.client import create_test_client

# Default workspace for tests
DEFAULT_WORKSPACE = "default"

# All configs expected after seeding: code-defined defaults + file-based configs from the test config store
EXPECTED_DEFAULT_CONFIGS = {"default", "content-safety", "self-check", "test-file-config"}

# Test-only config store for exercising the file-based seeding path
_CONFIG_STORE_PATH = Path(__file__).resolve().parent / "test-config-store"

# The unique file-based config name present in the test config store
_FILE_BASED_TEST_CONFIG = "test-file-config"


def _guardrails_config() -> GuardrailsServiceConfig:
    """Guardrails config with config_store_path set to the test config store (for integration tests)."""
    if not _CONFIG_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Config store not found at {_CONFIG_STORE_PATH}. "
            "File-based config tests require the test-config-store directory."
        )
    return GuardrailsServiceConfig(config_store_path=_CONFIG_STORE_PATH)


# Timeout for waiting for service readiness and for config store population
SERVICE_READY_TIMEOUT_SECONDS = 10
CONFIG_STORE_POPULATED_TIMEOUT_SECONDS = 30


def wait_for_service_ready(client: TestClient, timeout: float = SERVICE_READY_TIMEOUT_SECONDS) -> None:
    """Polls /health/ready until the service reports ready (before background startup runs)."""
    start = time.time()
    while time.time() - start < timeout:
        response = client.get("/health/ready")
        if response.status_code == 200:
            return
        time.sleep(0.1)

    raise TimeoutError(f"Service not ready after {timeout} seconds")


def wait_for_default_configs(
    client: TestClient,
    expected: set[str] = EXPECTED_DEFAULT_CONFIGS,
    timeout: float = CONFIG_STORE_POPULATED_TIMEOUT_SECONDS,
) -> None:
    """Polls the configs endpoint until all expected default configs are present.

    After running the platform seed task, the code-defined default configs (default,
    content-safety, self-check) are in Entity Store. This waits for them to appear
    so tests can assert on them.
    """
    start = time.time()
    last_names: set[str] = set()
    while time.time() - start < timeout:
        response = client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        if response.status_code != 200:
            time.sleep(0.2)
            continue
        data = response.json()
        configs = data.get("data") or []
        last_names = {c.get("name") for c in configs if c.get("name")}
        if expected <= last_names:
            return
        time.sleep(0.2)

    raise TimeoutError(f"Expected configs {expected} not all present after {timeout}s. Last had: {last_names}")


async def _run_platform_seed_async(client: TestClient) -> None:
    """Run platform seed using the test app's dependency provider (no HelixSeedService)."""
    app = cast(FastAPI, client.app)
    guardrails_svc = getattr(app.state, "guardrails_service", None)
    if guardrails_svc is None:
        raise RuntimeError("guardrails_service not on app.state")
    provider = guardrails_svc.dependency_provider
    entity_client = provider.get_entity_client(as_service="platform-seed")
    sdk = provider.get_sdk_client(as_service="platform-seed")
    if entity_client is None or sdk is None:
        raise RuntimeError("entity_client or sdk is None for platform-seed")
    # Only run guardrails seed; evaluator and data_designer can hang or be slow in test (network/FS).
    config = HelixSeedConfig(
        guardrails_config_store_path=_CONFIG_STORE_PATH,
        evaluator_enabled=False,
    )
    await run_platform_seed(entity_client, sdk, config)


def _run_platform_seed_for_test_client(client: TestClient) -> None:
    """Synchronously run the platform seed task using the test client's in-memory app."""

    async def run_with_timeout() -> None:
        await asyncio.wait_for(_run_platform_seed_async(client), timeout=30.0)

    asyncio.run(run_with_timeout())


@pytest.fixture(scope="module")
def http_client() -> Generator[TestClient, None, None]:
    """TestClient with GuardrailsService.

    Sets CONFIG_STORE_PATH to the test config store so both code-defined defaults and
    file-based configs are seeded. Platform seed runs once in the fixture setup.

    Module-scoped: building the 5-service test app and running platform seed costs
    ~14s per setup. Tests use uuid-suffixed config names so they don't collide.
    """
    os.environ["CONFIG_STORE_PATH"] = str(_CONFIG_STORE_PATH)
    try:
        with create_test_client(
            GuardrailsService,
            AuthService,
            SecretsService,
            FilesService,
            ModelsService,
            client_type=TestClient,
            service_configs={
                GuardrailsService: _guardrails_config(),
            },
        ) as client:
            wait_for_service_ready(client)
            _run_platform_seed_for_test_client(client)
            wait_for_default_configs(client)
            yield client
    finally:
        os.environ.pop("CONFIG_STORE_PATH", None)


def _generate_guardrail_config(name: str | None = None):
    """Generate test guardrail config."""
    return {
        "name": name or f"test-config-{uuid.uuid4().hex[:8]}",
        "description": "Guardrail Config for TestGuardrailConfigs integration tests",
        "data": {
            "models": [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "meta/llama-3.3-70b-instruct",
                },
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "meta/llama-3.1-nemoguard-8b-content-safety",
                },
                {
                    "type": "topic_control",
                    "engine": "nim",
                    "model": "meta/llama-3.1-nemoguard-8b-topic-control",
                },
            ],
            "prompts": [
                {
                    "task": "content_safety_check_input $model=content_safety",
                    "content": "Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response.",
                    "output_parser": "nemoguard_parse_prompt_safety",
                    "max_tokens": 100,
                },
                {
                    "task": "content_safety_check_output $model=content_safety",
                    "content": "Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response.",
                    "output_parser": "nemoguard_parse_prompt_safety",
                    "max_tokens": 100,
                },
                {
                    "task": "topic_safety_check_input $model=topic_control",
                    "content": "You are to act as a customer service agent, providing users with factual information in accordance to the knowledge base. Your role is to ensure that you respond only to relevant queries.",
                },
            ],
            "rails": {
                "input": {
                    "flows": [
                        "content safety check input $model=content_safety",
                        "topic safety check input $model=topic_control",
                    ]
                },
                "output": {
                    "flows": [
                        "content safety check output $model=content_safety",
                    ]
                },
            },
        },
    }


class TestGuardrailsOpenAPI:
    """Tests for guardrails routes in OpenAPI spec."""

    def test_guardrail_configs_routes_in_openapi(self, http_client: TestClient):
        """Test that guardrail config endpoints are documented in OpenAPI spec."""
        response = http_client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        paths = spec.get("paths", {})

        # Verify guardrail config endpoints are present (OpenAPI uses {workspace} template)
        assert "/apis/guardrails/v2/workspaces/{workspace}/configs" in paths
        assert "get" in paths["/apis/guardrails/v2/workspaces/{workspace}/configs"]
        assert "post" in paths["/apis/guardrails/v2/workspaces/{workspace}/configs"]
        assert "get" in paths["/apis/guardrails/v2/workspaces/{workspace}/configs/{name}"]

    def test_guardrail_checks_routes_in_openapi(self, http_client: TestClient):
        """Test that guardrail check endpoints are documented in OpenAPI spec."""
        response = http_client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        paths = spec.get("paths", {})

        # Verify check endpoint is present
        assert "/apis/guardrails/v2/workspaces/{workspace}/checks" in paths
        assert "post" in paths["/apis/guardrails/v2/workspaces/{workspace}/checks"]

    def test_guardrail_inference_routes_not_in_openapi(self, http_client: TestClient):
        """Test that deprecated guardrail inference endpoints are not documented in OpenAPI spec."""
        response = http_client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        paths = spec.get("paths", {})
        schemas = spec.get("components", {}).get("schemas", {})

        assert "/apis/guardrails/v2/workspaces/{workspace}/chat/completions" not in paths
        assert "/apis/guardrails/v2/workspaces/{workspace}/completions" not in paths
        assert "GuardrailChatCompletionRequest" not in schemas
        assert "GuardrailChatCompletionResponse" not in schemas
        assert "GuardrailChatCompletionStreamResponse" not in schemas
        assert "GuardrailCompletionRequest" not in schemas
        assert "GuardrailCompletionResponse" not in schemas
        assert "GuardrailCompletionStreamResponse" not in schemas


class TestGuardrailsDefaultConfigs:
    """Tests for default guardrails configs loaded on startup."""

    def test_default_configs_are_loaded(self, http_client: TestClient):
        """Test that default configs (default, content-safety, self-check) are loaded."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert "sort" in data

        configs_by_name = {config["name"]: config for config in data["data"]}

        assert len(configs_by_name) == len(EXPECTED_DEFAULT_CONFIGS), (
            f"Expected {len(EXPECTED_DEFAULT_CONFIGS)} configs, got {len(configs_by_name)}"
        )

        # Verify all expected default configs are present with entity metadata
        for expected_name in EXPECTED_DEFAULT_CONFIGS:
            assert expected_name in configs_by_name, f"Expected config '{expected_name}' not found"
            config = configs_by_name[expected_name]
            assert config.get("id")
            assert config.get("created_at")
            assert config.get("updated_at")

    def test_default_config_has_passthrough_true(self, http_client: TestClient):
        """Test that the 'default' config has passthrough=true."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        data = response.json()
        default_config = next((c for c in data["data"] if c["name"] == "default"), None)
        assert default_config is not None, "Default config not found"

        # Default config should have passthrough enabled
        assert default_config["data"]["passthrough"] is True

    def test_content_safety_config_has_models_and_rails(self, http_client: TestClient):
        """Test that the 'content-safety' config has the NemoGuard model and input/output rails."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        data = response.json()
        cs_config = next((c for c in data["data"] if c["name"] == "content-safety"), None)
        assert cs_config is not None, "content-safety config not found"

        # Should have a content_safety model
        models = cs_config["data"]["models"]
        content_safety_model = next((m for m in models if m["type"] == "content_safety"), None)
        assert content_safety_model is not None, "content-safety config should have a content_safety model"
        assert content_safety_model["model"] == f"{SYSTEM_WORKSPACE}/nvidia-llama-3-1-nemotron-safety-guard-8b-v3"

        # Should have both input and output rails
        input_flows = cs_config["data"]["rails"]["input"]["flows"]
        assert any("content safety check input" in f for f in input_flows)

        output_flows = cs_config["data"]["rails"]["output"]["flows"]
        assert any("content safety check output" in f for f in output_flows)

        # Should be passthrough
        assert cs_config["data"]["passthrough"] is True

    def test_self_check_config_has_llm_model(self, http_client: TestClient):
        """Test that the 'self-check' config has a main model and input rail configured."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        data = response.json()
        self_check_config = next((c for c in data["data"] if c["name"] == "self-check"), None)
        assert self_check_config is not None, "Self-check config not found"

        # Self-check config should have models configured
        models = self_check_config["data"]["models"]
        assert len(models) > 0, "Self-check config should have at least one model"
        main_model = next((m for m in models if m["type"] == "main"), None)
        assert main_model is not None, "Self-check config should have a main model"

        # Should have a self check input rail
        input_flows = self_check_config["data"]["rails"]["input"]["flows"]
        assert "self check input" in input_flows

        # Should be passthrough
        assert self_check_config["data"]["passthrough"] is True

    def test_configs_have_workspace(self, http_client: TestClient):
        """Test that all seeded configs are in the system workspace."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        data = response.json()

        # Ensure we have configs (don't pass vacuously on empty list)
        assert len(data["data"]) > 0, "Expected at least one config to be present"

        for config in data["data"]:
            assert config["workspace"] == SYSTEM_WORKSPACE, (
                f"Config '{config['name']}' has workspace '{config['workspace']}', expected '{SYSTEM_WORKSPACE}'"
            )


class TestGuardrailConfigs:
    """Tests for guardrails config CRUD operations."""

    def test_create_config(self, http_client: TestClient):
        """Test creating a guardrail config."""
        unique_name = f"test-config-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        response = http_client.post(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data)

        assert response.status_code == 201, f"Create failed: {response.text}"
        data = response.json()
        assert data["name"] == unique_name

        # Verify entity metadata fields are populated
        assert data.get("id")
        assert data.get("created_at")
        assert data.get("updated_at")

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

    def test_get_config(self, http_client: TestClient):
        """Test getting a guardrail config by name."""
        unique_name = f"get-config-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        # Create config
        create_response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data
        )
        assert create_response.status_code == 201
        created_data = create_response.json()
        config_id = create_response.json()["id"]

        # Get config
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == config_id
        assert data["name"] == unique_name

        # Verify entity metadata fields match the create response
        assert data["id"] == created_data["id"]
        assert data["created_at"] == created_data["created_at"]
        assert data.get("updated_at")

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

    def test_list_configs(self, http_client: TestClient):
        """Test listing guardrail configs."""
        unique_name = f"list-config-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        # Create config
        create_response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data
        )
        assert create_response.status_code == 201
        config_id = create_response.json()["id"]

        # List configs
        response = http_client.get("/apis/guardrails/v2/workspaces/default/configs")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "sort" in data
        # Should contain at least our created config
        assert any(c["id"] == config_id for c in data["data"])

        # Verify all configs in the list have entity metadata fields populated
        for config in data["data"]:
            assert config.get("id")
            assert config.get("created_at")
            assert config.get("updated_at")

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

    def test_update_config(self, http_client: TestClient):
        """Test updating a guardrail config."""
        unique_name = f"update-config-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        # Create config
        create_response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data
        )
        assert create_response.status_code == 201
        created_data = create_response.json()

        # Update config
        patch_data = {"description": "Updated description"}
        response = http_client.patch(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}", json=patch_data
        )
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"

        # Verify entity metadata fields are preserved
        assert data["id"] == created_data["id"]
        assert data["created_at"] == created_data["created_at"]
        assert data["updated_at"] >= created_data["created_at"]

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

    def test_delete_config(self, http_client: TestClient):
        """Test deleting a guardrail config."""
        unique_name = f"delete-config-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        # Create config
        create_response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data
        )
        assert create_response.status_code == 201

        # Delete config
        response = http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")
        assert response.status_code == 200

        # Verify config is deleted
        get_response = http_client.get(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")
        assert get_response.status_code == 404

    def test_get_nonexistent_config_returns_404(self, http_client: TestClient):
        """Test getting a non-existent config returns 404."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/fake-config-id")
        assert response.status_code == 404

    def test_delete_nonexistent_config_returns_404(self, http_client: TestClient):
        """Test deleting a non-existent config returns 404."""
        response = http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/fake-config-id")
        assert response.status_code == 404

    def test_update_nonexistent_config_returns_404(self, http_client: TestClient):
        """Test updating a non-existent config returns 404."""
        patch_data = {"description": "Updated description"}
        response = http_client.patch(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/fake-config-id", json=patch_data
        )
        assert response.status_code == 404

    def test_create_duplicate_config_returns_409(self, http_client: TestClient):
        """Test that creating a config with an already-existing name returns 409."""
        unique_name = f"dup-config-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        # First creation succeeds
        response = http_client.post(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data)
        assert response.status_code == 201

        # Second creation with the same name must conflict
        response = http_client.post(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data)
        assert response.status_code == 409

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

    def test_create_config_with_minimal_fields(self, http_client: TestClient):
        """Test creating a config with only the required name field returns 201 with null data."""
        unique_name = f"minimal-{uuid.uuid4().hex[:8]}"

        response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs",
            json={"name": unique_name},
        )
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["name"] == unique_name
        assert data.get("data") is None
        assert data.get("description") is None

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

    def test_create_config_missing_name_returns_422(self, http_client: TestClient):
        """Test that omitting the required name field returns 422."""
        response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs",
            json={"description": "No name provided"},
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
        assert "name" in response.text.lower()

    def test_delete_then_recreate_same_name_succeeds(self, http_client: TestClient):
        """Test that a name can be reused after the original config is deleted."""
        unique_name = f"recycle-{uuid.uuid4().hex[:8]}"
        config_data = _generate_guardrail_config(name=unique_name)

        # Create, then delete
        http_client.post(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data)
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")

        # Re-create with the same name must succeed (not 409)
        response = http_client.post(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs", json=config_data)
        assert response.status_code == 201, f"Expected 201 after delete, got {response.status_code}: {response.text}"

        # Cleanup
        http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{unique_name}")


class TestDeprecatedGuardrailsInferenceEndpoints:
    """Tests that deprecated direct guardrails inference endpoints are not registered."""

    @pytest.mark.parametrize(
        "path",
        [
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/chat/completions",
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/completions",
        ],
    )
    def test_deprecated_inference_endpoints_return_404(self, http_client: TestClient, path: str):
        response = http_client.post(path, json={})
        assert response.status_code == 404, f"Expected 404, got {response.status_code}: {response.text}"


class TestGuardrailsChecksValidationErrors:
    """Tests that validation errors are returned correctly by the `/checks` endpoint."""

    def test_checks_missing_messages_returns_422(self, http_client: TestClient):
        """Missing messages in a checks request should return 422."""
        response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/checks",
            json={"model": "meta/llama3-70b-instruct"},
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.json()}"
        assert "messages" in str(response.json()).lower()

    def test_checks_missing_model_returns_422(self, http_client: TestClient):
        """Missing model in a checks request should return 422."""
        response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/checks",
            json={"messages": [{"role": "user", "content": "Hello!"}]},
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.json()}"
        assert "model" in str(response.json()).lower()

    def test_checks_nonexistent_config_id_returns_error(self, http_client: TestClient):
        """Referencing a config_id that does not exist in a checks request should return 400."""
        response = http_client.post(
            f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/checks",
            json={
                "model": "meta/llama3-70b-instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
                "guardrails": {"config_id": "this-config-does-not-exist"},
            },
        )
        assert response.status_code == 400, (
            f"Expected 400 for unknown config_id, got {response.status_code}: {response.json()}"
        )


class TestGuardrailsConfigPagination:
    """Tests for guardrails config pagination."""

    def test_configs_list_has_pagination(self, http_client: TestClient):
        """Test that config listing returns pagination info."""
        response = http_client.get("/apis/guardrails/v2/workspaces/default/configs")
        assert response.status_code == 200

        data = response.json()
        assert "pagination" in data
        assert "sort" in data
        assert "page" in data["pagination"]
        assert "page_size" in data["pagination"]
        assert "total_results" in data["pagination"]

    def test_configs_list_pagination_params(self, http_client: TestClient):
        """Test that pagination parameters work."""
        response = http_client.get("/apis/guardrails/v2/workspaces/default/configs?page=1&page_size=2")
        assert response.status_code == 200

        data = response.json()
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 2


class TestGuardrailsConfigSorting:
    """Tests that the sort parameter actually orders results correctly."""

    def _create_configs(self, http_client: TestClient, count: int = 3) -> list[str]:
        """Create `count` configs and return their names for cleanup."""
        names = []
        for i in range(count):
            name = f"sort-test-{uuid.uuid4().hex[:8]}"
            resp = http_client.post(
                f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs",
                json={"name": name, "description": f"Sorting test config {i}"},
            )
            assert resp.status_code == 201, f"Failed to create config: {resp.text}"
            names.append(name)
        return names

    def _delete_configs(self, http_client: TestClient, names: list[str]) -> None:
        for name in names:
            http_client.delete(f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs/{name}")

    def test_sort_created_at_ascending(self, http_client: TestClient):
        """sort=created_at returns configs in ascending created_at order."""
        names = self._create_configs(http_client)
        try:
            response = http_client.get(
                f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs?sort=created_at&page_size=100"
            )
            assert response.status_code == 200
            timestamps = [c["created_at"] for c in response.json()["data"]]
            assert len(timestamps) >= 2, "Need at least 2 configs to verify ordering"
            assert timestamps == sorted(timestamps)
        finally:
            self._delete_configs(http_client, names)

    def test_sort_created_at_descending(self, http_client: TestClient):
        """sort=-created_at returns configs in descending created_at order."""
        names = self._create_configs(http_client)
        try:
            response = http_client.get(
                f"/apis/guardrails/v2/workspaces/{DEFAULT_WORKSPACE}/configs?sort=-created_at&page_size=100"
            )
            assert response.status_code == 200
            timestamps = [c["created_at"] for c in response.json()["data"]]
            assert len(timestamps) >= 2, "Need at least 2 configs to verify ordering"
            assert timestamps == sorted(timestamps, reverse=True)
        finally:
            self._delete_configs(http_client, names)


class TestGuardrailsFileBasedSeeding:
    """Tests for the file-based seeding path (populate_config_store).

    Uses the test-config-store fixture directory which contains:
    - test-file-config/  — a unique file-based config (verifies the path works)
    - default/           — same name as a code-defined config (verifies code-defined config takes precedence)
    """

    def test_file_based_config_is_loaded(self, http_client: TestClient) -> None:
        """File-based configs from CONFIG_STORE_PATH are created in Entity Store."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        names = {c["name"] for c in response.json()["data"]}
        assert _FILE_BASED_TEST_CONFIG in names, f"Expected file-based config '{_FILE_BASED_TEST_CONFIG}' in {names}"

    def test_file_based_config_has_inline_data(self, http_client: TestClient) -> None:
        """File-based configs are loaded at startup and stored with inline data."""
        response = http_client.get(
            f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs/{_FILE_BASED_TEST_CONFIG}"
        )
        assert response.status_code == 200
        config = response.json()

        assert "files_url" not in config, "files_url should not be exposed in the API response"
        assert config.get("data") is not None, "File-based config should have inline data populated at startup"

    def test_file_based_config_has_auto_generated_description(self, http_client: TestClient) -> None:
        """File-based configs receive an auto-generated description of the form '{name} guardrail config'."""
        response = http_client.get(
            f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs/{_FILE_BASED_TEST_CONFIG}"
        )
        assert response.status_code == 200
        config = response.json()

        assert config["description"] == f"{_FILE_BASED_TEST_CONFIG} guardrail config"

    def test_code_defined_config_takes_precedence_over_file_based(self, http_client: TestClient) -> None:
        """Code-defined defaults are not overwritten by a file-based entry with the same name.

        The test config store contains a 'default/' directory without passthrough: true.
        The code-defined 'default' config has passthrough: true and is seeded first.
        populate_config_store should skip 'default' because it already exists in the system workspace.
        """
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs/default")
        assert response.status_code == 200
        config = response.json()

        # The code-defined default has passthrough: true; the file-based one does not.
        # If code-defined took precedence, this must be True.
        assert config["data"]["passthrough"] is True, (
            "Code-defined 'default' config should have passthrough=true; "
            "file-based entry appears to have overwritten it"
        )
        # Code-defined configs use inline data, not files_url
        assert config.get("files_url") is None, "Code-defined config should not have files_url"

    def test_code_defined_configs_still_present_alongside_file_based(self, http_client: TestClient) -> None:
        """All code-defined defaults are still present when file-based seeding also runs."""
        response = http_client.get(f"/apis/guardrails/v2/workspaces/{SYSTEM_WORKSPACE}/configs")
        assert response.status_code == 200

        names = {c["name"] for c in response.json()["data"]}
        for expected in EXPECTED_DEFAULT_CONFIGS:
            assert expected in names, f"Code-defined config '{expected}' missing after file-based seeding"
