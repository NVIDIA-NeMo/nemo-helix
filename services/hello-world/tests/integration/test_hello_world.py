# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the hello-world service.

These tests verify:
- Basic hello endpoint functionality
- Job API routes are properly registered in OpenAPI
- Jobs can be created and executed
- Message entity CRUD operations

Uses the create_test_client pattern for fast in-memory testing.
"""

import time
import uuid
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from nemo_helix import NeMoHelix
from nhx.hello_world.service import HelloWorldService
from nhx.testing.client import SDKTestClientAdapter, create_test_client

# Default workspace for tests
DEFAULT_WORKSPACE = "default"
# Platform mounts hello-world at /apis/hello-world
HELLO_WORLD_API_PREFIX = "/apis/hello-world"

# Skip reason for job execution tests
JOBS_SKIP_REASON = "TODO: Need new pattern for configuring jobs SDK to use ASGI transport"


@pytest.fixture(scope="module")
def http_client() -> Generator[TestClient, None, None]:
    """TestClient with HelloWorldService."""
    with create_test_client(
        HelloWorldService,
        client_type=TestClient,
    ) as client:
        yield client


@pytest.fixture(scope="module")
def sdk(http_client: TestClient) -> NeMoHelix:
    """SDK client backed by the test client."""
    return NeMoHelix(base_url="http://testserver", http_client=SDKTestClientAdapter(http_client))


class TestHelloWorld:
    """Tests for the hello-world service endpoints."""

    def test_hello_endpoint_returns_message(self, sdk: NeMoHelix):
        """Test that /apis/hello-world/v2/workspaces/{workspace}/hello returns the expected message."""
        response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{DEFAULT_WORKSPACE}/hello")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert data["message"] == f"Hello World from workspace '{DEFAULT_WORKSPACE}'"

    def test_hello_endpoint_content_type(self, sdk: NeMoHelix):
        """Test that /apis/hello-world/v2/workspaces/{workspace}/hello returns JSON content type."""
        response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{DEFAULT_WORKSPACE}/hello")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_hello_endpoint_in_openapi(self, sdk: NeMoHelix):
        """Test that /apis/hello-world/v2/workspaces/{workspace}/hello is documented in OpenAPI spec."""
        response = sdk._client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        hello_path_key = f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{{workspace}}/hello"
        assert hello_path_key in spec.get("paths", {})

        # Verify it's tagged correctly
        hello_path = spec["paths"][hello_path_key]
        assert "get" in hello_path
        assert "Hello" in hello_path["get"].get("tags", [])


class TestHelloWorldJobs:
    """Tests for the hello-world job endpoints."""

    def test_jobs_routes_in_openapi(self, sdk: NeMoHelix):
        """Test that job endpoints are documented in OpenAPI spec."""
        response = sdk._client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        paths = spec.get("paths", {})

        # Verify job endpoints are present (platform mounts at /apis/hello-world)
        jobs_path = f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{{workspace}}/jobs"
        assert jobs_path in paths
        assert "post" in paths[jobs_path]
        assert "get" in paths[jobs_path]

    def test_jobs_schema_in_openapi(self, sdk: NeMoHelix):
        """Test that job schemas are in OpenAPI spec."""
        response = sdk._client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        schemas = spec.get("components", {}).get("schemas", {})

        # Verify job-related schemas are present
        assert "HelloWorldJobConfig" in schemas
        assert "HelloWorldJobRequest" in schemas

    def test_job_config_schema_has_message_field(self, sdk: NeMoHelix):
        """Test that HelloWorldJobConfig has message field."""
        response = sdk._client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        schemas = spec.get("components", {}).get("schemas", {})
        job_config = schemas.get("HelloWorldJobConfig", {})
        properties = job_config.get("properties", {})

        assert "message" in properties
        assert properties["message"].get("type") == "string"

    @pytest.mark.skip(reason=JOBS_SKIP_REASON)
    def test_create_job_and_wait_for_completion(self, sdk: NeMoHelix):
        """Test that a job can be created and reaches completed status."""
        job_request = {
            "name": "e2e-hello-world-job",
            "spec": {
                "message": "hello from e2e test",
            },
        }

        # Create the job
        response = sdk._client.post(
            f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{DEFAULT_WORKSPACE}/jobs", json=job_request
        )
        assert response.status_code == 201
        data = response.json()

        job_id = data["id"]
        assert data["status"] == "created"

        # Poll until job reaches a terminal status
        timeout = 60
        start = time.time()
        terminal_statuses = {"completed", "error", "cancelled", "paused"}

        while time.time() - start < timeout:
            response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job_id}")
            assert response.status_code == 200
            job = response.json()
            status = job["status"]

            if status in terminal_statuses:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Job {job_id} did not reach terminal status within {timeout}s")

        assert job["status"] == "completed", f"Expected job to complete, got status: {job['status']}"


class TestHelloWorldMessages:
    """Tests for the hello-world message entity endpoints."""

    def test_messages_routes_in_openapi(self, sdk: NeMoHelix):
        """Test that message endpoints are documented in OpenAPI spec."""
        response = sdk._client.get("/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        paths = spec.get("paths", {})

        # Verify message endpoints are present (platform mounts at /apis/hello-world)
        messages_path = f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{{workspace}}/messages"
        messages_name_path = f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{{workspace}}/messages/{{name}}"
        assert messages_path in paths
        assert "post" in paths[messages_path]
        assert "get" in paths[messages_path]
        assert messages_name_path in paths

    def test_message_crud_lifecycle(self, sdk: NeMoHelix):
        """Test full CRUD lifecycle for messages."""
        # Use default workspace (auto-created by entity-store)
        test_workspace = DEFAULT_WORKSPACE
        test_name = f"test-message-{uuid.uuid4().hex[:8]}"

        # CREATE
        message_data = {
            "name": test_name,
            "message": "Hello from e2e test",
            "description": "Test message",
        }
        response = sdk._client.post(
            f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages", json=message_data
        )
        assert response.status_code == 201, f"Create failed: {response.text}"
        created = response.json()
        assert created["name"] == test_name
        assert created["message"] == "Hello from e2e test"
        assert "id" in created

        # READ (single)
        response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages/{test_name}")
        assert response.status_code == 200, f"Get failed: {response.text}"
        fetched = response.json()
        assert fetched["id"] == created["id"]
        assert fetched["message"] == "Hello from e2e test"

        # LIST
        response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages")
        assert response.status_code == 200, f"List failed: {response.text}"
        messages = response.json()
        assert any(m["id"] == created["id"] for m in messages)

        # UPDATE
        update_data = {"message": "Updated message"}
        response = sdk._client.patch(
            f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages/{test_name}", json=update_data
        )
        assert response.status_code == 200, f"Update failed: {response.text}"
        updated = response.json()
        assert updated["message"] == "Updated message"

        # DELETE
        response = sdk._client.delete(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages/{test_name}")
        assert response.status_code == 204, f"Delete failed: {response.text}"

        # Verify deleted
        response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages/{test_name}")
        assert response.status_code == 404

    @pytest.mark.skip(
        reason="TODO: Re-enable once entity store supports unique constraint on (workspace_id, entity_type, name)"
    )
    def test_create_duplicate_message_fails(self, sdk: NeMoHelix):
        """Test that creating a duplicate message returns 409."""
        test_workspace = DEFAULT_WORKSPACE
        test_name = f"dup-message-{uuid.uuid4().hex[:8]}"

        message_data = {
            "name": test_name,
            "message": "First message",
        }

        # Create first message
        response = sdk._client.post(
            f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages", json=message_data
        )
        assert response.status_code == 201

        # Try to create duplicate
        response = sdk._client.post(
            f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages", json=message_data
        )
        assert response.status_code == 409

        # Cleanup
        sdk._client.delete(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{test_workspace}/messages/{test_name}")

    def test_get_nonexistent_message_returns_404(self, sdk: NeMoHelix):
        """Test that getting a non-existent message returns 404."""
        response = sdk._client.get(f"{HELLO_WORLD_API_PREFIX}/v2/workspaces/{DEFAULT_WORKSPACE}/messages/fake-message")
        assert response.status_code == 404
