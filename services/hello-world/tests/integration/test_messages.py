# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HelloWorld message endpoints."""

from typing import Generator

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.hello_world.service import HelloWorldService
from nhx.testing import create_test_client


@pytest.fixture
def sdk() -> Generator[NeMoHelix, None, None]:
    """Create SDK client for testing."""
    with create_test_client(HelloWorldService, client_type=NeMoHelix) as client:
        yield client


class TestMessagesEndpoints:
    """Tests for message CRUD endpoints."""

    def test_create_message(self, sdk: NeMoHelix):
        """Test POST /apis/hello-world/v2/workspaces/{workspace_id}/messages creates a message."""
        response = sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "description": "Test message", "message": "Hello Test"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "my-message"
        assert data["workspace"] == "default"
        assert data["description"] == "Test message"
        assert data["message"] == "Hello Test"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.skip(
        reason="TODO: Re-enable once entity store supports unique constraint on (workspace_id, entity_type, name)"
    )
    def test_create_message_conflict(self, sdk: NeMoHelix):
        """Test POST /apis/hello-world/v2/workspaces/{workspace_id}/messages returns 409 on duplicate."""
        # Create first message
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "message": "Hello"},
        )

        # Try to create duplicate
        response = sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "message": "Hello Again"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_list_messages(self, sdk: NeMoHelix):
        """Test GET /apis/hello-world/v2/workspaces/{workspace_id}/messages lists messages."""
        # Create some messages
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "message-1", "message": "Hello 1"},
        )
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "message-2", "message": "Hello 2"},
        )

        response = sdk._client.get("/apis/hello-world/v2/workspaces/default/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        names = {c["name"] for c in data}
        assert names == {"message-1", "message-2"}

    def test_list_messages_filters_by_workspace(self, sdk: NeMoHelix):
        """Test GET /apis/hello-world/v2/workspaces/{workspace_id}/messages only returns messages for that workspace."""
        # Create additional workspace for filtering test
        client_from_platform(sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name="workspace-2")
        ).data()

        # Create message in default workspace
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "message-1", "message": "Hello 1"},
        )
        # Create message in workspace-2
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/workspace-2/messages",
            json={"name": "message-2", "message": "Hello 2"},
        )

        response = sdk._client.get("/apis/hello-world/v2/workspaces/default/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "message-1"

    def test_get_message(self, sdk: NeMoHelix):
        """Test GET /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} returns message."""
        # Create a message
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "description": "Test", "message": "Hello"},
        )

        response = sdk._client.get("/apis/hello-world/v2/workspaces/default/messages/my-message")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "my-message"
        assert data["description"] == "Test"
        assert data["message"] == "Hello"

    def test_get_message_not_found(self, sdk: NeMoHelix):
        """Test GET /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} returns 404 if not found."""
        response = sdk._client.get("/apis/hello-world/v2/workspaces/default/messages/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_update_message(self, sdk: NeMoHelix):
        """Test PATCH /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} updates message."""
        # Create a message
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "description": "Original", "message": "Hello"},
        )

        # Update it
        response = sdk._client.patch(
            "/apis/hello-world/v2/workspaces/default/messages/my-message",
            json={"description": "Updated", "message": "Hello Updated"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated"
        assert data["message"] == "Hello Updated"

    def test_update_message_partial(self, sdk: NeMoHelix):
        """Test PATCH /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} with partial update."""
        # Create a message
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "description": "Original", "message": "Hello"},
        )

        # Update only message
        response = sdk._client.patch(
            "/apis/hello-world/v2/workspaces/default/messages/my-message",
            json={"message": "New Message"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Original"  # Unchanged
        assert data["message"] == "New Message"

    def test_update_message_not_found(self, sdk: NeMoHelix):
        """Test PATCH /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} returns 404 if not found."""
        response = sdk._client.patch(
            "/apis/hello-world/v2/workspaces/default/messages/nonexistent",
            json={"message": "Updated"},
        )

        assert response.status_code == 404

    def test_delete_message(self, sdk: NeMoHelix):
        """Test DELETE /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} deletes message."""
        # Create a message
        sdk._client.post(
            "/apis/hello-world/v2/workspaces/default/messages",
            json={"name": "my-message", "message": "Hello"},
        )

        # Delete it
        response = sdk._client.delete("/apis/hello-world/v2/workspaces/default/messages/my-message")

        assert response.status_code == 204

        # Verify it's gone
        get_response = sdk._client.get("/apis/hello-world/v2/workspaces/default/messages/my-message")
        assert get_response.status_code == 404

    def test_delete_message_not_found(self, sdk: NeMoHelix):
        """Test DELETE /apis/hello-world/v2/workspaces/{workspace_id}/messages/{name} returns 404 if not found."""
        response = sdk._client.delete("/apis/hello-world/v2/workspaces/default/messages/nonexistent")

        assert response.status_code == 404
