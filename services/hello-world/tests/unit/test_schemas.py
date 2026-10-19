# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hello world API schemas."""

from nhx.hello_world.api.v1.hello.schemas import HelloResponse


class TestHelloResponse:
    """Tests for HelloResponse schema."""

    def test_hello_response_creation(self):
        """Test creating a HelloResponse."""
        response = HelloResponse(message="Hello World")

        assert response.message == "Hello World"

    def test_hello_response_dict(self):
        """Test HelloResponse serialization."""
        response = HelloResponse(message="Test message")
        data = response.model_dump()

        assert data == {"message": "Test message"}

    def test_hello_response_json(self):
        """Test HelloResponse JSON serialization."""
        response = HelloResponse(message="Hello")
        json_str = response.model_dump_json()

        assert '"message":"Hello"' in json_str
