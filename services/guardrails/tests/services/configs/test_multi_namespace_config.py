# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from unittest.mock import MagicMock

import pytest
from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.guardrails.app.services.configs.registry import ConfigRegistry
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.services.rails.service import RailsService
from nhx.guardrails.app.services.utils import normalize_config_ids


@pytest.fixture
def config_registry():
    registry = ConfigRegistry()
    return registry


@pytest.fixture
def rails_registry():
    registry = RailsRegistry()
    return registry


@pytest.fixture
def rails_service(config_registry, rails_registry):
    service = RailsService(config_registry=config_registry, rails_registry=rails_registry)
    return service


@pytest.fixture
def mock_db():
    """Create a mock database with methods for testing."""
    # TODO: check whether nhx_persistence has such fixture

    db = MagicMock()

    # Storage for our mock entities
    db.configs = {}

    # Mock the add method
    async def add_mock(obj):
        # set an ID if not present
        if not getattr(obj, "id", None):
            obj.id = str(uuid.uuid4())

        # Store by {namespace/name}
        key = f"{obj.namespace}/{obj.name}"
        db.configs[key] = obj
        return obj

    # mock the get method
    async def get_mock(cls, **kwargs):
        namespace = kwargs.get("namespace", DEFAULT_WORKSPACE)
        name = kwargs.get("name")

        if not name:
            return None

        key = f"{namespace}/{name}"
        return db.configs.get(key)

    # mock the update method
    async def update_mock(obj):
        key = f"{obj.namespace}/{obj.name}"
        if key in db.configs:
            db.configs[key] = obj
        return obj

    # mock the delete method
    async def delete_mock(obj):
        key = f"{obj.namespace}/{obj.name}"
        if key in db.configs:
            del db.configs[key]
        return obj

    # mock the list method
    async def list_mock(cls):
        return list(db.configs.values())

    db.add = add_mock
    db.get = get_mock
    db.update = update_mock
    db.delete = delete_mock
    db.list = list_mock

    return db


class TestMultiNamespaceConfig:
    """Test the multi-namespace support in the config service."""

    def test_normalize_config_ids_with_namespaces(self):
        """Test normalizing config IDs with different namespaces."""

        config_ids = ["config1", "nvidia/config2", "default/config3"]
        normalized = normalize_config_ids(config_ids, default_workspace=DEFAULT_WORKSPACE)

        assert "default/config1" in normalized
        assert "nvidia/config2" in normalized
        assert "default/config3" in normalized

        # All without namespace
        config_ids = ["config1", "config2", "config3"]
        normalized = normalize_config_ids(config_ids, default_workspace=DEFAULT_WORKSPACE)

        assert "default/config1" in normalized
        assert "default/config2" in normalized
        assert "default/config3" in normalized

        # All with namespace
        config_ids = ["nvidia/config1", "default/config2", "test/config3"]
        normalized = normalize_config_ids(config_ids, default_workspace=DEFAULT_WORKSPACE)

        assert "nvidia/config1" in normalized
        assert "default/config2" in normalized
        assert "test/config3" in normalized

        # NOTE: what if default namespace changes
        # this test fails and we can spot it
