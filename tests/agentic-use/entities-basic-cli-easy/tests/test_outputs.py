# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that entity CRUD operations were performed correctly.

Checks:
- harbor-test-model was deleted (should not exist)
- harbor-final-dataset exists with correct data

TODO(mstaats): We need to verify the agentic path in the future, not just the end state.
"""

import os

import pytest
from nemo_helix_plugin.entities.client import EntitiesClient

WORKSPACE = "default"


@pytest.fixture
def client() -> EntitiesClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return EntitiesClient(base_url=nhx_base_url, workspace=WORKSPACE)


def test_harbor_test_model_deleted(client: EntitiesClient) -> None:
    """Test that harbor-test-model was deleted after CRUD operations."""
    response = client.list_entities(entity_type="model")
    entity_names = [e.name for e in response.page().items]
    assert "harbor-test-model" not in entity_names, (
        f"Entity 'harbor-test-model' should have been deleted but still exists! Found: {entity_names}"
    )


def test_harbor_final_dataset_exists(client: EntitiesClient) -> None:
    """Test that harbor-final-dataset was created and has correct data."""
    response = client.get_entity_by_name(name="harbor-final-dataset", entity_type="dataset").data()
    assert response.name == "harbor-final-dataset", (
        f"Expected entity name 'harbor-final-dataset', got '{response.name}'"
    )
    assert response.entity_type == "dataset", f"Expected entity type 'dataset', got '{response.entity_type}'"
    assert response.data.get("format") == "jsonl", f"Expected data format 'jsonl', got '{response.data.get('format')}'"
    assert response.data.get("size") == 1000, f"Expected data size 1000, got '{response.data.get('size')}'"
