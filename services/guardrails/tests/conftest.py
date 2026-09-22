# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for Guardrails service tests."""

import pytest
from fastapi.testclient import TestClient
from nhx.guardrails.service import GuardrailsService
from nhx.testing import create_test_client

# ============================================================================
# Test Client Fixture
# ============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the Guardrails service."""
    with create_test_client(GuardrailsService, client_type=TestClient) as tc:
        yield tc


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Auto-marks tests based on their location:

    - Tests in e2e/ directories get the 'e2e' marker
    - Tests in integration/ directories get the 'integration' marker
    - Tests without category markers get the 'unit' marker
    """
    # Category markers that determine test type
    category_markers = {"unit", "e2e", "integration", "regression", "canary", "slow", "skip_in_ci"}

    for item in items:
        # Get current marker names
        marker_names = {marker.name for marker in item.iter_markers()}

        # Auto-mark tests in e2e directories
        if "/e2e/" in str(item.fspath):
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")

        # Auto-mark tests in integration directories
        elif "/integration/" in str(item.fspath):
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
                marker_names.add("integration")

        # Auto-mark tests without category markers as unit tests
        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)
