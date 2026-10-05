# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for entities service."""

import pytest
from nhx.common.config import Configuration
from nhx.core.entities.config import EntitiesConfig
from nhx.testing.blockbuster import blockbuster_fixture


@pytest.fixture(autouse=True)
def _disable_principal_role_bindings_cache():
    """So tests that add or change role bindings see a fresh list (cache off, same as explicit env)."""
    Configuration.set_override(EntitiesConfig(principal_bindings_cache_enabled=False))
    try:
        yield
    finally:
        Configuration.clear_override(EntitiesConfig)


# Enable BlockBuster to detect blocking calls in async code
blockbuster = blockbuster_fixture(autouse=True)

# Service-specific fixtures are in integration/conftest.py
