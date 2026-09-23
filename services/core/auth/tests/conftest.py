# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for auth service."""

import pytest
from nhx.testing.blockbuster import blockbuster_fixture

# Enable BlockBuster to detect blocking calls in async code
blockbuster = blockbuster_fixture(autouse=True)


@pytest.fixture
def auth_config():
    """Fixture for auth service config."""
    from nhx.core.auth.config import AuthConfig

    return AuthConfig()
