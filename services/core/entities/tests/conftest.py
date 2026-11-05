# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for entities service."""

from nhx.testing.blockbuster import blockbuster_fixture

# Enable BlockBuster to detect blocking calls in async code
blockbuster = blockbuster_fixture(autouse=True)

# Service-specific fixtures are in integration/conftest.py
