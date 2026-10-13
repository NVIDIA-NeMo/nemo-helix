# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EntitiesService health and readiness."""

import pytest
from nhx.core.entities.service import EntitiesService


@pytest.mark.unit
@pytest.mark.asyncio
class TestEntitiesServiceHealth:
    """Test entities service is_ready() and DB health flag."""

    async def test_is_ready_false_when_db_conn_unhealthy(self):
        """is_ready() returns False when _db_conn_healthy is False (default)."""
        service = EntitiesService()
        assert service._db_conn_healthy is False
        result = await service.is_ready()
        assert result is False

    async def test_is_ready_true_when_db_conn_healthy(self):
        """is_ready() returns True when _db_conn_healthy is True."""
        service = EntitiesService()
        service._db_conn_healthy = True
        result = await service.is_ready()
        assert result is True

    async def test_is_ready_reflects_db_conn_healthy_changes(self):
        """is_ready() reflects _db_conn_healthy; /status uses is_ready()."""
        service = EntitiesService()
        service._db_conn_healthy = True
        assert await service.is_ready() is True
        service._db_conn_healthy = False
        assert await service.is_ready() is False
