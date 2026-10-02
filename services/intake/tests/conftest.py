# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for Intake tests using MockEntityClient."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from nhx.intake.api.v2.experiments.dependencies import get_evaluation_rollup_repository
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.service import IntakeService
from nhx.testing import create_test_client


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Create an entity-focused test client without requiring a live ClickHouse."""
    monkeypatch.setattr(IntakeService, "is_ready", AsyncMock(return_value=True))
    intake_config = IntakeConfig(
        clickhouse_config=ClickHouseConfig(url="http://127.0.0.1:1"),
    )
    with create_test_client(
        IntakeService,
        client_type=TestClient,
        dependency_overrides={get_evaluation_rollup_repository: lambda: None},
        service_configs={IntakeService: intake_config},
    ) as tc:
        yield tc
