# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``nemo_files`` NAT telemetry exporter."""

from __future__ import annotations

import pytest
from nemo_agents_plugin.telemetry.files_service_exporter import (
    NemoFilesTelemetryExporterConfig,
    nemo_files_telemetry_exporter,
)
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY


@pytest.mark.asyncio
async def test_nemo_files_telemetry_exporter_retries_transient_upload_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo.test")
    config = NemoFilesTelemetryExporterConfig(workspace="ws", agent_name="agent")

    async with nemo_files_telemetry_exporter(config, None) as exporter:
        files_client = exporter._files_client
        assert files_client.base_url == "http://nemo.test"
        assert files_client.retry == PLATFORM_DEFAULT_RETRY_POLICY
