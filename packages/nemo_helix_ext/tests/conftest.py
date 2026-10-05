# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _disable_cli_telemetry_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI telemetry side effects out of non-telemetry tests."""
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")


@pytest.fixture(autouse=True)
def isolated_config(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate tests from local config files and env vars.

    To skip isolation for a specific test, use:
        @pytest.mark.use_real_config
        def test_that_needs_real_config():
            ...
    """
    if request.node.get_closest_marker("use_real_config"):
        return

    # Clear all NHX_ env vars
    for var in list(os.environ):
        if var.startswith("NHX_"):
            monkeypatch.delenv(var, raising=False)

    # Point to an empty config file
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_file))
