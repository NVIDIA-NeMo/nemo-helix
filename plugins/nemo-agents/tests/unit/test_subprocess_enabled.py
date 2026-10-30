# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_agents_plugin.config import AgentsConfig


def test_subprocess_deployments_are_enabled_by_default() -> None:
    assert AgentsConfig().subprocess_enabled is True


def test_subprocess_deployments_can_be_turned_off_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_AGENTS_SUBPROCESS_ENABLED", "false")

    assert AgentsConfig().subprocess_enabled is False
