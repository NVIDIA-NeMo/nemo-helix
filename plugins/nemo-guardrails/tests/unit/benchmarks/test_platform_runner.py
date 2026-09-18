# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_guardrails_plugin.benchmarks import platform_runner
from nemo_guardrails_plugin.benchmarks.constants import (
    NMP_BENCHMARK_CONTROLLERS,
    NMP_BENCHMARK_SERVICES,
)
from nmp.platform_runner.config import PlatformAppConfig


def test_platform_runner_uses_benchmark_topology(monkeypatch) -> None:
    captured_config: PlatformAppConfig | None = None

    def fake_run_platform(*, config: PlatformAppConfig) -> None:
        nonlocal captured_config
        captured_config = config

    monkeypatch.setattr(platform_runner, "run_platform", fake_run_platform)

    platform_runner.main()

    assert captured_config is not None
    assert tuple(captured_config.services or ()) == NMP_BENCHMARK_SERVICES
    assert tuple(captured_config.controllers or ()) == NMP_BENCHMARK_CONTROLLERS
    assert captured_config.host == "127.0.0.1"
    assert captured_config.port == 8080
