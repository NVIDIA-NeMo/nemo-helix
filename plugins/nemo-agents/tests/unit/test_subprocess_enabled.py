# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nemo_agents_plugin import config as agents_config_module
from nemo_agents_plugin.config import AgentsConfig
from nemo_helix_plugin.config import Runtime


@pytest.fixture
def platform_runtime(monkeypatch: pytest.MonkeyPatch):
    def set_runtime(runtime: Runtime) -> None:
        monkeypatch.setattr(agents_config_module, "get_platform_config", lambda: SimpleNamespace(runtime=runtime))

    return set_runtime


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [(Runtime.DOCKER, True), (Runtime.NONE, True), (Runtime.KUBERNETES, False)],
)
def test_unset_follows_platform_runtime(platform_runtime, runtime: Runtime, expected: bool) -> None:
    platform_runtime(runtime)

    assert AgentsConfig().resolved_subprocess_enabled() is expected


def test_explicit_true_wins_on_kubernetes(platform_runtime) -> None:
    platform_runtime(Runtime.KUBERNETES)

    assert AgentsConfig(subprocess_enabled=True).resolved_subprocess_enabled() is True


def test_explicit_false_wins_off_kubernetes(platform_runtime) -> None:
    platform_runtime(Runtime.DOCKER)

    assert AgentsConfig(subprocess_enabled=False).resolved_subprocess_enabled() is False


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_environment_variable_sets_subprocess_enabled(
    monkeypatch: pytest.MonkeyPatch, platform_runtime, value: str, expected: bool
) -> None:
    platform_runtime(Runtime.KUBERNETES if expected else Runtime.DOCKER)
    monkeypatch.setenv("NEMO_AGENTS_SUBPROCESS_ENABLED", value)

    assert AgentsConfig().subprocess_enabled is expected
    assert AgentsConfig().resolved_subprocess_enabled() is expected
