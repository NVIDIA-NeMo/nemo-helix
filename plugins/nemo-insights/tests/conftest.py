# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the ``nemo insights`` CLI tests; the harness is in ``_insights_cli.py``."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import typer
from _insights_cli import CONFIGURED_DEFAULT, CONFIGURED_FAST, FakeInsightsAPI, WireState, app_with_state
from nemo_helix_plugin.nooa_model_client import ConfiguredModelRefs
from nemo_insights_plugin import cli


@pytest.fixture
def api() -> FakeInsightsAPI:
    return FakeInsightsAPI()


@pytest.fixture
def state(api: FakeInsightsAPI) -> WireState:
    return WireState(api)


@pytest.fixture
def app(state: WireState) -> typer.Typer:
    return app_with_state(state)


@pytest.fixture
def configured_models(monkeypatch: pytest.MonkeyPatch) -> Callable[[], ConfiguredModelRefs]:
    def refs() -> ConfiguredModelRefs:
        return ConfiguredModelRefs(default=CONFIGURED_DEFAULT, fast=CONFIGURED_FAST)

    monkeypatch.setattr(cli, "configured_model_refs", refs)
    return refs


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$NHX_WORKSPACE`` is a real resolver input; keep the developer's out of these tests."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)
