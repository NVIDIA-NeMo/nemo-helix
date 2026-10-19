# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from nemo_helix_ext.cli.telemetry.events import _CI_ENV_VARS


@pytest.fixture(autouse=True)
def _isolate_telemetry_env(_disable_cli_telemetry_by_default: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear inherited env that would make telemetry tests depend on the runner.

    Without this, tests that assert the default ``is_ci`` value pass locally but fail when the
    suite runs under GitHub Actions (which sets ``CI`` and ``GITHUB_ACTIONS``). Tests that
    exercise CI detection or telemetry opt-out set those variables explicitly, which overrides
    this fixture.
    """
    for var in _CI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)
    monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)
    yield
