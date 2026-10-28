# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nemo_rl_plugin.contributor import RlContributor
from typer.testing import CliRunner, Result


@pytest.fixture(autouse=True)
def _platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://nhx.test")


def _preflight(*profiles: SimpleNamespace) -> Result:
    jobs = MagicMock()
    jobs.__enter__.return_value.list_execution_profiles.return_value = list(profiles)
    with patch("nhx.customization_common.cli.dry_run.JobsClient", return_value=jobs):
        return CliRunner().invoke(RlContributor().get_cli(), ["preflight"])


def test_rl_preflight_requires_a_kubernetes_backend() -> None:
    result = _preflight(SimpleNamespace(provider="gpu", profile="default", backend="docker"))

    assert result.exit_code == 1
    assert "rl runs on Kubernetes only" in result.output


def test_rl_preflight_passes_on_kubernetes() -> None:
    result = _preflight(SimpleNamespace(provider="gpu", profile="default", backend="kubernetes_job"))

    assert result.exit_code == 0, result.output
    assert "Usable: gpu/default (kubernetes_job)." in result.output
