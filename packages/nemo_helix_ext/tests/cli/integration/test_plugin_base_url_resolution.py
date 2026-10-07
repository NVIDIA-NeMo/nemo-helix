# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin commands target the platform the global ``nemo`` options resolve.

Plugins no longer take a per-command ``--base-url``: they ask the CLI state,
and ``CLIContext`` resolves ``nemo --base-url`` / ``$NHX_BASE_URL`` / the
config file. Each half has unit coverage (``nemo_agents_plugin`` against a
stand-in state, ``CLIContext`` on its own), but only a run through the real
root app proves the two are connected. A plugin that went back to building its
own client from the environment would pass both unit suites and still hit the
wrong host, so these drive ``nemo agents list`` end to end against a real
config file.
"""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml
from nemo_helix_ext.cli.app import app
from typer.testing import CliRunner

CONFIG_HOST = "config-host.test"

_EMPTY_PAGE = {
    "data": [],
    "pagination": {"page": 1, "page_size": 0, "current_page_size": 0, "total_pages": 1, "total_results": 0},
}

runner = CliRunner()


@pytest.fixture
def config_file(isolated_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real config file whose context points at :data:`CONFIG_HOST`.

    Depends on ``isolated_config`` so it runs after that fixture has cleared
    ``NHX_*`` env vars and pointed ``NHX_CONFIG_FILE`` at an empty file.
    """
    del isolated_config
    path = tmp_path / "nemo-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "current_context": "team",
                "clusters": [{"name": "test-cluster", "base_url": f"http://{CONFIG_HOST}"}],
                "users": [{"name": "me", "type": "no-auth"}],
                "contexts": [{"name": "team", "cluster": "test-cluster", "user": "me"}],
            }
        )
    )
    monkeypatch.setenv("NHX_CONFIG_FILE", str(path))
    return path


def _invoke_agents_list(*global_args: str) -> httpx.Request:
    """Run ``nemo [global_args] agents list`` and return the request it issued."""
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_EMPTY_PAGE)

    transport = httpx.MockTransport(_handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    with patch.object(httpx, "Client", _factory):
        result = runner.invoke(app, [*global_args, "agents", "list", "-f", "json"])

    assert result.exit_code == 0, result.output
    assert captured, "no HTTP request was issued"
    return captured[0]


def test_plugin_command_targets_the_config_file_base_url(config_file: Path) -> None:
    del config_file
    assert _invoke_agents_list().url.host == CONFIG_HOST


def test_global_base_url_flag_overrides_the_config_file(config_file: Path) -> None:
    del config_file
    request = _invoke_agents_list("--base-url", "http://flag-host.test:1111")
    assert request.url.host == "flag-host.test"
    assert request.url.port == 1111


def test_nhx_base_url_overrides_the_config_file(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del config_file
    monkeypatch.setenv("NHX_BASE_URL", "http://env-host.test:2222")
    assert _invoke_agents_list().url.host == "env-host.test"


def test_legacy_nemo_base_url_does_not_redirect_the_cli(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``NEMO_BASE_URL`` belonged to the removed per-command flag; the CLI ignores it."""
    del config_file
    monkeypatch.setenv("NEMO_BASE_URL", "http://legacy-host.test:3333")
    assert _invoke_agents_list().url.host == CONFIG_HOST
