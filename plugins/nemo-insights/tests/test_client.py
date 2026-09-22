# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_insights_plugin.client import AsyncInsightsClient
from nemo_insights_plugin.platform_client import make_client
from nemo_insights_plugin.schema import CreateAnalysisRunRequest
from nemo_platform_ext.auth.helpers import NMPOIDCConfig

REMOTE_URL = "https://nemo-platform.example.com"


@pytest.mark.asyncio
async def test_async_client_submits_analysis_run() -> None:
    body = CreateAnalysisRunRequest(agent="research-agent", default_model="default/model", fast_model="default/fast")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/apis/insights/v2/workspaces/default/analysis-runs"
        payload = json.loads(request.content)
        assert payload["agent"] == body.agent
        assert payload["default_model"] == body.default_model
        assert payload["fast_model"] == body.fast_model
        assert "spec" not in payload
        return httpx.Response(
            201,
            json={
                "run": {"name": "analysis-run", "workspace": "default", "agent": body.agent},
                "job": {"name": "analysis-run", "status": "created"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncInsightsClient(base_url=REMOTE_URL, http_client=http_client)
        response = (await client.create_analysis_run(workspace="default", body=body)).data()
    assert response.run.name == "analysis-run"
    assert response.job_status == "created"


def test_remote_no_auth_ignores_unrelated_local_oauth_context() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("nemo_insights_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch(
            "nemo_insights_plugin.platform_client.discover_nmp_config",
            return_value=NMPOIDCConfig(auth_enabled=False),
        ),
        patch("nemo_insights_plugin.platform_client.AsyncNeMoPlatform") as client_cls,
    ):
        client = make_client(REMOTE_URL)

    client_cls.assert_called_once_with(base_url=REMOTE_URL)
    assert client is client_cls.return_value


def test_remote_auth_uses_local_oauth_context() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("nemo_insights_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch(
            "nemo_insights_plugin.platform_client.discover_nmp_config",
            return_value=NMPOIDCConfig(
                auth_enabled=True,
                client_id="nemo-cli",
                token_endpoint="https://auth.example.com/token",
            ),
        ),
        patch("nemo_insights_plugin.platform_client.AsyncNeMoPlatform") as client_cls,
    ):
        client = make_client(REMOTE_URL)

    client_cls.assert_called_once_with(base_url=REMOTE_URL, config_path=config_path)
    assert client is client_cls.return_value


def test_remote_auth_rejects_http_before_credential_bootstrap() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True
    base_url = "http://nemo-platform.example.com"

    with (
        patch("nemo_insights_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch(
            "nemo_insights_plugin.platform_client.discover_nmp_config",
            return_value=NMPOIDCConfig(
                auth_enabled=True,
                client_id="nemo-cli",
                token_endpoint="https://auth.example.com/token",
            ),
        ),
        patch("nemo_insights_plugin.platform_client.AsyncNeMoPlatform") as client_cls,
        pytest.raises(ValueError, match="non-HTTPS remote URL"),
    ):
        make_client(base_url)

    client_cls.assert_not_called()


def test_remote_discovery_failure_raises_controlled_error() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("nemo_insights_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch(
            "nemo_insights_plugin.platform_client.discover_nmp_config",
            side_effect=httpx.ConnectError("connection refused"),
        ),
        patch("nemo_insights_plugin.platform_client.AsyncNeMoPlatform") as client_cls,
        pytest.raises(RuntimeError, match="could not discover NeMo Platform auth configuration"),
    ):
        make_client(REMOTE_URL)

    client_cls.assert_not_called()


def test_remote_no_auth_allows_http_without_credentials() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True
    base_url = "http://nemo-platform.example.com"

    with (
        patch("nemo_insights_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch(
            "nemo_insights_plugin.platform_client.discover_nmp_config",
            return_value=NMPOIDCConfig(auth_enabled=False),
        ),
        patch("nemo_insights_plugin.platform_client.AsyncNeMoPlatform") as client_cls,
    ):
        client = make_client(base_url)

    client_cls.assert_called_once_with(base_url=base_url)
    assert client is client_cls.return_value
