# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_insights_plugin.platform_client import make_client
from nemo_insights_plugin.sdk_resources.analysis_jobs import AsyncAnalysisJobsClient, CreateAnalysisJobRequest
from nemo_insights_plugin.types import AnalyzeSpec
from nemo_platform_ext.auth.helpers import NMPOIDCConfig

REMOTE_URL = "https://nemo-platform.example.com"


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


def _analysis_spec() -> AnalyzeSpec:
    return AnalyzeSpec(
        agent="research-agent",
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
        update_analysis_config=True,
    )


def _analysis_jobs_client(handler: httpx.AsyncBaseTransport) -> tuple[AsyncAnalysisJobsClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=handler)
    client = AsyncAnalysisJobsClient(base_url=REMOTE_URL, http_client=http_client)
    return client, http_client


@pytest.mark.asyncio
async def test_async_analysis_jobs_client_posts_to_insights_job_route() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body == {
            "name": "analysis-job",
            "spec": {
                "agent": "research-agent",
                "default_model": "default/gpt-5",
                "fast_model": "default/gpt-5-mini",
                "update_analysis_config": True,
            },
            "custom_fields": {"insights_analysis_agent": "research-agent"},
        }
        return httpx.Response(
            201,
            request=request,
            json={
                "name": "analysis-job",
                "spec": body["spec"],
                "custom_fields": body["custom_fields"],
            },
        )

    client, http_client = _analysis_jobs_client(httpx.MockTransport(handler))
    try:
        job = (
            await client.create_analysis_job(
                workspace="default",
                body=CreateAnalysisJobRequest(
                    name="analysis-job",
                    spec=_analysis_spec(),
                    custom_fields={"insights_analysis_agent": "research-agent"},
                ),
            )
        ).data()
    finally:
        await http_client.aclose()

    assert requests[0].method == "POST"
    assert requests[0].url.path == "/apis/insights/v2/workspaces/default/jobs/analyze-job"
    assert job.name == "analysis-job"
    assert job.spec.agent == "research-agent"


@pytest.mark.asyncio
async def test_async_analysis_jobs_client_lists_with_json_filter() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        filter_param = request.url.params["filter"]
        assert json.loads(filter_param) == {"status": {"$in": ["active"]}}
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [
                    {
                        "name": "analysis-job",
                        "spec": _analysis_spec().model_dump(mode="json"),
                        "status": "active",
                        "custom_fields": {"insights_analysis_agent": "research-agent"},
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 100,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
                "sort": "-created_at",
                "filter": {"status": {"$in": ["active"]}},
            },
        )

    client, http_client = _analysis_jobs_client(httpx.MockTransport(handler))
    try:
        page = (
            await client.list_analysis_jobs(
                workspace="default",
                query_params={
                    "page_size": 100,
                    "filter": json.dumps({"status": {"$in": ["active"]}}),
                },
            )
        ).page()
    finally:
        await http_client.aclose()

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/apis/insights/v2/workspaces/default/jobs/analyze-job"
    assert page.items[0].spec.agent == "research-agent"
