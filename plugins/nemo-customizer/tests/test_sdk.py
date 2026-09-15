# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from nemo_automodel_plugin.sdk.resources import AsyncAutomodelCustomization, AutomodelCustomization
from nemo_customizer.sdk.resources import (
    AsyncCustomization,
    Customization,
    _coerce_health_payload,
    customization_sdk_resources,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.customization_contributor import CustomizationContributorSDKResources
from nemo_platform_plugin.sdk import NemoPluginSDKResources


class _AutomodelContributorStub:
    def get_sdk_resources(self) -> CustomizationContributorSDKResources:
        return CustomizationContributorSDKResources(
            sync_resource=AutomodelCustomization,
            async_resource=AsyncAutomodelCustomization,
        )


class _ContributorWithoutSdk:
    def get_sdk_resources(self) -> None:
        return None


class _InvalidCustomizationResource:
    def __init__(self, _context: object) -> None:
        pass


class _InvalidContributorStub:
    def get_sdk_resources(self) -> CustomizationContributorSDKResources:
        return CustomizationContributorSDKResources(sync_resource=_InvalidCustomizationResource)


def _recording_customization_transport() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/apis/customization/v2/workspaces/team-a/automodel/jobs":
            return httpx.Response(
                201,
                request=request,
                json={
                    "id": "job-id",
                    "name": "job-a",
                    "workspace": "team-a",
                    "spec": {"model": "default/qwen"},
                    "status": "created",
                },
            )
        return httpx.Response(404, request=request, json={"detail": "unexpected request"})

    return httpx.MockTransport(handler), requests


def test_customization_sdk_resources_entry_point_shape() -> None:
    assert isinstance(customization_sdk_resources, NemoPluginSDKResources)
    sync_resource = customization_sdk_resources.sync_resource
    async_resource = customization_sdk_resources.async_resource
    assert sync_resource is not None
    assert async_resource is not None

    platform = NeMoPlatform(base_url="http://localhost:8000", workspace="default")
    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={},
    ):
        assert isinstance(sync_resource(platform), Customization)


def test_customization_composes_automodel_when_contributor_present() -> None:
    from nemo_platform_plugin.client.client import NemoClient

    client = NemoClient(base_url="http://localhost:8000", workspace="default")

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={"automodel": _AutomodelContributorStub()},
    ):
        customization = Customization.from_client(client)

    assert customization.automodel.jobs is not None


def test_customization_skips_contributors_without_sdk() -> None:
    from nemo_platform_plugin.client.client import NemoClient

    client = NemoClient(base_url="http://localhost:8000", workspace="default")

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={"noop": _ContributorWithoutSdk()},
    ):
        customization = Customization.from_client(client)

    assert "noop" not in customization.contributors


def test_customization_composes_contributor_resources_on_typed_nemo_client() -> None:
    from nemo_platform_plugin.client.client import NemoClient

    client = NemoClient(base_url="http://localhost:8000", workspace="default")

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={"automodel": _AutomodelContributorStub()},
    ):
        customization = Customization.from_client(client)

    assert customization.automodel.jobs is not None


def test_customization_accepts_generated_nemo_platform_owner() -> None:
    transport, requests = _recording_customization_transport()
    platform = NeMoPlatform(
        base_url="http://localhost:8000",
        workspace="team-a",
        http_client=httpx.Client(transport=transport),
    )

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={"automodel": _AutomodelContributorStub()},
    ):
        response = Customization.from_platform(platform).automodel.jobs.create(
            spec={"model": "default/qwen"},
            name="job-a",
        )

    assert response.data().name == "job-a"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://localhost:8000/apis/customization/v2/workspaces/team-a/automodel/jobs"


def test_customization_keeps_dynamic_contributors_in_mapping() -> None:
    from nemo_platform_plugin.client.client import NemoClient

    client = NemoClient(base_url="http://localhost:8000", workspace="default")

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={"third_party": _AutomodelContributorStub()},
    ):
        customization = Customization.from_client(client)

    assert "third_party" not in customization.__dict__
    assert customization.contributors["third_party"].jobs is not None


def test_customization_rejects_contributor_resource_without_jobs() -> None:
    from nemo_platform_plugin.client.client import NemoClient

    client = NemoClient(base_url="http://localhost:8000", workspace="default")

    with (
        patch(
            "nemo_customizer.sdk.resources.discover_customization_contributors",
            return_value={"invalid": _InvalidContributorStub()},
        ),
        pytest.raises(TypeError, match="must be a CustomizationBackendResource"),
    ):
        Customization.from_client(client)


def test_plugin_status_hits_versioned_hub_healthz() -> None:
    import httpx
    from nemo_platform_plugin.client.client import NemoClient

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"plugin": "customization", "status": "ok", "contributors": ["automodel"]},
        )

    client = NemoClient(
        base_url="http://localhost:8000",
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={},
    ):
        status = Customization.from_client(client).plugin_status()

    assert requests[0].method == "GET"
    assert str(requests[0].url) == "http://localhost:8000/apis/customization/v2/healthz"
    assert status["contributors"] == ["automodel"]


def test_plugin_status_rejects_non_object_payload() -> None:
    with pytest.raises(TypeError):
        _coerce_health_payload(["not", "an", "object"])


async def test_async_plugin_status_hits_versioned_hub_healthz() -> None:
    from nemo_platform_plugin.client.client import AsyncNemoClient

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"plugin": "customization", "status": "ok", "contributors": []},
        )

    client = AsyncNemoClient(
        base_url="http://localhost:8000",
        workspace="default",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={},
    ):
        status = await AsyncCustomization.from_client(client).plugin_status()

    assert requests[0].method == "GET"
    assert str(requests[0].url) == "http://localhost:8000/apis/customization/v2/healthz"
    assert status["status"] == "ok"


async def test_async_customization_accepts_generated_nemo_platform_owner() -> None:
    transport, requests = _recording_customization_transport()
    platform = AsyncNeMoPlatform(
        base_url="http://localhost:8000",
        workspace="team-a",
        http_client=httpx.AsyncClient(transport=transport),
    )

    with patch(
        "nemo_customizer.sdk.resources.discover_customization_contributors",
        return_value={"automodel": _AutomodelContributorStub()},
    ):
        response = await AsyncCustomization.from_platform(platform).automodel.jobs.create(
            spec={"model": "default/qwen"},
            name="job-a",
        )

    assert response.data().name == "job-a"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://localhost:8000/apis/customization/v2/workspaces/team-a/automodel/jobs"
