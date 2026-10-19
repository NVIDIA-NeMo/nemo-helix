# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import AbstractContextManager
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
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.customization_contributor import CustomizationContributorSDKResources
from nemo_helix_plugin.sdk import NemoPluginSDKResources


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


def _contributors(**contributors: object) -> AbstractContextManager[object]:
    return patch("nemo_customizer.sdk.resources.discover_customization_contributors", return_value=contributors)


@pytest.fixture
def client() -> NemoClient:
    return NemoClient(base_url="http://localhost:8000", workspace="default")


def test_customization_sdk_resources_entry_point_shape(client: NemoClient) -> None:
    assert isinstance(customization_sdk_resources, NemoPluginSDKResources)
    sync_resource = customization_sdk_resources.sync_resource
    async_resource = customization_sdk_resources.async_resource
    assert sync_resource is not None
    assert async_resource is not None

    with _contributors():
        assert isinstance(sync_resource(client), Customization)


def test_customization_composes_automodel_when_contributor_present(client: NemoClient) -> None:
    with _contributors(automodel=_AutomodelContributorStub()):
        customization = Customization.from_client(client)

    assert customization.automodel.jobs is not None


def test_customization_skips_contributors_without_sdk(client: NemoClient) -> None:
    with _contributors(noop=_ContributorWithoutSdk()):
        customization = Customization.from_client(client)

    assert "noop" not in customization.contributors


def test_customization_keeps_dynamic_contributors_in_mapping(client: NemoClient) -> None:
    with _contributors(third_party=_AutomodelContributorStub()):
        customization = Customization.from_client(client)

    assert "third_party" not in customization.__dict__
    assert customization.contributors["third_party"].jobs is not None


def test_customization_rejects_contributor_resource_without_jobs(client: NemoClient) -> None:
    with (
        _contributors(invalid=_InvalidContributorStub()),
        pytest.raises(TypeError, match="must be a CustomizationBackendResource"),
    ):
        Customization.from_client(client)


def test_plugin_status_hits_versioned_hub_healthz() -> None:
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

    with _contributors():
        status = Customization.from_client(client).plugin_status()

    assert requests[0].method == "GET"
    assert str(requests[0].url) == "http://localhost:8000/apis/customization/v2/healthz"
    assert status["contributors"] == ["automodel"]


def test_plugin_status_rejects_non_object_payload() -> None:
    with pytest.raises(TypeError):
        _coerce_health_payload(["not", "an", "object"])


async def test_async_plugin_status_hits_versioned_hub_healthz() -> None:
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

    with _contributors():
        status = await AsyncCustomization.from_client(client).plugin_status()

    assert requests[0].method == "GET"
    assert str(requests[0].url) == "http://localhost:8000/apis/customization/v2/healthz"
    assert status["status"] == "ok"
