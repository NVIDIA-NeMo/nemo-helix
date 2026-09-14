# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Customization SDK hub — composes contributor backends for the customization namespace."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Self, TypeVar

from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.customization_contributor import (
    CustomizationContributor,
    CustomizationContributorSDKResources,
    CustomizationSDKResourceFactory,
)
from nemo_platform_plugin.discovery import discover_customization_contributors
from nemo_platform_plugin.sdk import NemoPluginSDKResources
from nmp.customization_common.sdk.client import (
    AsyncCustomizationBackendResource,
    AsyncCustomizationClient,
    AsyncCustomizationSDKContext,
    CustomizationBackendResource,
    CustomizationClient,
    CustomizationSDKContext,
    make_async_customization_sdk_context,
    make_customization_sdk_context,
)

logger = logging.getLogger(__name__)

ContextT = TypeVar("ContextT", CustomizationSDKContext, AsyncCustomizationSDKContext)
ResourceT = TypeVar("ResourceT", CustomizationBackendResource, AsyncCustomizationBackendResource)


def _coerce_health_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("customization health response must be a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _mount_contributor_sdk_resources(
    context: ContextT,
    contributors: dict[str, CustomizationContributor],
    resource_selector: Callable[
        [CustomizationContributorSDKResources],
        CustomizationSDKResourceFactory | None,
    ],
    resource_type: type[ResourceT],
) -> dict[str, ResourceT]:
    resources: dict[str, ResourceT] = {}
    for key in sorted(contributors.keys()):
        contributor = contributors[key]
        sdk_resources = contributor.get_sdk_resources()
        if sdk_resources is None:
            continue
        resource_cls = resource_selector(sdk_resources)
        if resource_cls is None:
            continue
        try:
            resource = resource_cls(context)
        except ImportError:
            logger.warning(
                "Customization contributor %r is installed but SDK resources are unavailable",
                key,
            )
            continue
        if not isinstance(resource, resource_type):
            raise TypeError(f"Customization contributor {key!r} SDK resource must be a {resource_type.__name__}")
        resources[key] = resource
    return resources


def _missing_resource_attribute(owner_type_name: str, name: str) -> AttributeError:
    return AttributeError(f"'{owner_type_name}' object has no attribute {name!r}")


def _require_resource_attribute(
    owner_type_name: str,
    name: str,
    resource: ResourceT | None,
) -> ResourceT:
    if resource is None:
        raise _missing_resource_attribute(owner_type_name, name)
    return resource


class Customization:
    """Sync customization SDK namespace."""

    def __init__(self, context: CustomizationSDKContext) -> None:
        self._customization_client: CustomizationClient = context.customization
        contributors = discover_customization_contributors()
        self._contributor_resources: dict[str, CustomizationBackendResource] = _mount_contributor_sdk_resources(
            context,
            contributors,
            lambda resources: resources.sync_resource,
            CustomizationBackendResource,
        )
        self.contributors: Mapping[str, CustomizationBackendResource] = MappingProxyType(self._contributor_resources)
        self._automodel: CustomizationBackendResource | None = self._contributor_resources.get("automodel")
        self._rl: CustomizationBackendResource | None = self._contributor_resources.get("rl")
        self._unsloth: CustomizationBackendResource | None = self._contributor_resources.get("unsloth")

    @classmethod
    def from_client(cls, client: NemoClient) -> Self:
        return cls(make_customization_sdk_context(client))

    @classmethod
    def from_platform(cls, platform: NeMoPlatform) -> Self:
        return cls.from_client(client_from_platform(platform, NemoClient))

    @property
    def automodel(self) -> CustomizationBackendResource:
        return _require_resource_attribute(type(self).__name__, "automodel", self._automodel)

    @property
    def rl(self) -> CustomizationBackendResource:
        return _require_resource_attribute(type(self).__name__, "rl", self._rl)

    @property
    def unsloth(self) -> CustomizationBackendResource:
        return _require_resource_attribute(type(self).__name__, "unsloth", self._unsloth)

    def plugin_status(self) -> dict[str, object]:
        """Return customization router health, including the registered contributors."""
        return _coerce_health_payload(
            self._customization_client.get_customization_health().data().model_dump(mode="json")
        )


class AsyncCustomization:
    """Async customization SDK namespace."""

    def __init__(self, context: AsyncCustomizationSDKContext) -> None:
        self._customization_client: AsyncCustomizationClient = context.customization
        contributors = discover_customization_contributors()
        self._contributor_resources: dict[str, AsyncCustomizationBackendResource] = _mount_contributor_sdk_resources(
            context,
            contributors,
            lambda resources: resources.async_resource,
            AsyncCustomizationBackendResource,
        )
        self.contributors: Mapping[str, AsyncCustomizationBackendResource] = MappingProxyType(
            self._contributor_resources
        )
        self._automodel: AsyncCustomizationBackendResource | None = self._contributor_resources.get("automodel")
        self._rl: AsyncCustomizationBackendResource | None = self._contributor_resources.get("rl")
        self._unsloth: AsyncCustomizationBackendResource | None = self._contributor_resources.get("unsloth")

    @classmethod
    def from_client(cls, client: AsyncNemoClient) -> Self:
        return cls(make_async_customization_sdk_context(client))

    @classmethod
    def from_platform(cls, platform: AsyncNeMoPlatform) -> Self:
        return cls.from_client(client_from_platform(platform, AsyncNemoClient))

    @property
    def automodel(self) -> AsyncCustomizationBackendResource:
        return _require_resource_attribute(type(self).__name__, "automodel", self._automodel)

    @property
    def rl(self) -> AsyncCustomizationBackendResource:
        return _require_resource_attribute(type(self).__name__, "rl", self._rl)

    @property
    def unsloth(self) -> AsyncCustomizationBackendResource:
        return _require_resource_attribute(type(self).__name__, "unsloth", self._unsloth)

    async def plugin_status(self) -> dict[str, object]:
        """Return customization router health, including the registered contributors."""
        response = await self._customization_client.get_customization_health()
        return _coerce_health_payload(response.data().model_dump(mode="json"))


customization_sdk_resources = NemoPluginSDKResources[Customization, AsyncCustomization](
    sync_resource=Customization.from_platform,
    async_resource=AsyncCustomization.from_platform,
)
