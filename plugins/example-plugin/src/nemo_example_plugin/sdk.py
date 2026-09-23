# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the example plugin.

Endpoints are defined in ``types.endpoints`` as decorated functions.
The client classes expose them as direct methods via ``method()`` wrappers.
"""

from __future__ import annotations

from nemo_example_plugin.types import endpoints
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.sdk import NemoPluginSDKResources


class _ExampleMethods:
    hello = method(endpoints.hello)
    create_item = method(endpoints.create_item)
    list_items = method(endpoints.list_items)
    get_item = method(endpoints.get_item)
    update_item = method(endpoints.update_item)
    delete_item = method(endpoints.delete_item)
    count = method(endpoints.count)
    upload_blob = method(endpoints.upload_blob)
    download_blob = method(endpoints.download_blob)


class ExampleClient(_ExampleMethods, NemoClient):
    """Sync client for the example plugin API."""


class AsyncExampleClient(_ExampleMethods, AsyncNemoClient):
    """Async client for the example plugin API."""


def _make_sync_resource(platform: NeMoHelix) -> ExampleClient:
    return client_from_platform(platform, ExampleClient)


def _make_async_resource(platform: AsyncNeMoHelix) -> AsyncExampleClient:
    return client_from_platform(platform, AsyncExampleClient)


example_sdk_resources = NemoPluginSDKResources(
    sync_resource=_make_sync_resource,
    async_resource=_make_async_resource,
)
