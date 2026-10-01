# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the example plugin API.

Endpoints are defined in ``types.endpoints`` as decorated functions; the
client classes expose them as direct methods via ``method()`` wrappers. The
CLI builds these from the ``nemo`` CLI state
(``cli_state(ctx).typed_client(ExampleClient)``), and ``sdk.py`` mounts them
on the platform SDK.
"""

from __future__ import annotations

from nemo_example_plugin.types import endpoints
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method


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
    create_middleware_config = method(endpoints.create_middleware_config)
    list_middleware_configs = method(endpoints.list_middleware_configs)
    get_middleware_config = method(endpoints.get_middleware_config)
    update_middleware_config = method(endpoints.update_middleware_config)
    delete_middleware_config = method(endpoints.delete_middleware_config)


class ExampleClient(_ExampleMethods, NemoClient):
    """Sync client for the example plugin API."""


class AsyncExampleClient(_ExampleMethods, AsyncNemoClient):
    """Async client for the example plugin API."""
