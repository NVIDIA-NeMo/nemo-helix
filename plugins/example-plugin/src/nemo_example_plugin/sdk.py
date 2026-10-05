# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the example plugin.

Mounts the typed clients from :mod:`nemo_example_plugin.client` on the
platform SDK.
"""

from __future__ import annotations

from nemo_example_plugin.client import AsyncExampleClient, ExampleClient
from nemo_helix_plugin.sdk import NemoPluginSDKResources

__all__ = ["AsyncExampleClient", "ExampleClient", "example_sdk_resources"]


example_sdk_resources = NemoPluginSDKResources(
    sync_resource=ExampleClient.from_client,
    async_resource=AsyncExampleClient.from_client,
)
