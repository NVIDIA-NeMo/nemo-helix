# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the external plugin install lifecycle.

Usage::

    client = CommunityPluginsClient(base_url="...")
    preview = client.preview_plugin(body=ManifestSource(url="https://adapter.example/plugin.yaml")).data()
    client.install_plugin(
        body=InstallRequest(url=..., manifest_sha256=preview.manifest_sha256, consent=True)
    )
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.community_plugins import endpoints


class _CommunityPluginsMethods:
    get_plugin_catalog = method(endpoints.get_plugin_catalog)
    preview_plugin = method(endpoints.preview_plugin)
    install_plugin = method(endpoints.install_plugin)
    list_installed_plugins = method(endpoints.list_installed_plugins)
    set_plugin_enabled = method(endpoints.set_plugin_enabled)
    uninstall_plugin = method(endpoints.uninstall_plugin)


class CommunityPluginsClient(_CommunityPluginsMethods, NemoClient):
    """Sync client for the external plugin install API."""


class AsyncCommunityPluginsClient(_CommunityPluginsMethods, AsyncNemoClient):
    """Async client for the external plugin install API."""
