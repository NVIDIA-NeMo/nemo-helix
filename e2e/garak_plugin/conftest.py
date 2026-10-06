# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for garak_plugin plugin e2e tests."""

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.garak_plugin.client import GarakPluginClient


@pytest.fixture
def garak_plugin_url(client: NemoClient) -> str:
    """Root URL for raw httpx calls to the garak_plugin plugin (bracketed filter params)."""
    return client.base_url + "/apis/garak-plugin"


@pytest.fixture
def garak_plugin(client: NemoClient) -> GarakPluginClient:
    """Typed garak_plugin client bound to the pooled platform."""
    return GarakPluginClient.from_client(client)
