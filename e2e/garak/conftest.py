# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for garak plugin e2e tests."""

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.garak.client import GarakClient


@pytest.fixture
def garak_url(client: NemoClient) -> str:
    """Root URL for raw httpx calls to the garak plugin (bracketed filter params)."""
    return client.base_url + "/apis/garak"


@pytest.fixture
def garak(client: NemoClient) -> GarakClient:
    """Typed garak client bound to the pooled platform."""
    return GarakClient.from_client(client)
