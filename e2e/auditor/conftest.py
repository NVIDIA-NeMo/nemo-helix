# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for auditor plugin e2e tests."""

import pytest
from nemo_helix_plugin.auditor.client import AuditorClient
from nemo_helix_plugin.client.client import NemoClient


@pytest.fixture
def auditor_url(client: NemoClient) -> str:
    """Root URL for raw httpx calls to the auditor plugin (bracketed filter params)."""
    return client.base_url + "/apis/auditor"


@pytest.fixture
def auditor(client: NemoClient) -> AuditorClient:
    """Typed auditor client bound to the pooled platform."""
    return AuditorClient.from_client(client)
