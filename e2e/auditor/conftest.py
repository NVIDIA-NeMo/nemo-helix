# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for auditor plugin e2e tests."""

import pytest
from nemo_helix import NeMoHelix


@pytest.fixture
def auditor_url(sdk: NeMoHelix) -> str:
    """Root URL for raw httpx calls to the auditor plugin (filter/sort params not in SDK)."""
    return str(sdk.base_url).rstrip("/") + "/apis/auditor"
