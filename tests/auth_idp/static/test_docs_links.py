# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR

pytestmark = [pytest.mark.auth_idp]


def test_auth_docs_link_to_contrib_references():
    content = Path("docs/auth/authentication/idp-integration.mdx").read_text()
    assert "contrib/auth/authentik" in content
    assert WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR in content
