# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.files.api import endpoint_helpers


def test_delegated_secrets_client_keeps_the_callers_access_key_id():
    jti = "ak_" + "b" * 32
    auth_client = AuthClient(
        principal=Principal(id="service-account:ci", access_key_jti=jti),
        config=AuthConfig(enabled=True),
    )
    client = MagicMock()

    with patch.object(endpoint_helpers, "AsyncSecretsClient"):
        endpoint_helpers.delegated_secrets_client(client, auth_client)

    headers = client.with_headers.call_args.args[0]
    assert headers["X-NHX-Principal-Id"] == "service:files"
    assert headers["X-NHX-Subject-Access-Key-Jti"] == jti
    # The caller's own key ID header is blanked so the service call isn't read as the key itself.
    assert headers["X-NHX-Access-Key-Jti"] == ""
