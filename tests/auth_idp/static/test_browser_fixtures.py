# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlparse
from urllib.request import urlopen

pytest_plugins = ("tests.auth_idp.browser_fixtures",)


def test_cross_origin_url_serves_bound_loopback_host(cross_origin_url: str) -> None:
    parsed = urlparse(cross_origin_url)
    assert parsed.hostname == "127.0.0.1"

    with urlopen(cross_origin_url, timeout=5) as response:
        body = response.read()

    assert response.status == 200
    assert b"Cross-origin auth test" in body
