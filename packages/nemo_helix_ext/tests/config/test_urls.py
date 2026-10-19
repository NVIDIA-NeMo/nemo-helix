# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_ext.config.urls import display_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8080", "http://localhost:8080"),
        ("http://localhost:8080/", "http://localhost:8080"),
        ("https://api.example.com/nhx/", "https://api.example.com/nhx"),
        ("https://s3cr3t-userinfo@api.example.com:8443/nhx", "https://api.example.com:8443/nhx"),
        ("https://api.example.com/?token=abc123#frag", "https://api.example.com"),
        ("http://[::1]:8080", "http://[::1]:8080"),
        ("http://[::1", "<invalid URL>"),
    ],
)
def test_display_url_strips_userinfo_query_and_fragment(url: str, expected: str):
    assert display_url(url) == expected
