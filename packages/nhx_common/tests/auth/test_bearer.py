# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.auth.bearer import MalformedBearerTokenError, parse_bearer_authorization_header


@pytest.mark.parametrize(
    ("auth_header", "expected"),
    [
        (None, None),
        ("", None),
        ("Basic token", None),
        ("Bearer token", "token"),
        ("bearer token", "token"),
        ("Bearer    token", "token"),
        (" Bearer token ", "token"),
    ],
)
def test_parse_bearer_authorization_header(auth_header, expected):
    assert parse_bearer_authorization_header(auth_header) == expected


@pytest.mark.parametrize(
    "auth_header",
    [
        "Bearer",
        "Bearer ",
        "Bearer token extra",
    ],
)
def test_parse_bearer_authorization_header_rejects_malformed_bearer(auth_header):
    with pytest.raises(MalformedBearerTokenError):
        parse_bearer_authorization_header(auth_header)
