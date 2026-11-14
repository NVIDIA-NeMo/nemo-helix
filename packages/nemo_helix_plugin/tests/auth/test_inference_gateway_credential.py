# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugins read a gateway call's credential exactly as the platform's auth does."""

from __future__ import annotations

import nhx.common.auth.bearer as bearer_mod
import pytest
from nemo_helix_plugin.auth import inference_gateway_credential


@pytest.mark.parametrize(
    ("headers", "credential"),
    [
        pytest.param({"Authorization": "Bearer tok"}, "tok", id="bearer"),
        pytest.param({"authorization": "bearer tok"}, "tok", id="any case"),
        pytest.param({"x-api-key": " key "}, "key", id="x-api-key"),
        pytest.param({"Authorization": "Bearer tok", "x-api-key": "key"}, "tok", id="bearer wins"),
        pytest.param({"Authorization": "Basic abc", "X-Api-Key": "key"}, "key", id="non-bearer falls back"),
        pytest.param({"Authorization": "Bearer a b", "x-api-key": "key"}, None, id="malformed bearer"),
        pytest.param({}, None, id="neither"),
    ],
)
def test_the_credential_is_the_bearer_token_or_else_the_api_key(
    headers: dict[str, str], credential: str | None
) -> None:
    assert inference_gateway_credential(headers) == credential


def test_nothing_is_read_outside_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without nhx_common the platform's bearer helpers can't be imported.
    monkeypatch.delattr(bearer_mod, "MalformedBearerTokenError")

    assert inference_gateway_credential({"Authorization": "Bearer tok"}) is None
