# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.auth.json_payload import JsonObjectDeserializationError, JsonObjectDeserializer


def test_json_object_deserializer_accepts_object_payload() -> None:
    assert JsonObjectDeserializer().deserialize(b'{"jwks_uri":"https://sso.example.com/jwks"}') == {
        "jwks_uri": "https://sso.example.com/jwks"
    }


def test_json_object_deserializer_rejects_non_object_payload() -> None:
    with pytest.raises(JsonObjectDeserializationError, match="JSON payload was not an object"):
        JsonObjectDeserializer().deserialize(b"[]")
