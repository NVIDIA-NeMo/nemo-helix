# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re

import pytest
from nemo_helix_plugin.k8s_naming import k8s_safe_name, workspace_name_identity

_HASH8 = re.compile(r"-[0-9a-f]{8}$")
_DNS_LABEL = re.compile(r"^[a-z](?:[a-z0-9-]*[a-z0-9])?$")


def _base_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def test_k8s_safe_name_always_includes_hash_suffix() -> None:
    name = k8s_safe_name("short-name")
    assert _HASH8.search(name)
    assert name == k8s_safe_name("short-name")


def test_k8s_safe_name_empty_base_uses_fallback_prefix() -> None:
    name = k8s_safe_name("")
    assert name.startswith("x-")
    assert len(name) <= 63


def test_k8s_safe_name_hash_input_differs_from_joined_base() -> None:
    base = "dep-foo-bar-baz"
    name_a = k8s_safe_name(base, hash_input=workspace_name_identity("foo", "bar-baz"))
    name_b = k8s_safe_name(base, hash_input=workspace_name_identity("foo-bar", "baz"))
    assert name_a != name_b


def test_k8s_safe_name_rejects_suffix_too_long_for_max_length() -> None:
    with pytest.raises(ValueError, match="max_length is too small"):
        k8s_safe_name("base", max_length=10, suffix="-" * 10)
