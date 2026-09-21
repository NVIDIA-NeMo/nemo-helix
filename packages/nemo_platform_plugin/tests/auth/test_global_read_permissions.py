# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the global_read permission marker (ASTD-526)."""

from nemo_platform_plugin.authz_format import flatten_permission_registry, global_read_permissions


def test_flatten_handles_nested_and_flat_keys() -> None:
    data = {
        "authz": {
            "permissions": {
                "models": {
                    "read": {"description": "Read models"},
                    "adapters": {"read": {"description": "Read model adapters"}},
                },
                "audit.configs.read": {"description": "Read audit configs"},
            }
        }
    }
    assert set(flatten_permission_registry(data)) == {
        "models.read",
        "models.adapters.read",
        "audit.configs.read",
    }


def test_flatten_empty_registry() -> None:
    assert flatten_permission_registry({}) == {}
    assert flatten_permission_registry({"authz": {"permissions": None}}) == {}


def test_global_read_permissions_selects_marked_only() -> None:
    data = {
        "authz": {
            "permissions": {
                "models": {
                    "read": {"description": "Read models", "global_read": True},
                    "create": {"description": "Create models"},
                    "adapters": {"read": {"description": "Read adapters", "global_read": True}},
                },
                "secrets": {"read": {"description": "Read secrets"}},
            }
        }
    }
    assert global_read_permissions(data) == ["models.adapters.read", "models.read"]


def test_global_read_permissions_requires_true_not_truthy() -> None:
    data = {
        "authz": {
            "permissions": {
                "models": {
                    "read": {"description": "Read models", "global_read": "yes"},
                    "list": {"description": "List models", "global_read": 1},
                }
            }
        }
    }
    assert global_read_permissions(data) == []


def test_global_read_permissions_empty_when_nothing_marked() -> None:
    data = {"authz": {"permissions": {"secrets": {"read": {"description": "Read secrets"}}}}}
    assert global_read_permissions(data) == []
