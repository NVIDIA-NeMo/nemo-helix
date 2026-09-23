# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.guardrails.app.services.utils import normalize_config_ids


def test_normalize_config_ids():
    config_ids = ["config1", "namespace/config2"]
    expected = [DEFAULT_WORKSPACE + "/config1", "namespace/config2"]
    result = normalize_config_ids(config_ids, default_workspace=DEFAULT_WORKSPACE)
    assert result == expected


def test_normalize_config_ids_empty():
    config_ids = []
    expected = []
    result = normalize_config_ids(config_ids, default_workspace=DEFAULT_WORKSPACE)
    assert result == expected


def test_normalize_config_ids_uses_supplied_workspace():
    """Unqualified IDs should be prefixed with the given workspace."""
    config_ids = ["my-config", "other-ws/qualified-config"]
    result = normalize_config_ids(config_ids, default_workspace="my-workspace")
    assert result == ["my-workspace/my-config", "other-ws/qualified-config"]


def test_normalize_config_ids_already_qualified_unchanged():
    """IDs that already carry a workspace prefix must not be modified."""
    config_ids = ["ws1/config-a", "ws2/config-b"]
    result = normalize_config_ids(config_ids, default_workspace="ws3")
    assert result == ["ws1/config-a", "ws2/config-b"]


def test_normalize_config_ids_rejects_malformed():
    """IDs with more than one slash or empty segments should raise ValueError."""
    import pytest

    with pytest.raises(ValueError):
        normalize_config_ids(["ws/name/extra"], default_workspace="ws")

    with pytest.raises(ValueError):
        normalize_config_ids(["ws/"], default_workspace="ws")
