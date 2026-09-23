# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for global-workspace read sharing (ASTD-526)."""

import pytest
from nmp.core.entities.api.v2.utils import can_read_global_workspace, describe_workspaces
from nmp.core.entities.utils.sharing import GLOBAL_WORKSPACE, is_globally_shareable


def test_global_workspace_is_default() -> None:
    assert GLOBAL_WORKSPACE == "default"


@pytest.mark.parametrize("entity_type", ["model", "adapter", "model_provider", "virtual_model"])
def test_registered_types_are_shareable(entity_type: str) -> None:
    assert is_globally_shareable(entity_type)


@pytest.mark.parametrize(
    "entity_type",
    ["agent_session", "platform_job", "role_binding", "access_key", "model_deployment"],
)
def test_unregistered_types_are_not_shareable(entity_type: str) -> None:
    assert not is_globally_shareable(entity_type)


def test_none_entity_type_is_not_shareable() -> None:
    assert not is_globally_shareable(None)


def test_shareable_type_is_readable_for_a_global_workspace_member() -> None:
    assert can_read_global_workspace({"team-a", "default"}, "model")


def test_shareable_type_is_not_readable_without_global_workspace_access() -> None:
    """Sharing resolves into your workspace; it does not grant access to the global one."""
    assert not can_read_global_workspace({"team-a"}, "model")


def test_unshareable_type_is_never_globally_readable() -> None:
    assert not can_read_global_workspace({"team-a", "default"}, "agent_session")


def test_full_access_can_read_the_global_workspace() -> None:
    assert can_read_global_workspace(None, "model")


def test_full_access_still_respects_the_type_registry() -> None:
    assert not can_read_global_workspace(None, "agent_session")


def test_describe_workspaces_counts_inaccessible_ones_without_naming_them() -> None:
    assert describe_workspaces(["team-a", "secret-b", "secret-c"], {"team-a"}) == "team-a, 2 you cannot access"


def test_describe_workspaces_names_all_with_full_access() -> None:
    assert describe_workspaces(["team-a", "team-b"], None) == "team-a, team-b"
