# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for global-workspace read sharing (ASTD-526)."""

import pytest
from nmp.core.entities.api.v2.utils import expand_readable_workspaces
from nmp.core.entities.utils.sharing import (
    GLOBAL_WORKSPACE,
    globally_shareable_entity_types,
    is_globally_shareable,
)


def test_global_workspace_is_default() -> None:
    assert GLOBAL_WORKSPACE == "default"


@pytest.mark.parametrize("entity_type", sorted(globally_shareable_entity_types()))
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


def test_shareable_type_gains_global_workspace() -> None:
    assert expand_readable_workspaces({"team-a"}, "model") == {"team-a", "default"}


def test_unshareable_type_is_untouched() -> None:
    assert expand_readable_workspaces({"team-a"}, "agent_session") == {"team-a"}


def test_full_access_passes_through() -> None:
    assert expand_readable_workspaces(None, "model") is None


def test_expansion_does_not_mutate_caller_set() -> None:
    accessible = {"team-a"}
    expand_readable_workspaces(accessible, "model")
    assert accessible == {"team-a"}


def test_member_of_global_workspace_is_unchanged() -> None:
    assert expand_readable_workspaces({"default", "team-a"}, "model") == {"default", "team-a"}
