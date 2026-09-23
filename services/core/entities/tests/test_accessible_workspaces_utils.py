# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for workspace access helpers in api/v2/utils.py."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from nmp.common.auth import AuthClient
from nmp.common.auth.dependencies import auth_client_context
from nmp.common.auth.models import Principal
from nmp.common.config import AuthConfig
from nmp.core.entities.api.v2.utils import _applicable_principal_strings, get_accessible_workspaces
from nmp.core.entities.entities import Entity

TEST_PRINCIPAL = "test-user@example.com"


def _binding_entity(workspace: str, role: str, revoked_at: str | None = None) -> Entity:
    now = datetime.now(timezone.utc)
    return Entity(
        entity_type="role_binding",
        id=f"binding-{workspace}-{role}",
        workspace=workspace,
        name=f"{TEST_PRINCIPAL}-{workspace}-{role}",
        data={
            "principal": TEST_PRINCIPAL,
            "workspace": workspace,
            "role": role,
            "revoked_at": revoked_at,
        },
        created_at=now,
        updated_at=now,
        db_version=1,
    )


def test_applicable_principal_strings_id_only() -> None:
    p = Principal(id="172d75ab-0866-4d10-b3ab-c42e37bf20b4", email=None, groups=[])
    assert _applicable_principal_strings(p) == ["172d75ab-0866-4d10-b3ab-c42e37bf20b4"]


def test_applicable_principal_strings_id_and_distinct_email() -> None:
    p = Principal(
        id="172d75ab-0866-4d10-b3ab-c42e37bf20b4",
        email="user@example.com",
        groups=[],
    )
    assert _applicable_principal_strings(p) == [
        "172d75ab-0866-4d10-b3ab-c42e37bf20b4",
        "user@example.com",
    ]


def test_applicable_principal_strings_dedupes_when_id_is_email_shaped() -> None:
    """If Principal-Id is already the email, do not duplicate."""
    p = Principal(id="same@example.com", email="same@example.com", groups=[])
    assert _applicable_principal_strings(p) == ["same@example.com"]


def test_applicable_principal_strings_includes_groups() -> None:
    p = Principal(
        id="sub-1",
        email="u@example.com",
        groups=["group-a", "group-b"],
    )
    assert _applicable_principal_strings(p) == ["sub-1", "u@example.com", "group-a", "group-b"]


def test_applicable_principal_strings_group_dedupes_against_id() -> None:
    p = Principal(id="dup", email=None, groups=["dup", "other"])
    assert _applicable_principal_strings(p) == ["dup", "other"]


@pytest.mark.asyncio
async def test_get_accessible_workspaces_excludes_revoked_bindings() -> None:
    bindings_by_principal = {
        TEST_PRINCIPAL: [
            _binding_entity("workspace-active", "viewer"),
            _binding_entity("workspace-revoked", "viewer", revoked_at="2026-01-01T00:00:00Z"),
        ],
    }

    async def list_entities(**kwargs):
        queried = kwargs["filter_op"].value
        return bindings_by_principal.get(queried, []), 0

    entity_repository = AsyncMock()
    entity_repository.list_entities.side_effect = list_entities

    principal = Principal(id=TEST_PRINCIPAL, email=None, groups=[])
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=True), http_client=None)
    token = auth_client_context.set(auth_client)
    try:
        accessible = await get_accessible_workspaces(entity_repository)
    finally:
        auth_client_context.reset(token)

    assert accessible == {"workspace-active"}
