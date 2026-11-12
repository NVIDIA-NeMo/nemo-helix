# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who may see whom: profile resolution is limited to people who share a workspace."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nhx.common.entities import ALL_WORKSPACES, SYSTEM_WORKSPACE, EntityClient
from nhx.core.auth.app.seeding import WILDCARD_PRINCIPAL
from nhx.core.auth.entities import RoleBindingEntity

_PAGE_SIZE = 1000


def _in(field: str, values: Collection[str]) -> ComparisonOperation:
    return ComparisonOperation(operator=FilterOperator.IN, field=field, value=sorted(values))


async def _active_bindings(entities_client: EntityClient, filter_operation: FilterOperation) -> list[RoleBindingEntity]:
    bindings: list[RoleBindingEntity] = []
    page = 1
    while True:
        result = await entities_client.list(
            RoleBindingEntity,
            workspace=ALL_WORKSPACES,
            filter_operation=filter_operation,
            page=page,
            page_size=_PAGE_SIZE,
        )
        bindings.extend(binding for binding in result.data if binding.revoked_at is None)
        if len(result.data) < _PAGE_SIZE:
            return bindings
        page += 1


def _explicit(keys: Collection[str]) -> set[str]:
    return {key for key in keys if key and key != WILDCARD_PRINCIPAL}


async def accounts_sharing_a_workspace(
    entities_client: EntityClient, caller_keys: Collection[str], candidates: Mapping[str, Collection[str]]
) -> set[str]:
    """The candidate account ids bound to at least one workspace the caller is bound to.

    Only explicit bindings count (by subject, email, or group). ``*`` bindings and the system workspace grant
    everyone access, so treating them as membership would let anyone resolve every account.
    """
    mine = _explicit(caller_keys)
    theirs = _explicit({key for keys in candidates.values() for key in keys})
    if not mine or not theirs:
        return set()
    workspaces = {
        binding.workspace
        for binding in await _active_bindings(entities_client, _in("data.principal", mine))
        if binding.workspace != SYSTEM_WORKSPACE
    }
    if not workspaces:
        return set()
    members = {
        binding.principal
        for binding in await _active_bindings(
            entities_client,
            LogicalOperation(
                operator=FilterOperator.AND,
                operations=[_in("workspace", workspaces), _in("data.principal", theirs)],
            ),
        )
    }
    return {account_id for account_id, keys in candidates.items() if members & set(keys)}
