# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity Store access shared by the resources that live there."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.client_provider import get_nemo_client
from nemo_helix_plugin.entities.base import EntityT, SyncEntityClient
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.iam.client import IAMClient
from nemo_helix_plugin.iam.types import AuthzRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest

from scaled_evals.api.repositories.base_repository import normalize_order
from scaled_evals.api.schemas.common import decode_cursor
from scaled_evals.api.settings import settings

SERVICE_NAME = "scaled-evals"

_client: SyncEntityClient | None = None
_client_lock = Lock()


def entity_client() -> SyncEntityClient:
    """Return the process-wide Entity Store client acting as the scaled-evals service."""
    global _client
    with _client_lock:
        if _client is None:
            _client = SyncEntityClient(
                EntitiesClient.from_client(get_nemo_client(as_service=SERVICE_NAME, internal=True))
            )
        return _client


@functools.cache
def private_workspace() -> str:
    """Return this service's private storage workspace, creating it on first use.

    The plugin authorizes reads of what it keeps there itself, so no user may
    reach those entities directly through the Entities API.

    Raises:
        RuntimeError: if someone else created the workspace or its storage role has not propagated.

    """
    name = settings.entity_store_private_workspace
    client = get_nemo_client(as_service=SERVICE_NAME, internal=True)
    workspaces_client = WorkspacesClient.from_client(client)
    try:
        workspaces_client.create_workspace(body=CreateWorkspaceRequest(name=name))
    except ConflictError:
        # Its creator is its Admin; a user who claimed the name first could read every entity.
        owner = workspaces_client.get_workspace(name=name).data().created_by
        if owner != f"service:{SERVICE_NAME}":
            raise RuntimeError(f"workspace {name!r} belongs to {owner!r}, not the {SERVICE_NAME} service") from None
    workspaces_client.create_workspace_member(
        workspace=name,
        body=CreateWorkspaceMemberRequest(
            principal=f"service:{SERVICE_NAME}",
            roles=["Admin", "ServiceEntityStore"],
        ),
    ).data()
    if platform_auth_enabled():
        decision = (
            IAMClient.from_client(client)
            .evaluate_authorization(
                entrypoint="has_role",
                body=AuthzRequest(
                    input={"principal_id": f"service:{SERVICE_NAME}", "workspace": name, "role": "ServiceEntityStore"}
                ),
            )
            .data()
        )
        if decision.result.get("has_role") is not True:
            raise RuntimeError(f"ServiceEntityStore role has not propagated to workspace {name!r}")
    return name


def list_all(
    client: SyncEntityClient,
    entity_type: type[EntityT],
    where: Mapping[str, Sequence[str | bool]] | None = None,
) -> list[EntityT]:
    """Return every entity of `entity_type` in the private workspace.

    `where` keeps only entities whose field holds one of the listed values. The
    values must not be None: the store does not promise to match a JSON null.
    A boolean must be its field's only value: the store's `$in` compares text,
    which never matches a JSON boolean, so a single value is sent as `$eq`.
    """
    operations: list[FilterOperation] = [
        ComparisonOperation(operator=FilterOperator.EQ, field=f"data.{field}", value=values[0])
        if len(values) == 1
        else ComparisonOperation(operator=FilterOperator.IN, field=f"data.{field}", value=list(values))
        for field, values in (where or {}).items()
    ]
    match = operations[0] if len(operations) == 1 else None
    if len(operations) > 1:
        match = LogicalOperation(operator=FilterOperator.AND, operations=operations)
    items: list[EntityT] = []
    page = 1
    while True:
        response = client.list(
            entity_type, workspace=private_workspace(), page=page, page_size=100, filter_operation=match
        )
        items.extend(response.data)
        if page >= response.pagination.total_pages:
            return items
        page += 1


def keyset_page(rows: list[dict[str, Any]], *, cursor: str | None, order: str, limit: int) -> list[dict[str, Any]]:
    """Return up to `limit + 1` rows after `cursor`, in `(created_at, id)` order.

    This matches the SQL keyset the list cursors encode, so `page_from_rows`
    pages these rows exactly as it paged the Postgres results.
    """
    descending = normalize_order(order) == "desc"
    ordered = sorted(rows, key=lambda row: (row["created_at"], row["id"]), reverse=descending)
    if (position := decode_cursor(cursor)) is not None:
        after = (position.created_at, position.id)
        ordered = [
            row
            for row in ordered
            if ((row["created_at"], row["id"]) < after if descending else (row["created_at"], row["id"]) > after)
        ]
    return ordered[: limit + 1]


def contains(query: str | None, *values: object) -> bool:
    """Return whether any value contains the stripped query, ignoring case; no query matches all."""
    if query is None or not (needle := query.strip().lower()):
        return True
    return any(needle in str(value or "").lower() for value in values)
