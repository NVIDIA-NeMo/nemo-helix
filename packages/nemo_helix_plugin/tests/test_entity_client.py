# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from nemo_helix_plugin.client.auth import ServicePrincipalTokenProvider
from nemo_helix_plugin.client.errors import BadRequestError, NotFoundError
from nemo_helix_plugin.entities import (
    EntityBase,
    EntityClient,
    EntityStoreError,
    SyncEntityClient,
    _convert_filter_obj_to_filter_str,
)
from nemo_helix_plugin.entities.client import AsyncEntitiesClient, EntitiesClient
from nemo_helix_plugin.entities.types import Entity
from pydantic import PrivateAttr, computed_field


class ExperimentGroup:
    __entity_type__ = "experiment_group"


def test_filter_obj_keeps_created_by_at_the_entity_root() -> None:
    assert _convert_filter_obj_to_filter_str(
        {
            "created_by": "session-owner",
            "deployment_id": "deployment-id",
        }
    ) == {
        "created_by": "session-owner",
        "data.deployment_id": "deployment-id",
    }


def _entities_page(group_counts: dict[str, int] | None = None) -> Mock:
    """Wrap a list response envelope so ``http_response.json()`` returns it.

    ``count_by`` reads ``group_counts`` off the raw envelope rather than the
    parsed item page, so the mock only needs the raw body.
    """
    body: dict[str, object] = {
        "data": [],
        "pagination": {
            "page": 1,
            "page_size": 1,
            "current_page_size": 0,
            "total_pages": 0,
            "total_results": 0,
        },
    }
    if group_counts is not None:
        body["group_counts"] = group_counts

    resp = Mock()
    resp.http_response = httpx.Response(200, json=body, request=httpx.Request("GET", "http://testserver"))
    return resp


def _error_page(status_code: int) -> Mock:
    """Wrap a non-2xx list response.

    Paginated responses defer ``raise_for_status`` to ``page()``/``items()``,
    which ``count_by`` never calls, so the status check has to be explicit.
    """
    resp = Mock()
    resp.http_response = httpx.Response(
        status_code,
        json={"detail": "boom"},
        request=httpx.Request("GET", "http://testserver"),
    )
    return resp


@pytest.mark.asyncio
async def test_count_by_returns_grouped_counts_for_shorthand_filter() -> None:
    mock_api = Mock()
    mock_api.list_entities = AsyncMock(return_value=_entities_page(group_counts={"insight-a": 2}))
    client = EntityClient(mock_api)

    counts = await client.count_by(
        ExperimentGroup,
        "insight_id",
        filter_obj={
            "insight_id": {"$in": ["insight-a"]},
            "is_deleted": False,
        },
    )

    assert counts == {"insight-a": 2}
    call = mock_api.list_entities.await_args
    assert call is not None
    query_params = call.kwargs["query_params"]
    assert query_params["filter"] == ('{"data.insight_id": {"$in": ["insight-a"]}, "data.is_deleted": false}')
    assert query_params["count_by"] == "data.insight_id"
    assert query_params["page_size"] == 1


@pytest.mark.asyncio
async def test_count_by_rejects_response_without_grouped_counts() -> None:
    mock_api = Mock()
    mock_api.list_entities = AsyncMock(return_value=_entities_page())
    client = EntityClient(mock_api)

    with pytest.raises(EntityStoreError, match="Grouped counts not found"):
        await client.count_by(ExperimentGroup, "insight_id")

    # No filter supplied, so the request must omit the filter param entirely.
    call = mock_api.list_entities.await_args
    assert call is not None
    assert "filter" not in call.kwargs["query_params"]


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(404, NotFoundError), (400, BadRequestError)],
)
@pytest.mark.asyncio
async def test_count_by_raises_http_error_rather_than_missing_counts(
    status_code: int, expected: type[Exception]
) -> None:
    """A failed request must surface as its HTTP error, not "counts not found"."""
    mock_api = Mock()
    mock_api.list_entities = AsyncMock(return_value=_error_page(status_code))
    client = EntityClient(mock_api)

    with pytest.raises(expected):
        await client.count_by(ExperimentGroup, "insight_id")


@pytest.mark.asyncio
async def test_count_by_rejects_non_direct_field() -> None:
    mock_api = Mock()
    client = EntityClient(mock_api)

    with pytest.raises(ValueError, match="direct entity data field"):
        await client.count_by(ExperimentGroup, "data.insight_id")


class _Child(EntityBase):
    """A child entity — addressed within its parent, not by name alone."""

    __entity_type__ = "child_probe"

    note: str = ""


def _stored_child(parent: str) -> Mock:
    """A server response for a child entity, as ``get``/``update`` receive it."""
    entity = Entity(
        entity_type="child_probe",
        id="child-1",
        workspace="default",
        parent=parent,
        name="child-1",
        data={"note": "before"},
        db_version=1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    resp = Mock()
    resp.data = Mock(return_value=entity)
    return resp


@pytest.mark.asyncio
async def test_parent_survives_the_get_update_round_trip() -> None:
    """``update`` takes no ``parent`` argument — it reads it off the entity.

    That only holds because ``get`` puts it there. If either half breaks, an update to a child
    entity silently addresses a root entity of the same name instead, so this pins both halves
    together rather than mocking one and asserting the other.
    """
    mock_api = Mock()
    mock_api.get_entity_by_name = AsyncMock(return_value=_stored_child("parent-1"))
    mock_api.update_entity_by_name = AsyncMock(return_value=_stored_child("parent-1"))
    client = EntityClient(mock_api)

    fetched = await client.get(_Child, "child-1", workspace="default", parent="parent-1")
    get_call = mock_api.get_entity_by_name.await_args
    assert get_call is not None
    assert get_call.kwargs["query_params"] == {"parent": "parent-1"}
    assert fetched.parent == "parent-1"

    fetched.note = "after"
    await client.update(fetched)

    update_call = mock_api.update_entity_by_name.await_args
    assert update_call is not None
    assert update_call.kwargs["query_params"] == {"parent": "parent-1"}


class _EntityWithAuthContext(EntityBase):
    source: str
    _auth_context: dict[str, object] | None = PrivateAttr(default=None)

    @computed_field
    @property
    def auth_context(self) -> dict[str, object] | None:
        return self._auth_context


class _ServiceTokenProvider(ServicePrincipalTokenProvider):
    def __init__(self, *, service_principal_id: str = "service:jobs", delegates_principal: bool = False) -> None:
        self._service_principal_id = service_principal_id
        self.delegates_principal = delegates_principal

    @property
    def service_principal_id(self) -> str:
        return self._service_principal_id

    @property
    def delegates_principal_identity(self) -> bool:
        return self.delegates_principal

    def get_access_token(self) -> str:
        return "service-token"


def _stored_entity_with_auth_context() -> Entity:
    return Entity(
        entity_type="test",
        id="entity-1",
        workspace="default",
        name="entity-1",
        data={
            "source": "test-source",
            "_auth_context": {
                "principal_id": "creator@example.com",
                "principal_email": "creator@example.com",
                "principal_groups": ["team-alpha"],
            },
        },
        db_version=1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_sync_entity_client_auth_context_uses_typed_default_headers() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        default_headers={"X-NHX-Principal-Id": "service:jobs"},
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context == {
        "principal_id": "creator@example.com",
        "principal_email": "creator@example.com",
        "principal_groups": ["team-alpha"],
    }


def test_sync_entity_client_auth_context_strips_delegated_trusted_service_headers() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        default_headers={
            "X-NHX-Principal-Id": "service:jobs",
            "X-NHX-Principal-On-Behalf-Of": "creator@example.com",
        },
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context is None


def test_sync_entity_client_auth_context_strips_malformed_service_header() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        default_headers={"X-NHX-Principal-Id": "service:"},
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context is None


def test_sync_entity_client_auth_context_uses_service_workload_token_provider() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        auth=_ServiceTokenProvider(),
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context == {
        "principal_id": "creator@example.com",
        "principal_email": "creator@example.com",
        "principal_groups": ["team-alpha"],
    }


def test_sync_entity_client_auth_context_strips_malformed_service_workload_token_provider() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        auth=_ServiceTokenProvider(service_principal_id="service:"),
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context is None


def test_sync_entity_client_auth_context_strips_user_bearer_token() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        auth="user-token",
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context is None


def test_sync_entity_client_auth_context_strips_delegated_service_workload_token_provider() -> None:
    entities_client = EntitiesClient(
        base_url="http://testserver",
        auth=_ServiceTokenProvider(delegates_principal=True),
    )
    client = SyncEntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        client.close()

    assert result.auth_context is None


@pytest.mark.asyncio
async def test_async_entity_client_auth_context_uses_typed_default_headers() -> None:
    entities_client = AsyncEntitiesClient(
        base_url="http://testserver",
        default_headers={"X-NHX-Principal-Id": "service:jobs"},
    )
    client = EntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        await client.close()

    assert result.auth_context == {
        "principal_id": "creator@example.com",
        "principal_email": "creator@example.com",
        "principal_groups": ["team-alpha"],
    }


@pytest.mark.asyncio
async def test_async_entity_client_auth_context_uses_service_workload_token_provider() -> None:
    entities_client = AsyncEntitiesClient(
        base_url="http://testserver",
        auth=_ServiceTokenProvider(),
    )
    client = EntityClient(entities_client)

    try:
        result = client._convert_api_entity_to_model(_stored_entity_with_auth_context(), _EntityWithAuthContext)
    finally:
        await client.close()

    assert result.auth_context == {
        "principal_id": "creator@example.com",
        "principal_email": "creator@example.com",
        "principal_groups": ["team-alpha"],
    }


@pytest.mark.asyncio
async def test_delete_forwards_parent() -> None:
    mock_api = Mock()
    mock_api.delete_entity_by_name = AsyncMock(return_value=Mock())
    client = EntityClient(mock_api)

    await client.delete(_Child, "child-1", workspace="default", parent="parent-1")

    call = mock_api.delete_entity_by_name.await_args
    assert call is not None
    assert call.kwargs["query_params"] == {"parent": "parent-1"}
