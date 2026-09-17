# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_platform_plugin.auth.access_keys.types import AccessKeyCreateResponse, AccessKeyEntityType
from nmp.common.auth.token_claims import TokenClaims
from nmp.common.entities import EntityConflictError, EntityNotFoundError
from nmp.core.auth.app.access_keys import AccessKeyNotFoundError, AccessKeyRegistry, AccessKeyStateConflictError
from nmp.core.auth.entities import AccessKeyEntity

NOW = datetime(2026, 8, 4, 18, 0, tzinfo=UTC)


def _record(
    *,
    jti: str = "ak_example",
    principal: str = "alice@example.com",
    revoked: bool = False,
    service_account: bool = False,
):
    return AccessKeyEntity(
        name=jti,
        workspace="system",
        key_name="ci-build",
        description="CI build automation",
        principal=principal,
        subject_principal="service-account:otel-collector" if service_account else None,
        entity_type="SERVICE_ACCOUNT" if service_account else "USER",
        issued_at=NOW,
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        status="REVOKED" if revoked else "ACTIVE",
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-platform-access-key"],
    )


def _suspended_record() -> AccessKeyEntity:
    return _record(jti="ak_suspended").model_copy(update={"status": "SUSPENDED"})


def _rotating_record(*, grace_period_expires_at: datetime | None = None) -> AccessKeyEntity:
    grace_period_expires_at = grace_period_expires_at or datetime.now(tz=UTC) + timedelta(days=2)
    return _record(jti="ak_rotating").model_copy(
        update={"status": "ROTATING", "grace_period_expires_at": grace_period_expires_at}
    )


def _expired_record() -> AccessKeyEntity:
    return _record(jti="ak_expired").model_copy(update={"expires_at": datetime(2000, 1, 1, tzinfo=UTC)})


def test_access_key_entity_migrates_legacy_revoked_at_to_status() -> None:
    record = AccessKeyEntity.model_validate(
        {
            "name": "ak_legacy_revoked",
            "workspace": "system",
            "principal": "alice@example.com",
            "issued_at": NOW,
            "expires_at": datetime(2030, 1, 1, tzinfo=UTC),
            "issuer": "https://platform.example.com/apis/auth",
            "audiences": ["nemo-platform-access-key"],
            "revoked_at": NOW,
        }
    )

    assert record.status == "REVOKED"


def test_access_key_entity_defaults_unrevoked_legacy_record_to_active() -> None:
    record = AccessKeyEntity.model_validate(
        {
            "name": "ak_legacy_active",
            "workspace": "system",
            "principal": "alice@example.com",
            "issued_at": NOW,
            "expires_at": datetime(2030, 1, 1, tzinfo=UTC),
            "issuer": "https://platform.example.com/apis/auth",
            "audiences": ["nemo-platform-access-key"],
            "revoked_at": None,
        }
    )

    assert record.status == "ACTIVE"


@pytest.mark.parametrize(
    ("entity_type", "principal", "subject_principal"),
    [
        ("SERVICE_ACCOUNT", "admin@example.com", None),
        ("SERVICE_ACCOUNT", "admin@example.com", "service-account:"),
        ("SERVICE_ACCOUNT", "admin@example.com", "alice@example.com"),
        ("SERVICE_ACCOUNT", "service:auth", "service-account:otel-collector"),
        ("SERVICE_ACCOUNT", "service-account:creator", "service-account:otel-collector"),
        ("USER", "alice@example.com", "service-account:otel-collector"),
    ],
)
def test_access_key_entity_rejects_inconsistent_identity_binding(
    entity_type: AccessKeyEntityType, principal: str, subject_principal: str | None
) -> None:
    with pytest.raises(ValueError):
        AccessKeyEntity(
            name="ak_invalid",
            workspace="system",
            principal=principal,
            subject_principal=subject_principal,
            entity_type=entity_type,
            issued_at=NOW,
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-platform-access-key"],
        )


@pytest.mark.asyncio
async def test_registry_persists_created_key_metadata() -> None:
    entity_client = AsyncMock()
    registry = AccessKeyRegistry(entity_client)
    key = AccessKeyCreateResponse(
        jti="ak_example",
        name="ci-build",
        principal="alice@example.com",
        created_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        description="CI build automation",
        status="ACTIVE",
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-platform-access-key"],
        scope=["intake", "entities"],
        token="secret-token",
        token_type="Bearer",
    )

    await registry.add(key)

    saved = entity_client.create.await_args.args[0]
    assert isinstance(saved, AccessKeyEntity)
    assert saved.name == "ak_example"
    assert saved.key_name == "ci-build"
    assert saved.description == "CI build automation"
    assert saved.scope == ["intake", "entities"]
    assert "secret-token" not in saved.model_dump_json()


@pytest.mark.asyncio
async def test_registry_separates_service_key_owner_from_token_subject() -> None:
    entity_client = AsyncMock()
    registry = AccessKeyRegistry(entity_client)
    key = AccessKeyCreateResponse(
        jti="ak_service",
        name="otel",
        principal="service-account:otel-collector",
        entity_type="SERVICE_ACCOUNT",
        created_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        description=None,
        status="ACTIVE",
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-platform-access-key"],
        token="secret-token",
        token_type="Bearer",
    )

    await registry.add(key, owner_principal="admin@example.com")

    saved = entity_client.create.await_args.args[0]
    assert saved.principal == "admin@example.com"
    assert saved.subject_principal == "service-account:otel-collector"
    assert saved.entity_type == "SERVICE_ACCOUNT"


@pytest.mark.asyncio
async def test_registry_discards_unreturned_key() -> None:
    entity_client = AsyncMock()
    registry = AccessKeyRegistry(entity_client)

    await registry.discard_unreturned("ak_orphaned")

    entity_client.delete.assert_awaited_once_with(
        AccessKeyEntity,
        name="ak_orphaned",
        workspace="system",
    )


@pytest.mark.asyncio
async def test_registry_discard_unreturned_is_idempotent_when_record_is_missing() -> None:
    entity_client = AsyncMock()
    entity_client.delete.side_effect = EntityNotFoundError("missing")
    registry = AccessKeyRegistry(entity_client)

    await registry.discard_unreturned("ak_missing")

    entity_client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_lists_principals_keys_with_status_across_pages() -> None:
    entity_client = AsyncMock()
    active_record = _record().model_copy(
        update={
            "audiences": ["nemo-platform-access-key", "nemo-platform-access-key"],
            "scope": ["intake", "intake"],
        }
    )
    entity_client.list.return_value = SimpleNamespace(
        data=[
            active_record,
            _record(jti="ak_service", principal="alice@example.com", service_account=True),
            _record(jti="ak_revoked", revoked=True),
        ],
        pagination=SimpleNamespace(total_pages=2),
    )
    registry = AccessKeyRegistry(entity_client)

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=2)

    assert [key.jti for key in result.data] == ["ak_example", "ak_revoked"]
    assert [key.status for key in result.data] == ["ACTIVE", "REVOKED"]
    assert result.data[0].audiences == ["nemo-platform-access-key"]
    assert result.data[0].scope == ["intake"]
    assert result.has_more
    entity_client.list.assert_awaited_once()
    assert entity_client.list.await_args.kwargs["page"] == 1
    assert entity_client.list.await_args.kwargs["page_size"] == 2
    assert entity_client.list.await_args.kwargs["filter_operation"].to_dict() == {
        "data.principal": {"$eq": "alice@example.com"}
    }


@pytest.mark.asyncio
async def test_registry_lists_all_service_accounts_when_include_service_accounts() -> None:
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[], pagination=SimpleNamespace(total_pages=1))
    registry = AccessKeyRegistry(entity_client)

    await registry.list_for_principal("admin@example.com", page=1, page_size=100, include_service_accounts=True)

    filter_operation = entity_client.list.await_args.kwargs["filter_operation"]
    assert filter_operation.to_dict() == {
        "$or": [
            {"data.principal": {"$eq": "admin@example.com"}},
            {"data.entity_type": {"$eq": "SERVICE_ACCOUNT"}},
        ]
    }


@pytest.mark.asyncio
async def test_registry_can_retrieve_later_list_page() -> None:
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(
        data=[_record(jti="ak_later")], pagination=SimpleNamespace(total_pages=12)
    )
    registry = AccessKeyRegistry(entity_client)

    result = await registry.list_for_principal("alice@example.com", page=11, page_size=25)

    assert [key.jti for key in result.data] == ["ak_later"]
    assert result.has_more
    assert entity_client.list.await_args.kwargs["page"] == 11
    assert entity_client.list.await_args.kwargs["page_size"] == 25


@pytest.mark.asyncio
async def test_registry_reports_expired_status() -> None:
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(
        data=[_expired_record()], pagination=SimpleNamespace(total_pages=1)
    )
    registry = AccessKeyRegistry(entity_client)

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=100)

    assert result.data[0].status == "EXPIRED"


@pytest.mark.asyncio
async def test_registry_reports_expired_at_timestamp_while_authentication_allows_clock_skew() -> None:
    record = _record().model_copy(update={"expires_at": datetime.now(tz=UTC) - timedelta(seconds=1)})
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(data=[record], pagination=SimpleNamespace(total_pages=1))
    entity_client.get.return_value = record
    registry = AccessKeyRegistry(entity_client)

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=100)

    assert result.data[0].status == "EXPIRED"
    assert await registry.is_active(record.name, record.principal)


@pytest.mark.asyncio
async def test_registry_revokes_owned_key_without_deleting_audit_record() -> None:
    entity_client = AsyncMock()
    original = _record()
    entity_client.get.return_value = original
    registry = AccessKeyRegistry(entity_client)

    assert await registry.revoke("ak_example", "alice@example.com")

    updated = entity_client.update.await_args.args[0]
    assert updated is not original
    assert original.status == "ACTIVE"
    assert updated.status == "REVOKED"
    entity_client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_concurrent_revoke_reports_existing_revocation() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record(), _record(revoked=True)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.revoke("ak_example", "alice@example.com")

    assert entity_client.get.await_count == 2
    entity_client.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_concurrent_revoke_with_hard_delete_treats_as_already_revoked() -> None:
    entity_client = AsyncMock()
    # First get succeeds; update conflicts; second get raises not-found (key deleted).
    entity_client.get.side_effect = [_record(), EntityNotFoundError("gone")]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.revoke("ak_example", "alice@example.com")

    assert entity_client.get.await_count == 2


@pytest.mark.asyncio
async def test_registry_concurrent_suspend_with_hard_delete_reports_not_found() -> None:
    entity_client = AsyncMock()
    # First get succeeds; update conflicts; second get raises not-found (key deleted).
    entity_client.get.side_effect = [_record(), EntityNotFoundError("gone")]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyNotFoundError, match="Scoped Access Key ak_example was not found"):
        await registry.suspend("ak_example", "alice@example.com")

    assert entity_client.get.await_count == 2


@pytest.mark.asyncio
async def test_registry_can_newly_revoke_expired_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _expired_record()
    registry = AccessKeyRegistry(entity_client)

    assert await registry.revoke("ak_expired", "alice@example.com")

    updated = entity_client.update.await_args.args[0]
    assert updated.status == "REVOKED"


@pytest.mark.asyncio
async def test_registry_active_authentication_updates_last_used_at() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record()
    registry = AccessKeyRegistry(entity_client)

    assert await registry.is_active("ak_example", "alice@example.com")

    updated = entity_client.update.await_args.args[0]
    assert updated.last_used_at is not None
    assert updated.last_used_at.tzinfo is UTC


@pytest.mark.asyncio
async def test_registry_rotating_authentication_updates_last_used_at() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(jti="ak_rotating").model_copy(
        update={
            "status": "ROTATING",
            "grace_period_expires_at": datetime.now(tz=UTC) + timedelta(minutes=5),
        }
    )
    registry = AccessKeyRegistry(entity_client)

    assert await registry.is_active("ak_rotating", "alice@example.com")

    updated = entity_client.update.await_args.args[0]
    assert updated.last_used_at is not None


@pytest.mark.asyncio
async def test_registry_rotating_authentication_bypasses_last_used_at_coalescing() -> None:
    # ROTATING keys skip the 5-minute coalescing window: last_used_at is the signal
    # callers rely on to confirm traffic has moved off a rotated-out key before
    # revoking it, so it must reflect every authenticated request during the
    # bounded grace window, not just one every 5 minutes.
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(jti="ak_rotating").model_copy(
        update={
            "status": "ROTATING",
            "grace_period_expires_at": datetime.now(tz=UTC) + timedelta(minutes=5),
            "last_used_at": datetime.now(tz=UTC) - timedelta(seconds=5),
        }
    )
    registry = AccessKeyRegistry(entity_client)

    assert await registry.is_active("ak_rotating", "alice@example.com")

    entity_client.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_last_used_at_update_retries_transient_conflict() -> None:
    stale, fresh = _record(), _record()
    stale._db_version = 1
    fresh._db_version = 2
    entity_client = AsyncMock()
    entity_client.get.side_effect = [stale, fresh]
    entity_client.update.side_effect = [EntityConflictError("entity version changed"), None]
    registry = AccessKeyRegistry(entity_client)

    assert await registry.is_active("ak_example", "alice@example.com")

    assert entity_client.get.await_count == 2
    assert entity_client.update.await_count == 2
    first_attempt, second_attempt = (call.args[0] for call in entity_client.update.await_args_list)
    assert first_attempt.db_version == 1
    assert second_attempt.db_version == 2
    assert first_attempt.last_used_at is not None
    assert second_attempt.last_used_at is not None
    assert second_attempt.last_used_at >= first_attempt.last_used_at


@pytest.mark.asyncio
async def test_registry_last_used_at_update_stops_when_conflict_reveals_revocation() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record(), _record(revoked=True)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.is_active("ak_example", "alice@example.com")

    assert entity_client.get.await_count == 2
    entity_client.update.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record",
    [
        _record(revoked=True),
        _suspended_record(),
        _expired_record(),
        _record(principal="bob@example.com"),
    ],
)
async def test_registry_rejected_authentication_does_not_update_last_used_at(record: AccessKeyEntity) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = record
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.is_active(record.name, "alice@example.com")

    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [EntityConflictError("entity version changed"), RuntimeError("storage unavailable")],
)
async def test_registry_last_used_at_update_failure_does_not_reject_authentication(failure: Exception) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record()
    entity_client.update.side_effect = failure
    registry = AccessKeyRegistry(entity_client)

    assert await registry.is_active("ak_example", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_reports_revoked_key_as_inactive() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(revoked=True)
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.is_active("ak_example", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_hides_missing_and_other_principals_keys() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(principal="bob@example.com")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyNotFoundError):
        await registry.revoke("ak_example", "alice@example.com")

    entity_client.get.side_effect = EntityNotFoundError("missing")
    assert not await registry.is_active("ak_missing", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_admin_override_lets_a_different_admin_revoke_a_service_bound_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(principal="admin-a@example.com", service_account=True)
    registry = AccessKeyRegistry(entity_client)

    assert await registry.revoke("ak_example", "admin-b@example.com", admin_override=AsyncMock(return_value=True))

    updated = entity_client.update.await_args.args[0]
    assert updated.status == "REVOKED"


@pytest.mark.asyncio
async def test_registry_admin_override_is_not_consulted_for_a_personal_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(principal="alice@example.com")
    registry = AccessKeyRegistry(entity_client)
    admin_override = AsyncMock(return_value=True)

    with pytest.raises(AccessKeyNotFoundError):
        await registry.revoke("ak_example", "bob@example.com", admin_override=admin_override)

    admin_override.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_admin_override_denied_still_hides_a_service_bound_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(principal="admin-a@example.com", service_account=True)
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyNotFoundError):
        await registry.revoke("ak_example", "admin-b@example.com", admin_override=AsyncMock(return_value=False))


@pytest.mark.asyncio
async def test_registry_service_key_creator_must_still_pass_admin_override() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(principal="admin-a@example.com", service_account=True)
    registry = AccessKeyRegistry(entity_client)
    admin_override = AsyncMock(return_value=False)

    with pytest.raises(AccessKeyNotFoundError):
        await registry.revoke("ak_example", "admin-a@example.com", admin_override=admin_override)

    admin_override.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_admin_override_is_not_consulted_when_the_caller_already_owns_the_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(principal="alice@example.com")
    registry = AccessKeyRegistry(entity_client)
    admin_override = AsyncMock(return_value=True)

    assert await registry.revoke("ak_example", "alice@example.com", admin_override=admin_override)

    admin_override.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_admin_override_lets_a_different_admin_suspend_and_unsuspend_a_service_bound_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [
        _record(principal="admin-a@example.com", service_account=True),
        _record(principal="admin-a@example.com", service_account=True).model_copy(update={"status": "SUSPENDED"}),
    ]
    registry = AccessKeyRegistry(entity_client)
    admin_override = AsyncMock(return_value=True)

    suspend_changed, suspend_status = await registry.suspend(
        "ak_example", "admin-b@example.com", admin_override=admin_override
    )
    assert suspend_changed
    assert suspend_status == "SUSPENDED"

    unsuspend_changed, unsuspend_status = await registry.unsuspend(
        "ak_example", "admin-b@example.com", admin_override=admin_override
    )
    assert unsuspend_changed
    assert unsuspend_status == "ACTIVE"


@pytest.mark.asyncio
async def test_registry_is_active_for_service_bound_key_checks_subject_not_owner() -> None:
    entity_client = AsyncMock()
    record = _record(principal="admin-a@example.com", service_account=True).model_copy(
        update={"subject_principal": "service-account:otel-collector"}
    )
    entity_client.get.return_value = record
    registry = AccessKeyRegistry(entity_client)

    # Authenticating as the service-account subject succeeds...
    assert await registry.is_active("ak_example", "service-account:otel-collector")
    # ...but the owning admin's principal is not itself a valid authentication subject.
    assert not await registry.is_active("ak_example", "admin-a@example.com")


@pytest.mark.asyncio
async def test_registry_backfills_missing_legacy_access_key_from_validated_claims() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = EntityNotFoundError("missing")
    entity_client.create.side_effect = lambda record: record
    registry = AccessKeyRegistry(entity_client)
    claims = TokenClaims(
        subject="alice@example.com",
        email=None,
        groups=[],
        scopes=[],
        raw_claims={
            "iss": "https://platform.example.com/apis/auth",
            "aud": ["nemo-platform-access-key", "nemo-platform-access-key"],
            "sub": "alice@example.com",
            "iat": 1_785_280_000,
            "nbf": 1_785_280_000,
            "exp": 1_893_456_000,
            "jti": "ak_legacy",
            "nmp_token_type": "access_key",
            "nmp_access_key": {"version": 1, "name": "legacy-key"},
        },
    )

    assert await registry.is_active("ak_legacy", "alice@example.com", claims=claims)

    saved = entity_client.create.await_args.args[0]
    assert saved.name == "ak_legacy"
    assert saved.key_name == "legacy-key"
    assert saved.description is None
    assert saved.principal == "alice@example.com"
    assert saved.issuer == "https://platform.example.com/apis/auth"
    assert saved.audiences == ["nemo-platform-access-key"]
    assert saved.issued_at == datetime.fromtimestamp(1_785_280_000, tz=UTC)
    assert saved.expires_at == datetime.fromtimestamp(1_893_456_000, tz=UTC)


@pytest.mark.asyncio
async def test_registry_reports_suspended_key_in_list() -> None:
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(
        data=[_suspended_record()], pagination=SimpleNamespace(total_pages=1)
    )
    registry = AccessKeyRegistry(entity_client)

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=100)

    assert result.data[0].status == "SUSPENDED"


@pytest.mark.asyncio
async def test_registry_expiration_prevents_suspension_state_changes() -> None:
    suspended_expired = _expired_record().model_copy(update={"status": "SUSPENDED"})
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(
        data=[suspended_expired], pagination=SimpleNamespace(total_pages=1)
    )
    entity_client.get.side_effect = [suspended_expired, _expired_record()]
    registry = AccessKeyRegistry(entity_client)

    listed = await registry.list_for_principal("alice@example.com", page=1, page_size=100)
    assert listed.data[0].status == "EXPIRED"

    unsuspend_changed, unsuspend_status = await registry.unsuspend("ak_expired", "alice@example.com")
    suspend_changed, suspend_status = await registry.suspend("ak_expired", "alice@example.com")
    assert not unsuspend_changed
    assert unsuspend_status == "EXPIRED"
    assert not suspend_changed
    assert suspend_status == "EXPIRED"
    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_revoke_transitions_suspended_key_to_revoked() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _suspended_record()
    registry = AccessKeyRegistry(entity_client)

    assert await registry.revoke("ak_suspended", "alice@example.com")

    updated = entity_client.update.await_args.args[0]
    assert updated.status == "REVOKED"


@pytest.mark.asyncio
async def test_registry_revoke_retries_transient_conflict_when_key_is_only_suspended() -> None:
    # A conflict from something unrelated (e.g. is_active's last_used_at update landing
    # between our read and update) shouldn't fail the revoke outright: since the key is
    # merely suspended (not revoked) on re-read, revoke should retry and succeed.
    stale, fresh = _record(), _suspended_record()
    stale._db_version = 1
    fresh._db_version = 2
    entity_client = AsyncMock()
    entity_client.get.side_effect = [stale, fresh]
    entity_client.update.side_effect = [EntityConflictError("entity version changed"), None]
    registry = AccessKeyRegistry(entity_client)

    assert await registry.revoke("ak_suspended", "alice@example.com")

    assert entity_client.get.await_count == 2
    assert entity_client.update.await_count == 2
    first_attempt, second_attempt = (call.args[0] for call in entity_client.update.await_args_list)
    assert first_attempt.db_version == 1
    assert second_attempt.db_version == 2


@pytest.mark.asyncio
async def test_registry_revoke_gives_up_after_persistent_conflict_when_key_is_only_suspended() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record(), *([_suspended_record()] * 3)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(EntityConflictError, match="entity version changed"):
        await registry.revoke("ak_suspended", "alice@example.com")

    assert entity_client.get.await_count == 4
    assert entity_client.update.await_count == 3


@pytest.mark.asyncio
async def test_registry_reports_suspended_key_as_inactive() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _suspended_record()
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.is_active("ak_suspended", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_suspends_and_unsuspends_key() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record(), _suspended_record()]
    registry = AccessKeyRegistry(entity_client)

    suspend_changed, suspend_status = await registry.suspend("ak_example", "alice@example.com")
    assert suspend_changed
    assert suspend_status == "SUSPENDED"
    assert entity_client.update.await_args_list[0].args[0].status == "SUSPENDED"
    unsuspend_changed, unsuspend_status = await registry.unsuspend("ak_suspended", "alice@example.com")
    assert unsuspend_changed
    assert unsuspend_status == "ACTIVE"
    assert entity_client.update.await_args_list[1].args[0].status == "ACTIVE"


@pytest.mark.asyncio
async def test_registry_suspension_operations_are_idempotent() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_suspended_record(), _record()]
    registry = AccessKeyRegistry(entity_client)

    suspend_changed, suspend_status = await registry.suspend("ak_suspended", "alice@example.com")
    assert not suspend_changed
    assert suspend_status == "SUSPENDED"
    unsuspend_changed, unsuspend_status = await registry.unsuspend("ak_example", "alice@example.com")
    assert not unsuspend_changed
    assert unsuspend_status == "ACTIVE"
    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["suspend", "unsuspend"])
async def test_registry_rejects_suspension_transition_for_revoked_key(operation: str) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record(revoked=True)
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyStateConflictError, match=f"cannot be {operation}ed"):
        await getattr(registry, operation)("ak_example", "alice@example.com")

    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "current"),
    [("suspend", _suspended_record()), ("unsuspend", _record())],
)
async def test_registry_concurrent_suspension_transition_reports_target_state(
    operation: str, current: AccessKeyEntity
) -> None:
    initial = _record() if operation == "suspend" else _suspended_record()
    entity_client = AsyncMock()
    entity_client.get.side_effect = [initial, current]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    changed, effective_status = await getattr(registry, operation)(initial.name, initial.principal)
    assert not changed
    assert effective_status == current.status
    assert entity_client.get.await_count == 2
    entity_client.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_concurrent_suspension_transition_reports_expiration() -> None:
    entity_client = AsyncMock()
    expired = _expired_record().model_copy(update={"name": "ak_example"})
    entity_client.get.side_effect = [_record(), expired]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    changed, effective_status = await registry.suspend("ak_example", "alice@example.com")

    assert not changed
    assert effective_status == "EXPIRED"
    assert entity_client.get.await_count == 2
    entity_client.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_concurrent_suspension_transition_rejects_revocation() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record(), _record(revoked=True)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyStateConflictError, match="cannot be suspended"):
        await registry.suspend("ak_example", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_suspend_retries_transient_conflict_from_unrelated_write() -> None:
    # A conflict from something unrelated (e.g. is_active's last_used_at update landing
    # between our read and update) shouldn't fail the suspension outright: since the
    # key is still eligible on re-read, suspend should retry and succeed.
    stale, fresh = _record(), _record()
    stale._db_version = 1
    fresh._db_version = 2
    entity_client = AsyncMock()
    entity_client.get.side_effect = [stale, fresh]
    entity_client.update.side_effect = [EntityConflictError("entity version changed"), None]
    registry = AccessKeyRegistry(entity_client)

    changed, effective_status = await registry.suspend("ak_example", "alice@example.com")

    assert changed
    assert effective_status == "SUSPENDED"
    assert entity_client.get.await_count == 2
    assert entity_client.update.await_count == 2
    first_attempt, second_attempt = (call.args[0] for call in entity_client.update.await_args_list)
    assert first_attempt.db_version == 1
    assert second_attempt.db_version == 2


@pytest.mark.asyncio
async def test_registry_suspend_gives_up_after_persistent_conflict() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record() for _ in range(4)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(EntityConflictError, match="entity version changed"):
        await registry.suspend("ak_example", "alice@example.com")

    assert entity_client.get.await_count == 4
    assert entity_client.update.await_count == 3


@pytest.mark.asyncio
async def test_registry_rejects_missing_current_access_key_record() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = EntityNotFoundError("missing")
    registry = AccessKeyRegistry(entity_client)
    claims = TokenClaims(
        subject="alice@example.com",
        email=None,
        groups=[],
        scopes=[],
        raw_claims={
            "iss": "https://platform.example.com/apis/auth",
            "aud": ["nemo-platform-access-key"],
            "sub": "alice@example.com",
            "iat": 1_785_280_000,
            "nbf": 1_785_280_000,
            "exp": 1_893_456_000,
            "jti": "ak_current",
            "nmp_token_type": "access_key",
            "nmp_access_key": {"version": 2, "name": "current-key"},
        },
    )

    assert not await registry.is_active("ak_current", "alice@example.com", claims=claims)
    entity_client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_reports_rotating_key_as_active_during_grace_period() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _rotating_record(grace_period_expires_at=datetime(2999, 1, 1, tzinfo=UTC))
    registry = AccessKeyRegistry(entity_client)

    assert await registry.is_active("ak_rotating", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_reports_rotated_key_as_inactive_after_grace_period() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _rotating_record(grace_period_expires_at=datetime(2000, 1, 1, tzinfo=UTC))
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.is_active("ak_rotating", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_is_active_does_not_extend_grace_deadline_by_auth_leeway() -> None:
    # is_active's 30s clock-skew leeway is meant only for validating a JWT's own exp
    # claim, not the server-side-only rotation grace deadline. A key just past its
    # grace deadline must be rejected immediately, not for another 30 seconds.
    entity_client = AsyncMock()
    entity_client.get.return_value = _rotating_record(
        grace_period_expires_at=datetime.now(tz=UTC) - timedelta(seconds=5)
    )
    registry = AccessKeyRegistry(entity_client)

    assert not await registry.is_active("ak_rotating", "alice@example.com")


@pytest.mark.asyncio
async def test_registry_reports_rotating_key_status_in_listing() -> None:
    entity_client = AsyncMock()
    entity_client.list.return_value = SimpleNamespace(
        data=[_rotating_record(grace_period_expires_at=datetime(2999, 1, 1, tzinfo=UTC))],
        pagination=SimpleNamespace(total_pages=1),
    )
    registry = AccessKeyRegistry(entity_client)

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=100)

    assert result.data[0].status == "ROTATING"


def test_registry_metadata_exposes_grace_expiry_for_rotating_key() -> None:
    grace_period_expires_at = datetime(2999, 1, 1, tzinfo=UTC)

    metadata = AccessKeyRegistry._metadata(_rotating_record(grace_period_expires_at=grace_period_expires_at))

    assert metadata.grace_period_expires_at == grace_period_expires_at


@pytest.mark.parametrize("status", ["ACTIVE", "REVOKED", "SUSPENDED"])
def test_registry_metadata_hides_stale_grace_expiry_when_not_rotating(status: str) -> None:
    record = _record().model_copy(
        update={
            "status": status,
            "grace_period_expires_at": datetime(2999, 1, 1, tzinfo=UTC),
        }
    )

    metadata = AccessKeyRegistry._metadata(record)

    assert metadata.grace_period_expires_at is None


@pytest.mark.asyncio
async def test_registry_get_rotatable_returns_active_record() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record()
    registry = AccessKeyRegistry(entity_client)

    record = await registry.get_rotatable("ak_example", "alice@example.com")

    assert record.name == "ak_example"
    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record",
    [_suspended_record(), _record(revoked=True), _rotating_record(), _expired_record()],
    ids=["suspended", "revoked", "rotating", "expired"],
)
async def test_registry_get_rotatable_rejects_non_active_key(record: AccessKeyEntity) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = record
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyStateConflictError):
        await registry.get_rotatable(record.name, record.principal)


@pytest.mark.asyncio
async def test_registry_begin_rotation_transitions_active_key_to_rotating() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _record()
    registry = AccessKeyRegistry(entity_client)

    result = await registry.begin_rotation(
        "ak_example", "alice@example.com", grace_period_seconds=3600, successor_jti="ak_successor"
    )

    assert isinstance(result, AccessKeyEntity)
    assert result.status == "ROTATING"
    assert result.grace_period_expires_at is not None
    assert result.grace_period_expires_at > datetime.now(tz=UTC)
    assert result.rotation_successor_jti == "ak_successor"
    updated = entity_client.update.await_args.args[0]
    assert result is updated
    assert updated.status == "ROTATING"
    assert updated.grace_period_expires_at == result.grace_period_expires_at


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record",
    [_suspended_record(), _record(revoked=True), _rotating_record(), _expired_record()],
    ids=["suspended", "revoked", "rotating", "expired"],
)
async def test_registry_begin_rotation_rejects_non_active_key(record: AccessKeyEntity) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = record
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyStateConflictError):
        await registry.begin_rotation(
            record.name, record.principal, grace_period_seconds=3600, successor_jti="ak_successor"
        )

    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_begin_rotation_caps_grace_deadline_at_natural_expiry() -> None:
    # Natural expiry always takes precedence over rotation grace (AIRCORE-985): a key
    # expiring sooner than the configured grace period must not have its reported
    # deadline pushed out past that natural expiry.
    soon_expiring = _record().model_copy(update={"expires_at": datetime.now(tz=UTC) + timedelta(minutes=10)})
    entity_client = AsyncMock()
    entity_client.get.return_value = soon_expiring
    registry = AccessKeyRegistry(entity_client)

    result = await registry.begin_rotation(
        "ak_example", "alice@example.com", grace_period_seconds=3600, successor_jti="ak_successor"
    )

    assert result.grace_period_expires_at == soon_expiring.expires_at


@pytest.mark.asyncio
async def test_registry_begin_rotation_retries_transient_conflict_from_unrelated_write() -> None:
    # A conflict from something unrelated (e.g. is_active's last_used_at update landing
    # between our read and update) shouldn't fail the rotation outright: since the
    # record is still rotatable on re-read, begin_rotation should retry and succeed.
    stale, fresh = _record(), _record()
    stale._db_version = 1
    fresh._db_version = 2
    entity_client = AsyncMock()
    entity_client.get.side_effect = [stale, fresh]
    entity_client.update.side_effect = [EntityConflictError("entity version changed"), None]
    registry = AccessKeyRegistry(entity_client)

    result = await registry.begin_rotation(
        "ak_example", "alice@example.com", grace_period_seconds=3600, successor_jti="ak_successor"
    )

    assert result.status == "ROTATING"
    assert entity_client.get.await_count == 2
    assert entity_client.update.await_count == 2
    first_attempt, second_attempt = (call.args[0] for call in entity_client.update.await_args_list)
    assert first_attempt.db_version == 1
    assert second_attempt.db_version == 2


@pytest.mark.asyncio
async def test_registry_begin_rotation_gives_up_after_persistent_conflict() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record() for _ in range(4)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(EntityConflictError):
        await registry.begin_rotation(
            "ak_example", "alice@example.com", grace_period_seconds=3600, successor_jti="ak_successor"
        )

    assert entity_client.get.await_count == 4
    assert entity_client.update.await_count == 3


@pytest.mark.asyncio
async def test_registry_begin_rotation_concurrent_conflict_surfaces_state_conflict_when_no_longer_rotatable() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = [_record(), _record(revoked=True)]
    entity_client.update.side_effect = EntityConflictError("entity version changed")
    registry = AccessKeyRegistry(entity_client)

    with pytest.raises(AccessKeyStateConflictError):
        await registry.begin_rotation(
            "ak_example", "alice@example.com", grace_period_seconds=3600, successor_jti="ak_successor"
        )
