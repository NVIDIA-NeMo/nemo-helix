# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nmp.core.auth.app import account_resolution
from nmp.core.auth.app.account_resolution import (
    AccountResolver,
    ServicePrincipalNotAllowedError,
)
from nmp.core.auth.config import AuthServiceConfig
from nmp.core.entities.app.repository.account_identity import AccountIdentityRecord


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def resolve_or_materialize(self, **kwargs) -> AccountIdentityRecord:
        self.calls.append(kwargs)
        return AccountIdentityRecord(
            account_id=f"account-for-{kwargs['subject']}",
            account_type=kwargs["account_type"],
            identity_id="account-identity-1",
            issuer=kwargs["issuer"],
            subject=kwargs["subject"],
        )


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> _FakeStore:
    store = _FakeStore()

    async def _store() -> _FakeStore:
        return store

    monkeypatch.setattr(account_resolution, "_account_identity_store", _store)
    return store


@pytest.mark.asyncio
async def test_resolves_user_descriptor_to_policy_fields(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "alice-subject",
            "principal_email": "alice@example.com",
            "identity_resolution": {
                "principal": {
                    "issuer": "https://idp.example.com",
                    "subject": "alice-subject",
                    "subject_claim": "sub",
                    "account_type": "user",
                    "primary_email": "alice@example.com",
                    "authz_aliases": ["alice-subject", "alice@example.com"],
                }
            },
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "principal",
        "actor_account_id": "account-for-alice-subject",
        "actor_aliases": ["alice-subject", "alice@example.com"],
    }
    assert fake_store.calls[0]["issuer"] == "https://idp.example.com"
    assert fake_store.calls[0]["subject"] == "alice-subject"


@pytest.mark.asyncio
async def test_resolved_account_input_skips_storage(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "alice-subject",
            "actor_account_id": "account-existing",
            "actor_aliases": ["alice-subject"],
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "principal",
        "actor_account_id": "account-existing",
        "actor_aliases": ["alice-subject"],
    }
    assert fake_store.calls == []


@pytest.mark.asyncio
async def test_extended_non_service_subject_materializes_user_account(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input({"principal_id": "auth0|abc"})

    assert context.to_policy_fields() == {
        "caller_kind": "principal",
        "actor_account_id": "account-for-auth0|abc",
    }
    assert fake_store.calls[0]["subject"] == "auth0|abc"
    assert fake_store.calls[0]["account_type"] == "user"


@pytest.mark.asyncio
async def test_input_caller_kind_is_ignored_for_resolved_account_context(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "alice-subject",
            "actor_account_id": "account-existing",
            "caller_kind": "service_principal",
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "principal",
        "actor_account_id": "account-existing",
    }
    assert fake_store.calls == []


@pytest.mark.asyncio
async def test_allowed_service_principal_materializes_service_account(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "service:jobs",
            "identity_resolution": {
                "principal": {
                    "issuer": "nemo:service",
                    "subject": "jobs",
                    "subject_claim": "service",
                    "account_type": "service",
                    "authz_aliases": ["service:jobs"],
                }
            },
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "service_principal",
        "actor_account_id": "account-for-jobs",
        "actor_aliases": ["service:jobs"],
    }
    assert fake_store.calls[0]["issuer"] == "nemo:service"
    assert fake_store.calls[0]["subject"] == "jobs"
    assert fake_store.calls[0]["account_type"] == "service"


@pytest.mark.asyncio
async def test_platform_seed_service_principal_is_allowed(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "service:platform-seed",
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "service_principal",
        "actor_account_id": "account-for-platform-seed",
        "actor_aliases": ["service:platform-seed"],
    }
    assert fake_store.calls[0]["issuer"] == "nemo:service"
    assert fake_store.calls[0]["subject"] == "platform-seed"
    assert fake_store.calls[0]["account_type"] == "service"


@pytest.mark.asyncio
@pytest.mark.parametrize("service_name", ["jobs-controller", "models-controller"])
async def test_platform_controller_service_principal_is_allowed(fake_store: _FakeStore, service_name: str):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input({"principal_id": f"service:{service_name}"})

    assert context.to_policy_fields() == {
        "caller_kind": "service_principal",
        "actor_account_id": f"account-for-{service_name}",
        "actor_aliases": [f"service:{service_name}"],
    }
    assert fake_store.calls[0]["subject"] == service_name
    assert fake_store.calls[0]["account_type"] == "service"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "principal_id",
    [
        "service:",
        "service:has spaces",
        "service:bad$name",
        "service:/path",
        "service:*",
    ],
)
async def test_malformed_service_principal_fails_closed_without_materializing(
    fake_store: _FakeStore,
    principal_id: str,
):
    resolver = AccountResolver(AuthServiceConfig())

    with pytest.raises(ServicePrincipalNotAllowedError):
        await resolver.resolve_authz_input({"principal_id": principal_id})

    assert fake_store.calls == []


@pytest.mark.asyncio
async def test_unknown_service_principal_fails_closed(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    with pytest.raises(ServicePrincipalNotAllowedError, match="Unknown service principal"):
        await resolver.resolve_authz_input(
            {
                "principal_id": "service:not-registered",
                "identity_resolution": {
                    "principal": {
                        "issuer": "nemo:service",
                        "subject": "not-registered",
                        "subject_claim": "service",
                        "account_type": "service",
                    }
                },
            }
        )

    assert fake_store.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "on_behalf_of",
    [
        "service:",
        "service:has spaces",
        "service:bad$name",
        "service:/path",
        "service:*",
    ],
)
async def test_malformed_on_behalf_service_principal_fails_closed_without_materializing(
    fake_store: _FakeStore,
    on_behalf_of: str,
):
    resolver = AccountResolver(AuthServiceConfig())

    with pytest.raises(ServicePrincipalNotAllowedError):
        await resolver.resolve_authz_input(
            {
                "principal_id": "alice-subject",
                "actor_account_id": "account-existing",
                "on_behalf_of_principal_id": on_behalf_of,
            }
        )

    assert fake_store.calls == []


@pytest.mark.asyncio
async def test_on_behalf_service_principal_materializes_service_account(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "service:jobs",
            "on_behalf_of_principal_id": "service:models",
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "service_principal",
        "actor_account_id": "account-for-jobs",
        "actor_aliases": ["service:jobs"],
        "subject_account_id": "account-for-models",
        "subject_aliases": ["service:models"],
    }
    assert fake_store.calls[1]["issuer"] == "nemo:service"
    assert fake_store.calls[1]["subject"] == "models"
    assert fake_store.calls[1]["account_type"] == "service"


@pytest.mark.asyncio
async def test_delegated_fallback_does_not_assign_subject_email_to_actor(fake_store: _FakeStore):
    resolver = AccountResolver(AuthServiceConfig())

    context = await resolver.resolve_authz_input(
        {
            "principal_id": "actor-subject",
            "on_behalf_of_principal_id": "delegated-subject",
            "principal_email": "delegated@example.com",
        }
    )

    assert context.to_policy_fields() == {
        "caller_kind": "principal",
        "actor_account_id": "account-for-actor-subject",
        "actor_aliases": ["actor-subject"],
        "subject_account_id": "account-for-delegated-subject",
        "subject_aliases": ["delegated-subject", "delegated@example.com"],
    }
    assert fake_store.calls[0]["subject"] == "actor-subject"
    assert fake_store.calls[0]["primary_email"] is None
    assert fake_store.calls[0]["display_name"] == "actor-subject"
    assert fake_store.calls[1]["subject"] == "delegated-subject"
    assert fake_store.calls[1]["primary_email"] == "delegated@example.com"
