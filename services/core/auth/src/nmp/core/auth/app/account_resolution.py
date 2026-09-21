# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable account resolution for authz policy input enrichment."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from nmp.core.auth.config import AuthServiceConfig
from nmp.core.entities.app.repository import (
    AccountIdentityConflictError,
    AccountIdentityStore,
    AccountIdentityUnavailableError,
    get_async_session_maker,
    initialize_async_engine,
)
from nmp.core.entities.config import EntitiesConfig
from nmp.core.entities.initialize import initialize_database

logger = logging.getLogger(__name__)

CallerKind = Literal["principal", "service_principal"]
AccountType = Literal["user", "service"]

_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_TRUSTED_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9@._\-:+/]+$")
_BUILT_IN_SERVICE_NAMES = {
    "auth",
    "entities",
    "files",
    "guardrails",
    "hello-world",
    "inference-gateway",
    "intake",
    "jobs",
    "jobs-controller",
    "models",
    "models-controller",
    "platform",
    "platform-seed",
    "secrets",
    "studio",
}


class AccountResolutionError(RuntimeError):
    """Raised when a principal cannot be resolved to a stable account."""


class ServicePrincipalNotAllowedError(AccountResolutionError):
    """Raised when a service principal is unknown or malformed."""


@dataclass(frozen=True)
class ResolvedAccountContext:
    """Account context copied into Rego input and the authz response."""

    actor_account_id: str | None = None
    actor_aliases: list[str] = field(default_factory=list)
    caller_kind: CallerKind = "principal"
    subject_account_id: str | None = None
    subject_aliases: list[str] = field(default_factory=list)

    def to_policy_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"caller_kind": self.caller_kind}
        if self.actor_account_id:
            fields["actor_account_id"] = self.actor_account_id
        if self.actor_aliases:
            fields["actor_aliases"] = list(self.actor_aliases)
        if self.subject_account_id:
            fields["subject_account_id"] = self.subject_account_id
        if self.subject_aliases:
            fields["subject_aliases"] = list(self.subject_aliases)
        return fields


def _dedupe_aliases(values: list[str | None]) -> list[str]:
    aliases: list[str] = []
    for value in values:
        if value is None:
            continue
        alias = value.strip()
        if not alias or alias == "*" or not _TRUSTED_IDENTIFIER_RE.match(alias):
            continue
        if alias not in aliases:
            aliases.append(alias)
    return aliases


def _string_value(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _list_of_strings(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return _dedupe_aliases([item if isinstance(item, str) else None for item in value])


def _caller_kind_from_principal_id(auth_input: Mapping[str, Any]) -> CallerKind:
    principal_id = _string_value(auth_input, "principal_id") or ""
    return "service_principal" if principal_id.startswith("service:") else "principal"


def _account_type_from_descriptor(descriptor: Mapping[str, Any]) -> AccountType:
    account_type = _string_value(descriptor, "account_type") or "user"
    if account_type == "user":
        return "user"
    if account_type == "service":
        return "service"
    raise AccountResolutionError("Identity descriptor has invalid account_type")


def _service_name_from_principal_id(principal_id: str) -> str | None:
    if not principal_id.startswith("service:"):
        return None
    service_name = principal_id.removeprefix("service:").strip()
    if not service_name or not _SERVICE_NAME_RE.match(service_name):
        return None
    return service_name


def _available_service_names(config: AuthServiceConfig) -> set[str]:
    names = set(_BUILT_IN_SERVICE_NAMES)
    names.update(config.allowed_service_principals)
    try:
        from nmp.platform_runner.registry import get_available_services

        names.update(get_available_services().keys())
    except Exception:
        logger.debug("Could not load platform service registry for service principal allowlist", exc_info=True)
    return names


def _descriptor_from_input(auth_input: Mapping[str, Any]) -> dict[str, Any] | None:
    principal_id = _string_value(auth_input, "principal_id")
    if principal_id is None:
        return None

    service_name = _service_name_from_principal_id(principal_id)
    if service_name is not None:
        return {
            "issuer": "nemo:service",
            "subject": service_name,
            "subject_claim": "service",
            "account_type": "service",
            "display_name": service_name,
            "authz_aliases": [principal_id],
        }

    email = _string_value(auth_input, "principal_email")
    if _string_value(auth_input, "on_behalf_of_principal_id") is not None:
        email = None
    return {
        "issuer": "nemo:trusted-header",
        "subject": principal_id,
        "subject_claim": "principal_id",
        "account_type": "user",
        "display_name": email or principal_id,
        "primary_email": email,
        "authz_aliases": _dedupe_aliases([principal_id, email]),
    }


def _on_behalf_descriptor_from_input(auth_input: Mapping[str, Any]) -> dict[str, Any] | None:
    on_behalf_of = _string_value(auth_input, "on_behalf_of_principal_id")
    if on_behalf_of is None:
        return None
    email = _string_value(auth_input, "principal_email")
    return {
        "issuer": "nemo:trusted-header",
        "subject": on_behalf_of,
        "subject_claim": "on_behalf_of_principal_id",
        "account_type": "user",
        "display_name": email or on_behalf_of,
        "primary_email": email,
        "authz_aliases": _dedupe_aliases([on_behalf_of, email]),
    }


async def _account_identity_store() -> AccountIdentityStore:
    try:
        session_maker = await get_async_session_maker()
    except RuntimeError:
        entities_config = EntitiesConfig.get()
        await initialize_async_engine(entities_config)
        await initialize_database(run_migrations=entities_config.run_migrations)
        session_maker = await get_async_session_maker()
    return AccountIdentityStore(session_maker)


class AccountResolver:
    """Resolve PDP input principals to stable account context."""

    def __init__(self, config: AuthServiceConfig) -> None:
        self.config = config

    async def resolve_authz_input(self, auth_input: Mapping[str, Any]) -> ResolvedAccountContext:
        """Resolve account context for a PDP input document."""
        context = ResolvedAccountContext(
            actor_account_id=_string_value(auth_input, "actor_account_id"),
            actor_aliases=_list_of_strings(auth_input, "actor_aliases"),
            caller_kind=_caller_kind_from_principal_id(auth_input),
            subject_account_id=_string_value(auth_input, "subject_account_id"),
            subject_aliases=_list_of_strings(auth_input, "subject_aliases"),
        )

        identity_resolution = auth_input.get("identity_resolution")
        resolution_map = identity_resolution if isinstance(identity_resolution, Mapping) else {}

        principal_descriptor = resolution_map.get("principal")
        if not isinstance(principal_descriptor, Mapping):
            principal_descriptor = _descriptor_from_input(auth_input)
        if context.actor_account_id is None and principal_descriptor is not None:
            account_id, aliases, resolved_caller_kind = await self._resolve_descriptor(principal_descriptor)
            context = ResolvedAccountContext(
                actor_account_id=account_id,
                actor_aliases=aliases,
                caller_kind=resolved_caller_kind,
                subject_account_id=context.subject_account_id,
                subject_aliases=context.subject_aliases,
            )

        on_behalf_descriptor = resolution_map.get("on_behalf_of")
        if not isinstance(on_behalf_descriptor, Mapping):
            on_behalf_descriptor = _on_behalf_descriptor_from_input(auth_input)
        if context.subject_account_id is None and on_behalf_descriptor is not None:
            account_id, aliases, _ = await self._resolve_descriptor(on_behalf_descriptor)
            context = ResolvedAccountContext(
                actor_account_id=context.actor_account_id,
                actor_aliases=context.actor_aliases,
                caller_kind=context.caller_kind,
                subject_account_id=account_id,
                subject_aliases=aliases,
            )

        return context

    async def _resolve_descriptor(self, descriptor: Mapping[str, Any]) -> tuple[str, list[str], CallerKind]:
        issuer = _string_value(descriptor, "issuer")
        subject = _string_value(descriptor, "subject")
        subject_claim = _string_value(descriptor, "subject_claim") or "sub"
        account_type = _account_type_from_descriptor(descriptor)
        if issuer is None or subject is None:
            raise AccountResolutionError("Identity descriptor is missing issuer or subject")

        caller_kind: CallerKind = "service_principal" if account_type == "service" else "principal"
        if account_type == "service":
            self._require_allowed_service(subject)

        claims_snapshot = descriptor.get("claims_snapshot")
        if not isinstance(claims_snapshot, dict):
            claims_snapshot = {}

        store = await _account_identity_store()
        try:
            record = await store.resolve_or_materialize(
                issuer=issuer,
                subject=subject,
                subject_claim=subject_claim,
                account_type=account_type,
                display_name=_string_value(descriptor, "display_name"),
                primary_email=_string_value(descriptor, "primary_email"),
                link_key=_string_value(descriptor, "link_key"),
                link_key_claim=_string_value(descriptor, "link_key_claim"),
                claims_snapshot=claims_snapshot,
                linked_via=_string_value(descriptor, "linked_via") or "resolver_materialization",
                linked_by=_string_value(descriptor, "linked_by"),
            )
        except (AccountIdentityConflictError, AccountIdentityUnavailableError) as exc:
            raise AccountResolutionError(str(exc)) from exc

        aliases = _list_of_strings(descriptor, "authz_aliases")
        if account_type == "service":
            aliases = _dedupe_aliases([f"service:{subject}", *aliases])
        else:
            aliases = _dedupe_aliases([subject, *aliases])
        return record.account_id, aliases, caller_kind

    def _require_allowed_service(self, service_name: str) -> None:
        if not _SERVICE_NAME_RE.match(service_name):
            raise ServicePrincipalNotAllowedError(f"Malformed service principal: service:{service_name}")
        if service_name not in _available_service_names(self.config):
            raise ServicePrincipalNotAllowedError(f"Unknown service principal: service:{service_name}")


async def resolve_authz_account_context(
    auth_input: Mapping[str, Any],
    config: AuthServiceConfig,
) -> ResolvedAccountContext:
    """Resolve stable account context for a PDP input document."""
    return await AccountResolver(config).resolve_authz_input(auth_input)
