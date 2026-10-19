# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing for NeMo principal identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MAX_PRINCIPAL_ID_LENGTH = 256
PRINCIPAL_ID_ALLOWED_CHARACTERS = "alphanumeric, @, ., -, _, :, +, /"
SERVICE_PRINCIPAL_PREFIX = "service:"
SERVICE_ACCOUNT_PRINCIPAL_PREFIX = "service-account:"

PrincipalIdentifierKind = Literal["principal", "service_principal", "service_account"]
CallerKind = Literal["principal", "service_principal"]
PRINCIPAL_IDENTIFIER_KIND_PRINCIPAL: PrincipalIdentifierKind = "principal"
PRINCIPAL_IDENTIFIER_KIND_SERVICE_PRINCIPAL: PrincipalIdentifierKind = "service_principal"
PRINCIPAL_IDENTIFIER_KIND_SERVICE_ACCOUNT: PrincipalIdentifierKind = "service_account"
CALLER_KIND_PRINCIPAL: CallerKind = "principal"
CALLER_KIND_SERVICE_PRINCIPAL: CallerKind = "service_principal"

_PRINCIPAL_ID_RE = re.compile(r"^[a-zA-Z0-9@._\-:+/]+$")
_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


class InvalidPrincipalIdentifier(ValueError):
    """Raised when a principal identifier is malformed."""


@dataclass(frozen=True)
class PrincipalIdentifier:
    """Parsed principal identifier with stable caller classification."""

    raw: str
    kind: PrincipalIdentifierKind
    service_name: str | None = None

    @property
    def caller_kind(self) -> CallerKind:
        return CALLER_KIND_SERVICE_PRINCIPAL if self.is_service_principal() else CALLER_KIND_PRINCIPAL

    def is_principal(self) -> bool:
        return self.kind == PRINCIPAL_IDENTIFIER_KIND_PRINCIPAL

    def is_service_principal(self) -> bool:
        return self.kind == PRINCIPAL_IDENTIFIER_KIND_SERVICE_PRINCIPAL

    def is_service_account(self) -> bool:
        return self.kind == PRINCIPAL_IDENTIFIER_KIND_SERVICE_ACCOUNT

    def is_privileged(self) -> bool:
        return self.is_service_principal()

    def is_service_identity(self) -> bool:
        return self.is_service_principal() or self.is_service_account()


def _invalid(label: str, message: str) -> InvalidPrincipalIdentifier:
    return InvalidPrincipalIdentifier(f"{label} {message}")


def validate_principal_identifier(value: str, *, label: str = "principal") -> str:
    """Return a stripped valid principal identifier or raise."""
    principal_id = value.strip()
    if not principal_id:
        raise _invalid(label, "is empty")
    if len(principal_id) > MAX_PRINCIPAL_ID_LENGTH:
        raise _invalid(label, f"exceeds maximum length of {MAX_PRINCIPAL_ID_LENGTH} characters")
    if not _PRINCIPAL_ID_RE.match(principal_id):
        raise _invalid(label, f"contains invalid characters; allowed: {PRINCIPAL_ID_ALLOWED_CHARACTERS}")
    return principal_id


def parse_service_name(value: str, *, label: str = "service principal") -> str:
    """Return a valid bare service name or raise."""
    service_name = value.strip()
    if not service_name or not _SERVICE_NAME_RE.match(service_name):
        raise _invalid(label, "is malformed")
    return service_name


def parse_principal_identifier(
    value: str,
    *,
    label: str = "principal",
    validate: bool = True,
) -> PrincipalIdentifier:
    """Parse and classify a principal identifier.

    Set ``validate=False`` for already-authenticated token subjects whose IdP
    subject syntax may be broader than trusted-header identifiers; reserved
    ``service:`` principals are still shape-validated in both modes.
    """
    principal_id = validate_principal_identifier(value, label=label) if validate else value.strip()
    if not principal_id:
        raise _invalid(label, "is empty")
    if len(principal_id) > MAX_PRINCIPAL_ID_LENGTH:
        raise _invalid(label, f"exceeds maximum length of {MAX_PRINCIPAL_ID_LENGTH} characters")
    if principal_id.startswith(SERVICE_PRINCIPAL_PREFIX):
        service_name = parse_service_name(
            principal_id.removeprefix(SERVICE_PRINCIPAL_PREFIX),
            label=label,
        )
        return PrincipalIdentifier(
            raw=principal_id,
            kind=PRINCIPAL_IDENTIFIER_KIND_SERVICE_PRINCIPAL,
            service_name=service_name,
        )
    if principal_id.startswith(SERVICE_ACCOUNT_PRINCIPAL_PREFIX):
        if not principal_id.removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX).strip():
            raise _invalid(label, "is malformed")
        return PrincipalIdentifier(raw=principal_id, kind=PRINCIPAL_IDENTIFIER_KIND_SERVICE_ACCOUNT)
    return PrincipalIdentifier(raw=principal_id, kind=PRINCIPAL_IDENTIFIER_KIND_PRINCIPAL)


def normalize_service_principal_identifier(value: str, *, label: str = "service principal") -> str:
    """Return a well-formed service principal id from either a bare service name or service:<name>."""
    service = value.strip()
    if service.startswith(SERVICE_PRINCIPAL_PREFIX):
        parsed = parse_principal_identifier(service, label=label)
        if parsed.is_service_principal():
            return parsed.raw
        raise _invalid(label, "is not a service principal")
    return f"{SERVICE_PRINCIPAL_PREFIX}{parse_service_name(service, label=label)}"


def is_service_principal(value: str) -> bool:
    """Return True only for well-formed internal service principals."""
    try:
        return parse_principal_identifier(value).is_service_principal()
    except InvalidPrincipalIdentifier:
        return False
