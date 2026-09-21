# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nmp.common.auth.principal_identifier import (
    PRINCIPAL_IDENTIFIER_KIND_PRINCIPAL,
    PRINCIPAL_IDENTIFIER_KIND_SERVICE_ACCOUNT,
    PRINCIPAL_IDENTIFIER_KIND_SERVICE_PRINCIPAL,
    InvalidPrincipalIdentifier,
    is_service_principal,
    normalize_service_principal_identifier,
    parse_principal_identifier,
)


@pytest.mark.parametrize(
    ("principal_id", "kind", "caller_kind", "service_name"),
    [
        ("user@example.com", "principal", "principal", None),
        ("service:jobs-controller", "service_principal", "service_principal", "jobs-controller"),
        ("service:models_controller.1", "service_principal", "service_principal", "models_controller.1"),
        ("service-account:otel-collector", "service_account", "principal", None),
    ],
)
def test_parse_principal_identifier_classifies_reserved_prefixes(
    principal_id: str,
    kind: str,
    caller_kind: str,
    service_name: str | None,
) -> None:
    parsed = parse_principal_identifier(principal_id)

    assert parsed.raw == principal_id
    assert parsed.kind == kind
    assert parsed.caller_kind == caller_kind
    assert parsed.service_name == service_name


@pytest.mark.parametrize(
    ("principal_id", "expected_kind", "is_principal", "is_service_principal_value", "is_service_account"),
    [
        ("user@example.com", PRINCIPAL_IDENTIFIER_KIND_PRINCIPAL, True, False, False),
        ("service:jobs", PRINCIPAL_IDENTIFIER_KIND_SERVICE_PRINCIPAL, False, True, False),
        ("service-account:otel-collector", PRINCIPAL_IDENTIFIER_KIND_SERVICE_ACCOUNT, False, False, True),
    ],
)
def test_parse_principal_identifier_exposes_kind_predicates(
    principal_id: str,
    expected_kind: str,
    is_principal: bool,
    is_service_principal_value: bool,
    is_service_account: bool,
) -> None:
    parsed = parse_principal_identifier(principal_id)

    assert parsed.kind == expected_kind
    assert parsed.is_principal() is is_principal
    assert parsed.is_service_principal() is is_service_principal_value
    assert parsed.is_service_account() is is_service_account


@pytest.mark.parametrize(
    "principal_id",
    [
        "service:",
        "service:  ",
        "service:has spaces",
        "service:bad$name",
        "service:/path",
        "service:*",
        "service:-starts-with-punctuation",
    ],
)
def test_parse_principal_identifier_rejects_malformed_service_principals(principal_id: str) -> None:
    with pytest.raises(InvalidPrincipalIdentifier):
        parse_principal_identifier(principal_id)
    assert is_service_principal(principal_id) is False


@pytest.mark.parametrize("principal_id", ["service-account:", "service-account:  "])
def test_parse_principal_identifier_rejects_empty_service_account_suffix(principal_id: str) -> None:
    with pytest.raises(InvalidPrincipalIdentifier, match="malformed"):
        parse_principal_identifier(principal_id)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("jobs", "service:jobs"),
        (" service:jobs-controller ", "service:jobs-controller"),
    ],
)
def test_normalize_service_principal_identifier(value: str, expected: str) -> None:
    assert normalize_service_principal_identifier(value) == expected


def test_normalize_service_principal_identifier_rejects_service_account() -> None:
    with pytest.raises(InvalidPrincipalIdentifier, match="malformed"):
        normalize_service_principal_identifier("service-account:otel-collector")


def test_parse_principal_identifier_can_classify_extended_non_service_subject_without_validation() -> None:
    with pytest.raises(InvalidPrincipalIdentifier, match="invalid characters"):
        parse_principal_identifier("auth0|abc")

    parsed = parse_principal_identifier("auth0|abc", validate=False)

    assert parsed.kind == "principal"
    assert parsed.caller_kind == "principal"
