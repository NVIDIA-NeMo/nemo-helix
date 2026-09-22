# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for workspace access helpers in api/v2/utils.py."""

from nhx.common.auth.models import Principal
from nhx.core.entities.api.v2.utils import _applicable_principal_strings


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
