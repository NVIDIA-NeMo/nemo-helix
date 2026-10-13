# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NeMo Helix authentication header names."""

AUTHORIZATION_HEADER = "Authorization"

TRUSTED_IDENTITY_HEADERS = frozenset(
    {
        "x-nhx-principal-id",
        "x-nhx-actor-account-id",
        "x-nhx-principal-email",
        "x-nhx-principal-groups",
        "x-nhx-actor-aliases",
        "x-nhx-principal-on-behalf-of",
        "x-nhx-principal-on-behalf-of-email",
        "x-nhx-principal-on-behalf-of-groups",
        "x-nhx-subject-account-id",
        "x-nhx-subject-aliases",
        "x-nhx-scopes",
    }
)

AUTHENTICATION_CONTEXT_HEADERS = frozenset({AUTHORIZATION_HEADER.lower(), *TRUSTED_IDENTITY_HEADERS})
