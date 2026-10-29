# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from enum import Enum

from nhx.common.config.base import OIDCConfig


class OIDCBearerPurpose(Enum):
    ACCESS_TOKEN = "access_token"
    ID_TOKEN = "id_token"
    UNSPECIFIED = "unspecified"
    INVALID = "invalid"


def oidc_bearer_purpose(header: Mapping[str, object], claims: Mapping[str, object]) -> OIDCBearerPurpose:
    purposes: set[OIDCBearerPurpose] = set()
    for value, allowed in (
        (
            header.get("typ", "JWT"),
            {
                "JWT": OIDCBearerPurpose.UNSPECIFIED,
                "at+jwt": OIDCBearerPurpose.ACCESS_TOKEN,
                "application/at+jwt": OIDCBearerPurpose.ACCESS_TOKEN,
            },
        ),
        (
            claims.get("typ", "JWT"),
            {
                "JWT": OIDCBearerPurpose.UNSPECIFIED,
                "Bearer": OIDCBearerPurpose.ACCESS_TOKEN,
                "ID": OIDCBearerPurpose.ID_TOKEN,
            },
        ),
        (
            claims.get("token_use", "unspecified"),
            {
                "unspecified": OIDCBearerPurpose.UNSPECIFIED,
                "access": OIDCBearerPurpose.ACCESS_TOKEN,
                "id": OIDCBearerPurpose.ID_TOKEN,
            },
        ),
    ):
        if not isinstance(value, str) or value not in allowed:
            return OIDCBearerPurpose.INVALID
        purpose = allowed[value]
        if purpose is not OIDCBearerPurpose.UNSPECIFIED:
            purposes.add(purpose)
    if len(purposes) > 1:
        return OIDCBearerPurpose.INVALID
    return next(iter(purposes), OIDCBearerPurpose.UNSPECIFIED)


def token_audiences(claims: Mapping[str, object]) -> list[str]:
    audience = claims.get("aud")
    if isinstance(audience, str) and audience:
        return [audience]
    if isinstance(audience, list) and audience and all(isinstance(value, str) and value for value in audience):
        return [value for value in audience if isinstance(value, str)]
    return []


def id_token_bearer_allowed(config: OIDCConfig, claims: Mapping[str, object]) -> bool:
    audiences = token_audiences(claims)
    if not audiences:
        return False
    authorized_party = claims.get("azp")
    client_id = claims.get("client_id")
    if authorized_party is not None and (not isinstance(authorized_party, str) or not authorized_party):
        return False
    if client_id is not None and (not isinstance(client_id, str) or not client_id):
        return False
    if authorized_party is not None and client_id is not None and authorized_party != client_id:
        return False
    selected_client = authorized_party or client_id
    if selected_client is None:
        if len(audiences) != 1:
            return False
        selected_client = audiences[0]
    if selected_client not in audiences:
        return False
    profiles = [profile for profile in (config.public_client, config.confidential_client) if profile is not None]
    matching_profiles = [profile for profile in profiles if profile.client_id == selected_client]
    return bool(matching_profiles) and all(profile.bearer_token_source == "id_token" for profile in matching_profiles)


def has_api_only_audience(config: OIDCConfig, claims: Mapping[str, object]) -> bool:
    if not config.audience:
        return False
    client_ids = {
        profile.client_id for profile in (config.public_client, config.confidential_client) if profile is not None
    }
    audiences = token_audiences(claims)
    return audiences == [config.audience] and config.audience not in client_ids
