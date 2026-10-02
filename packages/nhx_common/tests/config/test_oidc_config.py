# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.config.base import OIDCConfig
from pydantic import ValidationError


def test_oidc_bearer_token_source_defaults_to_access_token() -> None:
    assert OIDCConfig().bearer_token_source == "access_token"


def test_oidc_bearer_token_source_accepts_id_token() -> None:
    assert OIDCConfig(bearer_token_source="id_token").bearer_token_source == "id_token"


def test_oidc_bearer_token_source_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match="bearer_token_source"):
        OIDCConfig(bearer_token_source="refresh_token")  # type: ignore[arg-type]
