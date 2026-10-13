# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI compatibility facade for shared OIDC token-provider primitives."""

from nemo_helix_plugin.client.oidc import (
    DEFAULT_REFRESH_MARGIN_SECONDS as DEFAULT_REFRESH_MARGIN_SECONDS,
)
from nemo_helix_plugin.client.oidc import (
    OIDCTokenProvider as OIDCTokenProvider,
)
from nemo_helix_plugin.client.oidc import (
    TokenPersistenceError as TokenPersistenceError,
)
from nemo_helix_plugin.client.oidc import (
    TokenRefreshError as TokenRefreshError,
)
from nemo_helix_plugin.client.oidc import (
    TokenSet as TokenSet,
)
from nemo_helix_plugin.client.oidc import (
    refresh_token_grant as refresh_token_grant,
)

__all__ = [
    "DEFAULT_REFRESH_MARGIN_SECONDS",
    "OIDCTokenProvider",
    "TokenPersistenceError",
    "TokenRefreshError",
    "TokenSet",
    "refresh_token_grant",
]
