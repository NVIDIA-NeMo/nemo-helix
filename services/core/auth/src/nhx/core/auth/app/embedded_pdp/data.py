# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy data management for the embedded WASM policy engine.

This module handles loading and refreshing authorization data for the policy engine.
"""

import contextvars
import logging
from typing import Any, Optional

from nhx.common.entities import EntityClient

from ..bundle import build_authorization_data
from .engine import set_policy_data

logger = logging.getLogger(__name__)

# When bundle_cache_seconds==0, evaluate_policy calls load_policy_data before every PDP eval.
# load_policy_data fetches role bindings via the entities API, which calls the PDP again.
# Without a re-entrancy guard that would recurse until RecursionError (PDP→entities→PDP→…).
_load_policy_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "nhx_embedded_pdp_load_policy_depth", default=0
)


async def apply_embedded_policy_document(
    entity_client_for_dynamic: Optional[EntityClient],
    *,
    skip_static_bootstrap: bool = False,
) -> dict[str, Any]:
    """Build authorization data and push it into the embedded PDP WASM store.

    This is the single implementation shared by the auth service refresh loop and
    per-request policy reload (``load_policy_data``).

    Args:
        entity_client_for_dynamic: Client used for ``build_authorization_data`` to fetch
            role bindings (typically ``entities_client.as_service("auth", internal=True)``
            or ``dependency_provider.get_service_entity_client("auth")``). Pass
            ``None`` to load static YAML only (after bootstrap when applicable).
        skip_static_bootstrap: If ``False`` (default), load static YAML first so the PDP
            can authorize entity fetches. If ``True``, skip that step — use on steady-state
            background refresh when WASM already holds a full document.

    Returns:
        The full authorization data dict last applied via ``set_policy_data``.

    Raises:
        Exception: If ``build_authorization_data`` fails (callers handle fallback where
            appropriate).
    """
    if not skip_static_bootstrap:
        bootstrap = await build_authorization_data(None)
        set_policy_data(bootstrap)
        authz = bootstrap.get("authz", {})
        logger.debug(
            "Embedded PDP static bootstrap applied (endpoints=%d, roles=%d)",
            len(authz.get("endpoints", {})),
            len(authz.get("roles", {})),
        )

    data = await build_authorization_data(entity_client_for_dynamic)
    set_policy_data(data)
    return data


async def load_policy_data(entities_client: Optional[EntityClient] = None) -> bool:
    """Load policy data from database into the policy engine.

    Args:
        entities_client: Optional EntityClient for fetching dynamic data

    Returns:
        True if policy data was loaded successfully, False otherwise.
        On failure, static data is still loaded (without dynamic role bindings).
    """
    depth = _load_policy_depth.get()
    if depth > 0:
        logger.debug(
            "Skipping nested load_policy_data (depth=%d): inner PDP eval during entity fetch "
            "for policy reload; bootstrap data from outer load is already in WASM.",
            depth,
        )
        return True

    reset_token = _load_policy_depth.set(depth + 1)
    try:
        return await _load_policy_data_impl(entities_client)
    finally:
        _load_policy_depth.reset(reset_token)


async def _load_policy_data_impl(entities_client: Optional[EntityClient] = None) -> bool:
    try:
        # Elevate to service principal for internal entity store access
        # Mark as internal to suppress access logging
        service_client = entities_client.as_service("auth", internal=True) if entities_client else None
        data = await apply_embedded_policy_document(
            service_client,
            skip_static_bootstrap=False,
        )
        principals_count = len(data.get("authz", {}).get("principals", {}))
        logger.info(
            "Policy data loaded successfully (principals: %d, has_entities_client: %s)",
            principals_count,
            entities_client is not None,
        )
        return True
    except Exception as e:
        logger.warning("Failed to load dynamic policy data: %s. Loading static data only.", e)
        # Load static data without dynamic role bindings
        try:
            data = await build_authorization_data(None)
            set_policy_data(data)
            logger.info("Static policy data loaded (dynamic data will be loaded on next refresh)")
            return False
        except Exception as fallback_error:
            logger.exception("Failed to load even static policy data: %s", fallback_error)
            return False
