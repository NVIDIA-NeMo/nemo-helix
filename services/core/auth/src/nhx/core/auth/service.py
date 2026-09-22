# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth service implementation."""

import asyncio
import logging
from typing import ClassVar, List, Optional

import nhx.core.auth.api.v2.access_keys.endpoints as access_keys
import nhx.core.auth.api.v2.authenticate as authenticate
from nhx.common.config import get_service_config
from nhx.common.service import RouterConfig, Service
from nhx.core.auth.api.v2 import workload_token_exchange
from nhx.core.auth.api.v2.bundle import endpoints as bundle
from nhx.core.auth.api.v2.discovery import endpoints as discovery
from nhx.core.auth.api.v2.iam import endpoints as iam
from nhx.core.auth.app.embedded_pdp.data import apply_embedded_policy_document
from nhx.core.auth.app.embedded_pdp.policy_wasm import ensure_embedded_policy_wasm
from nhx.core.auth.config import AuthServiceConfig

logger = logging.getLogger(__name__)


class AuthService(Service[AuthServiceConfig]):
    """Authentication and Authorization service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities"]

    def __init__(self):
        """Initialize the auth service."""
        super().__init__(name="auth", module_name="nhx.core.auth")
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._policy_refresh_healthy: bool = False

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the auth service."""
        config = get_service_config(AuthServiceConfig)
        routers = [
            RouterConfig(iam.router, tag="IAM", description="Identity and Access Management endpoints"),
            RouterConfig(bundle.router, tag="Bundle", description="OPA bundle endpoints"),
            RouterConfig(discovery.router, tag="Discovery", description="Platform configuration discovery endpoints"),
            RouterConfig(
                authenticate.router, tag="Authentication", description="Bearer token authentication endpoints"
            ),
            RouterConfig(access_keys.router, tag="Scoped Access Keys", description="Scoped Access Key endpoints"),
            RouterConfig(
                workload_token_exchange.router,
                tag="Workload Identity",
                description="Workload identity token exchange endpoints",
            ),
        ]

        # Only register authz routes when auth is enabled and using embedded policy engine
        if config.enabled and config.policy_decision_point_provider == "embedded":
            from nhx.core.auth.api.v2.authz import endpoints as authz

            routers.append(RouterConfig(authz.router, tag="Authorization", description="Policy evaluation endpoints"))

        return routers

    async def on_startup(self) -> None:
        """Initialize the auth service on startup."""
        config = get_service_config(AuthServiceConfig)
        if not config.enabled:
            logger.info("Auth disabled - skipping embedded policy engine initialization")
            return
        if config.policy_decision_point_provider == "embedded":
            ensure_embedded_policy_wasm(auto_build=config.embedded_pdp_auto_build_wasm)

    async def _start_refresh_loop(self) -> asyncio.Task:
        """Start background task to periodically refresh policy data."""
        config = self.service_config
        # Config should usually be set, but if not, use a default of 30 seconds
        refresh_interval = config.policy_data_refresh_interval if config else 30
        service_client = self.dependency_provider.get_service_entity_client("auth")

        logger.info("Starting auth policy data refresh loop")

        async def refresh_loop():
            while True:
                try:
                    logger.debug("Refreshing auth policy data")
                    # Bootstrap static YAML only when we do not yet have a successful full
                    # refresh (cold start or recovery after failure). On steady-state refreshes
                    # we skip this: the WASM already holds the last document, so entity fetches
                    # stay authorized without briefly replacing dynamic principals with bootstrap-only
                    # data. Same loader as ``load_policy_data`` / ``apply_embedded_policy_document``.
                    data = await apply_embedded_policy_document(
                        service_client,
                        skip_static_bootstrap=self._policy_refresh_healthy,
                    )
                    self._policy_refresh_healthy = True
                    principals_count = len(data.get("authz", {}).get("principals", {}))
                    logger.debug("Policy data refreshed (principals: %d)", principals_count)
                    await asyncio.sleep(float(refresh_interval))
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception(
                        "Policy data refresh failed. Will retry in %ss",
                        refresh_interval,
                    )
                    self._policy_refresh_healthy = False
                    await asyncio.sleep(float(refresh_interval))

        refresh_task = asyncio.create_task(refresh_loop())
        logger.info("Policy data refresh loop started (interval=%ds)", refresh_interval)
        return refresh_task

    async def startup(self) -> None:
        """Background startup task - load policy after entities is ready. Role bindings are seeded by platform-seed."""
        config = get_service_config(AuthServiceConfig)

        # Skip if auth is disabled or not using embedded provider
        if not config.enabled:
            logger.info("Auth not enabled, skipping initialization")
            return

        if config.policy_decision_point_provider != "embedded":
            logger.info("Not using embedded auth policy engine, skipping initialization")
            return

        self._refresh_task = await self._start_refresh_loop()

    async def is_ready(self) -> bool:
        """Return True if healthy. When auth and embedded PDP are enabled, depends on policy refresh success."""
        config = get_service_config(AuthServiceConfig)
        if not config.enabled or config.policy_decision_point_provider != "embedded":
            return True
        return self._policy_refresh_healthy

    async def on_shutdown(self) -> None:
        """Cleanup on service shutdown."""
        config = get_service_config(AuthServiceConfig)
        if config.enabled and config.policy_decision_point_provider == "embedded":
            if self._refresh_task and not self._refresh_task.done():
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass
            self._refresh_task = None
            logger.info("Policy data refresh loop stopped")
        await super().on_shutdown()
