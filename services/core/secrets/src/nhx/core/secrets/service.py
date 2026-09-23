# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secrets service implementation."""

import logging
from typing import ClassVar, List

from nhx.common.service import RouterConfig, Service
from nhx.core.secrets.config import SecretsServiceConfig

logger = logging.getLogger(__name__)


class SecretsService(Service[SecretsServiceConfig]):
    """Secrets service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth"]

    def __init__(self):
        """Initialize the secrets service."""
        super().__init__(name="secrets", module_name="nhx.core.secrets")

    @property
    def title(self) -> str:
        return "NeMo Helix Secrets Microservice"

    @property
    def description(self) -> str:
        return "Service for secrets management."

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the secrets service."""
        from nhx.core.secrets.api.v2.admin import endpoints as admin_api
        from nhx.core.secrets.api.v2.secrets import endpoints as secrets_api

        return [
            RouterConfig(
                secrets_api.router,
                tag="Secrets",
                description="Secrets management endpoints",
            ),
            RouterConfig(
                admin_api.router,
                tag="Secrets Admin",
                description="Administrative endpoints for secrets management",
            ),
        ]
