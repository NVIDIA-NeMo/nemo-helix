# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs service implementation."""

import logging
from typing import ClassVar, List

from nhx.common.service import RouterConfig, Service
from nhx.core.jobs.config import JobsServiceConfig

logger = logging.getLogger(__name__)


class JobsService(Service[JobsServiceConfig]):
    """Jobs service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth", "secrets", "files"]

    def __init__(self):
        """Initialize the jobs service."""
        super().__init__(name="jobs", module_name="nhx.core.jobs")

    @property
    def title(self) -> str:
        return "NeMo Helix Jobs Microservice"

    @property
    def description(self) -> str:
        return "Service for job scheduling and execution management."

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the jobs service."""
        from nhx.core.jobs.api.v2.jobs import endpoints

        return [
            RouterConfig(
                endpoints.router,
                tag="Jobs",
                description="Job management endpoints",
            ),
        ]
