# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello World service implementation."""

from typing import ClassVar, List

from nhx.common.service import RouterConfig, Service
from nhx.hello_world.api.v1.hello import endpoints as hello
from nhx.hello_world.api.v1.messages import endpoints as messages
from nhx.hello_world.api.v2.jobs import endpoints as jobs
from nhx.hello_world.config import HelloWorldConfig


class HelloWorldService(Service[HelloWorldConfig]):
    """Hello World service for testing the platform."""

    dependencies: ClassVar[List[str]] = ["entities", "auth", "jobs", "secrets", "files"]

    def __init__(self):
        """Initialize the hello world service."""
        super().__init__(name="hello-world", module_name="nhx.hello_world")

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the hello world service."""
        return [
            RouterConfig(
                hello.router,
                prefix="/v2/workspaces/{workspace}",
                tag="Hello",
                description="Hello World greeting endpoints",
            ),
            RouterConfig(
                jobs.router,
                prefix="/v2/workspaces/{workspace}",
                tag="Jobs",
                description="Hello World job endpoints",
            ),
            RouterConfig(
                messages.router,
                prefix="/v2/workspaces/{workspace}",
                tag="Messages",
                description="Hello World message endpoints",
            ),
        ]
