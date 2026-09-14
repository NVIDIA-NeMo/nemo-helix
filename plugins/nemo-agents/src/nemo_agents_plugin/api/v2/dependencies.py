# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency injection for the Agents plugin API.

Re-exports the standard NeMo Platform dependencies so route handlers can import from
a single location.
"""

from fastapi import Depends
from nemo_agents_plugin.spec_revision import files_client_for
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.dependencies import get_sdk_client
from nemo_platform_plugin.entity_client import get_entity_client
from nemo_platform_plugin.files.client import AsyncFilesClient


def get_files_client(sdk: AsyncNeMoPlatform = Depends(get_sdk_client)) -> AsyncFilesClient | None:
    """Provide a Files service client sharing the request's SDK transport, or None."""
    return files_client_for(sdk)


__all__ = ["get_entity_client", "get_files_client"]
