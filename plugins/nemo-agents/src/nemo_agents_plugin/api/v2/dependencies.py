# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency injection for the Agents plugin API.

Re-exports the standard NeMo Helix dependencies so route handlers can import from
a single location.
"""

from fastapi import Depends
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_helix_plugin.files.client import AsyncFilesClient


def get_files_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    """Provide a Files service client sharing the request's platform client transport."""
    return AsyncFilesClient.from_client(client)


__all__ = ["get_entity_client", "get_files_client"]
