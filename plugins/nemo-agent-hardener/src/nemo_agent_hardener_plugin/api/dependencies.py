# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dependencies for Agent Hardener API routes."""

from nemo_helix_plugin.client_provider import get_nemo_client
from nemo_helix_plugin.files.client import FilesClient


def get_service_files_client() -> FilesClient:
    """Provide a request-scoped Files client acting as Agent Hardener."""
    return FilesClient.from_client(get_nemo_client(as_service="agent-hardener", internal=True))
