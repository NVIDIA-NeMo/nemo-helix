# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task to test access to a fileset in a specified workspace.

This task is used for E2E testing of auth propagation. It attempts to
retrieve a fileset and reports whether access was granted or denied.
"""

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nhx.common.client_factory import get_nemo_client
from nhx.common.jobs.config import get_task_config
from pydantic import BaseModel


class AccessFilesetConfig(BaseModel):
    """Configuration for the access_fileset task."""

    workspace: str
    fileset: str


def run(*, client: NemoClient | None = None) -> int:
    """Attempt to access a fileset in the specified workspace.

    Args:
        client: Optional platform client for dependency injection (for testing).
            If None, uses get_nemo_client().

    Returns:
        Exit code:
        - 0: Successfully accessed the fileset
        - 1: Access denied (403) or other error
    """
    try:
        config = get_task_config(AccessFilesetConfig)
        client = client or get_nemo_client()

        print(f"Attempting to access fileset '{config.fileset}' in workspace '{config.workspace}'")

        files = FilesClient.from_client(client)
        fileset = files.get_fileset(workspace=config.workspace, name=config.fileset).data()

        print(f"Successfully accessed fileset: {fileset.name}")
        return 0

    except NemoHTTPError as e:
        if e.status_code == 403:
            print(f"Access denied (403 Forbidden): {e}")
        elif e.status_code == 404:
            print(f"Fileset not found (404): {e}")
        else:
            print(f"API error ({e.status_code}): {e}")
        return 1

    except Exception as e:
        print(f"Task failed with unexpected error: {e}")
        return 1
