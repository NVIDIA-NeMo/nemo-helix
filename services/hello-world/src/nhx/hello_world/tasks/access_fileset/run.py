# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task to test access to a fileset in a specified workspace.

This task is used for E2E testing of auth propagation. It attempts to
retrieve a fileset and reports whether access was granted or denied.
"""

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nhx.common.jobs.config import get_task_config
from nhx.common.sdk_factory import get_platform_sdk
from pydantic import BaseModel


class AccessFilesetConfig(BaseModel):
    """Configuration for the access_fileset task."""

    workspace: str
    fileset: str


def run(*, sdk: NeMoHelix | None = None) -> int:
    """Attempt to access a fileset in the specified workspace.

    Args:
        sdk: Optional SDK instance for dependency injection (for testing).
            If None, uses get_platform_sdk().

    Returns:
        Exit code:
        - 0: Successfully accessed the fileset
        - 1: Access denied (403) or other error
    """
    try:
        config = get_task_config(AccessFilesetConfig)
        sdk = sdk or get_platform_sdk()

        print(f"Attempting to access fileset '{config.fileset}' in workspace '{config.workspace}'")

        files = client_from_platform(sdk, FilesClient)
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
