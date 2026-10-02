# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from filesets import FilesetFileSystem
from nemo_helix_plugin.client.adapter import SyncHelixClient, client_from_platform
from nemo_helix_plugin.files.client import FilesClient


def make_filesystem(sdk: SyncHelixClient) -> FilesetFileSystem:
    return FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
