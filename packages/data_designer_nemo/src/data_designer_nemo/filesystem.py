# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from filesets import FilesetFileSystem
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient


def make_filesystem(client: NemoClient) -> FilesetFileSystem:
    return FilesetFileSystem(client=FilesClient.from_client(client))
