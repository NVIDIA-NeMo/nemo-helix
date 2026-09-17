# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from filesets import FilesetFileSystem
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import FilesClient


def make_filesystem(sdk: NeMoPlatform) -> FilesetFileSystem:
    return FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
