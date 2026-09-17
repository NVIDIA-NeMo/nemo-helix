# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from data_designer_nemo.filesystem import make_filesystem
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.files.client import FilesClient


def test_make_filesystem_uses_sync_client_for_sync_sdk() -> None:
    sdk = Mock(spec=NeMoPlatform)
    files_client = Mock()
    filesystem = Mock()

    with (
        patch("data_designer_nemo.filesystem.client_from_platform", return_value=files_client) as client_from_platform,
        patch("data_designer_nemo.filesystem.FilesetFileSystem", return_value=filesystem) as fileset_filesystem,
    ):
        result = make_filesystem(sdk)

    assert result is filesystem
    client_from_platform.assert_called_once_with(sdk, FilesClient)
    fileset_filesystem.assert_called_once_with(client=files_client)
