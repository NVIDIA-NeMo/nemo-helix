# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from data_designer_nemo.filesystem import make_filesystem
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient


def test_make_filesystem_uses_sync_client_for_sync_sdk() -> None:
    client = Mock(spec=NemoClient)
    files_client = Mock()
    filesystem = Mock()

    with (
        patch("data_designer_nemo.filesystem.client_from_platform", return_value=files_client) as client_from_platform,
        patch("data_designer_nemo.filesystem.FilesetFileSystem", return_value=filesystem) as fileset_filesystem,
    ):
        result = make_filesystem(client)

    assert result is filesystem
    client_from_platform.assert_called_once_with(client, FilesClient)
    fileset_filesystem.assert_called_once_with(client=files_client)
