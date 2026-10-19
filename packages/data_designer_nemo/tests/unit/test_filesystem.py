# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from data_designer_nemo.filesystem import make_filesystem
from nemo_helix_plugin.client.client import NemoClient


def test_make_filesystem_uses_sync_client_for_sync_client() -> None:
    client = Mock(spec=NemoClient)
    files_client = Mock()
    filesystem = Mock()

    with (
        patch("data_designer_nemo.filesystem.FilesClient.from_client", return_value=files_client) as from_client,
        patch("data_designer_nemo.filesystem.FilesetFileSystem", return_value=filesystem) as fileset_filesystem,
    ):
        result = make_filesystem(client)

    assert result is filesystem
    from_client.assert_called_once_with(client)
    fileset_filesystem.assert_called_once_with(client=files_client)
