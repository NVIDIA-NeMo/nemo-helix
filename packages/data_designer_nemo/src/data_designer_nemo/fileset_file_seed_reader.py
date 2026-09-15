# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar

import duckdb
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from data_designer_nemo.filesystem import make_filesystem
from nemo_platform import NeMoPlatform

workspace_cvar = ContextVar[str | None]("workspace_cvar", default=None)


class FilesetFileSeedReader(SeedReader[FilesetFileSeedSource]):
    # The Data Designer library discovers seed-reader plugins by instantiating them with no args.
    # Within this plugin we always inject an SDK and pass a collection of readers that replaces
    # any library-produced default collection of readers.
    def __init__(self, sdk: NeMoPlatform | None = None) -> None:
        self._sdk = sdk

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        if self._sdk is None:
            raise RuntimeError("FilesetFileSeedReader requires an injected NeMo Platform SDK")

        filesystem = make_filesystem(self._sdk)
        conn = duckdb.connect()
        conn.register_filesystem(filesystem)
        return conn

    def get_dataset_uri(self) -> str:
        path = self.source.path
        if self._requires_workspace_prefix(path):
            path = f"{self._get_workspace()}/{path}"

        return f"fileset://{path}"

    def _requires_workspace_prefix(self, path: str) -> bool:
        user_provided_fileset = path.split("#")[0]
        components = user_provided_fileset.split("/")
        return len(components) == 1

    def _get_workspace(self) -> str:
        workspace = workspace_cvar.get()
        if workspace is None:
            raise ValueError(
                "FilesetFileSeedSource path does not include a workspace and "
                "a workspace could not be inferred from the current context"
            )
        return workspace
