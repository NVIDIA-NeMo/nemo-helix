# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.resources.person_reader import PersonReader
from data_designer_nemo.filesystem import make_filesystem
from data_designer_nemo.nemotron_personas import (
    get_locale_fileset_file_ref,
)
from nemo_helix_plugin.client.client import NemoClient

if TYPE_CHECKING:
    # Importing duckdb opens its default connection, which aborts in some
    # sandboxes, so keep it out of import time.
    import duckdb


class FilesetsPersonReader(PersonReader):
    """Provides DuckDB access to Nemotron personas datasets via filesets.

    DuckDB calls into the fileset filesystem synchronously, so this reader
    only accepts a sync :class:`NemoClient`.
    """

    def __init__(self, client: NemoClient):
        self._client = client

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        import duckdb

        filesystem = make_filesystem(self._client)
        conn = duckdb.connect()
        conn.register_filesystem(filesystem)
        return conn

    def get_dataset_uri(self, locale: str) -> str:
        return f"fileset://{get_locale_fileset_file_ref(locale)}"
