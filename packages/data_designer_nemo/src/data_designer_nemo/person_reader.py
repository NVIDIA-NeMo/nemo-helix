# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import duckdb
from data_designer.engine.resources.person_reader import PersonReader
from data_designer_nemo.filesystem import make_filesystem
from data_designer_nemo.nemotron_personas import (
    get_locale_fileset_file_ref,
)
from nemo_helix_plugin.client.adapter import SyncHelixClient


class FilesetsPersonReader(PersonReader):
    """Provides DuckDB access to Nemotron personas datasets via filesets.

    DuckDB calls into the fileset filesystem synchronously, so this reader
    only accepts a sync :class:`NemoClient`.
    """

    def __init__(self, sdk: SyncHelixClient):
        self._sdk = sdk

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        filesystem = make_filesystem(self._sdk)
        conn = duckdb.connect()
        conn.register_filesystem(filesystem)
        return conn

    def get_dataset_uri(self, locale: str) -> str:
        return f"fileset://{get_locale_fileset_file_ref(locale)}"
