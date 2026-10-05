# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync context backing real Data Designer engine workloads."""

from data_designer.engine.resources.person_reader import PersonReader
from data_designer.engine.resources.seed_reader import (
    DirectorySeedReader,
    FileContentsSeedReader,
    HuggingFaceSeedReader,
    SeedReader,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer_nemo.fileset_file_seed_reader import FilesetFileSeedReader
from data_designer_nemo.fileset_filesystem_provider import (
    FilesetFileSystemProvider,
)
from data_designer_nemo.person_reader import FilesetsPersonReader
from data_designer_nemo.secret_resolver import NHXSecretResolver
from nemo_helix_plugin.client.adapter import SyncHelixClient


class DataDesignerExecutionContext:
    """Sync-only context for the upstream Data Designer engine."""

    def __init__(
        self,
        client: SyncHelixClient,
        workspace: str,
        *,
        validated_roots: set[str] | None = None,
    ) -> None:
        self._client = client
        self._workspace = workspace
        self._validated_filesystem_roots = set(validated_roots or ())

    def get_secret_resolver(self) -> SecretResolver:
        return NHXSecretResolver(self._client, self._workspace)

    def get_seed_readers(self) -> list[SeedReader]:
        provider = FilesetFileSystemProvider(
            self._client,
            workspace=self._workspace,
            validated_roots=self._validated_filesystem_roots,
        )
        return [
            HuggingFaceSeedReader(),
            FilesetFileSeedReader(self._client),
            DirectorySeedReader(fs_provider=provider),
            FileContentsSeedReader(fs_provider=provider),
        ]

    def get_person_reader(self) -> PersonReader | None:
        return FilesetsPersonReader(self._client)


def create_execution_context(
    client: SyncHelixClient,
    workspace: str,
    *,
    validated_roots: set[str] | None = None,
) -> DataDesignerExecutionContext:
    return DataDesignerExecutionContext(client, workspace, validated_roots=validated_roots)
