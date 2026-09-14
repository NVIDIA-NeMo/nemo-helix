# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize immutable stored Harbor tasksets for source-adapter consumers."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

from nemo_evaluator.api.schemas import TasksetRef
from nemo_evaluator.harbor.resolution import resolve_harbor_source
from nemo_evaluator.harbor.tasks import PinnedHarborTaskset
from nemo_evaluator_sdk.agent_eval.taskset_sources import TasksetSourceAdapter, TasksetSourceMaterialization
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.files.client import AsyncFilesClient


class EvaluatorTasksetSourceAdapter:
    schemes = frozenset({"nemo-evaluator-taskset"})

    def __init__(self, *, client: AsyncNeMoPlatform, workspace: str):
        self._sdk = client
        self._workspace = workspace

    async def materialize(self, source_uri: str, *, destination_root: Path) -> TasksetSourceMaterialization:
        if any(ord(char) <= 32 or ord(char) == 127 for char in source_uri):
            raise ValueError("Source URI must not contain whitespace or control characters")
        uri = urlsplit(source_uri)
        workspace = unquote(uri.netloc)
        name = unquote(uri.path)
        digest = unquote(uri.fragment)
        if (
            uri.scheme not in self.schemes
            or workspace != self._workspace
            or uri.query
            or not name.startswith("/")
            or name.count("/") != 1
            or name[1:] in {"", ".", ".."}
            or "?" in source_uri
        ):
            raise ValueError("Expected a pinned nemo-evaluator-taskset URI in the configured workspace")
        source = PinnedHarborTaskset(taskset_ref=TasksetRef(f"{workspace}/{name[1:]}#{digest}"))
        canonical_uri = f"nemo-evaluator-taskset://{workspace}/{name[1:]}#{digest}"
        members = await resolve_harbor_source(
            source, entity_client=EntityClient(client_from_platform(self._sdk, AsyncEntitiesClient))
        )

        from nemo_evaluator.harbor.materialization import materialize_harbor_tasks

        materialized = await materialize_harbor_tasks(
            members, files_client=client_from_platform(self._sdk, AsyncFilesClient), destination_root=destination_root
        )
        root = materialized.dataset_root.resolve(strict=True)
        if not root.is_relative_to(destination_root.resolve(strict=True)):
            raise ValueError("Materialized dataset escaped destination_root")
        return TasksetSourceMaterialization(
            source_uri=canonical_uri,
            materialized_root=root,
            revision_digest=digest,
            member_digests={member.task_id: member.source.revision_digest for member in materialized.members},
            materialization_digest=materialized.materialization_digest,
        )


def factory(*, client: AsyncNeMoPlatform, workspace: str) -> TasksetSourceAdapter:
    return EvaluatorTasksetSourceAdapter(client=client, workspace=workspace)
