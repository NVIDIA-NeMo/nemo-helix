# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Files API upload of converted environment + Gym dataset packages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.metadata import (
    EnvironmentMetadataContent,
    FilesetMetadata,
    environment_metadata_from_manifest,
)
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nhx.rl.tasks.environment.validate import load_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadedEnvironmentRefs:
    environment: str
    dataset: str


def _upload_tree(client: FilesClient, *, workspace: str, fileset: str, root: Path) -> int:
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        client.upload_file(
            workspace=workspace,
            name=fileset,
            path=rel,
            content=path.read_bytes(),
        )
        count += 1
    return count


def upload_converted_packages(
    *,
    environment_root: Path,
    dataset_dir: Path,
    workspace: str,
    environment_name: str,
    dataset_name: str,
    base_url: str,
    api_key: str | None = None,
) -> UploadedEnvironmentRefs:
    """Create environment + dataset FileSets and upload package trees.

    Requires a reachable Files service (``base_url``). Intended for CLI hosts
    with platform credentials — not for deny-default training clusters.
    """
    manifest = load_manifest(environment_root)
    env_meta = environment_metadata_from_manifest(manifest.model_dump(mode="python"))
    # NemoClient takes the bearer token as `auth`; passing `api_key` raises TypeError.
    client = FilesClient(base_url=base_url, workspace=workspace, auth=api_key)

    client.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=environment_name,
            purpose=FilesetPurpose.ENVIRONMENT,
            description=f"Environment package for {manifest.metadata.name}",
            metadata=FilesetMetadata(environment=env_meta),
        ),
        exist_ok=True,
    )
    n_env = _upload_tree(client, workspace=workspace, fileset=environment_name, root=environment_root)
    logger.info("Uploaded %d files to environment fileset %s/%s", n_env, workspace, environment_name)

    client.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=dataset_name,
            purpose=FilesetPurpose.DATASET,
            description=f"Gym JSONL dataset for {manifest.metadata.name}",
        ),
        exist_ok=True,
    )
    n_ds = _upload_tree(client, workspace=workspace, fileset=dataset_name, root=dataset_dir)
    logger.info("Uploaded %d files to dataset fileset %s/%s", n_ds, workspace, dataset_name)

    return UploadedEnvironmentRefs(
        environment=f"{workspace}/{environment_name}",
        dataset=f"{workspace}/{dataset_name}",
    )


# Re-export for type checkers / callers that construct metadata manually.
__all__ = [
    "EnvironmentMetadataContent",
    "UploadedEnvironmentRefs",
    "upload_converted_packages",
]
