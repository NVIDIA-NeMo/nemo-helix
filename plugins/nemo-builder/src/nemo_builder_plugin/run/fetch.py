# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build fetch``: the first step. It downloads each build context onto the work volume."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path, PurePosixPath

from nemo_builder_plugin.run.context import (
    job_identity,
    read_step_config,
    split_fileset_ref,
    work_mount,
)
from nemo_builder_plugin.steps import ContextSource, FetchStepConfig, WorkLayout
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import ListFilesQueryParams
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)


def _safe_destination(root: Path, relative_path: str) -> Path:
    """Join a tenant-controlled fileset path to ``root``, refusing one that escapes it."""
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"fileset path {relative_path!r} escapes the context directory")
    return candidate


def _download_fileset(
    client: FilesClient,
    *,
    workspace: str,
    name: str,
    context_path: str | None,
    destination: Path,
) -> int:
    query: ListFilesQueryParams | None = {"path": context_path} if context_path else None
    listing = client.list_files(workspace=workspace, name=name, query_params=query).data()

    count = 0
    for entry in listing.data:
        target = _safe_destination(destination, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        response = client.download_file(workspace=workspace, name=name, path=entry.path)
        target.write_bytes(response.read())
        count += 1
    return count


def fetch_source(client: FilesClient, layout: WorkLayout, source: ContextSource, *, workspace: str) -> int:
    source_workspace, name = split_fileset_ref(source.fileset, workspace)
    fileset_dir = Path(layout.fileset(source.fileset))
    context_dir = Path(layout.context(source))
    fileset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "fetching fileset %s/%s%s -> %s",
        sanitize_for_log(source_workspace),
        sanitize_for_log(name),
        sanitize_for_log(f" ({source.context_path})" if source.context_path else ""),
        sanitize_for_log(context_dir),
    )
    # Files reports paths from the fileset's root even when the listing is narrowed to the context
    # path, so they are written under the fileset's directory, not the context's.
    count = _download_fileset(
        client,
        workspace=source_workspace,
        name=name,
        context_path=source.context_path,
        destination=fileset_dir,
    )
    context_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fetched %d file(s)", count)
    return count


def clear_earlier_attempts(layout: WorkLayout) -> None:
    """Remove whatever an earlier run of this job left in its slice of the work volume.

    A job submitted again under the same name gets the same slice.
    """
    for directory in (layout.root / "context", layout.outputs):
        path = Path(directory)
        if path.exists():
            logger.info("removing %s left by an earlier run of this job", sanitize_for_log(path))
            shutil.rmtree(path)


def main() -> int:
    config = FetchStepConfig.model_validate(read_step_config())
    workspace, _ = job_identity()

    # As the submitter, never a service identity, which could read any fileset: this is the only
    # check that the submitter may read the filesets the request names.
    client = client_from_platform(get_task_nemo_client("builder"), FilesClient)
    layout = WorkLayout(PurePosixPath(work_mount()))
    clear_earlier_attempts(layout)

    for source in config.sources:
        fetch_source(client, layout, source, workspace=workspace)

    return 0
