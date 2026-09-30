# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build fetch`` -- step 1. Trusted. Holds a Files client and nothing else.

**This step is the whole control against a caller naming another tenant's fileset.** It resolves
every source **as the submitting principal**, so a request for a fileset the submitter cannot read
fails here, at the API, rather than succeeding and mounting the stolen data faithfully.

``subPath`` does not help with that attack and it is worth being explicit about why: if the fetch
succeeds, mounting "the right subPath" mounts exactly the data that should never have been
downloaded. The mount is a partition between *this job's* groups; it is not an authorization
boundary against other tenants. This is.

It holds no registry credential and no pod RBAC, and it runs no caller-authored code -- it copies
bytes onto a volume.
"""

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
    """Resolve a fileset-supplied path under ``root``, refusing anything that escapes.

    The path comes from a fileset listing, which a tenant controls. A `..` or absolute component
    would otherwise write outside this job's slice of a volume shared by every build.
    """
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
    # `.data()` is a method on the response wrapper, not an attribute -- iterating the wrapper
    # directly iterates a bound method and silently yields nothing useful.
    listing = client.list_files(workspace=workspace, name=name, query_params=query).data()

    count = 0
    for entry in listing.data:
        target = _safe_destination(destination, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # `.read()`, not `bytes(...)`: download_file returns a streaming response object.
        response = client.download_file(workspace=workspace, name=name, path=entry.path)
        target.write_bytes(response.read())
        count += 1
    return count


def fetch_source(client: FilesClient, layout: WorkLayout, source: ContextSource, *, workspace: str) -> int:
    """Download one source to ``layout.context(source)``, and return how many files it has.

    Files reports entry paths relative to the fileset root even when the listing is narrowed to
    ``context_path``, so entries are written under the *fileset's* directory -- and a subtree
    then lands at ``layout.context(source)`` because that is where the layout puts it, inside its
    fileset. Writing them under the context directory instead would nest the subtree inside
    itself (``tests/tests/Dockerfile``).
    """
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

    The slice is keyed by the job's name, and a job deleted and submitted again under the same
    name gets the same slice. A context would then build from files since removed from its
    fileset, and a layout from the earlier run would be published if this run's build of it
    fails. This step runs first, so it is the one that starts the slice clean.
    """
    for directory in (layout.root / "context", layout.outputs):
        path = Path(directory)
        if path.exists():
            logger.info("removing %s left by an earlier run of this job", sanitize_for_log(path))
            shutil.rmtree(path)


def main() -> int:
    config = FetchStepConfig.model_validate(read_step_config())
    workspace, _ = job_identity()

    # As the SUBMITTER. `get_task_nemo_client` forwards the submitting principal -- via
    # on-behalf-of today, via workload-identity token exchange where that is enabled. It is
    # deliberately not a service identity: a service identity would read any fileset in the
    # deployment, which is the confused deputy this step exists to remove.
    client = client_from_platform(get_task_nemo_client("builder"), FilesClient)
    layout = WorkLayout(PurePosixPath(work_mount()))
    clear_earlier_attempts(layout)

    for source in config.sources:
        fetch_source(client, layout, source, workspace=workspace)

    return 0
