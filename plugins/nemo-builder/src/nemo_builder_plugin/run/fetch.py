# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nmp-build fetch`` -- step 1. Trusted. Holds a Files client and nothing else.

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
from pathlib import Path

from nemo_builder_plugin.run.context import (
    context_hash,
    job_identity,
    read_step_config,
    split_fileset_ref,
)
from nemo_builder_plugin.steps import FetchStepConfig
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client_provider import get_task_nemo_client
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import ListFilesQueryParams

logger = logging.getLogger(__name__)

#: Written beside each fetched context. See the note at the bottom of this module.
CONTEXT_HASH_FILE = ".nmp-context-hash"


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


def main() -> int:
    config = FetchStepConfig.model_validate(read_step_config())
    workspace, _ = job_identity()

    # As the SUBMITTER. `get_task_nemo_client` forwards the submitting principal -- via
    # on-behalf-of today, via workload-identity token exchange where that is enabled. It is
    # deliberately not a service identity: a service identity would read any fileset in the
    # deployment, which is the confused deputy this step exists to remove.
    client = client_from_platform(get_task_nemo_client("builder"), FilesClient)

    dest_root = Path(config.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    for source in config.sources:
        source_workspace, name = split_fileset_ref(source.fileset, workspace)
        destination = dest_root / source.fileset
        if source.context_path:
            destination = destination / source.context_path
        destination.mkdir(parents=True, exist_ok=True)

        logger.info(
            "fetching fileset %s/%s%s -> %s",
            source_workspace,
            name,
            f" ({source.context_path})" if source.context_path else "",
            destination,
        )
        count = _download_fileset(
            client,
            workspace=source_workspace,
            name=name,
            context_path=source.context_path,
            destination=destination,
        )

        digest = context_hash(destination)
        (destination.parent / f"{destination.name}{CONTEXT_HASH_FILE}").write_text(digest)
        logger.info("fetched %d file(s), context hash %s", count, digest)

    return 0


# --- `source_digest` does not currently reach the image, and that is a real gap. ---
#
# RFC 001 wants the context hash carried as an image LABEL, so the reconciler can recover it from
# the config blob. That needs the hash to reach kaniko, and kaniko is launched by `supervise` --
# which deliberately mounts no volume at all, so it cannot read what this step just wrote.
#
# Three ways out, none free:
#   1. Mount ONLY the hash file into `supervise`. Narrow, but it weakens "the control plane for
#      the sandbox reads nothing" from a structural property to a reviewed exception.
#   2. Have the compiler pass it -- impossible, the context does not exist at compile time.
#   3. Let the SANDBOX compute and label it. Rejected: the sandbox runs caller-authored `RUN`,
#      so a hash it reports describes whatever it wants it to describe.
#
# For now the hash is computed and written beside the context, where it is available to anything
# that mounts the volume, and `ContainerImage.source_digest` stays None. The field is advisory
# and never identity, so an absent value costs provenance detail rather than correctness.
