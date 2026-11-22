# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Files-service storage for the canonical serialized MetricBundle.

The evals service owns the lifecycle of a stored metric's payload. Each
upload lands in its own uniquely-named fileset, so an upload never overwrites an
existing metric's bytes and a failed/abandoned upload can always be cleaned up
by deleting exactly the fileset it created. The
:class:`~nemo_evals.entities.MetricBundleEntity` holds the resulting read
reference and Fileset ID; the bytes stay in Files.
"""

from __future__ import annotations

import logging
import uuid

import nemo_evals.shared.metric_bundles.cloudpickle  # noqa: F401
import nemo_evals.shared.metric_bundles.inline  # noqa: F401

# Importing the payload modules registers their bundle payload kinds in the
# bundle registry so MetricBundle payloads round-trip through validation.
from nemo_evals._fileset_receipt import FilesetReceipt
from nemo_evals.shared.metric_bundles.bundles import MetricBundle
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from pydantic import ValidationError

#: Filename of the serialized bundle stored within each metric's fileset.
BUNDLE_FILENAME = "bundle.json"

#: Prefix for the per-upload filesets backing stored metrics.
FILESET_PREFIX = "metric-bundle"

logger = logging.getLogger(__name__)


class MetricBundleStorageError(RuntimeError):
    """Raised when a stored metric bundle cannot be written, read, or verified."""


def _new_fileset_name() -> str:
    """Return a unique fileset name for a single metric-bundle upload.

    Intentionally independent of the metric name: the uuid alone guarantees
    uniqueness and keeps the name well within the fileset name length limit
    regardless of how long the (up to 255-char) metric name is. The
    metric/workspace association is recorded on the entity's ``bundle_ref`` and
    in the fileset description.
    """
    return f"{FILESET_PREFIX}.{uuid.uuid4().hex}"


def parse_bundle_ref(bundle_ref: str) -> tuple[str, str, str]:
    """Split a ``workspace/fileset#path`` reference into its parts."""
    if "#" not in bundle_ref:
        raise MetricBundleStorageError(f"invalid metric bundle reference (missing fragment): {bundle_ref!r}")
    location, path = bundle_ref.split("#", 1)
    if "/" not in location:
        raise MetricBundleStorageError(f"invalid metric bundle reference (missing workspace): {bundle_ref!r}")
    workspace, fileset = location.split("/", 1)
    if not workspace or not fileset or not path:
        raise MetricBundleStorageError(f"invalid metric bundle reference: {bundle_ref!r}")
    return workspace, fileset, path


async def store_bundle(client: AsyncFilesClient, workspace: str, name: str, bundle: MetricBundle) -> FilesetReceipt:
    """Serialize and upload a metric bundle, returning its read reference and allocation ID.

    Each call creates a new, uniquely-named fileset, so callers can safely roll
    back by deleting the captured allocation ID.
    """
    fileset = _new_fileset_name()
    body = bundle.model_dump_json().encode("utf-8")
    try:
        description = f"Stored metric bundle for {workspace}/{name}."
        allocated = (
            await client.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(name=fileset, description=description[:255]),
            )
        ).data()
    except Exception as exc:
        raise MetricBundleStorageError(f"failed to create fileset for metric bundle {workspace}/{name}") from exc
    try:
        await client.upload_file(
            content=body,
            path=BUNDLE_FILENAME,
            name=fileset,
            workspace=workspace,
        )
    except Exception as exc:
        # Roll back the just-created (now-empty) fileset so a failed upload
        # doesn't leak it, then surface a typed storage error.
        try:
            await delete_bundle_by_id(client, workspace=workspace, fileset_id=allocated.id)
        except Exception:
            logger.warning(
                "Failed to clean up fileset after a failed metric bundle upload; storage may be leaked",
                extra={"fileset": fileset},
                exc_info=True,
            )
        raise MetricBundleStorageError(f"failed to upload metric bundle to {workspace}/{fileset}") from exc
    # Construct the reference ourselves so storage/retrieval stay self-consistent
    # regardless of the service's file_ref string format.
    return FilesetReceipt(ref=f"{workspace}/{fileset}#{BUNDLE_FILENAME}", fileset_id=allocated.id)


async def load_bundle(client: AsyncFilesClient, bundle_ref: str, *, expected_digest: str | None = None) -> MetricBundle:
    """Download and reconstruct a stored metric bundle from its Files reference.

    When ``expected_digest`` is provided, the reconstructed payload digest is
    verified against it to detect drift or corruption.
    """
    workspace, fileset, path = parse_bundle_ref(bundle_ref)
    try:
        response = await client.download_file(path=path, name=fileset, workspace=workspace)
        data = await response.read()
    except Exception as exc:
        raise MetricBundleStorageError(f"failed to download metric bundle from {bundle_ref!r}") from exc
    try:
        bundle = MetricBundle.model_validate_json(data)
    except (ValidationError, ValueError) as exc:
        # ValueError also covers MetricBundlingError (unsupported/invalid payload kind).
        raise MetricBundleStorageError(f"stored metric bundle at {bundle_ref!r} is corrupt or unreadable") from exc
    if expected_digest is not None and bundle.payload.digest != expected_digest:
        raise MetricBundleStorageError(
            f"metric bundle digest mismatch for {bundle_ref!r}: "
            f"expected {expected_digest}, found {bundle.payload.digest}"
        )
    return bundle


async def delete_bundle_by_id(client: AsyncFilesClient, *, workspace: str, fileset_id: str) -> None:
    """Delete the metric bundle allocation captured when Files created it."""
    await client.delete_fileset_by_id(fileset_id=fileset_id, workspace=workspace)
