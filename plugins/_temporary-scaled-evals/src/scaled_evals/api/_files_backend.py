# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Files-service-backed artifact storage for scaled-evals.

Replaces raw object storage (S3/GCS) with NeMo Helix **filesets**. The public
``scaled_evals.api.s3`` module is a thin facade over this backend; every one of its
existing function signatures is preserved so callers and the test suite are unchanged.

Object keys map onto ``(fileset, path)`` as follows (see ``split_key``):

* ``evaluations/{id}/...``            -> fileset ``se-eval-{id}``,            path is the remainder
* ``{task_id}/rev/{n}/...``          -> fileset ``se-task-{task_id}``,        path is ``rev/{n}/...``
* ``switchyard-contexts/...``        -> fileset ``se-build-context``,        path is the remainder
* anything else (benchmark archives) -> fileset ``se-shared``,               path is the full key

Filesets all live in one service workspace (``settings.files_workspace``); scaled-evals keeps
owning its own ``owner_id`` tenancy and the task-pack size/quota guardrails on top of Files
file metadata. This is the "single service workspace" default; a future entity-owned-workspace
model resolves the workspace per-request in the ``_workspace`` seam instead.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scaled_evals.api.settings import settings

if TYPE_CHECKING:
    from filesets.resources import FilesResource


# ---------------------------------------------------------------------------
# Key -> (fileset, path) mapping
# ---------------------------------------------------------------------------

_EVAL_KEY = re.compile(r"^evaluations/(?P<eval_id>[^/]+)/(?P<path>.+)$")
_TASK_KEY = re.compile(r"^(?P<task_id>[^/]+)/(?P<path>rev/.+)$")
_SWITCHYARD_KEY = re.compile(r"^switchyard-contexts/(?P<path>.+)$")

_SHARED_FILESET = "se-shared"

# Files entity names cap at 63 chars total. Every id component is prefixed with
# ``se-<kind>-`` (longest is ``se-build-`` at 9 chars, but ``se-task-``/``se-eval-``
# at 8 are the ones fed a sanitized id), so bound the id component so the full name
# can never exceed the limit. 54 leaves headroom for the 8-char prefix + a 63 cap.
_MAX_FILESET_ID_LEN = 54


@dataclass(frozen=True, slots=True)
class FilesetRef:
    """A file addressed within a fileset."""

    fileset: str
    path: str


def _sanitize_fileset_id(raw: str) -> str:
    r"""Turn an arbitrary id into a valid fileset-name component.

    Files entity names must match ``^[a-z](?!.*--)[a-z0-9\-@.+_]{1,62}(?<!-)$`` — lowercase
    start, no doubled dashes, no trailing dash. scaled-evals ids can carry uppercase and other
    characters, so normalize deterministically: lowercase, replace runs of disallowed chars with
    a single dash, collapse doubled dashes, and trim leading/trailing dashes. Deterministic so a
    given evaluation/task always maps to the same fileset. The full name is ``se-<kind>-<id>``;
    the ``se-<kind>-`` prefix guarantees the required leading lowercase letter.

    NOTE: real scaled-evals ids are ``{prefix}_{uuid4().hex}`` (all lowercase), so lowercasing is
    a no-op in practice and cannot collide; it only defends against a future id scheme.
    """
    lowered = raw.lower()
    collapsed = re.sub(r"[^a-z0-9@.+_]+", "-", lowered)
    collapsed = re.sub(r"-{2,}", "-", collapsed).strip("-")
    collapsed = collapsed or "unknown"
    if len(collapsed) > _MAX_FILESET_ID_LEN:
        # Clamp long ids so ``se-<kind>-<id>`` stays within the 63-char entity limit,
        # appending a short deterministic digest so two long ids that share a prefix
        # can't collide onto the same fileset. Re-trim any trailing dash the cut left.
        digest = hashlib.sha1(collapsed.encode()).hexdigest()[:8]
        head = collapsed[: _MAX_FILESET_ID_LEN - len(digest) - 1].rstrip("-")
        collapsed = f"{head}-{digest}"
    return collapsed


def _reject_reserved_chars(key: str) -> None:
    """Reject keys the Files SDK would mis-parse.

    The Files path parser treats ``#`` as the ``fileset#path`` reference delimiter, and a
    leading ``fileset://`` as a full workspace/fileset reference, even when fileset/workspace
    are passed as explicit kwargs. A key carrying either would be silently re-routed to a
    different fileset/path (or another workspace entirely), so reject both up front rather
    than let an artifact land somewhere unexpected.
    """
    if "#" in key:
        raise ValueError(f"object key may not contain '#': {key!r}")
    if key.startswith("fileset://"):
        raise ValueError(f"object key may not start with 'fileset://': {key!r}")


def split_key(object_key: str) -> FilesetRef:
    """Map a legacy flat object key onto a ``(fileset, path)`` pair.

    Pure and total: any key resolves to some fileset/path so no caller can address
    storage the mapping doesn't cover. Keys containing ``#`` or beginning with
    ``fileset://`` are rejected (the Files SDK would treat them as reference syntax).
    """
    key = object_key.lstrip("/")
    _reject_reserved_chars(key)
    if match := _EVAL_KEY.match(key):
        return FilesetRef(f"se-eval-{_sanitize_fileset_id(match['eval_id'])}", match["path"])
    if match := _TASK_KEY.match(key):
        return FilesetRef(f"se-task-{_sanitize_fileset_id(match['task_id'])}", match["path"])
    if match := _SWITCHYARD_KEY.match(key):
        return FilesetRef("se-build-context", match["path"])
    return FilesetRef(_SHARED_FILESET, key)


def fileset_prefix(object_prefix: str) -> tuple[str, str]:
    """Map a legacy key *prefix* onto ``(fileset, path_prefix)`` for listing/deletion.

    A prefix like ``evaluations/{id}/artifacts/`` resolves to that eval's fileset and the
    remaining path prefix. A bare ``evaluations/{id}/`` resolves to the fileset root ("").
    """
    key = object_prefix.lstrip("/")
    _reject_reserved_chars(key)
    if match := re.match(r"^evaluations/(?P<eval_id>[^/]+)/(?P<path>.*)$", key):
        return f"se-eval-{_sanitize_fileset_id(match['eval_id'])}", match["path"]
    if match := re.match(r"^(?P<task_id>[^/]+)/(?P<path>rev/.*)$", key):
        return f"se-task-{_sanitize_fileset_id(match['task_id'])}", match["path"]
    if match := re.match(r"^switchyard-contexts/(?P<path>.*)$", key):
        return "se-build-context", match["path"]
    return _SHARED_FILESET, key


# ---------------------------------------------------------------------------
# SDK / FilesResource access
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _files_resource() -> FilesResource:
    """Return the process-wide FilesResource, built from a service-authed platform SDK.

    Authenticates as ``service:scaled-evals`` with the internal marker; when scaled-evals
    runs inside the platform the SDK provider routes URLs and reuses shared clients.
    """
    from nemo_helix_plugin.sdk_provider import get_platform_sdk

    sdk = get_platform_sdk(as_service="scaled-evals", internal=True)
    return _files_resource_for(sdk)


def _files_resource_for(sdk: Any) -> FilesResource:
    """Build (and attach) a FilesResource on an arbitrary SDK handle.

    Extracted so tests can inject an in-memory-Files-backed SDK.
    """
    from filesets.resources import FilesResource
    from nemo_helix_plugin.client.adapter import client_from_platform
    from nemo_helix_plugin.files.client import FilesClient

    existing = sdk.__dict__.get("files")
    if isinstance(existing, FilesResource):
        return existing
    resource = FilesResource(sdk, files_client=client_from_platform(sdk, FilesClient))
    sdk.__dict__["files"] = resource
    return resource


def set_files_resource(resource: FilesResource | None) -> None:
    """Override the cached FilesResource (tests). Pass ``None`` to reset."""
    _files_resource.cache_clear()
    if resource is not None:
        _OVERRIDE["resource"] = resource
    else:
        _OVERRIDE.pop("resource", None)


def set_workspace_override(workspace: str | None) -> None:
    """Override the fileset workspace (tests). Pass ``None`` to reset to settings."""
    if workspace is not None:
        _OVERRIDE_WS["workspace"] = workspace
    else:
        _OVERRIDE_WS.pop("workspace", None)


_OVERRIDE: dict[str, FilesResource] = {}
_OVERRIDE_WS: dict[str, str] = {}


def _files() -> FilesResource:
    if "resource" in _OVERRIDE:
        return _OVERRIDE["resource"]
    return _files_resource()


def _workspace(object_key: str | None = None) -> str:  # noqa: ARG001 - key reserved for the entity-owned-workspace seam
    """Resolve the platform workspace that owns a given artifact's fileset.

    This is the SINGLE indirection point for fileset tenancy, and deliberately so.

    Today every fileset lives in one service workspace (``settings.files_workspace``), and
    scaled-evals keeps owning its own ``owner_id`` tenancy + upload quotas on top of Files.
    The end state is different: the scaled-evals evaluation/task becomes a first-class
    platform *entity* owned by a workspace, and this resolver returns that entity's workspace
    so Files-native RBAC isolates tenants. When that lands, ONLY this function changes — it
    will map ``object_key`` (which carries the evaluation/task id) to the owning entity's
    workspace. The key mapping (``split_key``) and every caller stay untouched.

    ``object_key`` is accepted now (unused) so that future per-entity resolution is a
    body-only change with no signature churn at the call sites.
    """
    return _OVERRIDE_WS.get("workspace") or settings.files_workspace


@lru_cache(maxsize=1)
def _not_found_errors() -> tuple[type[BaseException], ...]:
    """Exception types that mean 'no such fileset or file'.

    The Files SDK raises ``nemo_helix_plugin.client.errors.NotFoundError`` (HTTP 404) for a
    missing fileset AND a missing file; ``FileNotFoundError`` is included defensively for the
    fsspec path. Resolved lazily so importing this module never requires the SDK to be present.
    """
    try:
        from nemo_helix_plugin.client.errors import NotFoundError

        return (NotFoundError, FileNotFoundError)
    except ImportError:  # pragma: no cover - SDK always present where the backend runs
        return (FileNotFoundError,)


# ---------------------------------------------------------------------------
# Primitive operations (transport). Higher-level redaction/manifest/archive
# logic stays in scaled_evals.api.s3 and calls these.
# ---------------------------------------------------------------------------


class MissingObjectError(RuntimeError):
    """Raised when an addressed fileset file does not exist."""

    def __init__(self, ref: FilesetRef) -> None:
        super().__init__(f"no such file: {ref.fileset}#{ref.path}")
        self.ref = ref


def put_bytes(object_key: str, body: bytes, *, content_type: str | None = None) -> None:
    ref = split_key(object_key)
    _files().upload_content(
        content=body,
        remote_path=ref.path,
        fileset=ref.fileset,
        workspace=_workspace(),
        fileset_auto_create=True,
    )


def get_bytes(object_key: str) -> bytes:
    ref = split_key(object_key)
    try:
        return _files().download_content(
            remote_path=ref.path,
            fileset=ref.fileset,
            workspace=_workspace(),
        )
    except _not_found_errors() as exc:
        raise MissingObjectError(ref) from exc


def object_size(object_key: str) -> int | None:
    ref = split_key(object_key)
    for item in _list(ref.fileset, ref.path):
        if item[0] == ref.path:
            return item[1]
    return None


def object_exists(object_key: str) -> bool:
    return object_size(object_key) is not None


def delete_object(object_key: str) -> None:
    ref = split_key(object_key)
    try:
        _files().delete(remote_path=ref.path, fileset=ref.fileset, workspace=_workspace())
    except _not_found_errors():
        # Deleting a missing object is a no-op, matching S3 DeleteObject semantics.
        return


def upload_file(path: Path, object_key: str, *, content_type: str | None = None) -> int:  # noqa: ARG001 - content_type kept for signature parity; Files infers type
    """Stream a local file into a fileset without buffering it in memory.

    Task packs can be up to 20 GiB, so this uses the fsspec filesystem's chunked
    ``put_file`` (which streams the local file) rather than reading the whole file
    into a bytes object. Ensures the destination fileset exists first, preserving the
    auto-create behavior the old object-store upload had.
    """
    from filesets.filesystem.filesystem import build_fileset_ref

    ref = split_key(object_key)
    resource = _files()
    resource._ensure_fileset_exists(_workspace(), ref.fileset)  # noqa: SLF001 - idempotent create; no public wrapper
    fileset_ref = build_fileset_ref(ref.path, workspace=_workspace(), fileset=ref.fileset)
    resource.fsspec.put_file(str(path), fileset_ref)
    return path.stat().st_size


def download_object(object_key: str, dest_path: str) -> None:
    """Stream a fileset object to a local file without buffering it in memory.

    Build workers fetch task packs (up to 20 GiB) through this, so it uses the fsspec
    filesystem's chunked ``get_file`` (which writes to ``dest_path`` as it streams)
    rather than materializing the whole object via ``get_bytes``.
    """
    from filesets.filesystem.filesystem import build_fileset_ref

    ref = split_key(object_key)
    fileset_ref = build_fileset_ref(ref.path, workspace=_workspace(), fileset=ref.fileset)
    try:
        _files().fsspec.get_file(fileset_ref, dest_path)
    except _not_found_errors() as exc:
        raise MissingObjectError(ref) from exc


def stream_object(object_key: str) -> Iterator[bytes]:
    # Bounded, streamed read via the fsspec filesystem so a large archive never
    # fully materializes in a request thread beyond one chunk at a time.
    ref = split_key(object_key)
    fs = _files().fsspec
    from filesets.filesystem.filesystem import build_fileset_ref

    fileset_ref = build_fileset_ref(ref.path, workspace=_workspace(), fileset=ref.fileset)
    try:
        with fs.open(fileset_ref, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    except _not_found_errors() as exc:
        raise MissingObjectError(ref) from exc


def _list(fileset: str, path_prefix: str) -> list[tuple[str, int]]:
    """List ``(path, size)`` within a fileset under a path prefix. Missing fileset -> []."""
    try:
        response = _files().list(fileset=fileset, workspace=_workspace())
    except _not_found_errors():
        return []
    items: list[tuple[str, int]] = []
    for entry in response.data:
        entry_path = entry.path
        if path_prefix and not entry_path.startswith(path_prefix):
            continue
        items.append((entry_path, _coerce_size(entry.size)))
    return items


def _coerce_size(size: object) -> int:
    """Best-effort byte-size coercion; a malformed/absent size reports 0, never raises."""
    if isinstance(size, int):
        return size
    if isinstance(size, (str, float)):
        try:
            return int(size)
        except (TypeError, ValueError):
            return 0
    return 0


def list_objects(prefix: str) -> list[dict[str, Any]]:
    """List objects below a legacy key prefix, reconstructing full keys.

    Returns the same shape as the legacy S3 lister: ``{"key", "size_bytes", "updated_at"}``.
    """
    fileset, path_prefix = fileset_prefix(prefix)
    # Reconstruct the legacy key head (everything before the in-fileset path).
    normalized = prefix.lstrip("/")
    head = normalized[: len(normalized) - len(path_prefix)] if path_prefix else normalized
    results: list[dict[str, Any]] = []
    for entry_path, size in _list(fileset, path_prefix):
        results.append(
            {
                "key": f"{head}{entry_path}" if head else entry_path,
                "size_bytes": size,
                "updated_at": None,
            }
        )
    return results


def readiness_probe() -> None:
    """Confirm Files is reachable by listing the shared fileset's workspace.

    Raises on failure so ``/v1/readyz`` can report ``object_store`` honestly. A missing
    shared fileset is fine (nothing has been written yet) — only transport failures raise.
    """
    try:
        _files().list(fileset=_SHARED_FILESET, workspace=_workspace())
    except _not_found_errors():
        return
