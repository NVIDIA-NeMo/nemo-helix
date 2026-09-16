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
from typing import Any

from filesets.resources import FilesResource

from scaled_evals.api.settings import settings

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


@dataclass(frozen=True, slots=True)
class UploadTarget:
    """Where a client uploads a file directly to Files (the broker-upload model).

    Structural analog of the old presigned-URL block: scaled-evals hands back the fileset
    coordinates + the SDK-facing remote path, and the client uploads out-of-band via
    ``sdk.files.upload(local_path=..., remote_path=remote_path)``. Bytes never transit the
    scaled-evals control plane.
    """

    workspace: str
    fileset: str
    path: str

    @property
    def remote_path(self) -> str:
        """The ``<workspace>/<fileset>#<path>`` ref the Files SDK accepts directly."""
        return f"{self.workspace}/{self.fileset}#{self.path}"


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

    The Files path parser treats ``#`` as the ``fileset#path`` reference delimiter, and
    ``fileset://`` as a full workspace/fileset reference, even when fileset/workspace are
    passed as explicit kwargs. Either token ANYWHERE in the key is dangerous: ``split_key``
    maps e.g. ``evaluations/{id}/fileset://other-ws/x`` to a path that itself begins with
    ``fileset://``, which the SDK would then re-route to a different workspace/fileset. Reject
    both tokens outright (not just a leading ``fileset://``) rather than let an artifact land
    somewhere unexpected.
    """
    if "#" in key:
        raise ValueError(f"object key may not contain '#': {key!r}")
    if "fileset://" in key:
        raise ValueError(f"object key may not contain 'fileset://': {key!r}")


def split_key(object_key: str) -> FilesetRef:
    """Map a legacy flat object key onto a ``(fileset, path)`` pair.

    Pure and total: any key resolves to some fileset/path so no caller can address
    storage the mapping doesn't cover. Keys containing ``#`` or ``fileset://`` anywhere are
    rejected (the Files SDK would treat them as reference syntax).
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


def upload_target(object_key: str, *, ensure_fileset: bool = True) -> UploadTarget:
    return default_backend().upload_target(object_key, ensure_fileset=ensure_fileset)


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
#
# The transport is an injectable object that holds its SDK handle + derived
# FilesResource on the instance (the house pattern — see nhx_common
# jobs.log_client, nemo-insights controller, core/models entity services).
# A process-wide default instance is lazily built via inject-or-default;
# tests construct their own FilesArtifactBackend(resource=...) and install it
# with use_backend(), so there is no module-global mutable state to leak.
# ---------------------------------------------------------------------------


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


class MissingObjectError(RuntimeError):
    """Raised when an addressed fileset file does not exist."""

    def __init__(self, ref: FilesetRef) -> None:
        super().__init__(f"no such file: {ref.fileset}#{ref.path}")
        self.ref = ref


def _resource_for(sdk: Any) -> FilesResource:
    """Build (and attach) a FilesResource on an SDK handle, reusing an attached one."""
    from nemo_helix_plugin.client.adapter import client_from_platform
    from nemo_helix_plugin.files.client import FilesClient

    existing = sdk.__dict__.get("files")
    if isinstance(existing, FilesResource):
        return existing
    resource = FilesResource(sdk, files_client=client_from_platform(sdk, FilesClient))
    sdk.__dict__["files"] = resource
    return resource


class FilesArtifactBackend:
    """Files-service transport for scaled-evals artifacts, bound to one SDK client.

    State (the SDK's FilesResource + the owning workspace) lives on the instance, derived
    from the platform config — not in module globals. Build with an explicit ``resource``
    (tests inject an in-memory-Files-backed one) or let it lazily service-auth as
    ``service:scaled-evals`` on first use. ``workspace`` overrides the configured default
    (``settings.files_workspace``); tests pin it to the in-memory service's workspace.
    """

    def __init__(self, *, resource: FilesResource | None = None, workspace: str | None = None) -> None:
        self._resource = resource
        self._workspace_name = workspace

    def _files(self) -> FilesResource:
        if self._resource is None:
            from nemo_helix_plugin.sdk_provider import get_platform_sdk

            sdk = get_platform_sdk(as_service="scaled-evals", internal=True)
            self._resource = _resource_for(sdk)
        return self._resource

    def workspace(self, object_key: str | None = None) -> str:  # noqa: ARG002 - key reserved for the entity-owned-workspace seam
        """Resolve the platform workspace that owns a given artifact's fileset.

        This is the SINGLE indirection point for fileset tenancy, and deliberately so.

        Today every fileset lives in one service workspace (``settings.files_workspace``), and
        scaled-evals keeps owning its own ``owner_id`` tenancy + upload quotas on top of Files.
        The end state is different: the scaled-evals evaluation/task becomes a first-class
        platform *entity* owned by a workspace, and this resolver returns that entity's workspace
        so Files-native RBAC isolates tenants. When that lands, ONLY this method changes — it
        will map ``object_key`` (which carries the evaluation/task id) to the owning entity's
        workspace. The key mapping (``split_key``) and every caller stay untouched.

        ``object_key`` is accepted now (unused) so that future per-entity resolution is a
        body-only change with no signature churn at the call sites.
        """
        return self._workspace_name or settings.files_workspace

    def upload_target(self, object_key: str, *, ensure_fileset: bool = True) -> UploadTarget:
        """Resolve the direct-to-Files upload coordinates for a key (broker-upload model).

        Returns the ``(workspace, fileset, path)`` the client uploads to via the Files SDK. When
        ``ensure_fileset`` (default), the fileset is created up front so the client's out-of-band
        upload cannot 404 on a missing fileset; pass ``False`` to skip the round-trip when the
        fileset is known to exist.
        """
        ref = split_key(object_key)
        workspace = self.workspace(object_key)
        if ensure_fileset:
            self._ensure_fileset(ref.fileset, workspace)
        return UploadTarget(workspace=workspace, fileset=ref.fileset, path=ref.path)

    def _ensure_fileset(self, fileset: str, workspace: str) -> None:
        """Idempotently create a fileset so a subsequent direct upload has a target."""
        from nemo_helix_plugin.files.types import CreateFilesetRequest

        self._files().client.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(name=fileset),
            exist_ok=True,
        )

    def put_bytes(self, object_key: str, body: bytes, *, content_type: str | None = None) -> None:  # noqa: ARG002 - content_type kept for signature parity; Files infers type
        ref = split_key(object_key)
        self._files().upload_content(
            content=body,
            remote_path=ref.path,
            fileset=ref.fileset,
            workspace=self.workspace(),
            fileset_auto_create=True,
        )

    def get_bytes(self, object_key: str) -> bytes:
        ref = split_key(object_key)
        try:
            return self._files().download_content(
                remote_path=ref.path,
                fileset=ref.fileset,
                workspace=self.workspace(),
            )
        except _not_found_errors() as exc:
            raise MissingObjectError(ref) from exc

    def object_size(self, object_key: str) -> int | None:
        ref = split_key(object_key)
        for item in self._list(ref.fileset, ref.path):
            if item[0] == ref.path:
                return item[1]
        return None

    def object_exists(self, object_key: str) -> bool:
        return self.object_size(object_key) is not None

    def delete_object(self, object_key: str) -> None:
        ref = split_key(object_key)
        try:
            self._files().delete(remote_path=ref.path, fileset=ref.fileset, workspace=self.workspace())
        except _not_found_errors():
            # Deleting a missing object is a no-op, matching S3 DeleteObject semantics.
            return

    def upload_file(self, path: Path, object_key: str, *, content_type: str | None = None) -> int:  # noqa: ARG002 - content_type kept for signature parity; Files infers type
        """Stream a local file into a fileset without buffering it in memory.

        Task packs can be up to 20 GiB, so this uses the fsspec filesystem's chunked
        ``put_file`` (which streams the local file) rather than reading the whole file
        into a bytes object. Ensures the destination fileset exists first, preserving the
        auto-create behavior the old object-store upload had.
        """
        from filesets.filesystem.filesystem import build_fileset_ref

        ref = split_key(object_key)
        resource = self._files()
        resource._ensure_fileset_exists(self.workspace(), ref.fileset)  # noqa: SLF001 - idempotent create; no public wrapper
        fileset_ref = build_fileset_ref(ref.path, workspace=self.workspace(), fileset=ref.fileset)
        resource.fsspec.put_file(str(path), fileset_ref)
        return path.stat().st_size

    def download_object(self, object_key: str, dest_path: str) -> None:
        """Stream a fileset object to a local file without buffering it in memory.

        Build workers fetch task packs (up to 20 GiB) through this, so it uses the fsspec
        filesystem's chunked ``get_file`` (which writes to ``dest_path`` as it streams)
        rather than materializing the whole object via ``get_bytes``.
        """
        from filesets.filesystem.filesystem import build_fileset_ref

        ref = split_key(object_key)
        fileset_ref = build_fileset_ref(ref.path, workspace=self.workspace(), fileset=ref.fileset)
        try:
            self._files().fsspec.get_file(fileset_ref, dest_path)
        except _not_found_errors() as exc:
            raise MissingObjectError(ref) from exc

    def stream_object(self, object_key: str) -> Iterator[bytes]:
        from filesets.filesystem.filesystem import build_fileset_ref

        # Bounded, streamed read via the fsspec filesystem so a large archive never
        # fully materializes in a request thread beyond one chunk at a time.
        ref = split_key(object_key)
        fileset_ref = build_fileset_ref(ref.path, workspace=self.workspace(), fileset=ref.fileset)
        try:
            with self._files().fsspec.open(fileset_ref, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        except _not_found_errors() as exc:
            raise MissingObjectError(ref) from exc

    def _list(self, fileset: str, path_prefix: str) -> list[tuple[str, int | None]]:
        """List ``(path, size)`` within a fileset under a path prefix. Missing fileset -> [].

        ``size`` is ``None`` when Files reports no usable size (preserved for the quota guard).
        """
        try:
            response = self._files().list(fileset=fileset, workspace=self.workspace())
        except _not_found_errors():
            return []
        items: list[tuple[str, int | None]] = []
        for entry in response.data:
            entry_path = entry.path
            if path_prefix and not entry_path.startswith(path_prefix):
                continue
            items.append((entry_path, _coerce_size(entry.size)))
        return items

    def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List objects below a legacy key prefix, reconstructing full keys.

        Returns the same shape as the legacy S3 lister: ``{"key", "size_bytes", "updated_at"}``.
        """
        fileset, path_prefix = fileset_prefix(prefix)
        # Reconstruct the legacy key head (everything before the in-fileset path).
        normalized = prefix.lstrip("/")
        head = normalized[: len(normalized) - len(path_prefix)] if path_prefix else normalized
        results: list[dict[str, Any]] = []
        for entry_path, size in self._list(fileset, path_prefix):
            results.append(
                {
                    "key": f"{head}{entry_path}" if head else entry_path,
                    "size_bytes": size,
                    "updated_at": None,
                }
            )
        return results

    def readiness_probe(self) -> None:
        """Confirm Files is reachable by listing the shared fileset's workspace.

        Raises on failure so ``/v1/readyz`` can report ``object_store`` honestly. A missing
        shared fileset is fine (nothing has been written yet) — only transport failures raise.
        """
        try:
            self._files().list(fileset=_SHARED_FILESET, workspace=self.workspace())
        except _not_found_errors():
            return


def _coerce_size(size: object) -> int | None:
    """Best-effort byte-size coercion; an absent/malformed size returns ``None``, never raises.

    ``None`` signals an UNVERIFIABLE size (not a zero-byte file) so the finalize quota guard
    can reject it as unsafe, matching the old object-store contract — coercing to 0 would let
    a pack whose size we cannot confirm slip past the size/quota check.
    """
    if isinstance(size, int):
        return size
    if isinstance(size, (str, float)):
        try:
            return int(size)
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Module-level default instance + function shims.
#
# The public callers (scaled_evals.api.artifacts) and the test suite use these
# module-level functions; each delegates to a single lazily-built default
# backend. Tests swap the default with use_backend() instead of mutating any
# free-floating global dict.
# ---------------------------------------------------------------------------

_default_backend: FilesArtifactBackend | None = None


def default_backend() -> FilesArtifactBackend:
    """Return the process-wide backend, building a service-authed one on first use."""
    global _default_backend
    if _default_backend is None:
        _default_backend = FilesArtifactBackend()
    return _default_backend


def use_backend(backend: FilesArtifactBackend | None) -> None:
    """Install a backend as the module default (tests). Pass ``None`` to reset."""
    global _default_backend
    _default_backend = backend


def put_bytes(object_key: str, body: bytes, *, content_type: str | None = None) -> None:
    default_backend().put_bytes(object_key, body, content_type=content_type)


def get_bytes(object_key: str) -> bytes:
    return default_backend().get_bytes(object_key)


def object_size(object_key: str) -> int | None:
    return default_backend().object_size(object_key)


def object_exists(object_key: str) -> bool:
    return default_backend().object_exists(object_key)


def delete_object(object_key: str) -> None:
    default_backend().delete_object(object_key)


def upload_file(path: Path, object_key: str, *, content_type: str | None = None) -> int:
    return default_backend().upload_file(path, object_key, content_type=content_type)


def download_object(object_key: str, dest_path: str) -> None:
    default_backend().download_object(object_key, dest_path)


def stream_object(object_key: str) -> Iterator[bytes]:
    return default_backend().stream_object(object_key)


def list_objects(prefix: str) -> list[dict[str, Any]]:
    return default_backend().list_objects(prefix)


def readiness_probe() -> None:
    default_backend().readiness_probe()
