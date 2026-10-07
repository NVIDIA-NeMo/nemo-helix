# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process GLiNER PII detector, served from Files pull-through weights.

The Anonymizer entity detector is GLiNER2 (``fastino/gliner2-privacy-filter-PII-multi``).
Upstream ``anonymizer`` ships a lazy, dependency-isolated GLiNER2 runtime
(``anonymizer.notebooks._runtime.create_anonymizer``) that downloads the pinned
checkpoint with ``huggingface_hub.snapshot_download`` and serves it on a loopback
OpenAI-compatible endpoint. We reuse that runtime unchanged and only control
*where the weights come from*: a system-owned HuggingFace-backed Fileset, so the
first fetch tees through the Files pull-through cache and later loads are local.

Rather than redirect upstream's hard-coded ``repo_id`` (there is no override
hook), we **prewarm the HuggingFace cache** for that repo id by downloading the
weights through Files first. Upstream's own ``snapshot_download`` then resolves
from the warmed cache with no further network. This keeps upstream untouched and
keeps the Fileset name stable and system-owned.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

# The upstream-pinned GLiNER2 checkpoint is imported from the library so a bump
# moves both the download target and the Fileset pin together.
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.notebooks.local_inference.gliner2 import MODEL_ID as GLINER_MODEL_ID
from anonymizer.notebooks.local_inference.gliner2 import MODEL_REVISION as GLINER_MODEL_REVISION
from data_designer.config.models import ModelProvider as DDModelProvider
from huggingface_hub import snapshot_download
from huggingface_hub.file_download import repo_folder_name
from nemo_anonymizer_plugin.app.errors import AnonymizerInternalError
from nemo_helix_plugin.client.adapter import AsyncHelixClient, SyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig, StorageConfigType
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose

logger = logging.getLogger(__name__)

#: System-owned Fileset that fronts the GLiNER weights for pull-through caching.
#: Static + ``nhx-``-prefixed in the ``default`` workspace so it is discoverable
#: by name across restarts and clearly platform-owned.
GLINER_FILESET_WORKSPACE = "default"
GLINER_FILESET_NAME = "nhx-anonymizer-gliner-pii"

#: Path suffix for the Files HuggingFace-Hub-compatible resolve API.
_FILES_HF_ENDPOINT_SUFFIX = "/apis/files/v2/hf"

#: Service principal used for the pull-through download (mirrors the model
#: weight-puller's ``service:models`` convention).
_GLINER_HF_TOKEN = "service:anonymizer"


def _files_hf_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{_FILES_HF_ENDPOINT_SUFFIX}"


def _gliner_storage_config() -> HuggingfaceStorageConfig:
    return HuggingfaceStorageConfig(
        repo_id=GLINER_MODEL_ID,
        repo_type="model",
        revision=GLINER_MODEL_REVISION,
    )


def _validate_existing_fileset_matches(storage: object) -> None:
    """Fail loudly if a pre-existing same-named Fileset is not our GLiNER source.

    A name collision with a user-created Fileset is not expected (the ``nhx-``
    prefix in ``default`` signals platform ownership), but if it happens we must
    not silently resolve weights against the wrong repo.
    """
    storage_type = getattr(storage, "type", None)
    repo_id = getattr(storage, "repo_id", None)
    if storage_type != StorageConfigType.HUGGINGFACE or repo_id != GLINER_MODEL_ID:
        raise AnonymizerInternalError(
            f"Fileset '{GLINER_FILESET_WORKSPACE}/{GLINER_FILESET_NAME}' already exists but is not the "
            f"expected HuggingFace-backed GLiNER source (got storage type {storage_type!r}, repo_id "
            f"{repo_id!r}; expected 'huggingface'/{GLINER_MODEL_ID!r}). Refusing to resolve weights "
            "against it."
        )


def ensure_gliner_fileset(sdk: SyncHelixClient) -> None:
    """Idempotently ensure the system-owned GLiNER Fileset exists (get-or-create).

    Validates that an existing same-named Fileset is actually our HuggingFace
    GLiNER source, else raises. Creating the entity is cheap metadata only — no
    weights move until the first resolve.
    """
    files = client_from_platform(sdk, FilesClient)
    try:
        existing = files.get_fileset(name=GLINER_FILESET_NAME, workspace=GLINER_FILESET_WORKSPACE)
        _validate_existing_fileset_matches(existing.storage)
        return
    except NotFoundError:
        pass

    try:
        files.create_fileset(
            workspace=GLINER_FILESET_WORKSPACE,
            body=CreateFilesetRequest(
                name=GLINER_FILESET_NAME,
                description="System-owned GLiNER2 PII detector weights (pull-through from HuggingFace).",
                purpose=FilesetPurpose.MODEL,
                storage=_gliner_storage_config(),
            ),
        )
    except NotFoundError:
        raise
    except Exception as exc:  # a concurrent creator won the race, or a transient create failure
        existing = files.get_fileset(name=GLINER_FILESET_NAME, workspace=GLINER_FILESET_WORKSPACE)
        _validate_existing_fileset_matches(existing.storage)
        logger.debug("GLiNER Fileset already present after create attempt (%s)", exc)


async def ensure_gliner_fileset_async(async_sdk: AsyncHelixClient) -> None:
    """Async counterpart of :func:`ensure_gliner_fileset` for the preview path."""
    files = client_from_platform(async_sdk, AsyncFilesClient)
    try:
        existing = await files.get_fileset(name=GLINER_FILESET_NAME, workspace=GLINER_FILESET_WORKSPACE)
        _validate_existing_fileset_matches(existing.storage)
        return
    except NotFoundError:
        pass

    try:
        await files.create_fileset(
            workspace=GLINER_FILESET_WORKSPACE,
            body=CreateFilesetRequest(
                name=GLINER_FILESET_NAME,
                description="System-owned GLiNER2 PII detector weights (pull-through from HuggingFace).",
                purpose=FilesetPurpose.MODEL,
                storage=_gliner_storage_config(),
            ),
        )
    except NotFoundError:
        raise
    except Exception as exc:
        existing = await files.get_fileset(name=GLINER_FILESET_NAME, workspace=GLINER_FILESET_WORKSPACE)
        _validate_existing_fileset_matches(existing.storage)
        logger.debug("GLiNER Fileset already present after create attempt (%s)", exc)


def _hf_hub_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _upstream_snapshot_dir() -> Path:
    """Cache snapshot dir upstream's ``snapshot_download(MODEL_ID, MODEL_REVISION)`` reads."""
    storage_folder = _hf_hub_cache_dir() / repo_folder_name(repo_id=GLINER_MODEL_ID, repo_type="model")
    return storage_folder / "snapshots" / GLINER_MODEL_REVISION


def is_gliner_cached() -> bool:
    """True when the pinned GLiNER snapshot is already materialized where upstream reads it."""
    snapshot = _upstream_snapshot_dir()
    return snapshot.is_dir() and any(snapshot.iterdir())


def _align_fileset_cache_to_upstream_repo(fileset_snapshot: Path) -> None:
    """Expose a Files-fetched snapshot under the repo id/revision upstream requests.

    The pull-through download lands under the Fileset's cache folder
    (``models--default--nhx-anonymizer-gliner-pii``) at the commit hash Files
    reports. Upstream, however, calls ``snapshot_download`` with the HuggingFace
    repo id and pinned SHA, so it reads ``models--fastino--…/snapshots/{SHA}``.
    We bridge the two by symlinking that path at the Files-fetched snapshot. Only
    the public on-disk cache layout is used (``repo_folder_name``); no cache
    internals are synthesized.
    """
    upstream_snapshot = _upstream_snapshot_dir()
    upstream_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if upstream_snapshot.exists() or upstream_snapshot.is_symlink():
        return
    upstream_snapshot.symlink_to(fileset_snapshot.resolve(), target_is_directory=True)


def prewarm_gliner_cache(base_url: str, *, on_download_start: Callable[[], None] | None = None) -> None:
    """Download the GLiNER weights through Files into the HuggingFace cache.

    Resolves the system-owned Fileset by its ``{workspace}/{name}`` ref through
    the Files HuggingFace resolve API — the first fetch tees upstream weights
    into the Files cache, later fetches serve locally — then aligns the result
    under the repo id/revision upstream reads so its own ``snapshot_download``
    is a cache hit.

    ``on_download_start`` fires only when a real (first-run) download is about to
    happen, so callers can surface a downloading state without logging on every
    warm invocation. No-op when the snapshot is already present.
    """
    if is_gliner_cached():
        return
    if on_download_start is not None:
        on_download_start()

    fileset_snapshot = snapshot_download(
        repo_id=f"{GLINER_FILESET_WORKSPACE}/{GLINER_FILESET_NAME}",
        revision=GLINER_MODEL_REVISION,
        endpoint=_files_hf_endpoint(base_url),
        token=_GLINER_HF_TOKEN,
        cache_dir=str(_hf_hub_cache_dir()),
    )
    _align_fileset_cache_to_upstream_repo(Path(fileset_snapshot))


def ensure_gliner_weights(sdk: SyncHelixClient, *, on_download_start: Callable[[], None] | None = None) -> None:
    """Ensure the GLiNER Fileset exists and its weights are cached locally (sync).

    Called on first *use* (not service startup). Convenience for the run-job,
    which has a sync SDK; the preview path ensures the fileset asynchronously and
    prewarms the cache off-thread instead.
    """
    ensure_gliner_fileset(sdk)
    prewarm_gliner_cache(str(sdk.base_url), on_download_start=on_download_start)


def build_gliner_anonymizer(
    *,
    model_configs_yaml: str,
    dd_providers: list[DDModelProvider] | None,
    artifact_path: str | Path | None = None,
) -> Anonymizer:
    """Build an :class:`Anonymizer` whose entity detector runs in-process on GLiNER2.

    Starts upstream's owned, dependency-isolated GLiNER2 loopback runtime and
    injects it as the detector — mirroring ``anonymizer.notebooks.create_anonymizer``
    but adding an ``artifact_path`` the run-job needs. **Weights must already be
    cached** (via :func:`ensure_gliner_weights` or :func:`prewarm_gliner_cache`);
    the isolated server then loads from cache with outgoing HuggingFace traffic
    disabled, so no network is needed here.
    """
    from anonymizer.notebooks._model_config import build_notebook_model_configuration
    from anonymizer.notebooks._runtime import _ensure_runtime, _runtime_lock

    # Weights are cached; keep the isolated server offline so it loads from cache
    # instead of re-resolving the pinned repo against public HuggingFace.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    with _runtime_lock:
        runtime = _ensure_runtime("cpu")
        configuration = build_notebook_model_configuration(
            model_configs=model_configs_yaml,
            model_providers=dd_providers,
            endpoint=runtime.endpoint,
        )
        return Anonymizer(
            model_configs=configuration.model_configs,
            model_providers=configuration.model_providers,
            artifact_path=artifact_path,
        )


def stop_gliner_runtime() -> None:
    """Stop the owned GLiNER2 runtime child process (safe to call repeatedly)."""
    from anonymizer.notebooks._runtime import stop_local_runtime

    stop_local_runtime()
