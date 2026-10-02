# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download Safe Synthesizer model weights from Files API into the HuggingFace cache."""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx
from nemo_helix_plugin.config import get_platform_config

logger = logging.getLogger(__name__)

DEFAULT_HF_HOME = "/app/.cache/huggingface"


@dataclass
class ModelFileset:
    """Configuration for a model fileset to download."""

    workspace: str
    name: str
    hf_model_id: str

    @property
    def fileset_ref(self) -> str:
        return f"{self.workspace}/{self.name}"


DEFAULT_MODEL_FILESETS = [
    ModelFileset(workspace="default", name="smollm3-3b", hf_model_id="HuggingFaceTB/SmolLM3-3B"),
    ModelFileset(
        workspace="default",
        name="gliner-gretel-bi-large",
        hf_model_id="gretelai/gretel-gliner-bi-large-v1.0",
    ),
    ModelFileset(workspace="default", name="bge-base-en", hf_model_id="BAAI/bge-base-en-v1.5"),
    ModelFileset(
        workspace="default",
        name="sentence-transformer-distiluse",
        hf_model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
    ),
]


def get_hf_cache_dir(hf_home: str | None = None) -> Path:
    """Get the HuggingFace hub cache directory."""
    hf_home = hf_home or os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    return Path(hf_home) / "hub"


def get_model_cache_path(model_id: str, hf_home: str | None = None) -> Path:
    """Get the cache path for a HuggingFace model ID."""
    return get_hf_cache_dir(hf_home) / f"models--{model_id.replace('/', '--')}"


def generate_snapshot_hash(fileset_name: str) -> str:
    """Generate a stable snapshot hash for a fileset-backed cache entry."""
    return hashlib.sha1(fileset_name.encode()).hexdigest()[:40]


def is_model_cached(model_id: str, hf_home: str | None = None) -> bool:
    """Check if a model is already cached locally."""
    snapshots_dir = get_model_cache_path(model_id, hf_home) / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(snapshot.is_dir() and any(snapshot.iterdir()) for snapshot in snapshots_dir.iterdir())


async def download_file(
    client: httpx.AsyncClient,
    files_api_url: str,
    workspace: str,
    fileset_name: str,
    file_path: str,
    dest_path: Path,
) -> None:
    """Download a single file from the Files API."""
    url = f"{files_api_url}/v2/workspaces/{workspace}/filesets/{fileset_name}/-/{file_path}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                await asyncio.to_thread(f.write, chunk)

    logger.debug("Downloaded: %s -> %s", file_path, dest_path)


async def list_fileset_files(
    client: httpx.AsyncClient,
    files_api_url: str,
    workspace: str,
    fileset_name: str,
) -> list[dict]:
    """List all files in a fileset."""
    url = f"{files_api_url}/v2/workspaces/{workspace}/filesets/{fileset_name}/files"
    response = await client.get(url)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload.get("files", []))


def _snapshot_target_path(snapshot_path: Path, file_path: str) -> Path:
    """Resolve a fileset path under a snapshot directory without allowing traversal."""
    relative_path = PurePosixPath(file_path)
    if (
        not file_path
        or not relative_path.parts
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"Invalid file path in fileset listing: {file_path!r}")

    snapshot_root = snapshot_path.resolve()
    target_path = (snapshot_root / Path(*relative_path.parts)).resolve()
    if not target_path.is_relative_to(snapshot_root):
        raise ValueError(f"Invalid file path in fileset listing: {file_path!r}")
    return target_path


async def download_model_fileset(
    client: httpx.AsyncClient,
    files_api_url: str,
    fileset: ModelFileset,
    hf_home: str | None = None,
    force: bool = False,
) -> bool:
    """Download a model fileset to the HuggingFace cache."""
    model_id = fileset.hf_model_id
    if not force and is_model_cached(model_id, hf_home):
        logger.info("Model already cached: %s", model_id)
        return True

    logger.info("Downloading model: %s from fileset %s", model_id, fileset.fileset_ref)
    try:
        files = await list_fileset_files(client, files_api_url, fileset.workspace, fileset.name)
        if not files:
            logger.warning("No files found in fileset: %s", fileset.fileset_ref)
            return False

        model_cache_path = get_model_cache_path(model_id, hf_home)
        snapshot_hash = generate_snapshot_hash(fileset.name)
        snapshot_path = model_cache_path / "snapshots" / snapshot_hash

        downloads = []
        for file_info in files:
            file_path = file_info["path"]
            downloads.append(
                (
                    file_path,
                    download_file(
                        client,
                        files_api_url,
                        fileset.workspace,
                        fileset.name,
                        file_path,
                        _snapshot_target_path(snapshot_path, file_path),
                    ),
                )
            )
        results = await asyncio.gather(
            *(download for _, download in downloads),
            return_exceptions=True,
        )
        failures = [
            (file_path, result)
            for (file_path, _), result in zip(downloads, results, strict=True)
            if isinstance(result, Exception)
        ]
        if failures:
            for file_path, error in failures:
                logger.error("Failed to download %s from fileset %s: %s", file_path, fileset.fileset_ref, error)
            return False

        refs_dir = model_cache_path / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(snapshot_hash, encoding="utf-8")

        logger.info("Successfully downloaded model: %s (%d files)", model_id, len(files))
        return True
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error downloading %s: %s", model_id, e.response.status_code)
        return False
    except Exception as e:
        logger.error("Error downloading %s: %s", model_id, e)
        return False


async def init_models(
    files_api_url: str,
    filesets: list[ModelFileset] | None = None,
    hf_home: str | None = None,
    force: bool = False,
    timeout: float = 600.0,
) -> dict[str, bool]:
    """Initialize model weights by downloading them from Files API."""
    filesets = filesets or DEFAULT_MODEL_FILESETS
    results = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        for fileset in filesets:
            results[fileset.hf_model_id] = await download_model_fileset(client, files_api_url, fileset, hf_home, force)
    return results


def init_models_sync(
    files_api_url: str | None = None,
    filesets: list[ModelFileset] | None = None,
    hf_home: str | None = None,
    force: bool = False,
) -> dict[str, bool]:
    """Synchronous wrapper for model initialization."""
    if files_api_url is None:
        files_api_url = get_platform_config().get_service_url("files")

    if not files_api_url:
        logger.warning(
            "Files API URL not configured. Set NHX_FILES_URL or pass files_api_url parameter. Skipping model download."
        )
        return {}

    logger.info("Initializing models from Files API: %s", files_api_url)
    return asyncio.run(
        init_models(
            files_api_url=files_api_url,
            filesets=filesets,
            hf_home=hf_home,
            force=force,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = init_models_sync()
    for model, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"{status}: {model}")
