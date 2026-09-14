# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fileset reference and download helpers for evaluator plugin execution."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

from filesets import AsyncFilesetFileSystem, FilesetFileSystem
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from pydantic import Field, RootModel

_GLOB_CHARS = {"*", "?", "["}


class FilesetRef(RootModel):
    """Reference to a persisted platform fileset, optionally with a file fragment."""

    root: str = Field(description="Reference to a Fileset (format: workspace/fileset-name).")

    def with_fragment(self, fragment: str) -> FilesetRef:
        """Return a new fileset reference with a file path fragment appended."""
        normalized_fragment = fragment.lstrip("/")
        if not normalized_fragment:
            raise ValueError("FilesetRef fragment cannot be empty.")
        if "#" in normalized_fragment:
            raise ValueError("FilesetRef fragment cannot contain '#'.")
        if "#" in self.root:
            raise ValueError("FilesetRef already includes a fragment.")
        return FilesetRef(root=f"{self.root}#{normalized_fragment}")


def is_fileset_glob_pattern(pattern: str) -> bool:
    """Return whether a fileset fragment contains glob wildcards."""
    return any(char in pattern for char in _GLOB_CHARS)


def matches_fileset_glob(filepath: str, pattern: str) -> bool:
    """Return whether a fileset-relative path matches a root-anchored glob pattern."""
    normalized_path = filepath.strip("/")
    normalized_pattern = pattern.strip("/")
    if not normalized_path or not normalized_pattern:
        return False
    return _match_path_parts(tuple(normalized_path.split("/")), tuple(normalized_pattern.split("/")))


def fileset_glob_prefix_dir(pattern: str) -> str:
    """Return the stable directory prefix before the first glob wildcard."""
    pattern = pattern.lstrip("/")
    if not pattern or not is_fileset_glob_pattern(pattern):
        return pattern

    first_wildcard = min(index for index, char in enumerate(pattern) if char in _GLOB_CHARS)
    prefix = pattern[:first_wildcard]
    if "/" not in prefix:
        return ""
    return prefix.rsplit("/", 1)[0]


def normalize_fileset_path(path: str) -> str:
    """Normalize a fileset ref into the local path used after download."""
    if "#" not in path:
        return path

    base, fragment = path.split("#", 1)
    fragment = fragment.lstrip("/")
    if not fragment:
        return base

    if is_fileset_glob_pattern(fragment):
        dir_prefix = fileset_glob_prefix_dir(fragment)
        return f"{base}/{dir_prefix}" if dir_prefix else base

    return f"{base}/{fragment}"


def _safe_child_path(base: Path, child: str) -> Path:
    """Return a child path only when it remains under base after resolution."""
    base_resolved = base.resolve()
    candidate = (base / child).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"Fileset path escapes destination: {child}")
    return candidate


async def download_dataset(
    client: AsyncNemoClient,
    dataset: FilesetRef,
    destination: str,
    recursive: bool = True,
) -> Path:
    """Download a FilesetRef dataset to a local directory."""
    files_client = AsyncFilesClient.from_client(client)
    fs = AsyncFilesetFileSystem(client=files_client)
    return await _download_fileset_ref(dataset, destination, recursive=recursive, fs=fs)


def download_dataset_sync(
    client: NemoClient,
    dataset: FilesetRef,
    destination: str,
    recursive: bool = True,
) -> Path:
    """Download a FilesetRef dataset to a local directory using the sync client."""
    return _download_fileset_ref_sync(client, dataset, destination, recursive=recursive)


def load_beir_dataset(path: str | Path) -> BeirDataset:
    """Validate and load a downloaded BEIR test fileset."""
    return BeirDataset.from_path(path)


def _match_path_parts(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts

    pattern_part = pattern_parts[0]
    remaining_pattern = pattern_parts[1:]
    if pattern_part == "**":
        return _match_path_parts(path_parts, remaining_pattern) or (
            bool(path_parts) and _match_path_parts(path_parts[1:], pattern_parts)
        )

    if not path_parts:
        return False
    return fnmatchcase(path_parts[0], pattern_part) and _match_path_parts(path_parts[1:], remaining_pattern)


async def _download_fileset_ref(
    dataset: FilesetRef,
    destination: str,
    recursive: bool = True,
    *,
    fs: AsyncFilesetFileSystem,
) -> Path:
    ref = dataset.root

    if "#" in ref:
        base_path, pattern = ref.split("#", 1)
        pattern = pattern.lstrip("/")

        if not pattern:
            return await _download_fileset_ref(
                FilesetRef(root=base_path),
                destination,
                recursive=recursive,
                fs=fs,
            )

        base_dest = _safe_child_path(Path(destination), base_path)
        base_dest.mkdir(parents=True, exist_ok=True)

        if is_fileset_glob_pattern(pattern):
            all_files = await fs._find(base_path)
            for file_path in all_files:
                if "#" in file_path:
                    relative_path = file_path.split("#", 1)[1]
                else:
                    relative_path = file_path.replace(f"{base_path}/", "", 1)
                if matches_fileset_glob(relative_path, pattern):
                    file_dest = _safe_child_path(base_dest, relative_path)
                    file_dest.parent.mkdir(parents=True, exist_ok=True)
                    await fs._get_file(file_path, str(file_dest))
            return base_dest

        full_remote_path = f"{base_path}/{pattern}"
        file_dest = _safe_child_path(base_dest, pattern)
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        await fs._get_file(full_remote_path, str(file_dest))
        return file_dest

    dest = _safe_child_path(Path(destination), normalize_fileset_path(ref))
    source = ref.rstrip("/") + "/"
    await fs._get(source, str(dest), recursive=recursive)
    return dest


def _download_fileset_ref_sync(
    client: NemoClient,
    dataset: FilesetRef,
    destination: str,
    recursive: bool = True,
) -> Path:
    files_client = FilesClient.from_client(client)
    fs = FilesetFileSystem(client=files_client)
    return _download_fileset_ref_with_sync_filesystem(dataset, destination, recursive=recursive, fs=fs)


def _download_fileset_ref_with_sync_filesystem(
    dataset: FilesetRef,
    destination: str,
    recursive: bool = True,
    *,
    fs: FilesetFileSystem,
) -> Path:
    ref = dataset.root

    if "#" in ref:
        base_path, pattern = ref.split("#", 1)
        pattern = pattern.lstrip("/")

        if not pattern:
            return _download_fileset_ref_with_sync_filesystem(
                FilesetRef(root=base_path),
                destination,
                recursive=recursive,
                fs=fs,
            )

        base_dest = _safe_child_path(Path(destination), base_path)
        base_dest.mkdir(parents=True, exist_ok=True)

        if is_fileset_glob_pattern(pattern):
            all_files = fs.find(base_path)
            for file_path in all_files:
                if "#" in file_path:
                    relative_path = file_path.split("#", 1)[1]
                else:
                    relative_path = file_path.replace(f"{base_path}/", "", 1)
                if matches_fileset_glob(relative_path, pattern):
                    file_dest = _safe_child_path(base_dest, relative_path)
                    file_dest.parent.mkdir(parents=True, exist_ok=True)
                    fs.get_file(file_path, str(file_dest))
            return base_dest

        full_remote_path = f"{base_path}/{pattern}"
        file_dest = _safe_child_path(base_dest, pattern)
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(full_remote_path, str(file_dest))
        return file_dest

    dest = _safe_child_path(Path(destination), normalize_fileset_path(ref))
    source = ref.rstrip("/") + "/"
    fs.get(source, str(dest), recursive=recursive)
    return dest
