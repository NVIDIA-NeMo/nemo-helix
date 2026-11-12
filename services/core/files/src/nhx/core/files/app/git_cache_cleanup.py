# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Removes Git fileset cache entries that no fileset uses."""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from nemo_helix_plugin.files.storage_config import is_commit_sha
from nhx.common.entities.client import EntityClient
from nhx.common.files.storage_config import GitStorageConfig, StorageConfigType
from nhx.core.files.app.backends.base import FileInfo, StorageImpl
from nhx.core.files.app.backends.git import (
    GIT_CACHE_PREFIX,
    commit_key,
    forget_listing,
    prune_unreferenced_fetch_repositories,
    url_digest,
)
from nhx.core.files.entities import Fileset
from nhx.core.files.exceptions import NotFoundError

logger = logging.getLogger(__name__)

_FILESET_PAGE_SIZE = 1000
_CACHE_KEY = re.compile(
    rf"(?P<entry>{re.escape(GIT_CACHE_PREFIX)}/(?P<digest>[0-9a-f]{{32}})/commits/[0-9a-f]{{40}}/[0-9a-f]{{16}})"
    r"(?:\.json|/files/.+)"
)


@dataclass
class GitCacheReferences:
    commit_keys: set[str] = field(default_factory=set)
    url_digests: set[str] = field(default_factory=set)
    unpinned_digests: set[str] = field(default_factory=set)

    def add(self, storage: GitStorageConfig) -> None:
        digest = url_digest(storage.url)
        self.url_digests.add(digest)
        if is_commit_sha(storage.revision):
            self.commit_keys.add(commit_key(storage.url, storage.revision, storage.path))
        else:
            self.unpinned_digests.add(digest)

    def protects(self, entry: str, digest: str) -> bool:
        return entry in self.commit_keys or digest in self.unpinned_digests

    def __or__(self, other: "GitCacheReferences") -> "GitCacheReferences":
        return GitCacheReferences(
            self.commit_keys | other.commit_keys,
            self.url_digests | other.url_digests,
            self.unpinned_digests | other.unpinned_digests,
        )


async def git_cache_references(entity_client: EntityClient) -> GitCacheReferences:
    references = GitCacheReferences()
    page = 1
    while True:
        result = await entity_client.list(
            Fileset,
            workspace="-",
            filter_obj={"storage.type": StorageConfigType.GIT.value},
            sort="created_at",
            page=page,
            page_size=_FILESET_PAGE_SIZE,
        )
        for fileset in result.data:
            if isinstance(fileset.storage, GitStorageConfig):
                references.add(fileset.storage)
        if page >= result.pagination.total_pages:
            return references
        page += 1


@dataclass
class _Entry:
    digest: str
    files: list[FileInfo] = field(default_factory=list)


def _cache_entries(files: list[FileInfo]) -> dict[str, _Entry]:
    entries: dict[str, _Entry] = {}
    for info in files:
        match = _CACHE_KEY.fullmatch(info.path)
        if match is not None:
            entries.setdefault(match["entry"], _Entry(match["digest"])).files.append(info)
    return entries


def _idle_since(entry: _Entry, cutoff: float) -> bool:
    return all(info.modified_at is not None and info.modified_at < cutoff for info in entry.files)


async def _remove_entry(store: StorageImpl, key: str, entry: _Entry) -> None:
    # The listing goes last, so an interrupted removal leaves the entry visible to the next sweep.
    for info in sorted(entry.files, key=lambda info: info.path.endswith(".json")):
        try:
            await store.delete(info.path)
        except NotFoundError:
            pass
    forget_listing(f"{key}.json")


async def sweep_git_cache(
    store: StorageImpl,
    load_references: Callable[[], Awaitable[GitCacheReferences]],
    *,
    grace_seconds: float,
    fetch_root: Path | None = None,
    now: Callable[[], float] = time.time,
) -> int:
    """Remove cache entries no Git fileset references and nothing has written within the grace period."""
    before = await load_references()
    cutoff = now() - grace_seconds
    entries = _cache_entries(await store.list_files(f"{GIT_CACHE_PREFIX}/"))
    candidates = {
        key: entry
        for key, entry in entries.items()
        if not before.protects(key, entry.digest) and _idle_since(entry, cutoff)
    }
    # Read again, so a fileset created while the cache was listed keeps its entry.
    references = before | await load_references()
    removed = 0
    for key, entry in candidates.items():
        if references.protects(key, entry.digest):
            continue
        try:
            await _remove_entry(store, key, entry)
        except Exception:
            logger.warning("Could not remove the unused git cache entry %s", key, exc_info=True)
            continue
        removed += 1
    if fetch_root is not None:
        await prune_unreferenced_fetch_repositories(fetch_root, references.url_digests, cutoff)
    return removed


async def run_git_cache_cleanup(
    store: StorageImpl,
    entity_client: EntityClient,
    *,
    interval_seconds: float,
    grace_seconds: float,
    fetch_root: Path,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            removed = await sweep_git_cache(
                store,
                lambda: git_cache_references(entity_client),
                grace_seconds=grace_seconds,
                fetch_root=fetch_root,
            )
        except Exception:
            logger.warning("Git cache cleanup failed; it will run again in %ss", interval_seconds, exc_info=True)
            continue
        if removed:
            logger.info("Removed %d unused git cache entries", removed)
