# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.api.common import SecretRef
from nhx.common.files.storage_config import LocalStorageConfig
from nhx.core.files.app.backends.git import (
    GitStorageConfig,
    GitStorageImpl,
    _listings,
    _remember_listing,
    commit_key,
    url_digest,
)
from nhx.core.files.app.backends.local import LocalStorageImpl
from nhx.core.files.app.git_cache_cleanup import GitCacheReferences, git_cache_references, sweep_git_cache

URL = "git@gitlab.example.com:org/repo.git"
OTHER_URL = "git@gitlab.example.com:org/other.git"
FIRST = "a" * 40
SECOND = "b" * 40
GRACE = 60.0
DAY = 24 * 60 * 60.0


def _config(url: str = URL, revision: str = FIRST, path: str = "") -> GitStorageConfig:
    return GitStorageConfig(
        url=url, revision=revision, path=path, ssh_key_secret=SecretRef("key"), known_hosts="host ssh-ed25519 AAAA"
    )


def _references(*configs: GitStorageConfig) -> GitCacheReferences:
    references = GitCacheReferences()
    for config in configs:
        references.add(config)
    return references


def _static(references: GitCacheReferences):
    async def load() -> GitCacheReferences:
        return references

    return load


async def _chunk(body: bytes) -> AsyncIterator[bytes]:
    yield body


@pytest.fixture
def store(tmp_path: Path) -> LocalStorageImpl:
    return LocalStorageImpl(LocalStorageConfig(path=str(tmp_path / "durable")))


async def _write_entry(store: LocalStorageImpl, key: str, *, age: float = DAY) -> list[str]:
    paths = [f"{key}.json", f"{key}/files/agent.yaml", f"{key}/files/nested/prompt.md"]
    for path in paths:
        await store.upload(path, _chunk(b"x"), content_length=1)
        _age(store, path, age)
    return paths


def _age(store: LocalStorageImpl, path: str, age: float) -> None:
    stamp = time.time() - age
    os.utime(Path(store.config.path) / path, (stamp, stamp))


async def _stored(store: LocalStorageImpl) -> set[str]:
    return {info.path for info in await store.list_files()}


@pytest.fixture(autouse=True)
def _forget_listings():
    yield
    _listings.clear()


class TestReferences:
    async def test_every_page_of_git_filesets_across_workspaces_is_read(self):
        pinned, unpinned = _config(path="agents/support"), _config(OTHER_URL, revision="main")
        entity_client = MagicMock()
        entity_client.list = AsyncMock(
            side_effect=[
                SimpleNamespace(data=[SimpleNamespace(storage=pinned)], pagination=SimpleNamespace(total_pages=2)),
                SimpleNamespace(data=[SimpleNamespace(storage=unpinned)], pagination=SimpleNamespace(total_pages=2)),
            ]
        )

        references = await git_cache_references(entity_client)

        assert references.commit_keys == {commit_key(URL, FIRST, "agents/support")}
        assert references.unpinned_digests == {url_digest(OTHER_URL)}
        assert references.url_digests == {url_digest(URL), url_digest(OTHER_URL)}
        calls = entity_client.list.await_args_list
        assert [call.kwargs["page"] for call in calls] == [1, 2]
        assert all(call.kwargs["workspace"] == "-" for call in calls)
        assert all(call.kwargs["filter_obj"] == {"storage.type": "git"} for call in calls)

    async def test_the_backend_writes_its_cache_where_a_reference_protects_it(self, tmp_path):
        config = _config(path="agents/support")
        impl = GitStorageImpl(config, {}, cache_root=tmp_path / "cache")
        cached_file = await impl.get_cache_path_key()
        assert cached_file is not None
        entry = impl._listing_key(FIRST).removesuffix(".json")

        assert cached_file == f"{entry}/files"
        assert _references(config).protects(entry, url_digest(URL))


class TestSweep:
    async def test_a_referenced_entry_is_kept_and_an_unreferenced_old_one_removed(self, store):
        kept = await _write_entry(store, commit_key(URL, FIRST, ""))
        await _write_entry(store, commit_key(URL, SECOND, ""))

        removed = await sweep_git_cache(store, _static(_references(_config())), grace_seconds=GRACE)

        assert removed == 1
        assert await _stored(store) == set(kept)

    async def test_an_unreferenced_entry_within_the_grace_period_is_kept(self, store):
        written = await _write_entry(store, commit_key(URL, SECOND, ""), age=GRACE / 2)

        assert await sweep_git_cache(store, _static(GitCacheReferences()), grace_seconds=GRACE) == 0
        assert await _stored(store) == set(written)

    async def test_an_entry_with_a_recently_cached_file_is_kept(self, store):
        written = await _write_entry(store, commit_key(URL, SECOND, ""))
        _age(store, written[1], GRACE / 2)

        assert await sweep_git_cache(store, _static(GitCacheReferences()), grace_seconds=GRACE) == 0
        assert await _stored(store) == set(written)

    async def test_a_fileset_created_while_the_cache_is_listed_keeps_its_entry(self, store):
        written = await _write_entry(store, commit_key(URL, FIRST, ""))
        load = AsyncMock(side_effect=[GitCacheReferences(), _references(_config())])

        assert await sweep_git_cache(store, load, grace_seconds=GRACE) == 0
        assert await _stored(store) == set(written)

    async def test_a_fileset_on_a_branch_keeps_every_entry_of_its_repository(self, store):
        kept = await _write_entry(store, commit_key(URL, SECOND, "agents"))
        await _write_entry(store, commit_key(OTHER_URL, SECOND, ""))

        references = _references(_config(revision="main"))
        assert await sweep_git_cache(store, _static(references), grace_seconds=GRACE) == 1
        assert await _stored(store) == set(kept)

    async def test_keys_outside_the_cache_layout_are_left_alone(self, store):
        unknown = [f"cache/git/{url_digest(URL)}/notes.txt", "cache/gitlab/listing.json", "filesets/default/a/b.txt"]
        for path in unknown:
            await store.upload(path, _chunk(b"x"), content_length=1)
            _age(store, path, DAY)

        assert await sweep_git_cache(store, _static(GitCacheReferences()), grace_seconds=GRACE) == 0
        assert await _stored(store) == set(unknown)

    async def test_an_empty_store_sweeps_nothing(self, store):
        assert await sweep_git_cache(store, _static(GitCacheReferences()), grace_seconds=GRACE) == 0

    async def test_a_removed_listing_is_forgotten_in_memory(self, store):
        key = commit_key(URL, SECOND, "")
        await _write_entry(store, key)
        _remember_listing(f"{key}.json", {})

        await sweep_git_cache(store, _static(GitCacheReferences()), grace_seconds=GRACE)

        assert f"{key}.json" not in _listings

    async def test_fetch_repositories_of_unreferenced_urls_are_pruned_once_idle(self, store, tmp_path):
        fetch_root = tmp_path / "fetch"
        referenced, idle, recent = (fetch_root / url_digest(url) for url in (URL, OTHER_URL, "git@h:x.git"))
        for repository in (referenced, idle, recent):
            repository.mkdir(parents=True)
        for repository in (referenced, idle):
            stamp = time.time() - DAY
            os.utime(repository, (stamp, stamp))

        await sweep_git_cache(store, _static(_references(_config())), grace_seconds=GRACE, fetch_root=fetch_root)

        assert (referenced.exists(), idle.exists(), recent.exists()) == (True, False, True)
