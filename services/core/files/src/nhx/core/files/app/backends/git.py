# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git storage backend for repositories read over SSH.

A fileset is pinned to a commit. That commit's listing of paths, blob ids and
sizes is written to the files service's default storage on first use. File
contents stream from a per-URL bare repository and are cached by the files
service like any external backend's. The bare repository can be deleted at
any time and only costs a fetch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Coroutine, Iterator
from contextlib import aclosing, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_helix_plugin.config import nhx_user_data_dir
from nemo_helix_plugin.files.storage_config import is_commit_sha
from nhx.common.files.storage_config import GitStorageConfig as GitStorageConfig
from nhx.core.files.app.backends.base import (
    REGULAR_FILE_MODES,
    ByteRange,
    FileInfo,
    StorageImpl,
)
from nhx.core.files.app.external_hosts import validate_external_host
from nhx.core.files.exceptions import (
    NotFoundError,
    StorageAccessError,
    StorageBackendError,
    StorageConfigError,
    StorageServerFault,
    StorageUnavailableError,
)

logger = logging.getLogger(__name__)

# Weak, so a URL's lock goes away once nothing holds or waits on it.
_repo_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

# git's own "infinite" depth, which deepens a shallow repository to the full history of a ref.
_FULL_HISTORY = 2147483647

# Bounds the git and ssh commands one files service runs at once, across every request; file streams are not counted.
_GIT_SLOTS = asyncio.Semaphore(16)


class GitBackendError(StorageBackendError):
    """Raised when git fails for a reason not covered below."""


class GitServerFault(StorageServerFault):
    """Raised when git cannot run on this server, such as a missing binary or a full disk."""


class GitAccessError(StorageAccessError):
    """Raised when the remote rejects the SSH key or its host key does not match."""


class GitConfigError(StorageConfigError):
    """Raised when the repository, revision, or directory does not exist."""


class GitUnavailableError(StorageUnavailableError):
    """Raised when the remote cannot be reached."""


# git closes every remote failure with this advice, which says nothing about the cause.
_GIT_BOILERPLATE = (
    "fatal: Could not read from remote repository",
    "Please make sure you have the correct access rights",
    "and the repository exists",
)

_SERVER_FAULTS = (
    "No space left on device",
    "Disk quota exceeded",
    "Read-only file system",
    # Only a local file; ssh's "Permission denied (publickey,password)" is matched before this.
    ": Permission denied",
)

_MISSING_FROM_REMOTE = (
    "does not appear to be a git repository",
    "Repository not found",
    "could not be found or you don't have permission",
    "not our ref",
    "unadvertised object",
    "couldn't find remote ref",
    "Not a valid object name",
    "not a tree object",
)

_NETWORK_FAILURES = (
    "Could not resolve hostname",
    "Connection refused",
    "Connection timed out",
    "Connection reset",
    "Connection closed",
    "kex_exchange_identification",
    "No route to host",
    "Host is down",
    "Network is unreachable",
    "Operation timed out",
    "not responding",
    "remote end hung up",
    "Broken pipe",
    "early EOF",
)


def _failure_detail(stderr: str) -> str:
    for line in stderr.splitlines():
        line = line.strip()
        if line and not line.startswith(_GIT_BOILERPLATE):
            return line
    return "no output"


def _classify_failure(stderr: str, subject: str) -> Exception:
    detail = _failure_detail(stderr)
    if "Host key verification failed" in stderr or "REMOTE HOST IDENTIFICATION HAS CHANGED" in stderr:
        return GitAccessError(f"The host key for {subject} does not match known_hosts")
    if "Permission denied (" in stderr or "Load key" in stderr:
        return GitAccessError(f"The SSH key was rejected for {subject}: {detail}")
    if any(marker in stderr for marker in _SERVER_FAULTS):
        return GitServerFault(f"git could not write its files on this server reading {subject}: {detail}")
    if any(marker in stderr for marker in _MISSING_FROM_REMOTE):
        return GitConfigError(f"{subject} does not exist, or the key cannot see it: {detail}")
    if any(marker in stderr for marker in _NETWORK_FAILURES):
        return GitUnavailableError(f"Could not reach {subject}: {detail}")
    return GitBackendError(f"git failed reading {subject}: {detail}")


_PEM_BLOCK = re.compile(r"-----BEGIN ([A-Z0-9 ]+)-----(.*?)-----END \1-----", re.DOTALL)


def normalize_private_key(key: str) -> str:
    """Restore the line breaks a single-line secret input strips from a pasted PEM key."""
    match = _PEM_BLOCK.search(key)
    if match is None:
        return key.replace("\r\n", "\n").strip() + "\n"
    label, body = match.group(1), "".join(match.group(2).split())
    lines = [body[start : start + 64] for start in range(0, len(body), 64)]
    return "\n".join([f"-----BEGIN {label}-----", *lines, f"-----END {label}-----"]) + "\n"


def _pick_ref(refs: dict[str, str], revision: str) -> str | None:
    # A peeled "^{}" line names an annotated tag's commit; the unpeeled line names the tag object.
    for name in (
        f"{revision}^{{}}",
        revision,
        f"refs/heads/{revision}",
        f"refs/tags/{revision}^{{}}",
        f"refs/tags/{revision}",
    ):
        if name in refs:
            return refs[name]
    return None


_GRACE_SECONDS = 2.0


async def stop_process_group(proc: asyncio.subprocess.Process) -> None:
    """Stop *proc*, started in its own session, with every child it spawned, such as ssh."""
    # SIGTERM first so git removes its lock files, such as shallow.lock in the shared cache.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # macOS answers EPERM, not ESRCH, when the group's only member has exited but is not yet reaped.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)
        with suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _GRACE_SECONDS)
            return


async def communicate_within(proc: asyncio.subprocess.Process, timeout: float) -> tuple[bytes, bytes]:
    """Wait for *proc*, stopping its whole process group if it overruns *timeout* or the caller is cancelled."""
    try:
        return await asyncio.wait_for(proc.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError):
        await asyncio.shield(stop_process_group(proc))
        raise


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    blob: str
    size: int


def _tree_entries(ls_tree_output: bytes) -> Iterator[_TreeEntry]:
    for raw_entry in ls_tree_output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            entry = raw_entry.decode()
        except UnicodeDecodeError:
            logger.debug("Skipping a non-UTF-8 file name in a git tree")
            continue
        meta, _, path = entry.partition("\t")
        mode, kind, blob, size = meta.split()
        if kind == "blob" and mode in REGULAR_FILE_MODES:
            yield _TreeEntry(path=path, blob=blob, size=int(size))


def default_git_cache_root() -> Path:
    return nhx_user_data_dir() / "git-cache"


def default_durable_storage() -> StorageImpl:
    """The files service's default storage, which also holds every external backend's file cache."""
    from nhx.core.files.app.backends.factory import storage_impl_factory
    from nhx.core.files.config import files_config

    return storage_impl_factory(files_config().default_storage_config, {})


_DURABLE_PREFIX = "cache/git"
_LISTING_VERSION = 1
_MAX_LISTED_FILES = 100_000
_FETCH_REPOSITORY_IDLE_SECONDS = 7 * 24 * 60 * 60
_LATEST_REF = "refs/nhx/latest"

# A path -> (blob id, size) listing of one commit, scoped to a fileset's directory.
Snapshot = dict[str, tuple[str, int]]


async def _single_chunk(body: bytes) -> AsyncIterator[bytes]:
    yield body


# Listings are immutable once written, so parsed ones are shared across requests.
_LISTING_MEMORY = 128
_listings: OrderedDict[str, Snapshot] = OrderedDict()


def _remember_listing(key: str, snapshot: Snapshot) -> None:
    _listings[key] = snapshot
    _listings.move_to_end(key)
    while len(_listings) > _LISTING_MEMORY:
        _listings.popitem(last=False)


# Strong references to work that outlives a cancelled request; the event loop keeps only weak ones.
_background_work: set[asyncio.Task[Any]] = set()


def _finish_even_if_cancelled[T](work: Coroutine[Any, Any, T]) -> Awaitable[T]:
    task = asyncio.ensure_future(work)
    _background_work.add(task)

    def _settle(done: asyncio.Task[Any]) -> None:
        _background_work.discard(done)
        if not done.cancelled() and done.exception() is not None:
            logger.debug("Git work finished after its request ended: %s", done.exception())

    task.add_done_callback(_settle)
    return asyncio.shield(task)


# Commits known to be in each fetch repository, so a read prepares a repository once per process.
_fetched_commits: dict[str, set[str]] = {}


async def _prune_fetch_repositories(root: Path, keep: Path) -> None:
    cutoff = time.time() - _FETCH_REPOSITORY_IDLE_SECONDS
    try:
        repos = await asyncio.to_thread(lambda: [entry for entry in root.iterdir() if entry.is_dir()])
    except OSError:
        return
    for repo in repos:
        lock = _repo_lock(repo)
        if repo == keep or lock.locked():
            continue
        async with lock:
            with suppress(OSError):
                if repo.stat().st_mtime < cutoff:
                    _fetched_commits.pop(str(repo), None)
                    await asyncio.to_thread(shutil.rmtree, repo, True)


def _repo_lock(repo: Path) -> asyncio.Lock:
    lock = _repo_locks.get(str(repo))
    if lock is None:
        lock = _repo_locks[str(repo)] = asyncio.Lock()
    return lock


async def _slice(stream: AsyncGenerator[bytes], byte_range: ByteRange | None) -> AsyncIterator[bytes]:
    async with aclosing(stream):
        offset = 0
        async for chunk in stream:
            if byte_range is None:
                yield chunk
                continue
            start, end = max(byte_range.start - offset, 0), byte_range.end - offset + 1
            offset += len(chunk)
            if start < len(chunk) and end > 0:
                yield chunk[start:end]
            if offset > byte_range.end:
                return


@dataclass
class GitStorageImpl(StorageImpl):
    config: GitStorageConfig
    secrets: dict[str, str] = field(repr=False)
    cache_root: Path = field(default_factory=default_git_cache_root)
    durable: StorageImpl | None = None
    allowed_protocols: str = "ssh"
    timeout_seconds: float = 120.0
    _resolved_sha: str | None = field(default=None, init=False, repr=False)
    _snapshot_memo: Snapshot | None = field(default=None, init=False, repr=False)

    @property
    def _store(self) -> StorageImpl:
        if self.durable is None:
            self.durable = default_durable_storage()
        return self.durable

    @property
    def _url_digest(self) -> str:
        return hashlib.sha256(self.config.url.encode()).hexdigest()[:32]

    @property
    def _repo_dir(self) -> Path:
        return self.cache_root / self._url_digest

    def _listing_key(self, sha: str) -> str:
        scope = hashlib.sha256(self.config.path.encode()).hexdigest()[:16]
        return f"{_DURABLE_PREFIX}/{self._url_digest}/commits/{sha}/{scope}.json"

    def _subject(self) -> str:
        scope = f"{self.config.url}#{self.config.path}" if self.config.path else self.config.url
        return f"{scope} at {self.config.revision}"

    def _base_env(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", "/"),
            "LC_ALL": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": self.allowed_protocols,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            # A path such as ":!x" would otherwise be read as pathspec magic rather than a file name.
            "GIT_LITERAL_PATHSPECS": "1",
            # A detached auto-gc would hold lock files in the fetch repository after its command returned.
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "gc.auto",
            "GIT_CONFIG_VALUE_0": "0",
            "GIT_CONFIG_KEY_1": "maintenance.auto",
            "GIT_CONFIG_VALUE_1": "false",
        }

    @contextmanager
    def _ssh_env(self) -> Iterator[dict[str, str]]:
        key = self.secrets.get("ssh_key", "")
        if not key.strip():
            raise GitConfigError(f"The SSH key secret for {self.config.url} is empty")

        with tempfile.TemporaryDirectory(prefix="nhx-git-ssh-") as tmp:
            key_path = Path(tmp) / "id"
            known_hosts_path = Path(tmp) / "known_hosts"
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(normalize_private_key(key))
            known_hosts_path.write_text(self.config.known_hosts.strip() + "\n")

            ssh_command = shlex.join(
                [
                    "ssh",
                    "-F",
                    os.devnull,
                    "-i",
                    str(key_path),
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "IdentityAgent=none",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    f"UserKnownHostsFile={known_hosts_path}",
                    "-o",
                    f"GlobalKnownHostsFile={os.devnull}",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                ]
            )
            yield {**self._base_env(), "GIT_SSH_COMMAND": ssh_command}

    async def _run(self, *args: str, env: dict[str, str]) -> tuple[int, bytes, str]:
        async with _GIT_SLOTS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    *args,
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise GitServerFault("git is not installed in the files service") from exc
            try:
                stdout, stderr = await communicate_within(proc, self.timeout_seconds)
            except TimeoutError:
                raise GitUnavailableError(
                    f"git timed out after {self.timeout_seconds:.0f}s reading {self._subject()}"
                ) from None
        return proc.returncode or 0, stdout, stderr.decode(errors="replace")

    async def _git(self, *args: str, env: dict[str, str], subject: str) -> bytes:
        returncode, stdout, stderr = await self._run(*args, env=env)
        if returncode != 0:
            raise _classify_failure(stderr, subject)
        return stdout

    async def _remote_git(self, *args: str, subject: str) -> bytes:
        with self._ssh_env() as env:
            return await self._git(*args, env=env, subject=subject)

    async def _ls_remote(self) -> str:
        revision = self.config.revision
        pinned = is_commit_sha(revision)
        # A bare tag pattern does not match the peeled `^{}` line that names an annotated tag's commit.
        patterns = ["HEAD"] if pinned else [revision, f"{revision}^{{}}"]
        output = await self._remote_git("ls-remote", "--", self.config.url, *patterns, subject=self.config.url)
        if pinned:
            return revision

        refs: dict[str, str] = {}
        for line in output.decode(errors="replace").splitlines():
            sha, _, name = line.partition("\t")
            refs[name] = sha
        sha = _pick_ref(refs, revision)
        if sha is None:
            raise GitConfigError(
                f"{self.config.url} has no branch or tag named {revision!r}; abbreviated SHAs are not supported"
            )
        return sha

    async def _commit(self) -> str:
        if self._resolved_sha is None:
            self._resolved_sha = (
                self.config.revision if is_commit_sha(self.config.revision) else await self._ls_remote()
            )
        return self._resolved_sha

    async def _has_commit(self, repo: Path, sha: str) -> bool:
        returncode, _, _ = await self._run("-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}", env=self._base_env())
        return returncode == 0

    async def _fetch(self, repo: Path, want: str, *, depth: int = 1) -> None:
        await self._remote_git(
            "-C",
            str(repo),
            "fetch",
            "--quiet",
            f"--depth={depth}",
            "--no-tags",
            "--no-recurse-submodules",
            "--no-write-fetch-head",
            "--",
            self.config.url,
            want,
            subject=f"{self.config.url} at {want}",
        )

    async def _fetch_with_fallback(self, repo: Path, sha: str) -> None:
        try:
            await self._fetch(repo, sha)
        except GitConfigError:
            # Some servers refuse a SHA in a fetch request; the ref it was resolved from still works.
            fallback = self.config.tracked_revision or (
                None if is_commit_sha(self.config.revision) else self.config.revision
            )
            if not fallback:
                raise
            await self._fetch(repo, fallback)
            if not await self._has_commit(repo, sha):
                # The ref has moved past the pinned commit, so only its history still holds it.
                await self._fetch(repo, fallback, depth=_FULL_HISTORY)

    async def _ensure_commit(self, repo: Path, sha: str) -> None:
        fetched = False
        async with _repo_lock(repo):
            if not await asyncio.to_thread((repo / "HEAD").exists):
                try:
                    await asyncio.to_thread(repo.parent.mkdir, parents=True, exist_ok=True)
                except OSError as exc:
                    raise GitServerFault(f"The git cache at {repo.parent} is not writable: {exc}") from exc
                await self._git("init", "--bare", "--quiet", str(repo), env=self._base_env(), subject=str(repo))
            if not await self._has_commit(repo, sha):
                await self._fetch_with_fallback(repo, sha)
                if not await self._has_commit(repo, sha):
                    raise GitConfigError(f"{self.config.url} has no commit {sha}")
                # Keeps the newest commit's objects, so the next fetch of this repository is a delta.
                await self._git(
                    "-C", str(repo), "update-ref", _LATEST_REF, sha, env=self._base_env(), subject=str(repo)
                )
                fetched = True
            _fetched_commits.setdefault(str(repo), set()).add(sha)
            with suppress(OSError):
                await asyncio.to_thread(os.utime, repo)
        if fetched:
            await _prune_fetch_repositories(self.cache_root, repo)

    async def _prepare_repository(self) -> None:
        repo, sha = self._repo_dir, await self._commit()
        if sha in _fetched_commits.get(str(repo), ()) and await asyncio.to_thread((repo / "HEAD").exists):
            return

        await _finish_even_if_cancelled(self._ensure_commit(repo, sha))

    async def resolve_config(self) -> GitStorageConfig:
        """Pin the revision to a commit SHA so the fileset cannot shift under a deployment."""
        sha = await self._commit()
        # A config already pinned keeps the branch or tag it was resolved from, for refreshes and fetch fallbacks.
        original = self.config.original_revision if is_commit_sha(self.config.revision) else None
        return self.config.model_copy(update={"revision": sha, "original_revision": original or self.config.revision})

    async def _snapshot(self) -> Snapshot:
        if self._snapshot_memo is None:
            sha = await self._commit()
            key = self._listing_key(sha)
            snapshot = await self._read_listing(key)
            if snapshot is None:
                snapshot = await _finish_even_if_cancelled(self._materialize(sha, key))
            self._snapshot_memo = snapshot
        return self._snapshot_memo

    async def _read_listing(self, key: str) -> Snapshot | None:
        if key in _listings:
            _listings.move_to_end(key)
            return _listings[key]
        try:
            stream = await self._store.download(key, None)
            raw = b"".join([chunk async for chunk in stream])
        except NotFoundError:
            return None
        try:
            listing = json.loads(raw)
            if listing.get("version") != _LISTING_VERSION:
                return None
            snapshot = {path: (blob, int(size)) for path, blob, size in listing["files"]}
        except (ValueError, TypeError, KeyError, AttributeError):
            logger.warning("Rebuilding an unreadable git listing at %s", key)
            return None
        _remember_listing(key, snapshot)
        return snapshot

    async def _materialize(self, sha: str, key: str) -> Snapshot:
        tree = f"{sha}:{self.config.path}" if self.config.path else sha

        await _finish_even_if_cancelled(self._ensure_commit(self._repo_dir, sha))
        output = await self._git(
            "-C", str(self._repo_dir), "ls-tree", "-r", "-l", "-z", tree, env=self._base_env(), subject=self._subject()
        )
        snapshot: Snapshot = {}
        for entry in _tree_entries(output):
            if len(snapshot) == _MAX_LISTED_FILES:
                raise GitConfigError(
                    f"{self._subject()} has more than {_MAX_LISTED_FILES} files; point the fileset at a directory"
                )
            snapshot[entry.path] = (entry.blob, entry.size)

        listing = {
            "version": _LISTING_VERSION,
            "commit": sha,
            "files": [[path, blob, size] for path, (blob, size) in sorted(snapshot.items())],
        }
        body = json.dumps(listing).encode()
        try:
            await self._store.upload(key, _single_chunk(body), content_length=len(body))
        except Exception:
            # Reads still work from memory and the fetch repository; the next process lists the commit again.
            logger.warning("Could not store the git listing at %s", key, exc_info=True)
        _remember_listing(key, snapshot)
        return snapshot

    async def _entry(self, path: str) -> tuple[str, int] | None:
        wanted = path.strip("/")
        if not wanted or any(segment in ("", ".", "..") for segment in wanted.split("/")):
            return None
        return (await self._snapshot()).get(wanted)

    async def _open_blob(self, repo: Path, blob: str, size: int) -> AsyncGenerator[bytes]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(repo),
                "cat-file",
                "blob",
                blob,
                env=self._base_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise GitServerFault("git is not installed in the files service") from exc
        assert proc.stdout is not None and proc.stderr is not None
        chunk_size = self.config.read_chunk_size
        try:
            # Read before returning, so a missing object fails before the response starts.
            first = (
                await asyncio.wait_for(proc.stdout.read(min(chunk_size, size)), self.timeout_seconds) if size else b""
            )
            if size and not first:
                stderr = (await communicate_within(proc, self.timeout_seconds))[1].decode(errors="replace")
                failure = _classify_failure(stderr, f"{blob} of {self._subject()}")
                if isinstance(failure, GitServerFault):
                    raise failure
                raise GitBackendError(f"The fetch repository for {self.config.url} could not read {blob}: {failure}")
        except BaseException as exc:
            await asyncio.shield(stop_process_group(proc))
            if isinstance(exc, TimeoutError):
                raise GitBackendError(
                    f"git sent nothing of {blob} of {self._subject()} for {self.timeout_seconds:.0f}s"
                ) from None
            raise

        async def stream() -> AsyncGenerator[bytes]:
            assert proc.stdout is not None
            remaining = size - len(first)
            try:
                if first:
                    yield first
                while remaining:
                    chunk = await proc.stdout.read(min(chunk_size, remaining))
                    if not chunk:
                        raise GitBackendError(f"git ended {blob} of {self._subject()} early")
                    remaining -= len(chunk)
                    yield chunk
            finally:
                if proc.returncode is None:
                    await asyncio.shield(stop_process_group(proc))

        return stream()

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        snapshot = await self._snapshot()
        wanted = path.strip("/") if path else ""
        files = [
            FileInfo(path=file_path, size=size)
            for file_path, (_blob, size) in snapshot.items()
            if not wanted or file_path == wanted or file_path.startswith(f"{wanted}/")
        ]
        if wanted and not files:
            raise NotFoundError(f"File not found for path: {path}")
        return files

    async def get_file(self, path: str) -> FileInfo:
        entry = await self._entry(path)
        if entry is None:
            raise NotFoundError(f"File not found for path: {path}")
        # Downloads look a file up before their short first-byte deadline starts, so a cold fetch belongs here.
        await self._prepare_repository()
        return FileInfo(path=path.strip("/"), size=entry[1])

    async def get_cache_path_key(self, path: str | None = None) -> str | None:
        # Keyed by blob, so an update caches only the files that changed.
        prefix = f"{_DURABLE_PREFIX}/{self._url_digest}/blobs"
        if path is None:
            return prefix
        entry = await self._entry(path)
        return None if entry is None else f"{prefix}/{entry[0][:2]}/{entry[0]}"

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        entry = await self._entry(path)
        if entry is None:
            raise NotFoundError(f"File not found for path: {path}")
        blob, size = entry
        await self._prepare_repository()
        return _slice(await self._open_blob(self._repo_dir, blob, size), byte_range)

    async def validate_storage(self):
        validate_external_host(self.config.remote.host_url)
        self._resolved_sha = await self._ls_remote()
        # ls-remote vouches for a branch or tag but not a commit SHA, and an empty directory reads like a missing one.
        if is_commit_sha(self.config.revision) or self.config.path:
            await self._snapshot()

    async def upload(
        self,
        path: str,
        fstream: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> FileInfo:
        raise NotImplementedError("Git upload is not implemented")

    async def delete(self, path: str) -> FileInfo:
        raise NotImplementedError("Git delete is not implemented")
