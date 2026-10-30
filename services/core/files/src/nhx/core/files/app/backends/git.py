# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git storage backend: filesets pinned to a commit of a repository read over SSH."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from dulwich.client import SSHGitClient, SSHVendor, StrangeHostname, SubprocessWrapper
from dulwich.errors import GitProtocolError, NotGitRepository, NotTreeError
from dulwich.object_store import BaseObjectStore, iter_tree_contents, tree_lookup_path
from dulwich.objects import Blob, Commit, ObjectID
from dulwich.protocol import DEFAULT_GIT_PROTOCOL_VERSION_FETCH
from dulwich.refs import Ref
from dulwich.repo import Repo
from nemo_helix_plugin.config import nhx_user_data_dir
from nemo_helix_plugin.files.storage_config import is_commit_sha
from nhx.common.files.storage_config import GitStorageConfig as GitStorageConfig
from nhx.core.files.app.backends.base import (
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

# Git's regular-file blob modes; directories, submodules and symlinks are not served as files.
_REGULAR_FILE_MODES = frozenset({0o100644, 0o100755})

_GIT_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="nhx-git")


class GitBackendError(StorageBackendError):
    """Raised when git fails for a reason not covered below."""


class GitServerFault(StorageServerFault):
    """Raised when git cannot run on this server, such as a missing ssh binary or a full disk."""


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
    "cannot run ssh",
    "ssh: not found",
    "ssh: command not found",
    "No user exists for uid",
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
    "unexpectedly closed the connection",
    "Broken pipe",
    "early EOF",
)

_SERVER_FAULT_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM})


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


_GIT_FAILURES = (GitProtocolError, NotGitRepository, OSError)


def _translate_failure(exc: Exception, subject: str, stderr: str = "") -> Exception:
    if isinstance(exc, GitProtocolError):
        # ssh's and the remote's stderr say why the connection ended; dulwich's own message rarely does.
        return _classify_failure(f"{stderr}\n{exc}", subject)
    if isinstance(exc, OSError) and exc.errno in _SERVER_FAULT_ERRNOS:
        return GitServerFault(f"git could not write its files on this server reading {subject}: {exc}")
    return GitBackendError(f"git failed reading {subject}: {exc}")


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
    # git's order for a name; a peeled "^{}" line names an annotated tag's commit.
    for name in (
        f"{revision}^{{}}",
        revision,
        f"refs/{revision}^{{}}",
        f"refs/{revision}",
        f"refs/tags/{revision}^{{}}",
        f"refs/tags/{revision}",
        f"refs/heads/{revision}",
    ):
        if name in refs:
            return refs[name]
    return None


_GRACE_SECONDS = 2.0
_MAX_STDERR_BYTES = 64 * 1024


async def stop_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop *proc*, started in its own session, with every child it spawned."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # macOS answers EPERM, not ESRCH, when the group's only member has exited but is not yet reaped.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)
        deadline = time.monotonic() + _GRACE_SECONDS
        while proc.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if proc.returncode is not None:
            return


class SessionSSHVendor(SSHVendor):
    """Runs ssh in a session of its own and keeps it, so a call that overruns can stop ssh from outside its thread."""

    def __init__(self, env: dict[str, str]) -> None:
        self._env = env
        self.processes: list[subprocess.Popen[bytes]] = []
        # dulwich closes a stderr pipe unread, and an undrained one can fill and stall ssh.
        self._stderr_files: list[IO[bytes]] = []

    def run_command(
        self,
        host: str,
        command: bytes,
        username: str | None = None,
        port: int | None = None,
        password: str | None = None,
        key_filename: str | None = None,
        ssh_command: str | None = None,
        protocol_version: int | None = None,
    ) -> SubprocessWrapper:
        args = [*shlex.split(ssh_command or "ssh"), "-x"]
        if port:
            args.extend(["-p", str(port)])
        version = DEFAULT_GIT_PROTOCOL_VERSION_FETCH if protocol_version is None else protocol_version
        if version > 0:
            args.extend(["-o", f"SetEnv GIT_PROTOCOL=version={version}"])
        destination = f"{username}@{host}" if username else host
        if destination.startswith("-"):
            raise StrangeHostname(hostname=destination)
        errors = tempfile.TemporaryFile()
        self._stderr_files.append(errors)
        try:
            proc = subprocess.Popen(
                [*args, destination, os.fsdecode(command)],
                bufsize=0,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                env=self._env,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise GitServerFault("ssh is not installed in the files service") from exc
        self.processes.append(proc)
        return SubprocessWrapper(proc)

    async def stop(self) -> None:
        await asyncio.gather(*(stop_process_group(proc) for proc in self.processes))

    def stderr(self) -> str:
        """What ssh and the remote wrote to stderr, once each ssh has exited."""
        for proc in self.processes:
            with suppress(subprocess.TimeoutExpired):
                proc.wait(_GRACE_SECONDS)
        chunks: list[str] = []
        for errors in self._stderr_files:
            errors.seek(0)
            chunks.append(errors.read(_MAX_STDERR_BYTES).decode(errors="replace"))
        return "\n".join(chunks)

    def close(self) -> None:
        for errors in self._stderr_files:
            errors.close()


def _regular_files(store: BaseObjectStore, tree_id: ObjectID) -> Iterator[tuple[str, ObjectID]]:
    for entry in iter_tree_contents(store, tree_id):
        if entry.mode not in _REGULAR_FILE_MODES:
            continue
        try:
            path = entry.path.decode()
        except UnicodeDecodeError:
            logger.debug("Skipping a non-UTF-8 file name in a git tree")
            continue
        yield path, entry.sha


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
_LATEST_REF = Ref(b"refs/nhx/latest")

Snapshot = dict[str, tuple[str, int]]


_BLOB_ID = re.compile(r"[0-9a-f]{40}")


def _parse_listing(raw: bytes, sha: str) -> Snapshot | None:
    try:
        listing = json.loads(raw)
        if listing.get("version") != _LISTING_VERSION or listing.get("commit") != sha:
            return None
        snapshot: Snapshot = {}
        for path, blob, size in listing["files"]:
            if not isinstance(path, str) or not isinstance(blob, str) or not _BLOB_ID.fullmatch(blob):
                return None
            if type(size) is not int or size < 0:
                return None
            snapshot[path] = (blob, size)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return snapshot


async def _single_chunk(body: bytes) -> AsyncIterator[bytes]:
    yield body


# Listings are immutable once written, so parsed ones are shared across requests.
_LISTED_FILES_IN_MEMORY = 250_000
_listings: OrderedDict[str, Snapshot] = OrderedDict()


def _remember_listing(key: str, snapshot: Snapshot) -> None:
    _listings[key] = snapshot
    _listings.move_to_end(key)
    while len(_listings) > 1 and sum(len(listing) for listing in _listings.values()) > _LISTED_FILES_IN_MEMORY:
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


_fetched_commits: dict[str, set[str]] = {}
_last_touched: dict[str, float] = {}
_TOUCH_INTERVAL_SECONDS = 60


async def _touch(repo: Path) -> None:
    now = time.monotonic()
    if now - _last_touched.get(str(repo), -_TOUCH_INTERVAL_SECONDS) >= _TOUCH_INTERVAL_SECONDS:
        _last_touched[str(repo)] = now
        with suppress(OSError):
            await asyncio.to_thread(os.utime, repo)


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
                    _last_touched.pop(str(repo), None)
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


@dataclass(frozen=True)
class _Remote:
    client: SSHGitClient
    path: str
    vendor: SessionSSHVendor


@dataclass
class GitStorageImpl(StorageImpl):
    config: GitStorageConfig
    secrets: dict[str, str] = field(repr=False)
    cache_root: Path = field(default_factory=default_git_cache_root)
    durable: StorageImpl | None = None
    ssh_binary: str = "ssh"
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

    @contextmanager
    def _remote(self) -> Generator[_Remote]:
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
                    self.ssh_binary,
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
            vendor = SessionSSHVendor(
                {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/"), "LC_ALL": "C"}
            )
            remote = self.config.remote
            client = SSHGitClient(
                remote.host, port=remote.port, username=remote.user, vendor=vendor, ssh_command=ssh_command
            )
            try:
                # git reads an scp-style path relative to the login directory and an ssh:// path from the root.
                yield _Remote(client, remote.path if remote.scp else f"/{remote.path}", vendor)
            finally:
                vendor.close()

    async def _call[T](self, work: Callable[[], T], subject: str, vendor: SessionSSHVendor | None = None) -> T:
        future = asyncio.get_running_loop().run_in_executor(_GIT_POOL, work)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.timeout_seconds)
        except (TimeoutError, asyncio.CancelledError) as exc:
            if vendor is not None:
                await asyncio.shield(vendor.stop())
            # The thread holds the repository until it returns, so the caller's lock has to outlast it.
            with suppress(Exception):
                await asyncio.shield(future)
            if isinstance(exc, TimeoutError):
                raise GitUnavailableError(
                    f"git timed out after {self.timeout_seconds:.0f}s reading {self._subject()}"
                ) from None
            raise
        except _GIT_FAILURES as exc:
            stderr = await asyncio.to_thread(vendor.stderr) if vendor is not None else ""
            raise _translate_failure(exc, subject, stderr) from exc

    async def _remote_refs(self) -> dict[str, str]:
        with self._remote() as remote:
            result = await self._call(
                lambda: remote.client.get_refs(remote.path.encode()), self.config.url, remote.vendor
            )
        if result.object_format == "sha256":
            raise GitConfigError(f"{self.config.url} uses SHA-256 object ids, which are not supported")
        return {name.decode(errors="replace"): sha.decode() for name, sha in result.refs.items() if sha is not None}

    async def _ls_remote(self) -> str:
        revision = self.config.revision
        sha = _pick_ref(await self._remote_refs(), revision)
        if sha is None:
            raise GitConfigError(
                f"{self.config.url} has no branch or tag named {revision!r}; abbreviated SHAs are not supported"
            )
        if not is_commit_sha(sha):
            raise GitConfigError(f"{self.config.url} uses SHA-256 object ids, which are not supported")
        return sha

    async def _commit(self) -> str:
        if self._resolved_sha is None:
            self._resolved_sha = (
                self.config.revision if is_commit_sha(self.config.revision) else await self._ls_remote()
            )
        return self._resolved_sha

    async def _has_commit(self, repo: Path, sha: str) -> bool:
        def check() -> bool:
            with Repo(str(repo)) as local:
                oid = ObjectID(sha.encode())
                return oid in local.object_store and local.object_store[oid].type_num == Commit.type_num

        return await self._call(check, str(repo))

    async def _fetch(self, repo: Path, sha: str) -> None:
        wanted = ObjectID(sha.encode())

        def only_the_commit(refs: Mapping[Ref, ObjectID], depth: int | None = None) -> list[ObjectID]:
            return [wanted]

        with self._remote() as remote:

            def fetch() -> None:
                with Repo(str(repo)) as local:
                    remote.client.fetch(remote.path, local, determine_wants=only_the_commit, depth=1)

            await self._call(fetch, f"{self.config.url} at {sha}", remote.vendor)

    async def _ensure_commit(self, repo: Path, sha: str, *, ask_remote: bool = False) -> None:
        fetched = False
        async with _repo_lock(repo):
            if not await asyncio.to_thread((repo / "HEAD").exists):
                try:
                    await asyncio.to_thread(repo.mkdir, parents=True, exist_ok=True)
                except OSError as exc:
                    raise GitServerFault(f"The git cache at {repo.parent} is not writable: {exc}") from exc
                await self._call(lambda: Repo.init_bare(str(repo)).close(), str(repo))
            present = await self._has_commit(repo, sha)
            if present and ask_remote:
                await self._fetch(repo, sha)
            elif not present:
                await self._fetch(repo, sha)
                if not await self._has_commit(repo, sha):
                    raise GitConfigError(f"{self.config.url} has no commit {sha}")

                def mark_latest() -> None:
                    with Repo(str(repo)) as local:
                        local.refs[_LATEST_REF] = ObjectID(sha.encode())

                # Keeps the newest commit's objects, so the next fetch of this repository is a delta.
                await self._call(mark_latest, str(repo))
                fetched = True
            _fetched_commits.setdefault(str(repo), set()).add(sha)
            with suppress(OSError):
                await asyncio.to_thread(os.utime, repo)
        if fetched:
            await _prune_fetch_repositories(self.cache_root, repo)

    async def _prepare_repository(self) -> None:
        repo, sha = self._repo_dir, await self._commit()
        if sha in _fetched_commits.get(str(repo), ()) and await asyncio.to_thread((repo / "HEAD").exists):
            await _touch(repo)
            return

        await _finish_even_if_cancelled(self._ensure_commit(repo, sha))

    async def resolve_config(self) -> GitStorageConfig:
        """Pin the revision to a commit SHA so the fileset cannot shift under a deployment."""
        sha = await self._commit()
        # Unconditional, as for GitHub: a client-supplied original_revision could repoint a pin on refresh.
        return self.config.model_copy(update={"revision": sha, "original_revision": self.config.revision})

    async def _snapshot(self) -> Snapshot:
        if self._snapshot_memo is None:
            sha = await self._commit()
            key = self._listing_key(sha)
            snapshot = await self._read_listing(key, sha)
            if snapshot is None:
                snapshot = await _finish_even_if_cancelled(self._materialize(sha, key))
            self._snapshot_memo = snapshot
        return self._snapshot_memo

    async def _read_listing(self, key: str, sha: str) -> Snapshot | None:
        if key in _listings:
            _listings.move_to_end(key)
            return _listings[key]
        try:
            stream = await self._store.download(key, None)
            raw = b"".join([chunk async for chunk in stream])
        except NotFoundError:
            return None
        snapshot = _parse_listing(raw, sha)
        if snapshot is None:
            logger.warning("Rebuilding an unreadable git listing at %s", key)
            return None
        _remember_listing(key, snapshot)
        return snapshot

    def _list_tree(self, sha: str) -> Snapshot:
        with Repo(str(self._repo_dir)) as local:
            store = local.object_store
            commit = store[ObjectID(sha.encode())]
            if not isinstance(commit, Commit):
                raise GitConfigError(f"{sha} of {self.config.url} is not a commit")
            tree_id = commit.tree
            if self.config.path:
                try:
                    mode, tree_id = tree_lookup_path(store.__getitem__, tree_id, self.config.path.encode())
                except (KeyError, NotTreeError):
                    mode = None
                if mode is None or not stat.S_ISDIR(mode):
                    raise GitConfigError(f"{self._subject()} has no directory {self.config.path!r}")

            snapshot: Snapshot = {}
            for path, blob in _regular_files(store, tree_id):
                if len(snapshot) == _MAX_LISTED_FILES:
                    raise GitConfigError(
                        f"{self._subject()} has more than {_MAX_LISTED_FILES} files; point the fileset at a directory"
                    )
                # Packs record only the compressed size, so each blob is inflated once per commit to measure it.
                _type, body = store.get_raw(blob)
                snapshot[path] = (blob.decode(), len(body))
            return snapshot

    async def _materialize(self, sha: str, key: str) -> Snapshot:
        await self._prepare_repository()
        snapshot = await self._call(lambda: self._list_tree(sha), self._subject())

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
        subject = f"{blob} of {self._subject()}"

        def read() -> bytes:
            with Repo(str(repo)) as local:
                try:
                    type_num, body = local.object_store.get_raw(ObjectID(blob.encode()))
                except KeyError:
                    raise GitBackendError(f"The fetch repository for {self.config.url} has no {subject}") from None
            if type_num != Blob.type_num:
                raise GitBackendError(f"{subject} is not a file")
            return body

        body = await self._call(read, subject)
        if len(body) != size:
            raise GitBackendError(f"{subject} is {len(body)} bytes, not the {size} its listing records")
        # Reads count as use, so idle pruning does not remove a repository that is still being read.
        await _touch(repo)
        chunk_size = self.config.read_chunk_size

        async def stream() -> AsyncGenerator[bytes]:
            for start in range(0, len(body), chunk_size):
                yield body[start : start + chunk_size]

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
        if is_commit_sha(self.config.revision):
            self._resolved_sha = self.config.revision
            # The fetch proves the key, and the server decides whether it may have this commit even if it is cached.
            await _finish_even_if_cancelled(self._ensure_commit(self._repo_dir, self._resolved_sha, ask_remote=True))
        else:
            self._resolved_sha = await self._ls_remote()
        # An empty directory reads like a missing one.
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
