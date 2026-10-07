# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git storage backend for repositories read over SSH.

A fileset is pinned to a commit. The first read materializes that commit into
the files service's durable storage: each file's bytes under its git blob id,
then a listing of paths, blob ids and sizes. Every later read uses only those,
so no git runs on the read path. A per-URL bare repository makes later fetches
incremental; it can be deleted at any time and only costs a full fetch.
"""

from __future__ import annotations

import asyncio
import base64
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
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from nemo_helix_plugin.config import nhx_user_data_dir
from nemo_helix_plugin.files.storage_config import SshRemote, is_commit_sha
from nemo_helix_plugin.files.types import SshHostKey
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
    StorageUnavailableError,
)

logger = logging.getLogger(__name__)

_repo_locks: dict[str, asyncio.Lock] = {}

# git's own "infinite" depth, which deepens a shallow repository to the full history of a ref.
_FULL_HISTORY = 2147483647

# Bounds the git and ssh processes one files service runs at once, across every request.
_GIT_SLOTS = asyncio.Semaphore(16)


class GitBackendError(StorageBackendError):
    """Raised when git fails for a reason not covered below."""


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


def _failure_detail(stderr: str) -> str:
    """The first stderr line that names the cause, skipping git's generic closing advice."""
    for line in stderr.splitlines():
        line = line.strip()
        if line and not line.startswith(_GIT_BOILERPLATE):
            return line
    return "no output"


def _classify_failure(stderr: str, subject: str) -> StorageBackendError:
    detail = _failure_detail(stderr)
    if "Host key verification failed" in stderr or "REMOTE HOST IDENTIFICATION HAS CHANGED" in stderr:
        return GitAccessError(f"The host key for {subject} does not match known_hosts")
    if "Permission denied (publickey" in stderr or "Load key" in stderr:
        return GitAccessError(f"The SSH key was rejected for {subject}: {detail}")
    if any(marker in stderr for marker in ("Could not resolve hostname", "Connection refused", "Connection timed out")):
        return GitUnavailableError(f"Could not reach {subject}: {detail}")
    if any(
        marker in stderr
        for marker in (
            "does not appear to be a git repository",
            "Repository not found",
            "not our ref",
            "unadvertised object",
            "couldn't find remote ref",
            "Not a valid object name",
            "not a tree object",
        )
    ):
        return GitConfigError(f"{subject} does not exist, or the key cannot see it: {detail}")
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


_KEY_TYPE_PREFERENCE = ("ssh-ed25519", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521", "ssh-rsa")


def ssh_fingerprint(key_base64: str) -> str:
    digest = hashlib.sha256(base64.b64decode(key_base64)).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def parse_keyscan_output(output: str) -> list[SshHostKey]:
    keys: list[SshHostKey] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3 or line.startswith("#"):
            continue
        _, key_type, key = fields[:3]
        keys.append(
            SshHostKey(key_type=key_type, fingerprint=ssh_fingerprint(key), known_hosts_line=" ".join(fields[:3]))
        )
    rank = {key_type: index for index, key_type in enumerate(_KEY_TYPE_PREFERENCE)}
    return sorted(keys, key=lambda host_key: rank.get(host_key.key_type, len(rank)))


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


async def scan_host_keys(remote: SshRemote, timeout_seconds: int = 10) -> list[SshHostKey]:
    """Fetch the host's public keys for the user to confirm, as ssh does on a first connection."""
    port = ["-p", str(remote.port)] if remote.port else []
    subject = remote.host_url.removeprefix("ssh://")
    try:
        proc = await asyncio.create_subprocess_exec(
            "ssh-keyscan",
            "-T",
            str(timeout_seconds),
            *port,
            remote.host,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise GitBackendError("ssh-keyscan is not installed in the files service") from exc
    try:
        stdout, _ = await communicate_within(proc, timeout_seconds + 5)
    except TimeoutError:
        raise GitUnavailableError(f"Timed out scanning host keys of {subject}") from None

    keys = parse_keyscan_output(stdout.decode(errors="replace"))
    if not keys:
        raise GitUnavailableError(f"{subject} returned no SSH host keys; check the host and port")
    return keys


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    blob: str
    size: int


def _tree_entries(ls_tree_output: bytes) -> Iterator[_TreeEntry]:
    """Regular-file entries of ``git ls-tree -r -l -z`` output, skipping names that are not UTF-8."""
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
    """The files service's default storage, where materialized commits live alongside other backends' caches."""
    from nhx.core.files.app.backends.factory import storage_impl_factory
    from nhx.core.files.config import files_config

    return storage_impl_factory(files_config().default_storage_config, {})


_DURABLE_PREFIX = "cache/git"
_LISTING_VERSION = 1
_FETCH_REPOSITORY_IDLE_SECONDS = 7 * 24 * 60 * 60
_LATEST_REF = "refs/nhx/latest"

# A path -> (blob id, size) listing of one commit, scoped to a fileset's directory.
Snapshot = dict[str, tuple[str, int]]


def _blob_key(blob: str) -> str:
    return f"{_DURABLE_PREFIX}/blobs/{blob[:2]}/{blob}"


async def _read_exactly(stream: asyncio.StreamReader, size: int, chunk_size: int) -> AsyncIterator[bytes]:
    remaining = size
    while remaining:
        chunk = await stream.read(min(chunk_size, remaining))
        if not chunk:
            raise GitBackendError("git ended a blob early while copying it to durable storage")
        remaining -= len(chunk)
        yield chunk


async def _single_chunk(body: bytes) -> AsyncIterator[bytes]:
    yield body


async def _prune_fetch_repositories(root: Path, keep: Path) -> None:
    """Remove fetch repositories idle for a week, each under its own lock so no fetch is using it."""
    cutoff = time.time() - _FETCH_REPOSITORY_IDLE_SECONDS
    with suppress(OSError):
        candidates = [entry for entry in await asyncio.to_thread(lambda: list(root.iterdir())) if entry != keep]
        for entry in candidates:
            lock = _repo_locks.setdefault(str(entry), asyncio.Lock())
            if lock.locked():
                continue
            async with lock:
                with suppress(OSError):
                    if entry.is_dir() and entry.stat().st_mtime < cutoff:
                        await asyncio.to_thread(shutil.rmtree, entry, True)


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
        return f"{_DURABLE_PREFIX}/commits/{self._url_digest}/{sha}/{scope}.json"

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
            return await self._run_now(*args, env=env)

    async def _run_now(self, *args: str, env: dict[str, str]) -> tuple[int, bytes, str]:
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
            raise GitBackendError("git is not installed in the files service") from exc
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

    async def _ls_remote(self) -> str:
        """Resolve the revision to a SHA, which also proves the key can read the repository."""
        revision = self.config.revision
        pinned = is_commit_sha(revision)
        # A bare tag pattern does not match the peeled `^{}` line that names an annotated tag's commit.
        patterns = ["HEAD"] if pinned else [revision, f"{revision}^{{}}"]
        with self._ssh_env() as env:
            output = await self._git("ls-remote", "--", self.config.url, *patterns, env=env, subject=self.config.url)
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
        with self._ssh_env() as env:
            await self._git(
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
                env=env,
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

    async def resolve_config(self) -> GitStorageConfig:
        """Pin the revision to a commit SHA so the fileset cannot shift under a deployment."""
        sha = await self._commit()
        return self.config.model_copy(update={"revision": sha, "original_revision": self.config.revision})

    async def _snapshot(self) -> Snapshot:
        """The pinned commit's listing, materializing the commit into durable storage on first use."""
        if self._snapshot_memo is None:
            sha = await self._commit()
            key = self._listing_key(sha)
            snapshot = await self._read_listing(key)
            if snapshot is None:
                snapshot = await self._materialize(sha, key)
            self._snapshot_memo = snapshot
        return self._snapshot_memo

    async def _read_listing(self, key: str) -> Snapshot | None:
        """The stored listing at *key*, or None when it is missing or unreadable and must be rebuilt."""
        try:
            stream = await self._store.download(key, None)
            raw = b"".join([chunk async for chunk in stream])
        except NotFoundError:
            return None
        try:
            listing = json.loads(raw)
            if listing.get("version") != _LISTING_VERSION:
                return None
            return {path: (blob, int(size)) for path, blob, size in listing["files"]}
        except (ValueError, TypeError, KeyError, AttributeError):
            logger.warning("Rebuilding an unreadable git listing at %s", key)
            return None

    async def _materialize(self, sha: str, key: str) -> Snapshot:
        repo = self._repo_dir
        async with _repo_locks.setdefault(str(repo), asyncio.Lock()):
            # Another request may have materialized this commit while this one waited for the lock.
            snapshot = await self._read_listing(key)
            if snapshot is not None:
                return snapshot
            await self._fetch_into(repo, sha)
            tree = f"{sha}:{self.config.path}" if self.config.path else sha
            output = await self._git(
                "-C", str(repo), "ls-tree", "-r", "-l", "-z", tree, env=self._base_env(), subject=self._subject()
            )
            entries = list(_tree_entries(output))
            await self._store_blobs(repo, entries)
            snapshot = {entry.path: (entry.blob, entry.size) for entry in entries}
            listing = {
                "version": _LISTING_VERSION,
                "commit": sha,
                "files": [[path, blob, size] for path, (blob, size) in sorted(snapshot.items())],
            }
            body = json.dumps(listing).encode()
            # Written last, so a listing only ever names blobs that are already stored.
            await self._store.upload(key, _single_chunk(body), content_length=len(body))
            with suppress(OSError):
                await asyncio.to_thread(os.utime, repo)
        await _prune_fetch_repositories(repo.parent, repo)
        return snapshot

    async def _fetch_into(self, repo: Path, sha: str) -> None:
        """Make *sha* available in the fetch repository, starting it over once if it is broken."""
        for attempt in (1, 2):
            try:
                if not await asyncio.to_thread((repo / "HEAD").exists):
                    try:
                        await asyncio.to_thread(repo.parent.mkdir, parents=True, exist_ok=True)
                    except OSError as exc:
                        raise GitBackendError(f"The git cache at {repo.parent} is not writable: {exc}") from exc
                    await self._git("init", "--bare", "--quiet", str(repo), env=self._base_env(), subject=str(repo))
                if not await self._has_commit(repo, sha):
                    await self._fetch_with_fallback(repo, sha)
                    if not await self._has_commit(repo, sha):
                        raise GitConfigError(f"{self.config.url} has no commit {sha}")
                # Keeps the newest commit's objects, so the next fetch of this repository is a delta.
                await self._git(
                    "-C", str(repo), "update-ref", _LATEST_REF, sha, env=self._base_env(), subject=str(repo)
                )
                return
            except GitBackendError:
                # A lock left by a killed git or a half-deleted directory: the repository is only an accelerator.
                if attempt == 2:
                    raise
                logger.warning("Starting the git fetch repository for %s over", self.config.url)
                await asyncio.to_thread(shutil.rmtree, repo, True)

    async def _store_blobs(self, repo: Path, entries: list[_TreeEntry]) -> None:
        """Copy the blobs durable storage lacks, so an update writes only the files that changed."""
        missing: dict[str, int] = {}
        for entry in entries:
            if entry.blob in missing:
                continue
            try:
                await self._store.get_file(_blob_key(entry.blob))
            except NotFoundError:
                missing[entry.blob] = entry.size
        if not missing:
            return

        async with _GIT_SLOTS:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(repo),
                "cat-file",
                "--batch",
                env=self._base_env(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            assert proc.stdin is not None and proc.stdout is not None
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    for blob in missing:
                        proc.stdin.write(f"{blob}\n".encode())
                        await proc.stdin.drain()
                        header = (await proc.stdout.readline()).split()
                        if len(header) != 3 or header[1] != b"blob":
                            raise GitBackendError(f"git could not read blob {blob} of {self._subject()}")
                        size = int(header[2])
                        await self._store.upload(
                            _blob_key(blob),
                            _read_exactly(proc.stdout, size, self.config.read_chunk_size),
                            content_length=size,
                        )
                        await proc.stdout.readexactly(1)
            except TimeoutError:
                raise GitUnavailableError(
                    f"Copying {self._subject()} to durable storage took over {self.timeout_seconds:.0f}s"
                ) from None
            finally:
                if proc.returncode is None:
                    await asyncio.shield(stop_process_group(proc))

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
        wanted = path.strip("/")
        if not wanted or any(segment in ("", ".", "..") for segment in wanted.split("/")):
            raise NotFoundError(f"File not found for path: {path}")
        entry = (await self._snapshot()).get(wanted)
        if entry is None:
            raise NotFoundError(f"File not found for path: {path}")
        return FileInfo(path=wanted, size=entry[1])

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        await self.get_file(path)
        blob, _size = (await self._snapshot())[path.strip("/")]
        return await self._store.download(_blob_key(blob), byte_range)

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
