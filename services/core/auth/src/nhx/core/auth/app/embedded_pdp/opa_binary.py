# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision native OPA without putting platform binaries in a universal wheel.

Explicit paths and PATH are operator-managed and trusted. Downloaded binaries
are checked against the checksum published with the pinned official release.
The checksum is cached alongside the asset for offline reuse (not a signature).
"""

import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

OPA_VERSION = "1.22.0"
_RELEASE_URL = f"https://github.com/open-policy-agent/opa/releases/download/v{OPA_VERSION}"
_MAX_BINARY_BYTES = 256 * 1024 * 1024
_DOWNLOAD_TIMEOUT = 60.0


def _asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if system not in {"Linux", "Darwin"} or arch is None:
        raise RuntimeError(f"No automatic OPA download for {system}/{machine}; configure auth.opa_binary_path")
    suffix = "_static" if system == "Linux" else ""
    return f"opa_{system.lower()}_{arch}{suffix}"


def _executable(path: Path) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"OPA binary is not an executable file: {path}")
    return path.resolve()


def _compatible(path: Path) -> Path:
    path = _executable(path)
    try:
        result = subprocess.run([str(path), "version"], capture_output=True, text=True, timeout=5, check=True)
        version = re.search(r"^Version:\s+(\d+)\.(\d+)\.(\d+)\s*$", result.stdout, re.MULTILINE)
        if version is None or not (1, 22, 0) <= tuple(map(int, version.groups())) < (2, 0, 0):
            raise ValueError("OPA >=1.22.0,<2 is required for validated managed runtime flags")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Incompatible OPA binary {path}: {exc}") from exc
    return path


def _checksum(value: str) -> str:
    # Official release checksum files may contain either a digest or sha256sum output.
    digest = value.split()[0] if value.split() else ""
    if not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise RuntimeError("Invalid official OPA SHA256 checksum")
    return digest.lower()


def _fetch(url: str, destination: Path, limit: int, deadline: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nemo-helix-opa-resolver"})
    with urllib.request.urlopen(request, timeout=10) as response, destination.open("wb") as output:
        if not response.geturl().startswith("https://"):
            raise RuntimeError("OPA download redirected to an insecure URL")
        size = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("OPA download exceeded its time budget")
            # read1 performs at most one buffered/raw read. Unlike read(n),
            # it cannot wait indefinitely for n bytes from a trickling peer.
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise RuntimeError("OPA download exceeds size limit")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _cached(binary: Path, checksum: Path) -> bool:
    if binary.is_symlink() or checksum.is_symlink():
        return False
    try:
        return (
            binary.is_file()
            and binary.stat().st_size <= _MAX_BINARY_BYTES
            and checksum.stat().st_size <= 1024
            and _digest(binary) == _checksum(checksum.read_text())
            and os.access(binary, os.X_OK)
        )
    except (OSError, ValueError, RuntimeError):
        return False


@contextmanager
def _cache_lock(directory: Path):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"OPA cache must be a directory owned by the current user: {directory}")
    directory.chmod(0o700)
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise RuntimeError("OPA cache lock must be a private regular file")
        os.fchmod(fd, 0o600)
        deadline = time.monotonic() + _DOWNLOAD_TIMEOUT
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for OPA cache lock") from None
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)


def resolve_opa_binary(binary_path: str | None = None) -> Path:
    """Return trusted provisioned OPA, validated cached OPA, or download OPA.

    A configured path fails closed rather than silently downloading a replacement.
    Compatible (>=1.22.0,<2) PATH binaries precede the private per-user release cache.
    Unsupported hosts must supply an executable explicitly. Network/provisioning
    failures are startup errors with offline provisioning guidance.
    """
    if binary_path is not None:
        return _compatible(Path(binary_path).expanduser())
    if provisioned := shutil.which("opa"):
        try:
            return _compatible(Path(provisioned))
        except RuntimeError:
            # A stale development PATH must not prevent pinned provisioning.
            pass

    asset = _asset_name()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    directory = cache_home / "nemo-helix" / "opa" / OPA_VERSION / asset
    binary = directory / "opa"
    checksum = directory / "opa.sha256"
    try:
        with _cache_lock(directory):
            return _resolve_cached(binary, checksum, asset)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot provision official OPA v{OPA_VERSION} ({asset}): {exc}. "
            "For offline/restricted hosts, provision OPA on PATH or set auth.opa_binary_path."
        ) from exc


def _resolve_cached(binary: Path, checksum: Path, asset: str) -> Path:
    if _cached(binary, checksum):
        return binary
    # Staging and publication share a filesystem. The cache lock also protects
    # the binary/checksum pair from being observed between the two replaces.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=binary.parent) as staging:
        staged_binary = Path(staging) / "opa"
        staged_checksum = Path(staging) / "opa.sha256"
        deadline = time.monotonic() + _DOWNLOAD_TIMEOUT
        _fetch(f"{_RELEASE_URL}/{asset}.sha256", staged_checksum, 1024, deadline)
        expected = _checksum(staged_checksum.read_text())
        _fetch(f"{_RELEASE_URL}/{asset}", staged_binary, _MAX_BINARY_BYTES, deadline)
        if _digest(staged_binary) != expected:
            raise RuntimeError("Official OPA binary SHA256 mismatch")
        staged_binary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        staged_checksum.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.replace(staged_checksum, checksum)
        os.replace(staged_binary, binary)
    return _executable(binary)
