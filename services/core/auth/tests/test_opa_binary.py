# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from nhx.core.auth.app.embedded_pdp import opa_binary


@pytest.fixture(autouse=True)
def isolated_resolver(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(opa_binary.shutil, "which", lambda _: None)
    monkeypatch.setattr(opa_binary.platform, "system", lambda: "Linux")
    monkeypatch.setattr(opa_binary.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        opa_binary.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="Version: 1.22.0\n"),
    )


def executable(path: Path) -> Path:
    path.write_bytes(b"operator managed")
    path.chmod(0o700)
    return path


def downloads(monkeypatch, payload=b"official binary", checksum=None):
    calls = []

    def fetch(url, destination, limit, deadline):
        calls.append(url)
        if url.endswith(".sha256"):
            destination.write_text(checksum or hashlib.sha256(payload).hexdigest())
        else:
            destination.write_bytes(payload)

    monkeypatch.setattr(opa_binary, "_fetch", fetch)
    return calls


def test_explicit_path_takes_precedence(monkeypatch, tmp_path):
    binary = executable(tmp_path / "configured")
    monkeypatch.setattr(opa_binary.shutil, "which", lambda _: pytest.fail("PATH lookup"))
    assert opa_binary.resolve_opa_binary(str(binary)) == binary


def test_invalid_explicit_path_never_downloads(tmp_path):
    with pytest.raises(RuntimeError, match="not an executable"):
        opa_binary.resolve_opa_binary(str(tmp_path / "missing"))


def test_path_binary(monkeypatch, tmp_path):
    binary = executable(tmp_path / "opa")
    monkeypatch.setattr(opa_binary.shutil, "which", lambda _: str(binary))
    assert opa_binary.resolve_opa_binary() == binary


@pytest.mark.parametrize("version", ["Version: 1.8.0", "unknown", "Version: 1.22.0-dev", "Version: 2.0.0"])
def test_incompatible_explicit_path_fails(monkeypatch, tmp_path, version):
    binary = executable(tmp_path / "old-opa")
    monkeypatch.setattr(
        opa_binary.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=version)
    )
    with pytest.raises(RuntimeError, match="Incompatible OPA binary"):
        opa_binary.resolve_opa_binary(str(binary))


def test_stale_path_falls_back_to_pinned_download(monkeypatch, tmp_path):
    binary = executable(tmp_path / "old-opa")
    monkeypatch.setattr(opa_binary.shutil, "which", lambda _: str(binary))
    monkeypatch.setattr(
        opa_binary.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="Version: 1.8.0"),
    )
    calls = downloads(monkeypatch)
    assert opa_binary.resolve_opa_binary() != binary
    assert len(calls) == 2


def test_version_probe_timeout_fails_explicit(monkeypatch, tmp_path):
    binary = executable(tmp_path / "hung-opa")

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(opa_binary.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Incompatible OPA binary"):
        opa_binary.resolve_opa_binary(str(binary))


def test_download_and_offline_cache_reuse(monkeypatch):
    calls = downloads(monkeypatch)
    binary = opa_binary.resolve_opa_binary()
    assert len(calls) == 2
    assert "/v1.22.0/opa_linux_amd64_static" in calls[-1]
    assert os.access(binary, os.X_OK)
    monkeypatch.setattr(opa_binary, "_fetch", lambda *args: pytest.fail("offline cache made network call"))
    assert opa_binary.resolve_opa_binary() == binary


@pytest.mark.parametrize("corruption", ["binary", "checksum", "permissions", "symlink"])
def test_invalid_cache_reprovisioned(monkeypatch, tmp_path, corruption):
    calls = downloads(monkeypatch)
    binary = opa_binary.resolve_opa_binary()
    if corruption == "binary":
        binary.write_bytes(b"tampered")
    elif corruption == "checksum":
        binary.with_name("opa.sha256").write_text("invalid")
    elif corruption == "permissions":
        binary.chmod(0o600)
    else:
        target = executable(tmp_path / "untrusted")
        binary.unlink()
        binary.symlink_to(target)
    assert opa_binary.resolve_opa_binary().read_bytes() == b"official binary"
    assert len(calls) == 4


@pytest.mark.parametrize("checksum", ["invalid", "0" * 64])
def test_checksum_failure_publishes_nothing(monkeypatch, tmp_path, checksum):
    downloads(monkeypatch, checksum=checksum)
    with pytest.raises(RuntimeError, match="checksum|SHA256"):
        opa_binary.resolve_opa_binary()
    assert not list(tmp_path.rglob("opa.sha256"))
    assert not list(tmp_path.rglob(".download-*"))


def test_concurrent_resolvers_publish_once(monkeypatch):
    calls = downloads(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        binaries = list(pool.map(lambda _: opa_binary.resolve_opa_binary(), range(4)))
    assert len(set(binaries)) == 1
    assert len(calls) == 2
    assert binaries[0].stat().st_mode & 0o777 == 0o700
    assert binaries[0].parent.stat().st_mode & 0o777 == 0o700


def test_cache_lock_timeout(monkeypatch, tmp_path):
    with opa_binary._cache_lock(tmp_path / "locked"):
        monkeypatch.setattr(opa_binary, "_DOWNLOAD_TIMEOUT", 0)
        with pytest.raises(TimeoutError, match="cache lock"):
            with opa_binary._cache_lock(tmp_path / "locked"):
                pytest.fail("acquired competing lock")


def test_cache_directory_symlink_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory owned"):
        with opa_binary._cache_lock(link):
            pytest.fail("accepted symlink cache")


def test_network_failure_has_offline_guidance(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError("network blocked")

    monkeypatch.setattr(opa_binary, "_fetch", fail)
    with pytest.raises(RuntimeError, match="network blocked.*auth.opa_binary_path"):
        opa_binary.resolve_opa_binary()
    assert not list(tmp_path.rglob(".download-*"))


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "aarch64", "opa_linux_arm64_static"),
        ("Darwin", "arm64", "opa_darwin_arm64"),
        ("Darwin", "x86_64", "opa_darwin_amd64"),
    ],
)
def test_platform_assets(monkeypatch, system, machine, expected):
    monkeypatch.setattr(opa_binary.platform, "system", lambda: system)
    monkeypatch.setattr(opa_binary.platform, "machine", lambda: machine)
    assert opa_binary._asset_name() == expected


def test_unsupported_platform_requires_provisioning(monkeypatch, tmp_path):
    monkeypatch.setattr(opa_binary.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="No automatic OPA download"):
        opa_binary.resolve_opa_binary()
    binary = executable(tmp_path / "opa")
    assert opa_binary.resolve_opa_binary(str(binary)) == binary


class Response(io.BytesIO):
    def geturl(self):
        return "https://release-assets.githubusercontent.com/opa"


def test_fetch_size_limit_and_timeout_parameter(tmp_path):
    with patch.object(opa_binary.urllib.request, "urlopen", return_value=Response(b"large")) as open_url:
        with pytest.raises(RuntimeError, match="size limit"):
            opa_binary._fetch("https://example.com/opa", tmp_path / "file", 2, float("inf"))
        assert open_url.call_args.kwargs["timeout"] == 10


def test_fetch_deadline(tmp_path):
    with patch.object(opa_binary.urllib.request, "urlopen", return_value=Response(b"binary")):
        with pytest.raises(TimeoutError, match="time budget"):
            opa_binary._fetch("https://example.com/opa", tmp_path / "file", 100, 0)


def test_fetch_rejects_insecure_redirect(tmp_path, monkeypatch):
    response = Response(b"binary")
    monkeypatch.setattr(response, "geturl", lambda: "http://example.com/opa")
    with patch.object(opa_binary.urllib.request, "urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="insecure"):
            opa_binary._fetch("https://example.com/opa", tmp_path / "file", 100, float("inf"))
