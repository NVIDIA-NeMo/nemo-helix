# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-cpython-source.py"


def load_module():
    spec = spec_from_file_location("collect_cpython_source", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_failures_are_not_swallowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()

    def fail_download(url: str, destination: Path) -> None:
        raise RuntimeError("download failed")

    monkeypatch.setattr(module, "python_version", lambda python: "3.12.11")
    monkeypatch.setattr(module, "download", fail_download)
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--python", "/fake/python", "--output", str(tmp_path)],
    )

    with pytest.raises(RuntimeError, match="download failed"):
        module.main()

    assert not (tmp_path / "manifests" / "missing-cpython-source.txt").exists()


def test_download_verifies_release_checksum_before_writing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    archive_url = "https://www.python.org/ftp/python/3.12.13/Python-3.12.13.tgz"
    destination = tmp_path / "Python-3.12.13.tgz"
    valid_archive = b"valid archive"
    tampered_archive = b"tampered archive"
    checksum = hashlib.sha256(valid_archive).hexdigest()
    release_page = f"""
        <table>
            <tr>
                <td><a href="{archive_url}">Download Gzipped source tarball</a></td>
                <td><code>{checksum[:32]} {checksum[32:]}</code></td>
            </tr>
        </table>
    """.encode()
    archive_attempts = 0

    class Response:
        def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.body = body
            self.headers = headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(url: str, timeout: int) -> Response:
        nonlocal archive_attempts
        if url == "https://www.python.org/downloads/release/python-31213/":
            return Response(gzip.compress(release_page), {"Content-Encoding": "gzip"})
        if url == archive_url:
            archive_attempts += 1
            if archive_attempts == 2:
                assert not destination.exists()
            return Response(tampered_archive if archive_attempts == 1 else valid_archive)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    module.download(archive_url, destination)

    assert archive_attempts == 2
    assert destination.read_bytes() == valid_archive
