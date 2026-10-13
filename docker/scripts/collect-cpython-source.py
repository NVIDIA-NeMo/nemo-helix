#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort CPython source archive collector for official Python base images."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReleaseFilesParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[list[str], list[str]]] = []
        self._in_row = False
        self._in_code = False
        self._links: list[str] = []
        self._checksums: list[str] = []
        self._code_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._in_row = True
            self._links = []
            self._checksums = []
            return
        if not self._in_row:
            return
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._links.append(href)
        elif tag == "code":
            self._in_code = True
            self._code_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_code:
            self._code_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "code" and self._in_code:
            self._checksums.append("".join(self._code_chunks))
            self._in_code = False
            self._code_chunks = []
        elif tag == "tr" and self._in_row:
            self.rows.append((self._links, self._checksums))
            self._in_row = False


def url_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    return PurePosixPath(urllib.parse.unquote(path)).name


def cpython_version_from_filename(filename: str) -> str:
    for suffix in (".tar.xz", ".tgz"):
        if filename.startswith("Python-") and filename.endswith(suffix):
            return filename.removeprefix("Python-").removesuffix(suffix)
    raise RuntimeError(f"could not infer CPython version from archive filename: {filename}")


def release_page_url(version: str) -> str:
    return f"https://www.python.org/downloads/release/python-{version.replace('.', '')}/"


def response_body(response) -> bytes:
    data = response.read()
    encoding = response.headers.get("Content-Encoding", "").lower()
    if encoding in {"", "identity"}:
        return data
    if encoding == "gzip":
        return gzip.decompress(data)
    if encoding == "deflate":
        return zlib.decompress(data)
    raise RuntimeError(f"unsupported content encoding: {encoding}")


def release_page_sha256(version: str, filename: str) -> str:
    metadata_url = release_page_url(version)
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        parser = ReleaseFilesParser()
        parser.feed(response_body(response).decode("utf-8", errors="replace"))

    for links, checksums in parser.rows:
        if not any(url_filename(link) == filename for link in links):
            continue
        for checksum in checksums:
            normalized = "".join(checksum.split()).lower()
            if SHA256_PATTERN.fullmatch(normalized):
                return normalized
    raise RuntimeError(f"could not find SHA-256 checksum for {filename} on {metadata_url}")


def verify_sha256(data: bytes, expected: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected.lower():
        raise RuntimeError(f"sha256 mismatch: expected {expected}, got {actual}")


def source_collection_enabled() -> bool:
    return os.environ.get("NHX_COLLECT_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}


def record_source_collection_disabled(output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "source-collection-disabled.txt").write_text(
        "source collection disabled; set NHX_COLLECT_SOURCES=1 to enable\n",
        encoding="utf-8",
    )


def python_version(python: str) -> str:
    return subprocess.check_output(
        [python, "-c", "import platform; print(platform.python_version())"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    filename = url_filename(url)
    expected_sha256 = release_page_sha256(cpython_version_from_filename(filename), filename)
    if destination.is_file() and destination.stat().st_size > 0:
        verify_sha256(destination.read_bytes(), expected_sha256)
        return
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            verify_sha256(data, expected_sha256)
            destination.write_bytes(data)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to identify")
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    version = python_version(args.python)
    filename = f"Python-{version}.tgz"
    url = f"https://www.python.org/ftp/python/{version}/{filename}"
    download(url, output_dir / "cpython" / filename)

    (manifests / "downloaded-cpython-source.txt").write_text(f"{filename}\t{url}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
