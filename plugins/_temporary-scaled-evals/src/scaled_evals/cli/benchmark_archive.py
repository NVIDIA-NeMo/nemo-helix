# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and safely materialize a benchmark archive without partial outputs."""

import hashlib
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import click
import httpx

from scaled_evals.cli.client import download_artifact


def save_benchmark_archive(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    archive_only: bool,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-download-", dir=dest.parent) as tmp:
        archive_path = Path(tmp) / "archive.tar.gz"
        download_artifact(client, url, archive_path)
        if expected_size_bytes is not None and archive_path.stat().st_size != expected_size_bytes:
            raise click.ClickException("downloaded archive size does not match the API; retry the download")
        if expected_sha256 is not None:
            with archive_path.open("rb") as body:
                actual_sha256 = hashlib.file_digest(body, "sha256").hexdigest()
            if actual_sha256 != expected_sha256:
                raise click.ClickException("downloaded archive checksum does not match the API; retry the download")
        if archive_only:
            # Exclusive creation protects an existing destination even if it appeared mid-download.
            dest.hardlink_to(archive_path)
            return
        unpacked = Path(tmp) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or not path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise click.ClickException("archive contains an unsafe path or file type")
                archive.extract(member, unpacked, filter="data")
        roots = list(unpacked.iterdir())
        if len(roots) != 1 or not roots[0].is_dir() or not (roots[0] / "result.json").is_file():
            raise click.ClickException("archive does not contain one Harbor experiment")
        if dest.exists():
            raise click.ClickException(f"destination already exists: {dest}")
        roots[0].rename(dest)
