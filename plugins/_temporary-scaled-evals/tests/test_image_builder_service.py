# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import tarfile
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build import image_builder_service


def _context(tmp_path: Path) -> Path:
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM python:3.13-slim-bookworm\n", encoding="utf-8")
    script = context / "run.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)
    return context


def test_context_hash_matches_builder_format(tmp_path: Path) -> None:
    context = _context(tmp_path)
    (context / "results").mkdir()
    (context / "results" / "included.txt").write_text("included", encoding="utf-8")

    expected = hashlib.sha256()
    for rel, mode in [
        ("Dockerfile", b"644"),
        ("results/included.txt", b"644"),
        ("run.sh", b"755"),
    ]:
        path = context / rel
        expected.update(rel.encode())
        expected.update(b"\0")
        expected.update(mode)
        expected.update(b"\0")
        expected.update(path.read_bytes())
        expected.update(b"\0")

    assert image_builder_service.compute_context_hash(context) == expected.hexdigest()


def test_archive_context_directory_is_deterministic_and_normalized(tmp_path: Path) -> None:
    context = _context(tmp_path)

    first = image_builder_service.archive_context_directory(context)
    second = image_builder_service.archive_context_directory(context)

    assert first == second


def test_archive_context_directory_can_normalize_source_dockerfile_path(tmp_path: Path) -> None:
    context = tmp_path / "server"
    context.mkdir()
    (context / ".git").write_text("gitdir: /private/tmp/not-source\n", encoding="utf-8")
    benchmark = context / "benchmark"
    benchmark.mkdir()
    source = benchmark / "server.Dockerfile"
    source.write_text("FROM python:3.13-slim-bookworm\n", encoding="utf-8")

    archive = image_builder_service.archive_context_directory(
        context,
        dockerfile_path="benchmark/server.Dockerfile",
    )

    names: set[str]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = set(tar.getnames())
        root_dockerfile = tar.extractfile("Dockerfile")
        source_dockerfile = tar.extractfile("benchmark/server.Dockerfile")
        assert root_dockerfile is not None
        assert source_dockerfile is not None
        assert root_dockerfile.read() == source_dockerfile.read() == source.read_bytes()

    assert ".git" not in names
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "context.tar.gz"
        archive_path.write_bytes(archive)
        metadata = image_builder_service.inspect_uploaded_archive_file(
            archive_path,
            dockerfile_path="benchmark/server.Dockerfile",
        )

    assert metadata.dockerfile_path == "benchmark/server.Dockerfile"
    assert metadata.dockerfile_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_archive_context_directory_prunes_venv_but_rejects_other_symlinks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    venv_bin = context / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to("/usr/bin/python3")
    (context / "keep.txt").write_text("real\n", encoding="utf-8")

    # A .venv with symlinked interpreter binaries is pruned entirely.
    archive = image_builder_service.archive_context_directory(context)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert all(not member.issym() and not member.islnk() for member in tar.getmembers())
    assert "keep.txt" in names
    assert "Dockerfile" in names
    assert not any(name.startswith(".venv") for name in names)

    # The hash walk applies the same prune, so pruning .venv is identity-neutral.
    bare_root = tmp_path / "bare"
    bare_root.mkdir()
    bare = _context(bare_root)
    (bare / "keep.txt").write_text("real\n", encoding="utf-8")
    assert image_builder_service.compute_context_hash(context) == image_builder_service.compute_context_hash(bare)

    # A symlink outside a pruned directory still fails loudly rather than being
    # silently dropped from the build context.
    (context / "link.txt").symlink_to("Dockerfile")
    with pytest.raises(image_builder_service.BuildError, match="refusing symlink"):
        image_builder_service.archive_context_directory(context)


@pytest.mark.parametrize("name", [".DS_Store", "._Dockerfile", "nested/._file"])
def test_archive_context_directory_rejects_macos_metadata(tmp_path: Path, name: str) -> None:
    context = _context(tmp_path)
    bad = context / name
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("metadata\n", encoding="utf-8")

    with pytest.raises(image_builder_service.BuildError, match="macOS metadata"):
        image_builder_service.archive_context_directory(context)


def test_resolve_uploaded_archive_rejects_macos_metadata_from_upload() -> None:
    with io.BytesIO() as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            dockerfile = b"FROM scratch\n"
            info = tarfile.TarInfo("Dockerfile")
            info.size = len(dockerfile)
            archive.addfile(info, io.BytesIO(dockerfile))
            metadata = b"metadata\n"
            bad = tarfile.TarInfo("._Dockerfile")
            bad.size = len(metadata)
            archive.addfile(bad, io.BytesIO(metadata))
        uploaded = raw.getvalue()

    with pytest.raises(image_builder_service.BuildError, match="macOS metadata"):
        image_builder_service.resolve_uploaded_archive(uploaded)
