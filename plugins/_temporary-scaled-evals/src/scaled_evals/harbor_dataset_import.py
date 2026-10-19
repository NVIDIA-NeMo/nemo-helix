# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable builder contexts for Harbor dataset task images.

Dataset-only runs have no user task pack to finalize.  This module creates the
minimal uploaded build context used to import a pinned upstream task image into
the target's managed image-build path. It intentionally never accepts mutable
tags.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

_DIGEST_RE = re.compile(r"^(?P<repository>[^@\s]+)@(?P<digest>sha256:[0-9a-f]{64})$")


@dataclass(frozen=True)
class HarborDatasetImageImport:
    """An immutable source identity that can be mirrored by a managed builder."""

    source_image: str
    source_repository: str
    source_digest: str

    @classmethod
    def parse(cls, source_image: str) -> HarborDatasetImageImport:
        value = source_image.strip().lower()
        match = _DIGEST_RE.fullmatch(value)
        if match is None:
            raise ValueError("dataset task image must be pinned as repository@sha256:<64 hex>")
        return cls(
            source_image=value,
            source_repository=match.group("repository"),
            source_digest=match.group("digest"),
        )


def build_image_import_context(
    import_: HarborDatasetImageImport,
    *,
    task_dir: Path | None = None,
    task_name: str | None = None,
) -> bytes:
    """Return a deterministic gzip build context that preserves the source image.

    The target builder signs or publishes the resulting image according to its
    policy. The metadata file makes the upstream identity explicit in the
    stored context and resulting build provenance.
    """
    files: dict[str, bytes] = {
        "Dockerfile": f"FROM {import_.source_image}\n".encode(),
        "scaled-evals-image-import.json": (
            json.dumps(
                {
                    "schema_version": 1,
                    "source_image": import_.source_image,
                    "source_digest": import_.source_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode(),
    }
    if task_dir is not None:
        if not task_name:
            raise ValueError("task_name is required when task_dir is provided")
        if not (task_dir / "task.toml").is_file():
            raise ValueError(f"Harbor task directory has no task.toml: {task_dir}")
        for path in sorted(task_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Harbor task contains unsupported symlink: {path}")
            if path.is_file():
                relative = path.relative_to(task_dir).as_posix()
                files[f"tasks/{task_name}/{relative}"] = path.read_bytes()
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()
