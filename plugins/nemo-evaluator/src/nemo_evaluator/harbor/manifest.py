# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned manifest codec and complete local-tree inspection for stored Harbor tasks."""

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from nemo_evaluator.api.task_definitions.harbor import TreeDigest, validate_tree_path
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

# Per-task ceilings; downloads enforce actual sizes as well as these declarations.
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 100_000
MAX_FILE_BYTES = 4 * 1024**3
MAX_TREE_BYTES = 16 * 1024**3
DOWNLOAD_CONCURRENCY = 8
CHUNK_BYTES = 1024 * 1024
TreePath = Annotated[str, AfterValidator(validate_tree_path)]


class TreeFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["file"] = "file"
    path: TreePath
    sha256: TreeDigest
    size: int = Field(ge=0, le=MAX_FILE_BYTES, strict=True)
    executable: int = Field(ge=0, le=0o111, strict=True)

    @field_validator("executable")
    @classmethod
    def _execute_bits(cls, value: int) -> int:
        if value & ~0o111:
            raise ValueError("Only ordinary executable bits are supported")
        return value


class TreeDirectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["directory"] = "directory"
    path: TreePath


class HarborTreeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["fileset-tree-v1"] = "fileset-tree-v1"
    task_dir: TreePath
    entries: tuple[Annotated[TreeFile | TreeDirectory, Field(discriminator="kind")], ...] = Field(
        min_length=1, max_length=MAX_ENTRIES
    )

    @field_validator("task_dir")
    @classmethod
    def _folder(cls, value: str) -> str:
        if "/" in value or value.casefold() == "task_template":
            raise ValueError("Invalid or reserved task folder")
        return value

    @model_validator(mode="after")
    def _membership(self) -> Self:
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths) or len(set(path.casefold() for path in paths)) != len(paths):
            raise ValueError("Manifest paths must be sorted and unique, including case-folded aliases")
        directories = {entry.path for entry in self.entries if isinstance(entry, TreeDirectory)}
        for entry in self.entries:
            if any(
                parent.as_posix() not in directories for parent in PurePosixPath(entry.path).parents if parent.parts
            ):
                raise ValueError(f"Missing directory or file ancestor: {entry.path}")
        if sum(entry.size for entry in self.entries if isinstance(entry, TreeFile)) > MAX_TREE_BYTES:
            raise ValueError("Tree exceeds total byte limit")
        return self

    def to_bytes(self) -> bytes:
        data = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest exceeds byte limit")
        return data

    @classmethod
    def from_bytes(cls, data: bytes, expected_digest: str) -> Self:
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest exceeds byte limit")
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise ValueError("Manifest checksum mismatch")
        result = cls.model_validate_json(data)
        if result.to_bytes() != data:
            raise ValueError("Manifest encoding must be canonical")
        return result


def inspect_tree(root: Path, *, task_dir: str | None = None) -> HarborTreeManifest:
    """Inspect a quiescent tree without following links or applying ignore rules."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Task root must be a real directory")
    entries: list[TreeFile | TreeDirectory] = []
    total = 0

    def visit(directory: Path) -> None:
        nonlocal total
        for path in sorted(directory.iterdir()):
            if len(entries) >= MAX_ENTRIES:
                raise ValueError("Tree exceeds entry limit")
            relative = validate_tree_path(path.relative_to(root).as_posix())
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                entries.append(TreeDirectory(path=relative))
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                if info.st_size > MAX_FILE_BYTES or total > MAX_TREE_BYTES:
                    raise ValueError("Tree exceeds byte limit")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                entries.append(
                    TreeFile(path=relative, sha256=digest, size=info.st_size, executable=info.st_mode & 0o111)
                )
            else:
                raise ValueError(f"Unsupported tree entry (links and special files are forbidden): {relative}")

    visit(root)
    return HarborTreeManifest(
        task_dir=task_dir or root.name, entries=tuple(sorted(entries, key=lambda item: item.path))
    )
