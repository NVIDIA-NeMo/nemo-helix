# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create Customizer inputs from local paths and HuggingFace ids.

Preparing a job normally means creating a fileset, uploading files, creating a
model entity, and only then writing the job JSON. This module collapses that into
one step, used by two surfaces:

- ``nemo customization --upload-model ... --upload-dataset ...`` creates the
  resources and prints the references to put in a job JSON.
- ``nemo customization <backend> submit --upload-model ... --upload-dataset ...``
  does the same and submits the job with the references already filled in.

Neither needs to know the backend. Files keep their local names, so a fileset
holds exactly what was on disk, and only ``submit`` cares where the reference
lands in the job JSON.

``--exist-ok`` follows the Files service: an existing fileset is reused **and its
files are left alone**. Editing a local dataset and re-running with
``--exist-ok`` therefore trains on the old contents. Callers are told this
explicitly, because silently training on stale data is worse than a slow upload.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.entity_naming import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_platform_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_platform_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_platform_plugin.models.types import CreateModelEntityRequest
from nemo_platform_plugin.schema import SecretRef

#: A HuggingFace repo id, e.g. ``Qwen/Qwen3-1.7B``. Two segments, no path separator beyond the first.
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_NAME_SAFE = re.compile(r"[^a-z0-9-]+")
_HYPHEN_RUN = re.compile(r"-{2,}")
_VALID_NAME = re.compile(NAME_PATTERN)
#: Suffixes worth dropping from a fileset name. A model version such as the
#: ``.7B`` in ``Qwen3-1.7B`` is not one, and dropping it would collide with ``Qwen3-1.5B``.
_DATA_SUFFIXES = frozenset({".jsonl", ".json", ".csv", ".tsv", ".parquet", ".txt", ".arrow"})


class FilesUploadClient(Protocol):
    """The Files service calls this module makes."""

    def get_fileset(self, *, workspace: str, name: str) -> Any: ...

    def create_fileset(self, *, workspace: str, body: Any, exist_ok: bool) -> Any: ...

    def upload_file(self, *, workspace: str, name: str, path: str, content: bytes) -> Any: ...


class ModelsRegistryClient(Protocol):
    """The Models service calls this module makes."""

    def create_model(self, *, workspace: str, body: Any, exist_ok: bool) -> Any: ...


class UploadError(RuntimeError):
    """Raised when a resource cannot be created from the given source."""


@dataclass(frozen=True, slots=True)
class SpecRefs:
    """Where a backend's job JSON holds the references this module creates."""

    #: Path to the model reference, e.g. ``("model",)`` or ``("model", "name")``.
    model: tuple[str, ...]
    #: Path to the training dataset reference, e.g. ``("dataset", "training")``.
    dataset: tuple[str, ...]
    #: Path to the validation dataset reference, when the backend has a separate one.
    dataset_validation: tuple[str, ...] | None = None
    #: Path to the environment reference, when the backend has one. GRPO only.
    environment: tuple[str, ...] | None = None


@dataclass(slots=True)
class UploadReport:
    """What was created and what was reused, so the caller can report it."""

    model_ref: str | None = None
    dataset_ref: str | None = None
    environment_ref: str | None = None
    created: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)

    @property
    def reused_files_untouched(self) -> bool:
        """True when a fileset was reused, so its files were not refreshed."""
        return bool(self.reused)


def fileset_name_for(source: str, *, prefix: str = "") -> str:
    """Derive a predictable fileset name from a local path or HuggingFace id.

    Only a real file's extension is dropped. A version like ``Qwen3-1.7B`` is not
    an extension, and truncating it would collide with ``Qwen3-1.5B``.

    The platform requires a name that starts with a letter and holds no repeated
    hyphen. A source such as ``2024-train.jsonl`` cannot satisfy that on its own,
    so *prefix* is prepended when the derived name would be rejected.
    """
    path = Path(source)
    stem = path.name or path.parent.name
    if path.suffix.lower() in _DATA_SUFFIXES:
        stem = path.stem
    name = _normalize(stem)
    if not _VALID_NAME.match(name) and prefix:
        name = _normalize(f"{prefix}-{name}")
    if not _VALID_NAME.match(name):
        raise UploadError(
            f"Cannot derive a valid fileset name from {source!r}. "
            f"{NAME_PATTERN_DESCRIPTION} Rename the source, or create the fileset yourself."
        )
    return name


def _normalize(value: str) -> str:
    """Lowercase, replace anything the platform disallows, and collapse hyphens."""
    return _HYPHEN_RUN.sub("-", _NAME_SAFE.sub("-", value.lower())).strip("-")


def is_huggingface_id(source: str) -> bool:
    """True when *source* looks like a HuggingFace repo id rather than a local path."""
    if Path(source).exists():
        return False
    return bool(_HF_REPO_ID.match(source))


def create_resources(
    *,
    model_source: str | None = None,
    dataset_source: str | None = None,
    environment_source: str | None = None,
    files: FilesUploadClient,
    models: ModelsRegistryClient,
    workspace: str,
    exist_ok: bool,
    hf_token_secret: str | None = None,
) -> UploadReport:
    """Create whichever resources were asked for, and report their references.

    A source is one local file or one directory, the same as ``nemo files
    upload`` takes, and a directory is uploaded recursively. To put several
    files in one fileset, pass the directory holding them.
    """
    report = UploadReport()
    if model_source is not None:
        report.model_ref = _create_model(
            model_source,
            files=files,
            models=models,
            workspace=workspace,
            exist_ok=exist_ok,
            report=report,
            hf_token_secret=hf_token_secret,
        )
    if dataset_source is not None:
        report.dataset_ref = _create_fileset_from_source(
            dataset_source,
            purpose=FilesetPurpose.DATASET,
            prefix="dataset",
            files=files,
            workspace=workspace,
            exist_ok=exist_ok,
            report=report,
            missing_hint=(
                "A dataset is uploaded from a local file or directory; convert a "
                "HuggingFace dataset to the backend's format first."
            ),
        )
    if environment_source is not None:
        report.environment_ref = _create_fileset_from_source(
            environment_source,
            purpose=FilesetPurpose.ENVIRONMENT,
            prefix="environment",
            files=files,
            workspace=workspace,
            exist_ok=exist_ok,
            report=report,
            missing_hint="An environment is uploaded from a local directory.",
        )
    return report


def write_refs(spec: dict, refs: SpecRefs, report: UploadReport) -> None:
    """Put the created references into *spec*, in place.

    One uploaded dataset fills both the training and the validation reference,
    because both splits live in the one fileset and the backend picks them apart
    by file name.
    """
    if report.model_ref is not None:
        _write_spec(spec, refs.model, report.model_ref)
    if report.dataset_ref is not None:
        _write_spec(spec, refs.dataset, report.dataset_ref)
        if refs.dataset_validation is not None:
            _write_spec(spec, refs.dataset_validation, report.dataset_ref)
    if report.environment_ref is not None and refs.environment is not None:
        _write_spec(spec, refs.environment, report.environment_ref)


def find_conflicts(
    spec: dict,
    refs: SpecRefs,
    *,
    model: bool,
    dataset: bool,
    environment: bool,
) -> list[str]:
    """Return the job JSON fields that a flag would overwrite.

    A reference given in both places is a contradiction, not an override: the
    two name different resources and only one can be right. The caller refuses
    rather than silently picking one.
    """
    checks: list[tuple[bool, tuple[str, ...] | None]] = [
        (model, refs.model),
        (dataset, refs.dataset),
        (dataset, refs.dataset_validation),
        (environment, refs.environment),
    ]
    conflicts = []
    for requested, path in checks:
        if not requested or path is None:
            continue
        existing = _read_spec(spec, path)
        if existing is not None:
            conflicts.append(f"{'.'.join(path)} ({existing})")
    return conflicts


def _read_spec(spec: dict, path: tuple[str, ...]) -> str | None:
    """Return the non-empty string at *path*, or None when it is absent."""
    node: object = spec
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, str) and node else None


def _create_model(
    source: str,
    *,
    files: FilesUploadClient,
    models: ModelsRegistryClient,
    workspace: str,
    exist_ok: bool,
    report: UploadReport,
    hf_token_secret: str | None,
) -> str:
    """Create the backing fileset and the model entity, and return the entity ref."""
    name = fileset_name_for(source, prefix="model")
    if is_huggingface_id(source):
        token = SecretRef(hf_token_secret) if hf_token_secret else None
        storage = HuggingfaceStorageConfig(repo_id=source, repo_type="model", token_secret=token)
        _create_fileset(
            files,
            name=name,
            purpose=FilesetPurpose.MODEL,
            workspace=workspace,
            exist_ok=exist_ok,
            report=report,
            storage=storage,
        )
    else:
        _create_fileset_from_source(
            source,
            purpose=FilesetPurpose.MODEL,
            prefix="model",
            files=files,
            workspace=workspace,
            exist_ok=exist_ok,
            report=report,
            missing_hint="A model source is a local path, or a HuggingFace repo id of the form 'owner/name'.",
        )

    _create_model_entity(models, name=name, fileset=f"{workspace}/{name}", workspace=workspace, exist_ok=exist_ok)
    return f"{workspace}/{name}"


def _create_fileset_from_source(
    source: str,
    *,
    purpose: FilesetPurpose,
    prefix: str,
    files: FilesUploadClient,
    workspace: str,
    exist_ok: bool,
    report: UploadReport,
    missing_hint: str,
) -> str:
    """Upload one local file or directory into a new fileset, and return its reference."""
    path = Path(source)
    if not path.exists():
        raise UploadError(f"Source {source!r} does not exist. {missing_hint}")

    name = fileset_name_for(source, prefix=prefix)
    created = _create_fileset(
        files,
        name=name,
        purpose=purpose,
        workspace=workspace,
        exist_ok=exist_ok,
        report=report,
    )
    if created:
        _upload_source(files, name=name, workspace=workspace, source=path)
    return f"{workspace}/{name}"


def _create_fileset(
    files: FilesUploadClient,
    *,
    name: str,
    purpose: FilesetPurpose,
    workspace: str,
    exist_ok: bool,
    report: UploadReport,
    storage: HuggingfaceStorageConfig | None = None,
) -> bool:
    """Create the fileset. Return True when it was created, False when reused."""
    if _fileset_exists(files, name=name, workspace=workspace):
        if not exist_ok:
            raise UploadError(
                f"Fileset {workspace}/{name} already exists. Pass --exist-ok to reuse it as it is, "
                "or delete it if you want it rebuilt from the local source."
            )
        report.reused.append(f"{workspace}/{name}")
        return False

    body = CreateFilesetRequest(name=name, purpose=purpose, storage=storage)
    files.create_fileset(workspace=workspace, body=body, exist_ok=exist_ok)
    report.created.append(f"{workspace}/{name}")
    return True


def _fileset_exists(files: FilesUploadClient, *, name: str, workspace: str) -> bool:
    """True when the fileset is already there.

    Only a 404 means absent. An auth or transport failure must surface rather
    than be read as "does not exist", which would send us on to create it.
    """
    try:
        files.get_fileset(workspace=workspace, name=name)
    except NotFoundError:
        return False
    return True


def _create_model_entity(
    models: ModelsRegistryClient,
    *,
    name: str,
    fileset: str,
    workspace: str,
    exist_ok: bool,
) -> None:
    body = CreateModelEntityRequest(name=name, fileset=fileset)
    models.create_model(workspace=workspace, body=body, exist_ok=exist_ok)


def _upload_source(files: FilesUploadClient, *, name: str, workspace: str, source: Path) -> None:
    """Upload *source* into the fileset, one request per file, keeping local names."""
    for local, remote in _files_to_upload(source):
        files.upload_file(workspace=workspace, name=name, path=remote, content=local.read_bytes())


def _files_to_upload(source: Path) -> Iterator[tuple[Path, str]]:
    """Yield each local file with the path it takes inside the fileset."""
    if source.is_file():
        yield source, source.name
        return
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        yield path, path.relative_to(source).as_posix()


def _write_spec(spec: dict, path: tuple[str, ...], value: str) -> None:
    node = spec
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
