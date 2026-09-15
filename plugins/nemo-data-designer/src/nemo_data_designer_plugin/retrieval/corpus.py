# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from data_designer_nemo.filesystem import make_filesystem
from filesets import FilesetPathError, build_fileset_ref, parse_fileset_ref
from nemo_platform import NeMoPlatform

_HF_PREFIX = "hf://"

#: Env var the job steps populate from the spec's ``hf_token_secret`` reference.
HF_TOKEN_ENVVAR = "HF_TOKEN"  # pragma: allowlist secret


def hf_token_from_env() -> str | None:
    """Read the Hugging Face token a job step received from its secret reference."""
    return os.environ.get(HF_TOKEN_ENVVAR) or None


def materialize_corpus(
    corpus: str,
    dest: Path,
    sdk: NeMoPlatform,
    workspace: str,
    hf_token: str | None = None,
    allow_local_path: bool = False,
) -> Path:
    """Stage a fileset or Hugging Face corpus into the job-owned destination.

    Local filesystem paths are only accepted when ``allow_local_path`` is true,
    which retrieval-preview uses for in-process no-upload loops.
    """
    if corpus.startswith(_HF_PREFIX):
        return _download_hf(corpus, dest, token=hf_token)

    if allow_local_path:
        local = Path(corpus)
        if local.exists():
            return local.resolve()

    return _download_fileset(corpus, dest=dest, sdk=sdk, workspace=workspace)


def _validated_hf_subdir(subdir: str) -> str:
    """Keep Hugging Face URI suffixes inside the snapshot staging directory."""
    posix = PurePosixPath(subdir.replace("\\", "/"))
    if posix.is_absolute() or any(part == ".." for part in posix.parts):
        raise ValueError(f"Invalid hf:// URI subdirectory {subdir!r}: must be a relative path without '..' components")
    return posix.as_posix()


def _download_hf(corpus: str, dest: Path, token: str | None) -> Path:
    rest = corpus[len(_HF_PREFIX) :]
    parts = rest.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid hf:// URI: {corpus}. Expected hf://org/dataset[@revision][/subdir]")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision = None
    if "@" in parts[1]:
        dataset_name, revision = parts[1].rsplit("@", 1)
        repo_id = f"{parts[0]}/{dataset_name}"
    subdir = _validated_hf_subdir(parts[2]) if len(parts) > 2 and parts[2] else None

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("hf:// corpus staging requires nemo-data-designer-plugin[retrieval-sdg].") from exc

    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            revision=revision,
            allow_patterns=f"{subdir}/**" if subdir else None,
            token=token,
        )
    )
    return local_dir / subdir if subdir else local_dir


def _download_fileset(corpus: str, dest: Path, sdk: NeMoPlatform, workspace: str) -> Path:
    try:
        fileset_workspace, fileset, fragment = parse_fileset_ref(corpus, workspace_fallback=workspace)
    except FilesetPathError as exc:
        raise ValueError(f"Invalid corpus reference {corpus!r}: {exc}") from exc

    if fileset_workspace != workspace:
        raise ValueError(f"Corpus fileset workspace {fileset_workspace!r} does not match job workspace {workspace!r}")

    root = build_fileset_ref(fragment, workspace=fileset_workspace, fileset=fileset)
    dest.mkdir(parents=True, exist_ok=True)
    fs = make_filesystem(sdk)
    fs.get(root, str(dest), recursive=True)
    return dest
