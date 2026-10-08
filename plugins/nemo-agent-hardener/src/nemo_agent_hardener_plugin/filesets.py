# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download + safely expand an uploaded NAT project bundle stored as a platform fileset.

An uploaded project is stored as a single-file fileset (one zip). The inspect endpoint, manifest
creation, and the war-game job all need the project on local disk, so this module downloads the whole
fileset and expands the zip with hardening (no absolute members, no symlinks, no traversal, bounded
size/entry count) — the archive is untrusted user input and is never executed here (only statically
scanned by ``agent-hardener inspect`` and later run inside the OpenShell sandbox).
"""

from __future__ import annotations

import logging
import stat
import subprocess
import tempfile
import uuid
import zipfile
from fnmatch import fnmatch
from pathlib import Path

from filesets import FilesetFileSystem
from nemo_agents_plugin.container.template import DOCKERIGNORE_TEMPLATE
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 10_000
_MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500 MB expanded — a NAT project, not a dataset.
_IGNORED_TOP_LEVEL = frozenset({"__MACOSX"})

# "What must not leave a project directory" is the same question `nemo agents package` answers for its
# Docker build context, so reuse its answer rather than keeping a second, weaker list in sync — it
# already covers credentials (.env, *.pem, *.key) alongside the usual caches and virtualenvs.
_UPLOAD_EXCLUDES = tuple(
    line.strip() for line in DOCKERIGNORE_TEMPLATE.splitlines() if line.strip() and not line.startswith("#")
)


def _is_absolute_member(name: str) -> bool:
    """Return whether a zip member name is an absolute path (POSIX, Windows, or drive-letter)."""
    return name.startswith(("/", "\\")) or (len(name) >= 2 and name[1] == ":")


def download_fileset(sdk: NeMoPlatform, ref: str, dest: Path) -> Path:
    """Download an entire fileset (all files) into *dest* using the sync platform SDK.

    Whole-fileset download only — Agent Hardener stores the project as one zip, so there is no
    fragment/glob handling (unlike the evaluator's dataset downloader).
    """
    fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
    dest.mkdir(parents=True, exist_ok=True)
    source = ref.rstrip("/") + "/"
    fs.get(source, str(dest), recursive=True)
    return dest


def upload_file_to_fileset(sdk: NeMoPlatform, local_path: Path, *, workspace: str, prefix: str = "hitlog") -> str:
    """Upload a single file into a freshly-created fileset and return its ``workspace/name`` ref.

    Used to persist a war-game's produced garak hitlog so a later run can replay it: platform
    persistent job storage is per-job, so the hitlog must live in a fileset to survive across runs.

    ``prefix`` names the fileset for what it holds. The ref is all a later reader sees, so a project
    bundle or a benign suite carrying a ``hitlog-`` name reads as the wrong artifact entirely.
    """
    files = client_from_platform(sdk, FilesClient)
    fileset_name = f"{prefix}-{uuid.uuid4().hex[:8]}"
    files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset_name))
    files.upload_file(
        name=fileset_name,
        workspace=workspace,
        path=local_path.name,
        content=local_path.read_bytes(),
    )
    return f"{workspace}/{fileset_name}"


def _is_excluded(relative_path: Path) -> bool:
    """Return whether *relative_path* matches an exclude pattern (``dir/`` forms match any segment)."""
    parts = relative_path.parts
    for pattern in _UPLOAD_EXCLUDES:
        if pattern.endswith("/"):
            if pattern.rstrip("/") in parts:
                return True
        elif any(fnmatch(part, pattern) for part in parts):
            return True
    return False


def delete_fileset(sdk: NeMoPlatform, ref: str) -> None:
    """Delete a fileset by ``workspace/name`` ref; never raises.

    Called when a manifest is deleted so its victim bundle doesn't outlive it. Best-effort by
    design: a manifest the user asked to delete should go even if its bundle is already gone.
    """
    workspace, _, name = ref.partition("/")
    if not workspace or not name:
        return
    try:
        client_from_platform(sdk, FilesClient).delete_fileset(name=name, workspace=workspace)
    except Exception:  # already deleted, or storage unavailable — the manifest still goes
        logger.warning("failed to delete fileset %s", ref, exc_info=True)


def _git_listed_files(root: Path) -> list[Path] | None:
    """Files git considers part of *root* (tracked + untracked, ignored excluded), or ``None``.

    A real project's ``.gitignore`` already states what shouldn't leave the machine — datasets, build
    output, local credentials — and knows far more than a static pattern list can. ``None`` means
    *root* isn't a git worktree (or git is unavailable), leaving the caller on the pattern fallback.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [root / name for name in proc.stdout.decode(errors="replace").split("\0") if name]


def upload_project_dir(sdk: NeMoPlatform, project_dir: Path, *, workspace: str) -> str:
    """Zip a local NAT project and upload it as a fileset; return its ``workspace/name`` ref.

    The counterpart to :func:`download_and_extract_project`: the manifest API and the war-game both
    take a project as a single-zip fileset, so a CLI user's local directory has to become one.

    Selection defers to git when the project is a repo, then applies the exclude patterns on top —
    the patterns still matter there, since a credential file that was never gitignored would
    otherwise be uploaded. Dotenv files are excluded deliberately and cost nothing: the run
    overwrites ``agent.secrets_file`` with a dotenv materialized from the platform secret store
    (``jobs/manifest.py``), so a bundled one is never read.
    """
    if not project_dir.is_dir():
        raise ValueError(f"project dir {str(project_dir)!r} does not exist")
    root = project_dir.resolve()

    candidates = _git_listed_files(root)
    if candidates is None:
        candidates = list(root.rglob("*"))
    files = sorted(
        path
        for path in candidates
        if path.is_file() and not path.is_symlink() and not _is_excluded(path.relative_to(root))
    )
    if not files:
        raise ValueError(f"project dir {str(project_dir)!r} has no files to upload")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{root.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, path.relative_to(root))
        return upload_file_to_fileset(sdk, archive, workspace=workspace, prefix="project")


def extract_zip_safely(zip_path: Path, dest: Path) -> Path:
    """Expand *zip_path* into *dest*, rejecting absolute/symlink/traversing members and oversized archives.

    The size cap sums ``ZipInfo.file_size``, which is the archive's own declared figure. That is a sound
    bound because ``zipfile`` also *reads* against it: a member claiming to be smaller than its data is
    truncated at the declared length and fails its CRC, so an under-reporting bomb cannot expand past
    the cap — it errors out instead.
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if len(infos) > _MAX_ENTRIES:
            raise ValueError(f"Project archive has too many entries ({len(infos)} > {_MAX_ENTRIES}).")
        total = 0
        for info in infos:
            name = info.filename
            if _is_absolute_member(name):
                raise ValueError(f"Archive member has an absolute path: {name!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Archive contains a symlink ({name!r}); not allowed.")
            if not (dest / name).resolve().is_relative_to(dest_resolved):
                raise ValueError(f"Archive member escapes the destination: {name!r}")
            total += info.file_size
            if total > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Project archive is too large when uncompressed (max 500 MB).")
        archive.extractall(dest)
    return dest


def download_and_extract_project(sdk: NeMoPlatform, ref: str, workdir: Path) -> Path:
    """Download the project fileset into *workdir*, expand its zip, and return the project root.

    Collapses a single wrapping top-level directory (the common ``repo-name/…`` zip layout) so the
    returned path is the installable project itself.
    """
    bundle_dir = download_fileset(sdk, ref, workdir / "bundle")
    zips = sorted(bundle_dir.rglob("*.zip"))
    if not zips:
        raise ValueError(f"Fileset {ref!r} contains no .zip project bundle.")
    extracted = extract_zip_safely(zips[0], workdir / "project")
    entries = [p for p in extracted.iterdir() if p.name not in _IGNORED_TOP_LEVEL]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted
