# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a run's frozen skills on the worker as the directories Harbor injects.

Submission froze each skill as a :class:`~nemo_helix_plugin.agents.skills.ResolvedSkill`. Here each
archive is downloaded from its ``archive_ref``, checked against its ``files_hash`` while it streams, and
extracted (which verifies the archive again, in full, before writing a byte) into its own
``<skill_name>`` directory. The directories become ``HarborRuntimeConfig.skills``.

The staging root is fresh per job and sits outside the result bundle tree, which the job deletes after
building its Harbor config and before the run.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from nemo_evals.harbor.archive_io import download_verified_async
from nemo_evals.jobs.utils import run_with_isolated_async_client
from nemo_helix_plugin.agents.skills import ResolvedSkill
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.skill_archive import DEFAULT_LIMITS, extract_skill_archive

#: Directory under the job's persistent storage that holds the staged skills; never part of the result bundle.
SKILLS_STORAGE_DIR = "injected-skills"
_ARCHIVES_DIR = "archives"
_SKILLS_DIR = "skills"


async def download_skill_archives(
    skills: Sequence[ResolvedSkill], destination: Path, files: AsyncFilesClient
) -> list[Path]:
    """Download each skill's archive into ``destination`` (which must exist), verifying its bytes.

    Raises:
        ValueError: An archive exceeds the size limit or does not hash to its ``files_hash``.
    """
    archives: list[Path] = []
    for index, skill in enumerate(skills):
        path = destination / f"{index}.tar"
        with path.open("xb") as output:
            await download_verified_async(
                files,
                skill.archive.archive_ref,
                output,
                limit=DEFAULT_LIMITS.max_archive_bytes,
                expected_digest=skill.archive.files_hash,
            )
        archives.append(path)
    return archives


def extract_skills(skills: Sequence[ResolvedSkill], archives: Sequence[Path], destination: Path) -> list[Path]:
    """Extract each downloaded archive under ``destination`` and return the skill directories, in order.

    Raises:
        SkillArchiveError: An archive breaks the format contract or its ``files_hash``, or its skill name
            is not the one submission froze.
    """
    directories: list[Path] = []
    for skill, archive in zip(skills, archives, strict=True):
        directory = extract_skill_archive(archive, destination, expected_files_hash=skill.archive.files_hash)
        if directory.name != skill.archive.skill_name:
            raise ValueError(
                f"skill {skill.ref.root} archive holds skill {directory.name!r}, not {skill.archive.skill_name!r}"
            )
        directories.append(directory)
    return directories


def reset_staging_root(root: Path) -> None:
    """Empty ``root`` and recreate it privately, so a retried job never sees a previous attempt's skills."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, mode=0o700)


def stage_resolved_skills(
    skills: Sequence[ResolvedSkill] | None, root: Path, async_client: AsyncNemoClient | None
) -> list[Path] | None:
    """Download, verify and extract the run's skills under ``root``; return their directories.

    None (the run does not manage skills) stays None and touches nothing. An empty list is a run with
    exactly no skills, so it stays an empty list.

    Raises:
        ValueError: Skills need downloading but there is no platform client, or an archive fails
            verification.
    """
    if skills is None:
        return None
    reset_staging_root(root)
    if not skills:
        return []
    if async_client is None:
        raise ValueError("staging skills needs a platform client to download their archives")
    archives_dir = root / _ARCHIVES_DIR
    skills_dir = root / _SKILLS_DIR
    archives_dir.mkdir(mode=0o700)
    skills_dir.mkdir(mode=0o700)
    archives = run_with_isolated_async_client(
        async_client,
        lambda client: download_skill_archives(skills, archives_dir, AsyncFilesClient.from_client(client)),
    )
    directories = extract_skills(skills, archives, skills_dir)
    skills_dir.chmod(0o755)
    return directories
