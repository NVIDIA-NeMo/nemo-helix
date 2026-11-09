# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base agent installer with default install logic."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from shutil import rmtree

import yaml
from nemo_helix_ext.cli.commands.skills.base import (
    INSTALLED_SKILL_PREFIX,
    Scope,
    Skill,
    installed_skill_name,
    validate_skill_name,
)

logger = logging.getLogger(__name__)

# One per skills root. A pruning install only ever removes directories recorded here.
INSTALLED_SKILLS_MANIFEST_NAME = ".nemo-helix-installed-skills.json"


def _is_single_path_segment(name: str) -> bool:
    """True when ``name`` can only ever resolve directly beneath a directory."""
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name


def _read_owned_directories(skills_root: Path) -> list[str]:
    """Return recorded owned directory names for ``skills_root``, tolerating a missing or corrupt manifest."""
    manifest_path = skills_root / INSTALLED_SKILLS_MANIFEST_NAME
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        if manifest_path.exists() or manifest_path.is_symlink():
            logger.warning(
                "Installed-skills manifest %s is unreadable (%s); treating as unrecorded", manifest_path, exc
            )
        return []
    owned = data.get("owned") if isinstance(data, dict) else None
    if not isinstance(owned, list) or not all(isinstance(name, str) for name in owned):
        logger.warning("Installed-skills manifest %s has an unexpected shape; treating as unrecorded", manifest_path)
        return []
    return owned


def _write_owned_directories(skills_root: Path, owned: list[str]) -> None:
    """Write the manifest last, after all writes and removals for this install."""
    skills_root.mkdir(parents=True, exist_ok=True)
    manifest_path = skills_root / INSTALLED_SKILLS_MANIFEST_NAME
    payload = json.dumps({"owned": sorted(set(owned))}, indent=2) + "\n"
    # Rename into place: a crash mid-write keeps the old record, and a planted symlink is replaced, not followed.
    fd, tmp_name = tempfile.mkstemp(dir=skills_root, prefix=INSTALLED_SKILLS_MANIFEST_NAME, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(payload)
        os.replace(tmp_name, manifest_path)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _remove_if_owned(skills_root: Path, name: str) -> bool:
    """Remove a recorded skill directory; True when it is gone, False when it was left alone."""
    if not _is_single_path_segment(name) or not name.startswith(INSTALLED_SKILL_PREFIX):
        return False
    target = skills_root / name
    root_resolved = skills_root.resolve()
    target_resolved = target.resolve()
    if target_resolved == root_resolved or root_resolved not in target_resolved.parents:
        return False  # Symlinks out of the skills root are never followed.
    if target.is_symlink():
        target.unlink()  # Removes the link only, never its target.
        return True
    try:
        rmtree(target)
    except FileNotFoundError:
        return True
    except NotADirectoryError:
        return False  # A plain file is not something this installer wrote.
    return True


def format_agent_skill_content(skill: Skill) -> str:
    """Return standard Agent Skills content with normalized NeMo metadata."""
    metadata: dict[str, object] = {"name": installed_skill_name(skill.name), "description": skill.description}
    if skill.preconditions:
        metadata["preconditions"] = skill.preconditions
    front_matter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
    return f"---\n{front_matter}---\n\n{skill.content}"


class BaseAgentInstaller:
    """Base class providing default install behavior: loop over skills, format, write."""

    name: str
    display_name: str
    supported_scopes: list[Scope]

    #: Skill directories removed by the most recent ``install`` call, so
    #: callers can report the removals. Reset on every call.
    pruned_directories: list[Path]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        raise NotImplementedError

    def _skills_root(self, scope: Scope, project_root: Path) -> Path:
        """The directory that holds installed skill directories, even when this install writes none."""
        return self.get_install_path(scope, project_root, "skill").parent.parent

    def format_content(self, skill: Skill) -> str:
        """Format skill content for this agent. Default: return raw content."""
        return skill.raw

    def install(self, scope: Scope, project_root: Path, skills: dict[str, Skill], prune: bool = False) -> list[Path]:
        """Install all skills and return the paths written.

        With ``prune``, also remove recorded directories that ``skills`` no longer contains.
        """
        paths: list[Path] = []
        self.pruned_directories = []
        pending: list[tuple[Skill, Path]] = []
        destinations: dict[Path, str] = {}
        for skill_name, skill in skills.items():
            validate_skill_name(skill_name)
            path = self.get_install_path(scope, project_root, skill_name)
            if path in destinations:
                raise ValueError(f"Multiple skills resolve to {path}: {destinations[path]!r} and {skill_name!r}")
            destinations[path] = skill_name
            pending.append((skill, path))
        preexisting = {path.parent for _, path in pending if path.parent.exists()}
        for skill, path in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.format_content(skill))
            self._copy_companion_files(skill, path)
            paths.append(path)

        # The skills root is the parent of the installed skill directory
        # (``SKILL.md``'s grandparent). Each root owns its own manifest.
        wrote: dict[Path, set[str]] = {}
        for path in paths:
            skills_root = path.parent.parent
            wrote.setdefault(skills_root, set()).add(path.parent.name)
        # A full install that ships nothing still has a root to prune, which the writes above never visit.
        if prune and not wrote:
            skills_root = self._skills_root(scope, project_root)
            if (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).exists():
                wrote[skills_root] = set()

        for skills_root, installed_dirs in wrote.items():
            owned = _read_owned_directories(skills_root)
            owned_folded = {name.casefold() for name in owned}
            # A directory that existed before this install and was never recorded may be the
            # user's own; overwrite its SKILL.md but never claim it, so it is never pruned.
            claimed = [
                name
                for name in installed_dirs
                if name.casefold() in owned_folded or skills_root / name not in preexisting
            ]
            kept: list[str] = []
            if prune:
                # Case-insensitive: on a case-insensitive filesystem a recorded name differing
                # only in case from one just written is the same directory.
                installed_folded = {installed.casefold() for installed in installed_dirs}
                for name in owned:
                    if name.casefold() in installed_folded:
                        continue  # Still shipped: overwritten above and recorded below, never removed.
                    try:
                        removed = _remove_if_owned(skills_root, name)
                    except OSError as exc:
                        logger.warning("Could not remove %s (%s); keeping it recorded", skills_root / name, exc)
                        removed = False
                    if not removed:
                        kept.append(name)
                    else:
                        self.pruned_directories.append(skills_root / name)
            else:
                kept = owned
            # Manifest last: a crash before this write leaves the previous
            # record intact, so the next full install finishes the removal.
            _write_owned_directories(skills_root, [*kept, *claimed])
        return paths

    def _copy_companion_files(self, skill: Skill, installed_path: Path) -> None:
        """Copy non-SKILL.md files from source_dir to the installed skill's directory."""
        if skill.source_dir is None:
            return
        target_dir = installed_path.parent
        for item in skill.source_dir.iterdir():
            if item.name == "SKILL.md":
                continue
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
