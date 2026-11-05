# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base agent installer with default install logic."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from nemo_helix_ext.cli.commands.skills.base import (
    INSTALLED_SKILL_PREFIX,
    Scope,
    Skill,
    installed_skill_name,
    validate_skill_name,
)

logger = logging.getLogger(__name__)

# Manifest file recording the skill directories this installer wrote, kept
# beside the installed skills (one per skills root, so project scope, user
# scope, and `--path` each get their own). Only files recorded here are ever
# removed by a pruning full install; directories written before this record
# existed are left alone.
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
    # A kept name can also be one this call wrote (overwrite), so dedupe.
    payload = json.dumps({"owned": sorted(set(owned))}, indent=2) + "\n"
    # Write to a fresh file in the same directory, then rename it into place:
    # a crash mid-write leaves the previous record intact, and a symlink
    # planted at the manifest path is replaced rather than followed.
    fd, tmp_name = tempfile.mkstemp(dir=skills_root, prefix=INSTALLED_SKILLS_MANIFEST_NAME, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(payload)
        os.replace(tmp_name, manifest_path)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _remove_if_owned(skills_root: Path, name: str) -> bool:
    """Remove ``skills_root/name`` when it is a directory this installer owns.

    Safety checks, in order: the name must be a single path segment carrying
    the installed-skill prefix (the only names this installer writes), the
    resolved target must stay inside ``skills_root`` (symlinks pointing out
    are left alone), and anything that is not a directory is left alone. A
    missing directory counts as already removed.
    """
    if not _is_single_path_segment(name) or not name.startswith(INSTALLED_SKILL_PREFIX):
        return False
    target = skills_root / name
    if not target.exists() and not target.is_symlink():
        return True  # Already gone; treat the recorded name as removed.
    root_resolved = skills_root.resolve()
    target_resolved = target.resolve()
    if target_resolved == root_resolved or root_resolved not in target_resolved.parents:
        return False  # Resolves outside the skills root: leave it alone.
    if target.is_symlink():
        target.unlink()  # The link itself lives inside the root; its target does not get touched.
        return True
    if not target.is_dir():
        return False  # A plain file is not a directory this installer wrote.
    shutil.rmtree(target)
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

    def format_content(self, skill: Skill) -> str:
        """Format skill content for this agent. Default: return raw content."""
        return skill.raw

    def install(self, scope: Scope, project_root: Path, skills: dict[str, Skill], prune: bool = False) -> list[Path]:
        """Install all skills. Returns list of paths written.

        Writes the new skill files first (including overwriting a same-named
        skill), then, when ``prune`` is true, removes directories a previous
        install recorded that this install set no longer contains, and
        finally writes the manifest. Only full installs (no ``--skill``)
        pass ``prune=True``; subset installs record what they wrote and never
        remove anything.
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

        for skills_root, installed_dirs in wrote.items():
            owned = _read_owned_directories(skills_root)
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
            _write_owned_directories(skills_root, [*kept, *installed_dirs])
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
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
