#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview, apply, or verify a plugin rename described by a JSON profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rename_common import git_file_set, git_paths, path_selected, read_text, repo_root, run_git


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    replacement: str
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def apply(self, path: Path, text: str) -> str:
        if not path_selected(path, self.include, self.exclude):
            return text
        return self.pattern.sub(self.replacement, text)


@dataclass(frozen=True)
class Profile:
    name: str
    replacements: dict[str, str]
    paths: dict[str, str]
    rules: tuple[Rule, ...]
    exclude: tuple[str, ...]
    notes: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> Profile:
        data = json.loads(path.read_text())
        replacements: dict[str, str] = {}
        paths: dict[str, str] = {}
        rules: list[Rule] = []
        # Library is optional; plugins with no companion library need only plugin.
        for section in (data["plugin"], data.get("library", {})):
            for key, destination in (("replacements", replacements), ("paths", paths)):
                for old, new in section.get(key, {}).items():
                    if not old or not new or old == new:
                        raise ValueError(f"Invalid {key} mapping: {old!r} -> {new!r}")
                    if old in destination and destination[old] != new:
                        raise ValueError(f"Conflicting mapping for {old!r}")
                    destination[old] = new
            for item in section.get("rules", []):
                rules.append(
                    Rule(
                        re.compile(item["pattern"], re.MULTILINE),
                        item["replacement"],
                        tuple(item.get("include", [])),
                        tuple(item.get("exclude", [])),
                    )
                )
        for old, new in paths.items():
            for value in (old, new):
                candidate = Path(value)
                if candidate.is_absolute() or ".." in candidate.parts or value == ".":
                    raise ValueError(f"Paths must be safe repo-relative prefixes: {value!r}")
        return cls(
            data["name"],
            replacements,
            paths,
            tuple(rules),
            tuple(data.get("exclude", [])),
            tuple(data.get("notes", [])),
        )

    def transform(self, path: Path, text: str) -> str:
        # One pass, longest first: library names cannot be swallowed by a module prefix,
        # and replacement values are not fed back into other literal mappings.
        if self.replacements:
            pattern = "|".join(re.escape(old) for old in sorted(self.replacements, key=len, reverse=True))
            text = re.sub(pattern, lambda match: self.replacements[match.group()], text)
        for rule in self.rules:
            text = rule.apply(path, text)
        return text

    def destination(self, path: Path) -> Path:
        for old in sorted(self.paths, key=len, reverse=True):
            prefix = Path(old)
            if path == prefix or prefix in path.parents:
                return Path(self.paths[old]) / path.relative_to(prefix)
        return path


@dataclass(frozen=True)
class Change:
    path: Path
    destination: Path
    content: bytes | None
    symlink_target: str | None = None


def plan(profile: Profile, include: tuple[str, ...], exclude: tuple[str, ...]) -> list[Change]:
    changes: list[Change] = []
    destinations: dict[Path, Path] = {}
    excluded = (*profile.exclude, *exclude)
    # Let Git prune excluded untracked directories before enumerating them.
    # Tracked files are still filtered below with the common glob semantics.
    git_excludes = [f"--exclude={pattern[:-2] if pattern.endswith('/**') else pattern}" for pattern in excluded]
    candidates = git_paths("ls-files", "-z", "--cached", "--others", "--exclude-standard", *git_excludes)
    for path in sorted(set(git_file_set(include, excluded, paths=candidates))):
        if not path.is_symlink() and not path.is_file():
            continue
        destination = profile.destination(path)
        if destination != path:
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"Cannot rename {path}: destination exists: {destination}")
            if destination in destinations:
                raise ValueError(f"Both {path} and {destinations[destination]} map to {destination}")
            destinations[destination] = path
            for parent in destination.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"Destination parent is not a directory: {parent}")
                if parent.is_symlink():
                    raise ValueError(f"Destination parent is a symlink: {parent}")
        content = None
        symlink_target = None
        if path.is_symlink():
            # Read the link itself, never its target's contents. Remap repo-local
            # targets lexically, including dangling links left by a partial rename.
            target = Path(os.readlink(path))
            absolute = Path(os.path.abspath(path.parent / target))
            try:
                relative = absolute.relative_to(Path.cwd())
            except ValueError:
                mapped = absolute
            else:
                mapped = Path.cwd() / profile.destination(relative)
            updated_target = str(mapped) if target.is_absolute() else os.path.relpath(mapped, destination.parent)
            if updated_target != str(target):
                symlink_target = updated_target
        else:
            text = read_text(path)
            if text is not None:
                updated = profile.transform(path, text)
                if updated != text:
                    content = updated.encode("utf-8")
        if destination != path or content is not None or symlink_target is not None:
            changes.append(Change(path, destination, content, symlink_target))
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true", help="fail if the profile would still change files")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--include-glob", action="append", default=[])
    parser.add_argument("--exclude-glob", action="append", default=[])
    args = parser.parse_args()
    try:
        profile = Profile.load(args.profile.resolve())
        os.chdir(repo_root(args.repo_dir))
        if not args.dry_run and not args.verify and not args.allow_dirty and run_git("status", "--short").stdout:
            raise ValueError("The worktree must be clean. Inspect existing changes before using --allow-dirty.")
        changes = plan(profile, tuple(args.include_glob), tuple(args.exclude_glob))
        print(profile.name)
        for change in changes:
            actions = (
                "content" if change.content is not None else "symlink" if change.symlink_target is not None else "path"
            )
            if change.destination != change.path:
                actions += f" -> {change.destination}"
            print(f"  {change.path}: {actions}")
        print(f"{len(changes)} files would change.")
        for note in profile.notes:
            print(f"Note: {note}")
        if args.verify:
            return int(bool(changes))
        if args.dry_run:
            return 0
        # Preflight all paths and content before making the first edit.
        for change in changes:
            if change.symlink_target is not None:
                change.path.unlink()
                change.path.symlink_to(change.symlink_target)
            elif change.content is not None:
                change.path.write_bytes(change.content)
            if change.destination != change.path:
                change.destination.parent.mkdir(parents=True, exist_ok=True)
                change.path.rename(change.destination)
        # Remove empty source directories, never recursively delete their contents.
        for parent in sorted(
            {p for c in changes for p in c.path.parents if p != Path(".")}, key=lambda p: len(p.parts), reverse=True
        ):
            try:
                parent.rmdir()
            except OSError:
                pass
        print("Rename applied. Run the same invocation with --verify, then review and validate the changed plugins.")
        return 0
    except (OSError, ValueError, KeyError, re.error) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
