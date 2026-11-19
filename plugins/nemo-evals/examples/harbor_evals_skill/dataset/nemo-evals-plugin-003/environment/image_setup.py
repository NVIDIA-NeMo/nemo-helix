# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove embedded skill and example material from the benchmark image."""

import json
import os
import re
import shutil
import subprocess
import sysconfig
import tomllib
from pathlib import Path

IMAGE_ROOTS = tuple(map(Path, ("/app", "/workspace", "/root", "/home/nvs", "/usr/share/nemo-helix/skills")))
ARCHIVE_ROOTS = tuple(map(Path, ("/usr/share/nemo-helix/source-distributions", "/app/.source-distributions")))
ACCOUNTS = {"harbor-agent": "/home/harbor-agent", "helix-task": "/var/lib/helix-task/service"}
EXCLUDED_SKILLS = {"evaluator-plugin", "nemo-evals-plugin", "nemo-evals"}
EVALS_SOURCE = re.compile(r"^(?:nemo[-_]evals(?:[-_]plugin)?|nemo[-_]helix(?:[-_]ext)?|nhx[-_]evals[-_]sdk)[-_]\d")
ARCHIVES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tgz", ".zip", ".whl")


def removal_path(path: Path, roots: tuple[Path, ...]) -> Path:
    if (
        not path.is_absolute()
        or not any(path != root and path.is_relative_to(root) for root in roots)
        or path.parent != path.parent.resolve()
    ):
        raise ValueError(f"Invalid removal path: {path}")
    return path


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Removal failed: {path}")


def main() -> None:
    settings = tomllib.loads(Path(__file__).with_name("skill-removals.toml").read_text())
    site_packages = Path(sysconfig.get_path("purelib"))
    required = [removal_path(site_packages / entry, (site_packages,)) for entry in settings["required_python_paths"]]
    optional = [removal_path(site_packages / entry, (site_packages,)) for entry in settings["optional_python_paths"]]
    optional += [removal_path(Path(entry), IMAGE_ROOTS) for entry in settings["optional_image_paths"]]
    for path in required:
        if not (path / "SKILL.md").is_file():
            raise RuntimeError(f"Expected embedded skill in the pinned base: {path}")
    for path in [*required, *optional]:
        remove(path)
    for root in ARCHIVE_ROOTS:
        if root.is_dir():
            if root != root.resolve():
                raise ValueError(f"Source archive root must be a real directory: {root}")
            for path in root.rglob("*"):
                if path.name.endswith(ARCHIVES) and EVALS_SOURCE.match(path.name):
                    remove(removal_path(path, (root,)))
            root.chmod(0o700)
    skill_roots = [*site_packages.glob("*/skills"), *(Path(home) / ".agents/skills" for home in ACCOUNTS.values())]
    for root in skill_roots:
        if any((root / name / "SKILL.md").is_file() for name in EXCLUDED_SKILLS):
            raise RuntimeError(f"An embedded Evals skill remains available under {root}")
    for account, home in ACCOUNTS.items():
        output = subprocess.check_output(
            [
                "setpriv",
                f"--reuid={account}",
                f"--regid={account}",
                "--clear-groups",
                "nemo",
                "skills",
                "list",
                "-f",
                "json",
            ],
            env={
                "PATH": os.environ["PATH"],
                "HOME": home,
                "XDG_CONFIG_HOME": home + "/.config",
                "NEMO_TELEMETRY_ENABLED": "false",
            },
            text=True,
        )
        registry = json.dumps(json.loads(output))
        if any(json.dumps(name) in registry for name in EXCLUDED_SKILLS):
            raise RuntimeError(f"The skill registry still serves an embedded Evals skill to {account}")


if __name__ == "__main__":
    main()
