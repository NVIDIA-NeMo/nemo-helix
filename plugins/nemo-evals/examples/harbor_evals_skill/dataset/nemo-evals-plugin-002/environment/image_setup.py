# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prune configured skill directories while building the shared task image."""

import json
import os
import shutil
import subprocess
import sysconfig
import tomllib
from pathlib import Path

SETTINGS = Path("/opt/helix-task/skill-removals.toml")
ACCOUNTS = {"harbor-agent": "/home/harbor-agent", "helix-task": "/var/lib/helix-task/service"}


def remove_skills(entries: list[str], *, roots: list[Path]) -> None:
    approved = []
    for entry in entries:
        path = Path(entry)
        if (
            not path.is_absolute()
            or path != path.resolve()
            or not any(path.is_relative_to(root) and path != root for root in roots)
            or not path.is_dir()
            or not (path / "SKILL.md").is_file()
            or (path / "SKILL.md").is_symlink()
        ):
            raise ValueError(f"expected an existing, absolute skill directory beneath an approved root: {path}")
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in approved):
            raise ValueError(f"overlapping skill removal: {path}")
        approved.append(path)
    for path in approved:
        shutil.rmtree(path)
        if path.exists():
            raise RuntimeError(f"skill removal failed: {path}")


def main() -> None:
    settings = tomllib.loads(SETTINGS.read_text())
    entries = settings["remove_skill_dirs"]
    if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
        raise ValueError("remove_skill_dirs must be a list of absolute paths")
    site_packages = Path(sysconfig.get_path("purelib"))
    roots = [
        *site_packages.glob("*/skills"),
        Path("/root/.agents/skills"),
        Path("/home/nvs/.agents/skills"),
        Path("/workspace/.agents/skills"),
        Path("/usr/share/nemo-helix/skills"),
        *(Path(home) / ".agents/skills" for home in ACCOUNTS.values()),
    ]
    remove_skills(entries, roots=[root for root in roots if root == root.resolve()])
    for archives in (Path("/usr/share/nemo-helix/source-distributions"), Path("/app/.source-distributions")):
        if archives.is_dir():
            os.chown(archives, 0, 0)
            archives.chmod(0o700)
    excluded = {"evaluator-plugin", "nemo-evals-plugin", "nemo-evals"}
    for root in roots:
        if root.exists() and any((root / name / "SKILL.md").is_file() for name in excluded):
            raise RuntimeError(f"an excluded Evals skill remains under {root}")
    for account, home in ACCOUNTS.items():
        result = subprocess.check_output(
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
        listed = json.dumps(json.loads(result))
        if any(json.dumps(name) in listed for name in excluded):
            raise RuntimeError(f"nemo skills list exposes an excluded Evals skill to {account}")


if __name__ == "__main__":
    main()
