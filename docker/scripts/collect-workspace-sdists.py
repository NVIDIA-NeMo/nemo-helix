#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build source distributions for local packages installed into a Python environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path


def source_collection_enabled() -> bool:
    return os.environ.get("NHX_COLLECT_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}


def record_source_collection_disabled(output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "source-collection-disabled.txt").write_text(
        "source collection disabled; set NHX_COLLECT_SOURCES=1 to enable\n",
        encoding="utf-8",
    )


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def uv_bin() -> str:
    return os.environ.get("UV_BIN", "uv")


def purelib_for(python: str) -> Path:
    script = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    return Path(run_text([python, "-c", script]).strip())


def direct_url_projects(purelib: Path) -> set[Path]:
    projects: set[Path] = set()
    for dist_info in purelib.glob("*.dist-info"):
        direct_url = dist_info / "direct_url.json"
        if not direct_url.is_file():
            continue
        try:
            data = json.loads(direct_url.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        url = str(data.get("url", ""))
        if not url.startswith("file://"):
            continue
        path = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path))
        if (path / "pyproject.toml").is_file():
            projects.add(path)
    return projects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    parser.add_argument("--python", help="Python interpreter whose installed local packages should be inspected")
    parser.add_argument("--project", action="append", default=[], help="Additional local project path to build")
    args = parser.parse_args()

    output_dir = Path(args.output)
    workspace_dir = output_dir / "workspace"
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    workspace_dir.mkdir(parents=True, exist_ok=True)

    projects: set[Path] = set()
    missing: list[str] = []
    if args.python:
        try:
            projects.update(direct_url_projects(purelib_for(args.python)))
        except subprocess.CalledProcessError as error:
            (manifests / "workspace-sdist-errors.txt").write_text(
                f"failed to inspect Python environment {args.python}: {error.output}\n",
                encoding="utf-8",
            )
    for project in args.project:
        path = Path(project)
        if not path.exists():
            missing.append(f"{path}\tproject path does not exist")
        elif not (path / "pyproject.toml").is_file():
            missing.append(f"{path}\tmissing pyproject.toml")
        else:
            projects.add(path)

    built: list[str] = []
    for project in sorted(projects):
        try:
            output = run_text([uv_bin(), "build", "--sdist", "--out-dir", str(workspace_dir), str(project)])
        except (OSError, subprocess.CalledProcessError) as error:
            if isinstance(error, subprocess.CalledProcessError) and error.output:
                message = error.output.strip()
            else:
                message = str(error)
            missing.append(f"{project}\t{message}")
            continue
        built.append(f"{project}\t{output.strip()}")

    (manifests / "built-workspace-sdists.txt").write_text("\n".join(built) + ("\n" if built else ""), encoding="utf-8")
    (manifests / "missing-workspace-sdists.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
