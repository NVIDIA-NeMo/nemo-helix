#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the manual Gym resource-server matrix against a local NeMo Platform."""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST = "e2e/test_gym_resource_server_matrix.py::test_gym_resource_server_job_matrix"


def worker_count(value: str) -> int:
    workers = int(value)
    if not 1 <= workers <= 16:
        raise argparse.ArgumentTypeError("must be between 1 and 16")
    return workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=os.environ.get("NMP_GYM_TASKS_IMAGE", "nmp-gym-tasks:latest"),
        help="task image used by NMP and as the Gym dataset source",
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        default=worker_count(os.environ.get("NEMO_GYM_MATRIX_WORKERS", "1")),
        help="concurrent pytest workers and NMP jobs (default: 1)",
    )
    parser.add_argument("--server", help="resource server, agent, or exact pair ID to run")
    return parser.parse_args()


def extract_example_datasets(image: str, destination: Path) -> Path:
    archive = destination / "examples.tar"
    export_command = """
set -eu
for site_packages in /opt/gym-venv/lib/python*/site-packages; do
  if [ -d "$site_packages/resources_servers" ]; then
    cd "$site_packages"
    tar -cf - resources_servers/*/data/example.jsonl
    exit 0
  fi
done
echo "Gym resources_servers package not found in task image" >&2
exit 1
"""
    with archive.open("wb") as output:
        subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", export_command],
            check=True,
            stdout=output,
        )
    with tarfile.open(archive) as examples:
        examples.extractall(destination, filter="data")
    resources_root = destination / "resources_servers"
    if not resources_root.is_dir():
        raise RuntimeError(f"Gym example datasets were not extracted from {image}")
    return resources_root


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"Run this script with Python 3.13, not {sys.version.split()[0]}")

    internal_base_url = os.environ.get("NMP_E2E_INTERNAL_BASE_URL", "http://host.docker.internal:8080")
    internal_base_url = internal_base_url.replace("://localhost", "://host.docker.internal")
    internal_base_url = internal_base_url.replace("://127.0.0.1", "://host.docker.internal")

    with tempfile.TemporaryDirectory(prefix="nemo-gym-matrix-") as temporary_directory:
        resources_root = extract_example_datasets(args.image, Path(temporary_directory))
        environment = os.environ.copy()
        environment.update(
            {
                "NMP_BASE_URL": os.environ.get("NMP_BASE_URL", "http://localhost:8080"),
                "NMP_E2E_INTERNAL_BASE_URL": internal_base_url,
                "NEMO_GYM_RUN_PLATFORM_MATRIX": "1",
                "NEMO_GYM_MATRIX_CASES": args.server or "",
                "NEMO_GYM_RESOURCES_ROOT": str(resources_root),
            }
        )
        pytest_args = [
            sys.executable,
            "-m",
            "pytest",
            TEST,
            "--run-e2e",
            "-n",
            str(args.workers),
            "--dist",
            "load",
            "-v",
            "-ra",
        ]
        return subprocess.run(pytest_args, cwd=REPO_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
