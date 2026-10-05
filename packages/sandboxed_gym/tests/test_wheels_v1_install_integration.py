# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the wheels-v1 installer with a real wheel and the real ``uv`` command.

Unit tests cover manifest handling and command construction. This focused test proves that the
resulting target is importable by both the running host and a child Python process without requiring
Docker or an external image.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import zipfile
from pathlib import Path

from sandboxed_gym.environment_package import WHEELS_V1_SUBDIR
from sandboxed_gym.runtime import gym_host_runtime as runtime

_PACKAGE_NAME = "wheels_v1_probe"
_WHEEL_NAME = f"{_PACKAGE_NAME}-1.0-py3-none-any.whl"
# A recognizable sentinel exported by the test wheel, proving imports came from the installed wheel.
_EXPECTED_VALUE = 42


def _write_wheels_v1_bundle(environment_dir: Path) -> None:
    # Reproduce the staged layout the Gym host receives: a manifest at the environment root and
    # vendored wheels under wheels/. No package build tool is needed because the fixture itself is
    # a small, valid ZIP-format wheel.
    config_path = "resources_servers/wheels_v1_probe/configs/wheels_v1_probe.yaml"
    config = environment_dir / config_path
    config.parent.mkdir(parents=True)
    config.write_text("wheels_v1_probe: {}\n", encoding="utf-8")
    (environment_dir / runtime.ENVIRONMENT_MANIFEST_FILENAME).write_text(
        f"format: wheels-v1\nconfig_paths:\n  - {config_path}\nmetadata:\n  name: wheels-v1-probe\n",
        encoding="utf-8",
    )
    wheels_dir = environment_dir / WHEELS_V1_SUBDIR
    wheels_dir.mkdir()
    with zipfile.ZipFile(wheels_dir / _WHEEL_NAME, "w") as archive:
        archive.writestr(f"{_PACKAGE_NAME}/__init__.py", f"VALUE = {_EXPECTED_VALUE}\n")
        archive.writestr(
            f"{_PACKAGE_NAME}-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {_PACKAGE_NAME}\nVersion: 1.0\n",
        )
        archive.writestr(
            f"{_PACKAGE_NAME}-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{_PACKAGE_NAME}-1.0.dist-info/RECORD", "")


def test_real_wheel_is_importable_by_host_and_child(tmp_path: Path, isolated_gym_host_process_state: None) -> None:
    # Recreate the two directories mounted into a real Gym host. `environment_dir` stands in for
    # read-only `/job/environment`, where the manifest and wheels are staged. `work_dir` stands in
    # for writable `/job/work`, where uv installs them. `tmp_path` keeps both isolated to this test.
    environment_dir = tmp_path / "environment"
    work_dir = tmp_path / "work"
    environment_dir.mkdir()
    work_dir.mkdir()
    # Populate the simulated environment mount with nemo-environment.yaml and one test wheel.
    _write_wheels_v1_bundle(environment_dir)

    # Do not mock subprocess.run here: this is the layer that proves the generated command works
    # with the real uv executable and produces an importable target directory.
    runtime._install_wheels_v1_dependencies(
        runtime._load_runtime_environment_package(str(environment_dir), required=True),
        str(work_dir),
    )

    try:
        # sys.path is updated for the already-running Gym host process.
        installed = importlib.import_module(_PACKAGE_NAME)
        assert installed.VALUE == _EXPECTED_VALUE

        # Gym creates a separate environment for each server from that server's requirements file.
        # Reproduce that uv install and prove it resolves the named requirement from the staged
        # wheelhouse. Package-index fallback remains available for Gym's core dependencies.
        requirements_path = work_dir / "requirements.txt"
        requirements_path.write_text(f"{_PACKAGE_NAME}==1.0\n", encoding="utf-8")
        server_install_dir = work_dir / "server-site-packages"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--target",
                str(server_install_dir),
                "--requirements",
                str(requirements_path),
            ],
            check=True,
            env=runtime.os.environ.copy(),
        )

        # Assert a child process can import from the independently installed server environment.
        child_env = runtime.os.environ.copy()
        child_env["PYTHONPATH"] = str(server_install_dir)
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {_PACKAGE_NAME}; assert {_PACKAGE_NAME}.VALUE == {_EXPECTED_VALUE}",
            ],
            check=True,
            env=child_env,
        )
    finally:
        # Avoid leaking the temporary package through Python's process-global module cache.
        sys.modules.pop(_PACKAGE_NAME, None)
