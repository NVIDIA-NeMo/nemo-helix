# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-owned environment package bootstrap (native-v1 / wheels-v1 / adapter-wheels-v1).

Validates package layout, then offline-installs wheels when the format requires it.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nhx.rl.schemas.environment import (
    AdapterWheelsV1Manifest,
    EnvironmentManifest,
    NativeV1Manifest,
    WheelsV1Manifest,
)
from nhx.rl.tasks.environment.allowlist import IMAGE_ADAPTER_ALLOWLIST
from nhx.rl.tasks.environment.validate import (
    EnvironmentPackageValidationError,
    load_manifest,
    offline_wheel_install_required,
    validate_package_layout,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    manifest: EnvironmentManifest
    environment_root: Path
    work_venv: Path | None
    image_config_root: str | None


def resolve_adapter_image_root(manifest: EnvironmentManifest) -> str | None:
    if not isinstance(manifest, AdapterWheelsV1Manifest):
        return None
    agent = manifest.adapter.agent
    if agent not in IMAGE_ADAPTER_ALLOWLIST:
        raise EnvironmentPackageValidationError(f"adapter.agent {agent!r} is not on the image allowlist")
    return manifest.adapter.image_config_root or IMAGE_ADAPTER_ALLOWLIST[agent]


def _offline_pip_install(wheels_dir: Path, *, target_venv: Path) -> None:
    python = target_venv / "bin" / "python"
    if not python.is_file():
        raise EnvironmentPackageValidationError(f"Missing venv python at {python}")
    wheels = sorted(wheels_dir.glob("*.whl"))
    if not wheels:
        raise EnvironmentPackageValidationError(f"No wheels to install under {wheels_dir}")
    cmd = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-index",
        f"--find-links={wheels_dir}",
        *[str(w) for w in wheels],
    ]
    logger.info("Offline wheel install: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_venv(work_venv: Path) -> None:
    if (work_venv / "bin" / "python").is_file():
        return
    work_venv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(work_venv)], check=True)


def bootstrap_environment_package(
    environment_root: Path,
    *,
    work_venv: Path | None = None,
    install_wheels: bool = True,
) -> BootstrapResult:
    """Validate + optionally install an environment FileSet package for Gym.

    Parameters
    ----------
    environment_root:
        Path to the unpacked environment FileSet (contains ``nemo-environment.yaml``).
    work_venv:
        Target venv for offline wheel installs (wheels-v1 / adapter-wheels-v1).
        Ignored for native-v1 when ``install_wheels`` is false.
    install_wheels:
        When false, only validate layout (useful for unit tests / dry-run).
    """
    environment_root = environment_root.resolve()
    manifest = load_manifest(environment_root)
    validate_package_layout(environment_root, manifest)

    if isinstance(manifest, NativeV1Manifest):
        for rel in manifest.config_paths:
            cfg = environment_root / rel
            if not cfg.is_file():
                raise EnvironmentPackageValidationError(f"native-v1 missing config: {rel}")
        logger.info("Bootstrapped native-v1 environment at %s", environment_root)
        return BootstrapResult(
            manifest=manifest,
            environment_root=environment_root,
            work_venv=None,
            image_config_root=None,
        )

    image_root = resolve_adapter_image_root(manifest)
    venv_path: Path | None = None
    if install_wheels and offline_wheel_install_required(manifest):
        if work_venv is None:
            # Never default inside environment_root. validate_package_layout rejects any
            # *.jsonl under the package, and site-packages routinely ships them, so a venv
            # placed in-tree makes every later bootstrap of the same PVC fail — which
            # includes job resume and rerun, since job storage is keyed by job id.
            work_venv = Path(tempfile.mkdtemp(prefix="nhx-rl-envpkg-")) / "venv"
        _ensure_venv(work_venv)
        _offline_pip_install(environment_root / "wheels", target_venv=work_venv)
        venv_path = work_venv

    if isinstance(manifest, (WheelsV1Manifest, AdapterWheelsV1Manifest)):
        logger.info(
            "Bootstrapped %s environment at %s (venv=%s)",
            manifest.format,
            environment_root,
            venv_path,
        )

    return BootstrapResult(
        manifest=manifest,
        environment_root=environment_root,
        work_venv=venv_path,
        image_config_root=image_root,
    )
