# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate environment FileSet packages on disk."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import yaml
from nmp.rl.schemas.environment import (
    ENVIRONMENT_MANIFEST_ADAPTER,
    AdapterWheelsV1Manifest,
    EnvironmentFormat,
    EnvironmentManifest,
    GymVerifiersDatasetRow,
)
from nmp.rl.tasks.environment.allowlist import IMAGE_ADAPTER_ALLOWLIST
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class EnvironmentPackageValidationError(ValueError):
    """Raised when an environment package fails validation."""


# Gym builds each per-server venv with ``uv venv --seed``, which installs these before any
# requirement is resolved. uv resolves them like any other package, so on a deny-default
# sandbox the vendored wheelhouse is their only source. Python 3.13 seeds pip alone.
VENV_SEED_PACKAGES = ("pip",)


def duplicate_wheel_distributions(wheels_dir: Path) -> dict[str, list[str]]:
    """Return ``{normalized distribution: [versions]}`` for anything vendored twice.

    A ``--find-links`` directory carrying several versions of one project is legal, so
    this is not an error -- but it means the package no longer pins what gets installed,
    and a consumer that pins every file outright cannot satisfy it at all.
    """
    versions: dict[str, list[str]] = defaultdict(list)
    for wheel in sorted(wheels_dir.glob("*.whl")):
        parts = wheel.name.split("-")
        if len(parts) < 5:
            continue
        versions[re.sub(r"[-_.]+", "-", parts[0]).lower()].append(parts[1])
    return {name: found for name, found in versions.items() if len(found) > 1}


def load_manifest(env_root: Path) -> EnvironmentManifest:
    manifest_path = env_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise EnvironmentPackageValidationError(f"Missing {MANIFEST_FILENAME} at environment root: {env_root}")
    return parse_manifest(manifest_path.read_text(encoding="utf-8"))


MANIFEST_FILENAME = "nemo-environment.yaml"


def parse_manifest(raw_yaml: bytes | str) -> EnvironmentManifest:
    """Parse and validate manifest bytes without touching the filesystem."""
    raw = yaml.safe_load(raw_yaml)
    if not isinstance(raw, dict):
        raise EnvironmentPackageValidationError(f"{MANIFEST_FILENAME} must be a mapping")
    try:
        return ENVIRONMENT_MANIFEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise EnvironmentPackageValidationError(f"Invalid {MANIFEST_FILENAME}: {exc}") from exc


def validate_manifest_against_listing(manifest: EnvironmentManifest, paths: Iterable[str]) -> None:
    """Check a manifest against a flat list of package-relative file paths.

    Everything here is decidable from the file names alone, so it runs against a FileSet
    listing at submit time as well as against a downloaded package in the job. Checks that
    need file contents or filesystem metadata (symlinks, path escapes) stay in
    :func:`validate_package_layout`.
    """
    entries = {p.lstrip("./") for p in paths}

    for path in sorted(entries):
        if path.endswith(".jsonl"):
            raise EnvironmentPackageValidationError(
                f"Prompt JSONL must not live in the environment package: {path}. "
                "Upload dataset rows as a separate dataset FileSet."
            )

    missing = [rel for rel in manifest.config_paths if rel not in entries]
    if missing:
        raise EnvironmentPackageValidationError(
            f"config_paths reference files that are not in the package: {', '.join(sorted(missing))}"
        )

    if offline_wheel_install_required(manifest):
        wheels = [p for p in entries if p.startswith("wheels/") and "/" not in p[len("wheels/") :]]
        if not wheels:
            raise EnvironmentPackageValidationError(
                f"format {manifest.format.value!r} installs its dependencies offline, so the package "
                "must carry a non-empty wheels/ directory"
            )
        non_wheels = sorted(p for p in wheels if not p.endswith(".whl"))
        if non_wheels:
            raise EnvironmentPackageValidationError(f"Non-wheel files in wheels/: {', '.join(non_wheels)}")

        # Warn, not raise: without these a package still installs wherever the sandbox can
        # reach an index, and rejecting would invalidate packages already uploaded. But on a
        # deny-default sandbox it surfaces as a 900s readiness timeout, so name it here.
        vendored = {re.sub(r"[-_.]+", "-", Path(name).name.split("-")[0]).lower() for name in wheels}
        missing_seed = [name for name in VENV_SEED_PACKAGES if name not in vendored]
        if missing_seed:
            logger.warning(
                "wheels/ does not vendor %s. Gym creates every per-server venv with "
                "`uv venv --seed`, which resolves these before any requirement, so a sandbox "
                "without egress would fail. Add the wheel(s) to wheels/; packages built by "
                "pi-to-gym-conversion carry them already.",
                ", ".join(missing_seed),
            )

    if isinstance(manifest, AdapterWheelsV1Manifest) and manifest.adapter.agent not in IMAGE_ADAPTER_ALLOWLIST:
        raise EnvironmentPackageValidationError(
            f"adapter.agent {manifest.adapter.agent!r} is not built into the training image. "
            f"Supported: {', '.join(sorted(IMAGE_ADAPTER_ALLOWLIST))}"
        )


def validate_package_layout(env_root: Path, manifest: EnvironmentManifest) -> None:
    """Validate on-disk layout for the declared environment format."""
    if not env_root.is_dir():
        raise EnvironmentPackageValidationError(f"Environment root is not a directory: {env_root}")

    # Filesystem-only checks run first: rglob does not follow symlinked directories, so an
    # escaping config_path is absent from the listing below and would otherwise be reported
    # as a plain missing file rather than as the containment breach it is.
    for rel in manifest.config_paths:
        cfg = env_root / rel
        if cfg.is_symlink():
            raise EnvironmentPackageValidationError(f"Symlinks are not allowed: {rel}")
        # Compare by path components, not string prefix: "/job/environment-attacker"
        # startswith("/job/environment") but is a different directory.
        if not cfg.resolve().is_relative_to(env_root.resolve()):
            raise EnvironmentPackageValidationError(f"config_paths escapes environment root: {rel}")

    validate_manifest_against_listing(
        manifest,
        (str(p.relative_to(env_root)) for p in env_root.rglob("*") if p.is_file()),
    )

    if offline_wheel_install_required(manifest):
        # Warn, not raise: several versions of a project in a find-links pool is legal, and
        # rejecting here would invalidate packages already uploaded. But it means the
        # package does not pin what gets installed, so say so where it can still be fixed
        # rather than leaving it to surface as a resolver failure mid-job.
        for name, versions in duplicate_wheel_distributions(env_root / "wheels").items():
            logger.warning(
                "wheels/ vendors %d versions of %s (%s). The installed version is then "
                "whatever the resolver picks. Regenerate the package so it vendors a "
                "single resolved closure.",
                len(versions),
                name,
                ", ".join(sorted(versions)),
            )


def validate_dataset_rows(
    rows: list[dict],
    *,
    expected_vf_env_id: str | None,
    expected_agent: str | None,
) -> list[GymVerifiersDatasetRow]:
    parsed: list[GymVerifiersDatasetRow] = []
    for row in rows:
        item = GymVerifiersDatasetRow.model_validate(row)
        if expected_agent and item.agent_ref.name != expected_agent:
            raise EnvironmentPackageValidationError(
                f"agent_ref.name {item.agent_ref.name!r} != expected {expected_agent!r}"
            )
        if expected_vf_env_id and item.vf_env_id != expected_vf_env_id:
            raise EnvironmentPackageValidationError(f"vf_env_id {item.vf_env_id!r} != expected {expected_vf_env_id!r}")
        parsed.append(item)
    return parsed


def offline_wheel_install_required(manifest: EnvironmentManifest) -> bool:
    return manifest.format in (EnvironmentFormat.WHEELS_V1, EnvironmentFormat.ADAPTER_WHEELS_V1)
