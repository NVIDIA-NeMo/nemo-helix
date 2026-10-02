# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the Gym environment package that runs a registered platform agent.

The staging step writes, into the job's ``environment/`` tree, a ``wheels-v1`` package holding the
platform's Gym agent component, the agent-instance config generated from the registered agent's
resolved Fabric config, the agent's Ethos files, and a wheelhouse with the Fabric harness the agent
needs. The sandboxed Gym host installs the wheelhouse offline and runs the component like any
package-supplied agent. A user's own environment package, if one was staged first, is extended
rather than replaced.
"""

from __future__ import annotations

import importlib.resources
import platform
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from nemo_evaluator.api.schemas import AgentRef
from nemo_evaluator.jobs.agent_spec import (
    REGISTERED_AGENT_GYM_COMPONENT,
    registered_agent_gym_config_path,
    registered_agent_gym_instance,
)
from nemo_evaluator.jobs.gym_environment_package import (
    CUSTOM_AGENT_SUBDIR,
    ENVIRONMENT_MANIFEST_FILENAME,
    EnvironmentFormat,
)
from pydantic import BaseModel, ConfigDict, Field

#: Where the sandboxed Gym host mounts the staged environment; absolute paths written into the
#: package must be host paths, not job-container paths.
ENVIRONMENT_MOUNT_PATH = "/job/environment"
WHEELHOUSE_SUBDIR = "wheels"
AGENT_FILES_SUBDIR = "agents"
#: The Gym host image's interpreter; wheels are downloaded for it, not for the job container's.
DEFAULT_WHEEL_PYTHON_VERSION = "3.13"

WheelDownloader = Callable[[Sequence[str], Sequence[str], Path, str, str | None], None]

#: Upper bound on one wheelhouse download. The staging step has no step-level deadline of its own, so a
#: stalled index must not hold the worker indefinitely; a healthy download takes well under a minute.
WHEEL_DOWNLOAD_TIMEOUT_S = 20 * 60


class GymRegisteredAgentPackageSpec(BaseModel):
    """What the staging step needs to assemble a registered agent's Gym package."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentRef = Field(description="The registered agent, workspace-qualified.")
    resolved_config: dict[str, Any] = Field(description="The Fabric config the agent resolved to at submit.")
    requirements: list[str] = Field(
        min_length=1,
        description="Requirement specifiers for the wheelhouse: the Fabric harness extra and companions.",
    )
    wheel_python_version: str = Field(default=DEFAULT_WHEEL_PYTHON_VERSION)
    wheel_architecture: str | None = Field(
        default=None,
        description="CPU architecture of the Gym host image (`x86_64` or `aarch64`); the job container's own "
        "when omitted. The host is Linux either way.",
    )


_ARCH_ALIASES = {"amd64": "x86_64", "arm64": "aarch64"}


def wheel_download_command(
    requirements: Sequence[str],
    constraints_file: Path | None,
    destination: Path,
    python_version: str,
    architecture: str | None,
) -> list[str]:
    """The ``pip download`` invocation for a wheelhouse the Gym host's interpreter can install offline.

    The host is Linux; ``architecture`` (``x86_64`` or ``aarch64``, the job container's when omitted)
    becomes the pair of manylinux tags a wheel may carry, because pip matches tags exactly.
    """
    arch = _ARCH_ALIASES.get(architecture or platform.machine(), architecture or platform.machine())
    return [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(destination),
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        f"cp{python_version.replace('.', '')}",
        "--platform",
        f"manylinux2014_{arch}",
        "--platform",
        f"manylinux_2_28_{arch}",
        *(["--constraint", str(constraints_file)] if constraints_file else []),
        *requirements,
    ]


def download_wheels(
    requirements: Sequence[str],
    constraints: Sequence[str],
    destination: Path,
    python_version: str,
    architecture: str | None,
) -> None:
    """Fill ``destination`` with wheels for ``requirements`` and everything they need, for the host's interpreter."""
    constraints_file = None
    if constraints:
        constraints_file = destination.parent / ".wheel-constraints.txt"
        constraints_file.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    command = wheel_download_command(requirements, constraints_file, destination, python_version, architecture)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=WHEEL_DOWNLOAD_TIMEOUT_S)
        if result.returncode != 0 and "No module named pip" in result.stderr:
            # The task image's interpreter has no pip; uv brings one for the invocation.
            result = subprocess.run(
                ["uv", "run", "--no-project", "--with", "pip", "python", *command[1:]],
                capture_output=True,
                text=True,
                timeout=WHEEL_DOWNLOAD_TIMEOUT_S,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"wheelhouse download for {list(requirements)} did not finish within {WHEEL_DOWNLOAD_TIMEOUT_S}s"
        ) from exc
    finally:
        if constraints_file is not None:
            constraints_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"wheelhouse download failed for {list(requirements)}: {result.stderr.strip()[-2000:]}")


def _component_source() -> str:
    return (importlib.resources.files("nemo_evaluator.gym_registered_agent") / "app.py").read_text(encoding="utf-8")


#: A copy of the Gym host image's ``docker/locks/nhx-gym-host/uv.lock``, kept byte-identical by a test.
HOST_LOCK_RESOURCE = "nhx-gym-host.uv.lock"


def _host_lock_text() -> str:
    return (importlib.resources.files("nemo_evaluator.gym_registered_agent") / HOST_LOCK_RESOURCE).read_text(
        encoding="utf-8"
    )


def host_gym_constraints(lock_text: str | None = None) -> list[str]:
    """Every distribution the sandboxed Gym host image has installed, pinned to its version.

    The wheelhouse resolves under these so the harness installs into the host without moving anything
    the host already has. Lock entries that are not distributions (the host's own virtual project, the
    sandboxed-gym package it installs from source) carry no pin.
    """
    lock = tomllib.loads(lock_text if lock_text is not None else _host_lock_text())
    pins = []
    for package in lock.get("package", []):
        source = package.get("source") or {}
        if not package.get("version") or any(key in source for key in ("virtual", "directory", "editable")):
            continue
        pins.append(f"{package['name']}=={package['version']}")
    return sorted(pins)


def _merged_manifest(root: Path, config_path: str, agent_name: str) -> dict[str, Any]:
    manifest_path = root / ENVIRONMENT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {
            "format": EnvironmentFormat.WHEELS_V1.value,
            "config_paths": [config_path],
            "metadata": {"name": f"registered-agent-{agent_name}", "description": f"Registered agent {agent_name}"},
        }
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("format") != EnvironmentFormat.WHEELS_V1.value:
        raise ValueError(
            f"a registered agent can only be added to a `{EnvironmentFormat.WHEELS_V1.value}` environment package; "
            f"this one is `{manifest.get('format')}`. Vendor the environment's wheels or drop the FileSet."
        )
    paths = list(manifest.get("config_paths") or [])
    if config_path not in paths:
        paths.append(config_path)
    manifest["config_paths"] = paths
    return manifest


def write_registered_agent_package(
    root: Path,
    spec: GymRegisteredAgentPackageSpec,
    *,
    agent_files: Path | None,
    download: WheelDownloader | None = None,
) -> str:
    """Write the component, its instance config, the agent's files and the wheelhouse under ``root``.

    Returns the package-relative path of the generated agent-instance config.
    """
    instance = registered_agent_gym_instance(spec.agent)
    config_path = registered_agent_gym_config_path(spec.agent)
    component_dir = root / CUSTOM_AGENT_SUBDIR / REGISTERED_AGENT_GYM_COMPONENT
    for taken in (component_dir, root / config_path):
        if taken.exists():
            raise ValueError(
                f"the environment package already contains {taken.relative_to(root).as_posix()}; a registered "
                f"agent's package cannot be added to one that ships its own `{REGISTERED_AGENT_GYM_COMPONENT}`"
            )
    (component_dir / "configs").mkdir(parents=True, exist_ok=True)
    (component_dir / "app.py").write_text(_component_source(), encoding="utf-8")
    (component_dir / "requirements.txt").write_text("\n".join(spec.requirements) + "\n", encoding="utf-8")

    base_dir = None
    if agent_files is not None:
        files_dir = component_dir / AGENT_FILES_SUBDIR / instance
        shutil.copytree(agent_files, files_dir, dirs_exist_ok=True)
        base_dir = f"{ENVIRONMENT_MOUNT_PATH}/{files_dir.relative_to(root).as_posix()}"

    component: dict[str, Any] = {
        "entrypoint": "app.py",
        "description": f"Registered platform agent {spec.agent.root}, run through NeMo Fabric.",
        "resources_server": {"type": "resources_servers", "name": "???"},
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "fabric_config": spec.resolved_config,
        "fabric_config_base_dir": base_dir,
    }
    instance_config = {instance: {CUSTOM_AGENT_SUBDIR: {REGISTERED_AGENT_GYM_COMPONENT: component}}}
    (root / config_path).write_text(yaml.safe_dump(instance_config, sort_keys=False), encoding="utf-8")

    wheelhouse = root / WHEELHOUSE_SUBDIR
    wheelhouse.mkdir(exist_ok=True)
    (download or download_wheels)(
        spec.requirements, host_gym_constraints(), wheelhouse, spec.wheel_python_version, spec.wheel_architecture
    )

    manifest = _merged_manifest(root, config_path, spec.agent.root.rpartition("/")[2])
    (root / ENVIRONMENT_MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return config_path
