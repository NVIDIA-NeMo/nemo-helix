# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Fabric adapter IDs a harness ``kind`` may name.

A kind is accepted when it is a stock short name, a stock adapter ID, or the ID
declared by an adapter manifest installed anywhere below the Python environment's
``share/nemo-fabric/`` data directory. Plugins ship their adapters there (Compass
installs ``adapters/helix/compass/analyst.fabric-adapter.json``), and Fabric discovers
installed adapters from the same tree, so an ID accepted here is one Fabric can load.

When ``ADAPTER_PYTHON`` names another interpreter, Fabric discovers adapters
from that interpreter's environment, so its tree is scanned too.

An adapter installed only in a custom agent image is invisible to the API
server, so a deploy or job that names its own image skips this check at create.
The check still runs inside the image, whose manifests are the ones Fabric loads.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import sysconfig
from pathlib import Path

from nemo_agents_plugin.agent_config import AgentConfig

logger = logging.getLogger(__name__)

HARNESS_ADAPTER_IDS = {
    "claude": "nvidia.fabric.claude",
    "codex": "nvidia.fabric.codex",
    "deepagents": "nvidia.fabric.langchain.deepagents",
    "hermes": "nvidia.fabric.hermes",
    "remote-agent": "nvidia.fabric.remote-agent",
    "pi": "nvidia.fabric.pi",
    "mini-swe-agent": "nvidia.fabric.mini-swe-agent",
    "nooa-bench-agent": "nvidia.fabric.nooa.bench-agent",
}

# Fabric searches this whole tree, at any depth, so a package may lay its manifests out however it likes.
ADAPTER_MANIFEST_ROOT = Path(sysconfig.get_path("data")) / "share" / "nemo-fabric"
# The tree also holds workflow target descriptors (``*.fabric-target.json``), which are not adapters.
ADAPTER_MANIFEST_PATTERN = "*.fabric-adapter.json"
# Fabric's setting for an interpreter whose environment holds the adapters (our Hermes images use one).
ADAPTER_PYTHON_ENV = "ADAPTER_PYTHON"

#: Appended where naming an image is the fix: deploy and execute job create.
CUSTOM_IMAGE_HINT = "An adapter installed only in a custom agent image is accepted when the request names that image."


class UnknownAdapterError(ValueError):
    """Raised when a harness kind names neither a stock nor an installed adapter."""


def installed_adapter_ids(root: Path | None = None) -> frozenset[str]:
    """Return the ``adapter_id`` of every adapter manifest installed under ``root``.

    ``root`` defaults to ``ADAPTER_MANIFEST_ROOT`` plus, when ``ADAPTER_PYTHON`` is
    set, that interpreter's tree. Each root is scanned once per process: installing
    an adapter means installing a package, which already takes a restart to load.

    The ID comes from the manifest, not its path (``nooa/nooa-bench.fabric-adapter.json``
    holds ``nvidia.fabric.nooa.bench-agent``), manifests may sit at any depth
    (``helix/compass/``), and one directory may hold several manifests (``nooa/``
    holds ``nooa`` and ``nooa-bench``). An unreadable
    manifest is skipped so one broken plugin can't block every other adapter.
    """
    if root is not None:
        return _scan_adapter_manifests(root)
    roots = [ADAPTER_MANIFEST_ROOT]
    adapter_python_root = _adapter_python_manifest_root(os.environ.get(ADAPTER_PYTHON_ENV, ""))
    if adapter_python_root is not None:
        roots.append(adapter_python_root)
    return frozenset().union(*(_scan_adapter_manifests(manifest_root) for manifest_root in roots))


@functools.cache
def _adapter_python_manifest_root(python: str) -> Path | None:
    """Return the manifest tree of the environment ``python`` runs in, or ``None`` if unset or unusable."""
    if not python:
        return None
    try:
        result = subprocess.run(
            [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('data'))"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Can't find Fabric adapters for %s=%s: %s", ADAPTER_PYTHON_ENV, python, error)
        return None
    return Path(result.stdout.strip()) / "share" / "nemo-fabric"


@functools.cache
def _scan_adapter_manifests(root: Path) -> frozenset[str]:
    adapter_ids: set[str] = set()
    for manifest in sorted(root.rglob(ADAPTER_MANIFEST_PATTERN)):
        try:
            adapter_id = json.loads(manifest.read_text(encoding="utf-8")).get("adapter_id")
        except (OSError, UnicodeDecodeError, ValueError, AttributeError) as error:
            logger.warning("Skipping unreadable Fabric adapter manifest %s: %s", manifest, error)
            continue
        if not isinstance(adapter_id, str) or not adapter_id:
            logger.warning("Skipping Fabric adapter manifest %s: it declares no adapter_id.", manifest)
            continue
        adapter_ids.add(adapter_id)
    return frozenset(adapter_ids)


def accepted_adapter_ids(root: Path | None = None) -> frozenset[str]:
    """Return every adapter ID a harness kind may name directly."""
    return frozenset(HARNESS_ADAPTER_IDS.values()) | installed_adapter_ids(root)


def resolve_adapter_id(kind: str, root: Path | None = None) -> str:
    """Return the Fabric adapter ID for a harness ``kind``.

    Raises:
        UnknownAdapterError: ``kind`` is neither a stock short name nor an accepted adapter ID.
    """
    if kind in HARNESS_ADAPTER_IDS:
        return HARNESS_ADAPTER_IDS[kind]
    accepted = accepted_adapter_ids(root)
    if kind in accepted:
        return kind
    raise UnknownAdapterError(
        f"Unsupported harness kind {kind!r}. Supported harness kinds: {', '.join(sorted(HARNESS_ADAPTER_IDS))}; "
        f"or an installed Fabric adapter ID: {', '.join(sorted(accepted))}."
    )


def validate_harness_kinds(config: AgentConfig) -> None:
    """Check that every configured harness names an accepted adapter.

    Run at deploy and job create so a typo fails the request, not the pod that
    later runs the agent. Every harness is checked, not just the default, since a
    deployment or invocation may select any of them.

    Raises:
        UnknownAdapterError: A harness kind names no accepted adapter.
    """
    for name, harness in config.harnesses.items():
        try:
            resolve_adapter_id(harness.kind)
        except UnknownAdapterError as error:
            raise UnknownAdapterError(f"Harness {name!r}: {error}") from error
