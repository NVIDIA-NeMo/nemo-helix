# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which distributions install a Fabric harness, pinned to the versions this service runs.

Shared by the Harbor path (``fabric_package`` for the installed agent) and the Gym path (the
wheelhouse of a registered agent's environment package).
"""

from __future__ import annotations

import importlib.metadata

#: Fabric distribution extra that installs each harness adapter into the task container.
FABRIC_ADAPTER_EXTRAS: dict[str, str] = {
    "nvidia.fabric.langchain.deepagents": "deepagents",
    "nvidia.fabric.codex": "codex",
    "nvidia.fabric.claude": "claude",
    "nvidia.fabric.hermes": "hermes-agent",
}

#: Distributions a harness leaves unpinned but must match the service: deepagents' ``langchain-mcp-adapters``
#: admits an ``mcp`` major it cannot import.
HARNESS_COMPANION_PINS: dict[str, tuple[str, ...]] = {
    "nvidia.fabric.langchain.deepagents": ("mcp", "langchain-mcp-adapters"),
}


def fabric_harness_package(adapter_id: str) -> str:
    """The requirements that install ``adapter_id``'s harness, pinned to this service's own versions.

    One or more whitespace-separated specifiers: the ``nemo-fabric`` extra, plus any companion
    distribution the harness needs at the version the service runs, when the service has it.
    """
    extra = FABRIC_ADAPTER_EXTRAS.get(adapter_id)
    if extra is None:
        raise ValueError(
            f"no known Fabric package extra installs harness {adapter_id!r}; set `agent_kwargs.fabric_package` "
            "to the requirement that does"
        )
    requirements = [f"nemo-fabric[{extra},relay]=={importlib.metadata.version('nemo-fabric')}"]
    for name in HARNESS_COMPANION_PINS.get(adapter_id, ()):
        try:
            requirements.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return " ".join(requirements)


def fabric_harness_requirements(adapter_id: str, *, companions: bool = True) -> list[str]:
    """:func:`fabric_harness_package` as a list of specifiers.

    ``companions=False`` returns the harness extra alone, for an installation whose companion versions
    are fixed by the target environment's own pins rather than by this service's.
    """
    requirements = fabric_harness_package(adapter_id).split()
    return requirements if companions else requirements[:1]
