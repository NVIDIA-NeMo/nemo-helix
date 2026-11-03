# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Dockerfile an agent layer builds: the task's own environment, then the agent installed into it.

Builder's builds hold no registry credential, so a layer can't start ``FROM`` the environment image the platform
built: it builds the environment again, from the task's own Dockerfile (a prebuilt image is public, so it starts
``FROM`` that), and appends one step to the final stage. That step runs the agent's own setup as root, with the
agent installer (a standalone Python with Harbor) and the agent to install read from Builder mounts, where the
generated Dockerfile itself is too. They are outside
the build context, so the task's ``COPY . .`` doesn't copy them and its ``.dockerignore`` doesn't hide them, and the
image holds none of them. The setup switches to the task's agent user for what it installs as the agent. The final
stage's ``USER`` is restored.
"""

from __future__ import annotations

import json
import re

#: Where the mounts' archive holds the installer, and where the build mounts it: the venv's interpreter path is
#: absolute.
INSTALLER_DIR = "nhx-agent-installer"
INSTALLER_MOUNT = "/opt/nhx-agent-installer"
#: Where the mounts' archive holds the agent to install, and where the build mounts it.
AGENT_DIR = "nhx-agent-layer"
AGENT_MOUNT = "/opt/nhx-agent-layer"
AGENT_FILE = "agent.json"
#: The generated Dockerfile's name, beside the agent to install: never in the context, where the task's
#: ``COPY . .`` would copy it.
LAYER_DOCKERFILE = "Dockerfile.nhx-agent-layer"
INSTALL_MODULE = "nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer_install"

_FROM = re.compile(r"^\s*FROM\s", re.IGNORECASE)
_USER = re.compile(r"^\s*USER\s+(?P<user>\S+)", re.IGNORECASE)


def layer_dockerfile(
    *, environment_dockerfile: str | None, base_image: str | None, image_user: str | None = None
) -> str:
    """The layer's Dockerfile, over the task's ``environment_dockerfile``, or ``FROM base_image`` for a prebuilt one.

    ``image_user`` is the environment image's ``USER`` as Builder recorded it, inherited ones included (empty for
    root), which the layer is left running as. When it wasn't recorded, the final stage's own ``USER`` is.
    """
    if (environment_dockerfile is None) == (base_image is None):
        raise ValueError("give exactly one of environment_dockerfile and base_image")
    base = environment_dockerfile if environment_dockerfile is not None else f"FROM {base_image}\n"
    user = (image_user or None) if image_user is not None else final_stage_user(base)
    install = (
        f"RUN {json.dumps([f'{INSTALLER_MOUNT}/venv/bin/python', '-m', INSTALL_MODULE, f'{AGENT_MOUNT}/{AGENT_FILE}'])}"
    )
    lines = [
        base.rstrip("\n"),
        "",
        "# Installed by NeMo Helix: the run's agent, so trials needn't install it.",
        "USER root",
    ]
    lines.append(install)
    if user is not None:
        lines.append(f"USER {user}")
    return "\n".join(lines) + "\n"


def final_stage_user(dockerfile: str) -> str | None:
    """The user the final stage of ``dockerfile`` leaves set, or None when it sets none."""
    user = None
    for line in _logical_lines(dockerfile):
        if _FROM.match(line):
            user = None
        elif match := _USER.match(line):
            user = match["user"]
    return user


def _logical_lines(dockerfile: str) -> list[str]:
    """Instructions with their ``\\`` continuations joined; comments dropped."""
    joined: list[str] = []
    current = ""
    for raw in dockerfile.splitlines():
        if raw.lstrip().startswith("#") and not current:
            continue
        if raw.rstrip().endswith("\\"):
            current += raw.rstrip()[:-1] + " "
            continue
        joined.append(current + raw)
        current = ""
    if current:
        joined.append(current)
    return joined
