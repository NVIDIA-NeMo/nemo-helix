# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render sandbox-runtime image profiles into Dockerfile fragments.

This is the packager's *mechanism*: it discovers a provider-supplied
:class:`~nemo_helix_plugin.sandbox.SandboxImageProfile` by name and turns its
declarative fields (apt packages, users) into shell fragments the Dockerfile
template interpolates. The provider (e.g. ``nemo-deployments[openshell]``) owns
*what* an image needs; this module owns *how* it is baked in, so
``nemo agents package`` never imports anything runtime-specific.
"""

from __future__ import annotations

import re

from nemo_helix_plugin.discovery import discover_sandbox_profiles
from nemo_helix_plugin.sandbox import SandboxImageProfile

# Profile values originate from trusted provider code, not end users, but these
# fragments are interpolated into a Dockerfile so validate against tight
# charsets as defense-in-depth against shell/Dockerfile injection.
_APT_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_-]*$")
_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")


def resolve_sandbox_profile(name: str) -> SandboxImageProfile:
    """Resolve a registered sandbox runtime by name, or raise ``ValueError``.

    The error lists the runtimes that are actually installed so the operator
    knows whether they are missing the provider package (e.g. the
    ``nemo-deployments[openshell]`` extra).
    """
    profiles = discover_sandbox_profiles()
    profile = profiles.get(name)
    if profile is None:
        available = ", ".join(sorted(profiles)) or "(none installed)"
        raise ValueError(
            f"unknown sandbox runtime {name!r}. Installed runtimes: {available}. "
            "Install the provider that registers it (e.g. 'nemo-deployments[openshell]')."
        )
    return profile


def _validate(token: str, pattern: re.Pattern[str], kind: str) -> str:
    if not pattern.match(token):
        raise ValueError(f"invalid {kind} {token!r} in sandbox profile {pattern.pattern!r}")
    return token


def render_apt_packages(profile: SandboxImageProfile) -> str:
    """Return the profile's apt packages as a space-joined, validated string."""
    return " ".join(_validate(pkg, _APT_RE, "apt package") for pkg in profile.apt_packages)


def render_user_setup(profile: SandboxImageProfile) -> str:
    """Return a single shell command creating the profile's users, or ``""``.

    Each user becomes ``groupadd ... && useradd ...``; multiple users are
    chained with ``&&``. The result is designed to sit after ``RUN `` on one
    logical Dockerfile line (with escaped continuations for readability).
    """
    parts: list[str] = []
    for user in profile.users:
        name = _validate(user.name, _NAME_RE, "user name")
        group = _validate(user.resolved_group(), _NAME_RE, "group name")
        home = _validate(user.resolved_home(), _PATH_RE, "home dir")
        shell = _validate(user.shell, _PATH_RE, "shell")
        sysflag = "--system " if user.system else ""
        groupadd = f"groupadd {sysflag}{group}"
        useradd = f"useradd {sysflag}--gid {group}"
        if user.create_home:
            useradd += f" --create-home --home-dir {home}"
        useradd += f" --shell {shell} {name}"
        parts.append(f"{groupadd} && \\\n    {useradd}")
    return " && \\\n    ".join(parts)
