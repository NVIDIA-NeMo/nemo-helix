# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-plugin contract for sandbox-runtime image requirements.

A sandbox runtime (e.g. OpenShell) runs an agent image under a supervisor that
imposes requirements on the image: a dedicated non-root user resolved by name,
extra OS packages for its network/policy setup, and so on. Those requirements
are *provider* knowledge, but the *packager* (``nemo agents package``) is what
physically bakes them into the image.

To keep the packager provider-agnostic, a provider plugin registers one
:class:`SandboxImageProfile` under the ``nemo.sandbox_profiles`` entry-point
group; the packager discovers it by name (via
:func:`nemo_helix_plugin.discovery.discover_sandbox_profiles`) and renders
whatever it is handed. The packager never imports the provider.

These are plain, dependency-light dataclasses so a provider can describe its
profile without importing the runtime's heavy client libraries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxUser:
    """A user (and its primary group) a sandbox runtime requires in the image.

    A runtime typically resolves the user by name, so the uid is not load-bearing;
    ``system=True`` keeps it out of the uid range a packager may reclaim for its own
    default runtime user.
    """

    name: str
    group: str | None = None
    """Primary group name. Defaults to :attr:`name` when ``None``."""
    system: bool = True
    create_home: bool = True
    home: str | None = None
    """Home directory. Defaults to ``/home/<name>`` when ``None``."""
    shell: str = "/bin/bash"

    def resolved_group(self) -> str:
        return self.group or self.name

    def resolved_home(self) -> str:
        return self.home or f"/home/{self.name}"


@dataclass(frozen=True)
class SandboxImageProfile:
    """Declarative image requirements for running under a sandbox runtime.

    Registered by a provider plugin under the ``nemo.sandbox_profiles``
    entry-point group and consumed by ``nemo agents package
    --sandbox-runtime <name>``.
    """

    name: str
    description: str = ""
    apt_packages: tuple[str, ...] = ()
    users: tuple[SandboxUser, ...] = ()
