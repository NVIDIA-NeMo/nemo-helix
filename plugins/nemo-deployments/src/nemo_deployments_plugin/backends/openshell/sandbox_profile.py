# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell sandbox image profile.

Registered under the ``nemo.sandbox_profiles`` entry-point group so
``nemo agents package --sandbox-runtime openshell`` can bake OpenShell's
supervisor requirements into an agent image without the packager importing
anything OpenShell-specific.

Intentionally dependency-light (no gRPC / client imports) so profile discovery
works even when the ``nemo-deployments[openshell]`` extra is not installed.
"""

from __future__ import annotations

from nemo_helix_plugin.sandbox import SandboxImageProfile, SandboxUser

# Without ``nftables`` the supervisor falls back to a degraded policy mode. The
# glibc >= 2.39 floor its binary needs is described rather than enforced: the
# packager's default base (noble) already satisfies it.
PROFILE = SandboxImageProfile(
    name="openshell",
    description=(
        "OpenShell sandbox supervisor: non-root 'sandbox' user, iproute2 for "
        "network-namespace setup, nftables for full policy enforcement, and a "
        "glibc >= 2.39 base image."
    ),
    apt_packages=("iproute2", "nftables"),
    users=(
        SandboxUser(
            name="sandbox",
            system=True,
            create_home=True,
            home="/home/sandbox",
            shell="/bin/bash",
        ),
    ),
)
