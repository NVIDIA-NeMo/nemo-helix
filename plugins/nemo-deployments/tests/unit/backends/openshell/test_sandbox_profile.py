# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenShell sandbox image profile and its entry-point registration."""

from __future__ import annotations

from nemo_deployments_plugin.backends.openshell.sandbox_profile import PROFILE


def test_openshell_profile_shape() -> None:
    assert PROFILE.name == "openshell"
    assert "iproute2" in PROFILE.apt_packages
    assert "nftables" in PROFILE.apt_packages
    sandbox_users = [u for u in PROFILE.users if u.name == "sandbox"]
    assert len(sandbox_users) == 1
    assert sandbox_users[0].system is True
    assert sandbox_users[0].resolved_home() == "/home/sandbox"


def test_openshell_profile_is_discoverable() -> None:
    # Registered under nemo.sandbox_profiles so the agent packager can find it
    # by name without importing anything OpenShell-specific.
    from nemo_helix_plugin import discovery

    discovery.discover.cache_clear()
    discovery.discover_entry_points.cache_clear()
    profiles = discovery.discover_sandbox_profiles()

    assert "openshell" in profiles
    assert profiles["openshell"].name == "openshell"
