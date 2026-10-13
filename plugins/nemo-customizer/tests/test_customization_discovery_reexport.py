# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_customizer.discovery import (
    CUSTOMIZATION_CONTRIBUTORS_GROUP,
    discover_customization_contributor_classes,
    discover_customization_contributors,
)
from nemo_helix_plugin.discovery import (
    discover_customization_contributors as platform_discover,
)


def test_reexport_matches_platform_discovery() -> None:
    assert discover_customization_contributors is platform_discover
    assert CUSTOMIZATION_CONTRIBUTORS_GROUP == "nemo.customization.contributors"
    discover_customization_contributors.cache_clear()
    assert isinstance(discover_customization_contributors(), dict)
    assert isinstance(discover_customization_contributor_classes(), dict)
