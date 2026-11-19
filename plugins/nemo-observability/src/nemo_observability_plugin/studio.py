# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio web UI registration for the observability plugin."""

from __future__ import annotations

from pathlib import Path

from nemo_helix_plugin.interface import StudioSpec


def get_studio_spec() -> StudioSpec:
    # 'self' only: the embedded UI is always served through this plugin's same-origin proxy,
    # never framed from the upstream's own origin.
    return StudioSpec(
        name="observability",
        bundle_path=Path(__file__).parent / "web" / "dist" / "index.js",
        frame_src=("'self'",),
    )
