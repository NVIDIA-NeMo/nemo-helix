# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registers the UserSim pages in Studio. The bundle is built from ``web/`` with ``pnpm build``."""

from pathlib import Path

from nemo_helix_plugin.interface import StudioSpec


def get_studio_spec() -> StudioSpec:
    return StudioSpec(name="user-sim", bundle_path=Path(__file__).parent / "web" / "dist" / "index.js")
