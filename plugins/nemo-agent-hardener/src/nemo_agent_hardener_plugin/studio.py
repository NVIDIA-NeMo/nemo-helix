# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio web UI registration for the agent-hardener plugin."""

from __future__ import annotations

from pathlib import Path

from nemo_platform_plugin.interface import StudioSpec


def get_studio_spec() -> StudioSpec:
    """Return the StudioSpec for the agent-hardener plugin's web UI.

    Uses ``__file__`` so the path resolves correctly for both editable
    (``uv pip install -e``) and wheel installs.
    """
    bundle_path = Path(__file__).parent / "web" / "dist" / "index.js"
    return StudioSpec(name="agent-hardener", bundle_path=bundle_path)
