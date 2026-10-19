# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin manifest — lightweight identity record derived by the platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PluginManifest:
    """Lightweight identity record for an installed NeMo Helix plugin.

    The platform derives one manifest per installed plugin by scanning all
    known surface entry-point groups (``nemo.services``, ``nemo.cli``, etc.).
    Plugin authors do **not** declare this — it is assembled automatically
    from the installing distribution's package metadata.

    Attributes:
        name: Entry-point key (e.g. ``"example"``).
        version: Distribution ``Version`` field, or ``""`` if unavailable.
        description: Distribution ``Summary`` field, or ``""`` if unavailable.
    """

    name: str
    version: str
    description: str = field(default="")


@dataclass
class StudioSpec:
    """Describes a plugin's Studio web UI contribution.

    Registered under the ``nemo.studio`` entry-point group as a zero-argument
    callable that returns an instance of this class.

    Attributes:
        name: Entry-point key matching the plugin name, e.g. ``"example"``.
        bundle_path: Absolute path to the plugin's built ``web/dist/index.js``
            on disk. Use ``Path(__file__).parent... / "web" / "dist" / "index.js"``
            so the path resolves correctly for both editable and wheel installs.
            ``None`` for plugins that have no web UI (Python-only plugins that
            still wish to appear in the ``/apis/plugins`` manifest).
    """

    name: str
    bundle_path: Path | None = field(default=None)
