# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled NeMo Helix skills, exposed via the ``nemo.skills`` entry point.

This package serves a dual role:

* As a Python package, it exports :func:`skills_dir`, the entry-point callable
  registered under ``nemo.skills`` in the package's ``pyproject.toml``. The
  skills registry uses the same discovery mechanism for both built-in and
  plugin-provided skills, so the platform appears as just one provider among
  many.

* As a data directory, it holds the bundled ``<skill-name>/SKILL.md`` files
  shipped with NeMo Helix. The vendor tool copies the entire directory into
  the SDK package, so the SDK ships these alongside the generated client code.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["skills_dir"]


def skills_dir() -> Path:
    """Return the directory containing this package's bundled skills."""
    return Path(__file__).parent
