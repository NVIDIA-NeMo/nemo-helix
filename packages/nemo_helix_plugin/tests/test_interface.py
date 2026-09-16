# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_helix_plugin.interface."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from nemo_helix_plugin.interface import StudioSpec


def test_studio_spec_stores_name_and_bundle_path():
    path = Path("/some/web/dist/index.js")
    spec = StudioSpec(name="example", bundle_path=path)
    assert spec.name == "example"
    assert spec.bundle_path == path


def test_studio_spec_is_dataclass():
    """StudioSpec should be a plain dataclass — no pydantic, no validation."""
    assert dataclasses.is_dataclass(StudioSpec)
    assert len(dataclasses.fields(StudioSpec)) == 2
