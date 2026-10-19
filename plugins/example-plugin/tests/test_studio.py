# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_example_plugin.studio."""

from __future__ import annotations

from nemo_example_plugin.studio import get_studio_spec
from nemo_helix_plugin.interface import StudioSpec


def test_get_studio_spec_returns_studio_spec():
    spec = get_studio_spec()
    assert isinstance(spec, StudioSpec)
    assert spec.name == "example"
    assert spec.bundle_path is not None
    assert spec.bundle_path.name == "index.js"
    assert "web" in spec.bundle_path.parts
    assert "dist" in spec.bundle_path.parts
