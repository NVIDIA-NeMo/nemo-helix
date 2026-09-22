# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized temporary exclusions for automatic root test discovery."""

from importlib.util import find_spec
from pathlib import Path

TEST_DISCOVERY_EXCLUSIONS: dict[Path, str] = {
    Path("plugins/nemo-agents/tests"): "Not installed by the normal root uv environment yet.",
}

if find_spec("nemo_insights_plugin") is None:
    TEST_DISCOVERY_EXCLUSIONS[Path("plugins/nemo-insights/tests")] = (
        "The optional insights dependency group is not installed in the root test environment."
    )
