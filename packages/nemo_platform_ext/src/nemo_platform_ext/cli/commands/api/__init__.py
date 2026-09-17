# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from nemo_platform_ext.cli.manifest import TopLevelEntry

API_TOP_LEVEL_ENTRIES = (
    TopLevelEntry(
        import_path=f"{__package__}.adapters:app",
        name="adapters",
        help="Manage adapters.",
        panel="Core plugins",
        kind="group",
        hidden=True,
    ),
    TopLevelEntry(
        import_path=f"{__package__}.experiments:app",
        name="experiments",
        help="Manage experiments.",
        panel="Functional plugins",
        kind="group",
        hidden=False,
    ),
    TopLevelEntry(
        import_path=f"{__package__}.files:app",
        name="files",
        help="Manage files.",
        panel="Core plugins",
        kind="group",
        hidden=False,
    ),
    TopLevelEntry(
        import_path=f"{__package__}.guardrail:app",
        name="guardrail",
        help="Manage guardrails.",
        panel="Functional plugins",
        kind="group",
        hidden=False,
    ),
    TopLevelEntry(
        import_path=f"{__package__}.inference:app",
        name="inference",
        help="Inference operations.",
        panel="Core plugins",
        kind="group",
        hidden=False,
    ),
    TopLevelEntry(
        import_path=f"{__package__}.intake:app",
        name="intake",
        help="Intake operations.",
        panel="Functional plugins",
        kind="group",
        hidden=False,
    ),
    TopLevelEntry(
        import_path=f"{__package__}.models:app",
        name="models",
        help="Manage models.",
        panel="Core plugins",
        kind="group",
        hidden=False,
    ),
)
