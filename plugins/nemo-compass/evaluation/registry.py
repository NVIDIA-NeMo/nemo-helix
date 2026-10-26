# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the evaluation subject registry (evaluations.toml)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Subject:
    """One evaluation subject: a name, a type, and its merged config."""

    name: str
    type: str
    config: dict[str, Any]


def load_registry(path: Path) -> dict[str, Subject]:
    """Parse the registry. Top-level scalars are shared defaults merged into every
    subject (e.g. ``base_url``); each table is a subject; a per-subject key overrides."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = {k: v for k, v in data.items() if not isinstance(v, dict)}
    subjects: dict[str, Subject] = {}
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        merged = {**defaults, **cfg}
        subjects[name] = Subject(name=name, type=str(merged.get("type", "")), config=merged)
    return subjects
