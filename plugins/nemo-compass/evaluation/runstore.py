# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persist and load a benchmark's last produced run (the bridge from `run` to `analyze`)."""

import json
from collections.abc import Mapping
from pathlib import Path


def save_run(path: Path, record: Mapping[str, object]) -> None:
    """Write a run record as JSON, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2), encoding="utf-8")


def load_run(path: Path) -> dict[str, object] | None:
    """Return the run record at ``path``, or ``None`` if it does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
