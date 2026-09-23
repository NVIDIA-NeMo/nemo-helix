# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration package for nemo_helix."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_helix_ext.config.models import ConfigParams, Context

__all__ = ["get_context"]


def get_context(
    config_path: Path | None = None,
    overrides: ConfigParams | None = None,
) -> Context:
    from nemo_helix_ext.config.config import get_context as _get_context

    return _get_context(config_path=config_path, overrides=overrides)
