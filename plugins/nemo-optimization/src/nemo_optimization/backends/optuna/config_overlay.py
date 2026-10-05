# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for shared dotted-path overlay helpers."""

from __future__ import annotations

from nemo_optimization.config_overlay import (
    apply_suggestions,
    get_by_dotted_path,
    nest_dotted_paths,
    set_by_dotted_path,
    strip_optimizer_only_fields,
    suggestions_to_profile_overlay,
)

__all__ = [
    "apply_suggestions",
    "get_by_dotted_path",
    "nest_dotted_paths",
    "set_by_dotted_path",
    "strip_optimizer_only_fields",
    "suggestions_to_profile_overlay",
]
