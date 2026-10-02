# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixture re-exports for the example plugin integration tests.

The module-scope helpers (``_igw_app_context``, ``_igw_extra_services``)
are re-imported so pytest can resolve :func:`igw_plugin_harness`'s
dependency chain. The default empty ``_igw_extra_services`` tuple
applies — no services beyond IGW + Models are mounted.
"""

from nhx.core.inference_gateway.testing.fixtures import (
    _igw_app_context,
    _igw_extra_services,
    igw_plugin_harness,
)

__all__ = [
    "_igw_app_context",
    "_igw_extra_services",
    "igw_plugin_harness",
]
