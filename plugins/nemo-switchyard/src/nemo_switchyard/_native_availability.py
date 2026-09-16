# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy detection of the native Switchyard Python bindings.

Default CI and the 0.7 image ship May ``switchyard.lib``. Native algorithms live
on ``switchyard_rust`` and must never be imported at module load of this plugin.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import cache
from types import ModuleType


@cache
def native_rust_available() -> bool:
    """Return True when the ``switchyard_rust`` distribution is importable."""
    return importlib.util.find_spec("switchyard_rust") is not None


def load_libsy() -> ModuleType:
    """Import ``switchyard_rust.libsy`` (CallModel / Done host surface).

    Raises ImportError when the native wheel is absent so callers can map that
    to HTTP 400 at VirtualModel upsert time.
    """
    return importlib.import_module("switchyard_rust.libsy")
