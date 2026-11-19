# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform version and revision helpers."""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version

_UNSTAMPED_VERSION = "0.0.0"
_RELEASE_VERSION_PATTERN = re.compile(r"v?\d+(\.\d+)+\S*")


def _resolve_version() -> str:
    env_version = os.environ.get("NHX_PLATFORM_VERSION", "").strip()
    if _RELEASE_VERSION_PATTERN.fullmatch(env_version) and env_version != _UNSTAMPED_VERSION:
        return env_version
    try:
        package_version = version("nemo-helix").strip()
    except PackageNotFoundError:
        package_version = ""
    if package_version and package_version != _UNSTAMPED_VERSION:
        return package_version
    return "dev"


__version__ = _resolve_version()


def get_platform_version() -> str:
    return __version__


def get_revision() -> str:
    return os.environ.get("NHX_CODE_REVISION", "dev").strip() or "dev"
