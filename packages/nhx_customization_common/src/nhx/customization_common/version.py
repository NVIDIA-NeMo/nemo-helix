# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version marker for the nhx-customization-common shared library."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("nhx-customization-common")
except PackageNotFoundError:
    __version__ = "0.0.0"
