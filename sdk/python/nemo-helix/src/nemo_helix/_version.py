# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version as _package_version

__title__ = "nemo_helix"
try:
    __version__ = _package_version("nemo-helix")
except PackageNotFoundError:
    __version__ = "0.0.0"
# Injected at release time for non-production builds; None for RC and production releases.
__image_tag__: str | None = None
