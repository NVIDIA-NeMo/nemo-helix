# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LibraryConfig:
    """nemo-automodel recipe config written by the training runner."""

    config_dict: dict[str, Any]
    config_path: Path
