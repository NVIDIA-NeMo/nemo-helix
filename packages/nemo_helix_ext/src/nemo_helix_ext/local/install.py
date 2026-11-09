# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How this NeMo Helix install was created."""

from __future__ import annotations

from pathlib import Path
from sys import prefix as sys_prefix


def services_extra_install_command() -> str:
    """Return the command that adds the packaged service dependencies here.

    ``uv tool install`` environments are managed by uv and are not meant to be
    edited with pip; re-running the tool install with the extra upgrades them
    in place. uv marks such an environment with a ``uv-receipt.toml`` at its
    root, which is what distinguishes it from an ordinary virtual environment.
    """
    if (Path(sys_prefix) / "uv-receipt.toml").is_file():
        return "uv tool install 'nemo-helix[all]'"
    return "pip install 'nemo-helix[all]'"
