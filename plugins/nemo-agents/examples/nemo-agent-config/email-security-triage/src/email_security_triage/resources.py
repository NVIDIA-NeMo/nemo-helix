# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access to the packaged Email Security Triage sample assets."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def sample_file(name: str) -> Traversable:
    """Return a packaged sample asset by name."""
    return files("email_security_triage").joinpath(name)
