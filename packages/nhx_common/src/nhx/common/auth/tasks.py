# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Authorization primitives for NeMo Helix tasks.
Auth context is passed to tasks via environment variables.
"""

from __future__ import annotations

from .models import NHX_PRINCIPAL_ENVVAR, Principal


def principal_from_env(env_var_name: str = NHX_PRINCIPAL_ENVVAR) -> Principal | None:
    """Checks env var for the auth principal and returns it.

    When a task runs in a container, the platform propagates the job creator's principal
    via the NHX_PRINCIPAL environment variable (JSON serialized `Principal` model):

        NHX_PRINCIPAL = {"id": "user@example.com", "email": "user@example.com", "groups": ["team-a"]}

    This helper returns that principal if it exists, or None otherwise. If NHX_PRINCIPAL is not set or invalid, return None.
    If the JSON is malformed, raises ValueError.

    Note: this doesn't set or modify the auth context var, it only returns the principal.
    """
    return Principal.from_env_var(env_var_name)
