# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedded Policy Decision Point interpreting the shared Rego source with regopy."""

from .data import apply_embedded_policy_document, load_policy_data
from .engine import (
    OPAPolicy,
    PolicyEngineError,
    RegoPolicy,
    evaluate,
    get_policy,
    get_valid_entrypoints,
    reload_policy,
    set_policy_data,
    validate_entrypoint,
)

__all__ = [
    # Engine
    "OPAPolicy",
    "PolicyEngineError",
    "RegoPolicy",
    "evaluate",
    "get_policy",
    "get_valid_entrypoints",
    "reload_policy",
    "set_policy_data",
    "validate_entrypoint",
    "load_policy_data",
    "apply_embedded_policy_document",
]
