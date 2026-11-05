# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-managed native OPA using the shared Rego source."""

from .data import apply_embedded_policy_document, load_policy_data
from .engine import (
    OPAPolicy,
    PolicyEngineError,
    RegoPolicy,
    check_policy,
    close_policy,
    evaluate,
    get_policy,
    get_valid_entrypoints,
    invalidate_policy_data,
    is_policy_ready,
    reload_policy,
    set_policy_data,
    start_policy,
    validate_entrypoint,
)

__all__ = [
    # Engine
    "OPAPolicy",
    "PolicyEngineError",
    "RegoPolicy",
    "check_policy",
    "close_policy",
    "evaluate",
    "get_policy",
    "get_valid_entrypoints",
    "invalidate_policy_data",
    "is_policy_ready",
    "reload_policy",
    "set_policy_data",
    "start_policy",
    "validate_entrypoint",
    "load_policy_data",
    "apply_embedded_policy_document",
]
