# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for garak plugin e2e tests."""

import uuid


def unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def minimal_audit_config(**overrides) -> dict:
    """Minimal AuditConfig body for CRUD tests (no name/workspace — SDK takes those separately)."""
    base: dict = {
        "system": {},
        "run": {"generations": 1},
        "plugins": {"probe_spec": "test.Test", "detector_spec": "auto"},
        "reporting": {},
    }
    base.update(overrides)
    return base


def minimal_audit_target(**overrides) -> dict:
    """Minimal AuditTarget body (fake endpoint — safe for CRUD tests that never invoke garak)."""
    base: dict = {
        "type": "openai",
        "model": "gpt-4o-mini",
        "options": {},
    }
    base.update(overrides)
    return base
