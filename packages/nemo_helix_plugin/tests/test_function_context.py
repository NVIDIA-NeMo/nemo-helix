# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`nemo_helix_plugin.function_context.FunctionContext`.

The context is a tiny dataclass; the value of these tests is documenting
what's *not* on it (no ``storage``, no ``results``, no ``principal``)
so the surface stays stable as additions are weighed.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from nemo_helix_plugin.function_context import FunctionContext


def test_minimal_construction() -> None:
    ctx = FunctionContext(workspace="default")
    assert ctx.workspace == "default"
    assert ctx.request_id is None


def test_with_request_id() -> None:
    ctx = FunctionContext(workspace="prod", request_id="req-123")
    assert ctx.request_id == "req-123"


def test_kw_only_construction() -> None:
    """Workspace must be passed by keyword — the dataclass is ``kw_only``."""
    with pytest.raises(TypeError):
        FunctionContext("default")  # ty: ignore[missing-argument,too-many-positional-arguments]  — intentional misuse


def test_field_set_is_minimal() -> None:
    """Guard the surface — adding a field requires updating the docs and skill.

    See ``packages/nemo_helix_plugin/src/nemo_helix_plugin/function_context.py`` for
    the rationale (no ``storage`` / ``results`` / ``principal`` slot in
    this PR).
    """
    assert {f.name for f in fields(FunctionContext)} == {"workspace", "request_id"}
