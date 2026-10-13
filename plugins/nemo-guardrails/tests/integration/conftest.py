# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for the guardrails plugin integration tests.

Re-exports the IGW harness fixtures so test modules don't have to
import them. ``_igw_extra_services`` is overridden below to mount
:class:`GuardrailsService` on the module-scoped app — entity-backed
guardrail-config tests need its CRUD routes. The module-scope helpers
(``_igw_app_context``, ``_igw_loopback_context``) are re-imported so
pytest can resolve the dependency chain from this conftest's scope.

``HF_HUB_OFFLINE`` is set at conftest import time, **before** the
``GuardrailsService`` import below — importing ``GuardrailsService``
transitively imports ``nemoguardrails``, which reaches HuggingFace at
import time if not told to stay offline. A function-scoped autouse
``monkeypatch`` fixture would be too late: it doesn't run until after
the module-scoped fixture setup that triggers these imports.
"""

import os

import pytest
from nhx.core.inference_gateway.testing.fixtures import (
    _igw_app_context,
    _igw_loopback_context,
    igw_loopback_harness,
    igw_plugin_harness,
)
from nhx.testing.client import ServiceFactory

# Must precede the ``nemoguardrails``-pulling import below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from nhx.guardrails.service import GuardrailsService  # noqa: E402

__all__ = [
    "_igw_app_context",
    "_igw_loopback_context",
    "igw_loopback_harness",
    "igw_plugin_harness",
]


@pytest.fixture(scope="module")
def _igw_extra_services() -> tuple[ServiceFactory, ...]:
    """Mount :class:`GuardrailsService` on the module-scoped IGW + Models app.

    Every integration test here gets Guardrails CRUD routes whether it
    uses them or not — the startup cost amortises across the module,
    so files that only touch inline configs pay almost nothing extra.
    """
    return (GuardrailsService,)
