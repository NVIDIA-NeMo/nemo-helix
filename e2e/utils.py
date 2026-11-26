# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for e2e tests."""

import json
import os
from typing import Any

import pytest


def collect_sse_chunks(response) -> list[dict[str, Any]]:
    """Collect JSON ``data:`` events from a text/event-stream response."""
    chunks: list[dict[str, Any]] = []
    for line in response.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: ") :]))

    return chunks


def kube_target_from_env() -> tuple[str, str]:
    """The kubectl context and release namespace of the cluster under test.

    ``NHX_E2E_KUBE_CONTEXT`` is required, and its absence fails the test: a test that reaches into the
    cluster must never fall back to whatever the current context happens to be. ``NHX_E2E_KUBE_NAMESPACE``
    defaults to ``nemo-helix``, the namespace the kind lanes install into.
    """
    context = os.environ.get("NHX_E2E_KUBE_CONTEXT")
    if not context:
        pytest.fail(
            "NHX_E2E_KUBE_CONTEXT is required: name the kubectl context of the cluster the platform runs on "
            "(tests never use the current context)"
        )
    return context, os.environ.get("NHX_E2E_KUBE_NAMESPACE") or "nemo-helix"
