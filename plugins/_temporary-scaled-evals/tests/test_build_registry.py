# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for registry readiness probe."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build import registry
from scaled_evals.api.settings import settings


def test_check_registry_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "image_registry", "registry:5000")
    monkeypatch.setattr(settings, "registry_insecure", True)

    class _Resp:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Resp())

    registry.check_registry()


def test_check_registry_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "image_registry", "registry:5000")
    monkeypatch.setattr(settings, "registry_insecure", True)

    def _fail(url: str, timeout: float) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _fail)

    with pytest.raises(RuntimeError, match="connection refused"):
        registry.check_registry()


def test_registry_v2_url_uses_https_when_not_insecure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "image_registry", "nvcr.io/nvidia/foo")
    monkeypatch.setattr(settings, "registry_insecure", False)
    assert registry._registry_v2_base_url() == "https://nvcr.io/nvidia/foo"
