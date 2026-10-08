# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import nemo_opensandbox_ext as ext
import pytest
from _ext_doubles import NAMESPACE
from _server_doubles import FakeK8sClient


@pytest.fixture
def k8s() -> FakeK8sClient:
    """The fake Kubernetes API the provider talks to."""
    return FakeK8sClient()


@pytest.fixture
def provider(k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch) -> ext.NemoServicesProvider:
    """The real provider over ``k8s``, checking pod status on every read instead of every 2 s."""
    monkeypatch.setattr(ext, "STATUS_CHECK_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(ext, "_active_provider", None)
    p = ext.NemoServicesProvider(k8s)  # ty: ignore[invalid-argument-type]
    p.namespace = NAMESPACE
    return p
