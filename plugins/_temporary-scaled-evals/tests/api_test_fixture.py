# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mounted-app fixture for scaled-evals API tests."""

import pytest

pytest.importorskip("scaled_evals")

from fastapi.testclient import TestClient
from nemo_scaled_evals_plugin.service import ScaledEvalsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from scaled_evals.api.runnability import caller_secret_readable

app = NemoServiceAdapter(ScaledEvalsService()).create_app()
# No platform Secrets service runs under unit tests; a test that needs an
# unreadable secret overrides this itself.
app.dependency_overrides[caller_secret_readable] = lambda: lambda _ref: True
v1 = app
client = TestClient(app)
