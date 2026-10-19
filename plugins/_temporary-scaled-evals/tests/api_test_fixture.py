# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mounted-app fixture for scaled-evals API tests."""

import pytest

pytest.importorskip("scaled_evals")

from fastapi.testclient import TestClient
from nemo_scaled_evals_plugin.service import ScaledEvalsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter

app = NemoServiceAdapter(ScaledEvalsService()).create_app()
v1 = app
client = TestClient(app)
