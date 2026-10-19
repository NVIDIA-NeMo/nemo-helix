# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rejected credential write must not echo the submitted secret back."""

from __future__ import annotations

import pytest

# This plugin is absent from `enabled-plugins`, so a default sync leaves it and its database
# driver uninstalled and the repo-wide test run still sweeps this directory. Skip rather than
# error there; the job that owns these tests installs the `scaled-evals` group first.
try:
    from fastapi.testclient import TestClient
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
    from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
    from scaled_evals.api.db import get_db
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

SECRET = "sk-do-not-echo-this-value"


def test_rejected_credential_write_does_not_echo_the_secret() -> None:
    """FastAPI's default 422 includes the offending value in ``input``.

    On this route that value is the whole submitted model, so the plaintext
    ``key`` lands in the response body. Sending both ``key`` and ``yaml``
    trips the XOR validator and is the cheapest way to reach that path.
    """
    app = NemoServiceAdapter(ScaledEvalsService()).create_app()
    # Body validation happens after dependency solving, so without this the
    # unreachable-Postgres 503 masks the 422 under test.
    app.dependency_overrides[get_db] = lambda: None
    client = TestClient(app)

    response = client.post(
        "/v1/credentials",
        json={"name": "leak-probe", "provider": "openai", "key": SECRET, "yaml": "also-set"},
    )

    assert response.status_code == 422, response.status_code
    assert SECRET not in response.text
    # The structural half of the error is what callers need, so keep it.
    assert response.json()["detail"][0]["loc"]
