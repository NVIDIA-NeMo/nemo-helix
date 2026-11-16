# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rejected request must not echo a submitted secret back."""

from __future__ import annotations

import pytest

# This plugin is absent from `enabled-plugins`, so a default sync leaves it and its database
# driver uninstalled and the repo-wide test run still sweeps this directory. Skip rather than
# error there; the job that owns these tests installs the `scaled-evals` group first.
try:
    from fastapi.testclient import TestClient
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
    from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
    from scaled_evals.api.db import get_db, get_request_db
    from scaled_evals.api.runnability import caller_secret_readable
    from scaled_evals.api.tenancy import caller_is_admin
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

SECRET = "sk-do-not-echo-this-value"


def test_raw_key_in_credential_reference_is_not_echoed() -> None:
    """FastAPI's default 422 includes the offending value in ``input``.

    A caller who pastes the API key itself where a secret reference belongs
    would get the plaintext key back in the response body.
    """
    app = NemoServiceAdapter(ScaledEvalsService()).create_app()
    # Body validation happens after dependency solving, so without this the
    # unreachable-Postgres 503 masks the 422 under test.
    app.dependency_overrides[get_db] = lambda: None
    app.dependency_overrides[get_request_db] = lambda: None
    app.dependency_overrides[caller_is_admin] = lambda: True
    app.dependency_overrides[caller_secret_readable] = lambda: lambda _ref: True
    client = TestClient(app)

    response = client.post(
        "/v1/evaluations",
        json={"name": "leak-probe", "task_id": "task_1", "task_revision": 1, "credentials": {"openai": SECRET}},
    )

    assert response.status_code == 422, response.status_code
    assert SECRET not in response.text
    # The structural half of the error is what callers need, so keep it.
    assert response.json()["detail"][0]["loc"]
