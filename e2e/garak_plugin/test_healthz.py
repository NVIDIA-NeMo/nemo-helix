# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E smoke test for the garak_plugin plugin healthz endpoint.

Verifies that the garak_plugin plugin loaded correctly in the running platform and
that its healthz response contains the expected keys. A 404 here means the
plugin failed to initialize.
"""

from nemo_helix_plugin.client.client import NemoClient


def test_garak_plugin_plugin_status(client: NemoClient) -> None:
    response = client._client.get("/apis/garak-plugin/v1/healthz")
    response.raise_for_status()
    status = response.json()

    assert status["plugin"] == "garak-plugin"
    assert status["status"] == "ok"
    assert "garak-plugin.scan" in status["jobs"]
    assert "garak_plugin_scan_config" in status["entities"]
    assert "garak_plugin_scan_target" in status["entities"]
