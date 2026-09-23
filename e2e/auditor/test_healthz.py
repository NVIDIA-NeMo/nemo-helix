# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E smoke test for the auditor plugin healthz endpoint.

Verifies that the auditor plugin loaded correctly in the running platform and
that its healthz response contains the expected keys. A 404 here means the
plugin failed to initialize.
"""

from nemo_helix import NeMoHelix


def test_auditor_plugin_status(sdk: NeMoHelix) -> None:
    status = sdk.auditor.plugin_status()

    assert status["plugin"] == "auditor"
    assert status["status"] == "ok"
    assert "auditor.audit" in status["jobs"]
    assert "auditor_audit_config" in status["entities"]
    assert "auditor_audit_target" in status["entities"]
