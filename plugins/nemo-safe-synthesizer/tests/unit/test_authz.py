# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the safe-synthesizer plugin (every route ruled, no problems)."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_safe_synthesizer_plugin.service import SafeSynthesizerService


def test_safe_synthesizer_authz_derivation_has_no_problems() -> None:
    pytest.importorskip("nemo_safe_synthesizer.config.job")
    contrib, problems, _warnings = _derive_service_contribution(SafeSynthesizerService())
    assert problems == []

    # Single job collection under the flat safe-synthesizer namespace.
    assert {
        "safe-synthesizer.create",
        "safe-synthesizer.list",
        "safe-synthesizer.read",
        "safe-synthesizer.delete",
        "safe-synthesizer.cancel",
    } <= set(contrib.permissions)

    # Pin verb->permission, including a custom result-download route (a read action), so a
    # mis-stamp is localized to this plugin.
    jobs = "/apis/safe-synthesizer/v2/workspaces/{workspace}/jobs"
    assert contrib.endpoints[jobs]["post"].permissions == ["safe-synthesizer.create"]
    assert contrib.endpoints[jobs]["get"].permissions == ["safe-synthesizer.list"]
    assert contrib.endpoints[f"{jobs}/{{name}}"]["delete"].permissions == ["safe-synthesizer.delete"]
    assert contrib.endpoints[f"{jobs}/{{job}}/results/summary/download"]["get"].permissions == ["safe-synthesizer.read"]

    # Every mounted route — including the custom result-download routes —
    # carries a valid rule (none falls through to deny).
    assert contrib.endpoints
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.deny is False
