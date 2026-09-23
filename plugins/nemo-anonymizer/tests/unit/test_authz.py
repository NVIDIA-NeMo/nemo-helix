# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the anonymizer plugin (every route ruled, no problems)."""

from __future__ import annotations

from nemo_anonymizer_plugin.service import AnonymizerService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution


def test_anonymizer_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(AnonymizerService())
    assert problems == []

    # Job-factory perms plus the preview function perm, all declared.
    assert {
        "anonymizer.create",
        "anonymizer.list",
        "anonymizer.read",
        "anonymizer.delete",
        "anonymizer.cancel",
        "anonymizer.preview",
    } <= set(contrib.permissions)

    # Pin verb->permission so a mis-stamp (e.g. create<->read) is localized to this plugin.
    jobs = "/apis/anonymizer/v2/workspaces/{workspace}/jobs/run"
    assert contrib.endpoints[jobs]["post"].permissions == ["anonymizer.create"]
    assert contrib.endpoints[jobs]["get"].permissions == ["anonymizer.list"]
    assert contrib.endpoints[f"{jobs}/{{name}}"]["delete"].permissions == ["anonymizer.delete"]
    preview = "/apis/anonymizer/v2/workspaces/{workspace}/preview"
    assert contrib.endpoints[preview]["post"].permissions == ["anonymizer.preview"]

    # Every mounted route carries a valid rule (none falls through to deny).
    assert contrib.endpoints
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.deny is False
