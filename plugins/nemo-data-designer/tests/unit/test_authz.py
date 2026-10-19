# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the data-designer plugin (every route ruled, no problems)."""

from __future__ import annotations

from nemo_data_designer_plugin.service import DataDesignerService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution


def test_data_designer_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(DataDesignerService())
    assert problems == []

    # Job-factory perms plus the preview function perm, all declared.
    assert {
        "data-designer.create",
        "data-designer.list",
        "data-designer.read",
        "data-designer.delete",
        "data-designer.cancel",
        "data-designer.preview",
    } <= set(contrib.permissions)

    # Pin verb->permission so a mis-stamp (e.g. create<->read) is localized to this plugin.
    jobs = "/apis/data-designer/v2/workspaces/{workspace}/jobs/create"
    assert contrib.endpoints[jobs]["post"].permissions == ["data-designer.create"]
    assert contrib.endpoints[jobs]["get"].permissions == ["data-designer.list"]
    assert contrib.endpoints[f"{jobs}/{{name}}"]["delete"].permissions == ["data-designer.delete"]
    preview = "/apis/data-designer/v2/workspaces/{workspace}/preview"
    assert contrib.endpoints[preview]["post"].permissions == ["data-designer.preview"]
    retrieval_preview = "/apis/data-designer/v2/workspaces/{workspace}/retrieval-preview"
    assert contrib.endpoints[retrieval_preview]["post"].permissions == ["data-designer.retrieval-preview"]
    for job_name in ("retrieval-generate", "retrieval-prepare", "retrieval-run"):
        retrieval_job = f"/apis/data-designer/v2/workspaces/{{workspace}}/jobs/{job_name}"
        assert contrib.endpoints[retrieval_job]["post"].permissions == ["data-designer.create"]
        assert contrib.endpoints[retrieval_job]["get"].permissions == ["data-designer.list"]

    # Every mounted route carries a valid rule (none falls through to deny).
    assert contrib.endpoints
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.deny is False
