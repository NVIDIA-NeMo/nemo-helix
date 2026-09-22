# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the evaluator plugin (every route ruled, no problems)."""

from __future__ import annotations

from nemo_evaluator.service import EvaluatorPluginService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution


def test_evaluator_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(EvaluatorPluginService())
    assert problems == []

    # Job-factory perms plus the hand-written hello read perm, all declared.
    assert {
        "evaluator.create",
        "evaluator.list",
        "evaluator.read",
        "evaluator.delete",
        "evaluator.cancel",
        "evaluator.hello.read",
    } <= set(contrib.permissions)

    # Pin the two hand-written routes (mirrors the auditor test's healthz spot-check).
    assert contrib.endpoints["/apis/evaluator/v1/healthz"]["get"].permissions == []
    assert contrib.endpoints["/apis/evaluator/v1/hello/{name}"]["get"].permissions == ["evaluator.hello.read"]

    # Pin the job-collection verb->permission mapping so a mis-stamp (e.g. create<->read) is
    # localized to this plugin, not only caught by the central factory test.
    jobs = "/apis/evaluator/v2/workspaces/{workspace}/evaluate/jobs"
    assert contrib.endpoints[jobs]["post"].permissions == ["evaluator.create"]
    assert contrib.endpoints[jobs]["get"].permissions == ["evaluator.list"]
    assert contrib.endpoints[f"{jobs}/{{name}}"]["delete"].permissions == ["evaluator.delete"]

    # Every mounted route carries a valid rule (none falls through to deny).
    assert contrib.endpoints
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.deny is False
