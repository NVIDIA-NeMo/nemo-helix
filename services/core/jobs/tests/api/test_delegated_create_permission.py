# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The actual compiled route policy replaces the former Python OBO branch."""

import pytest
from nhx.testing.policy import PolicyHarness


@pytest.fixture
def policy() -> PolicyHarness:
    harness = PolicyHarness()
    harness.document["authz"]["principals"]["alice"] = {"workspaces": {"team": ["Editor"]}}
    harness.install(harness.document)
    return harness


@pytest.mark.parametrize(
    "principal,obo,workspace,allowed",
    [
        ("service:evals", "alice", "team", True),
        ("service:evals", "alice", "other", False),
        ("service:evals", "bob", "team", False),
        ("service:evals", "", "other", True),
        ("alice", "", "team", True),
    ],
)
def test_jobs_create_uses_subject_permissions(
    policy: PolicyHarness, principal: str, obo: str, workspace: str, allowed: bool
) -> None:
    result = policy.evaluate(
        "allow",
        {
            "principal_id": principal,
            "on_behalf_of_principal_id": obo,
            "caller_kind": "service_principal" if principal.startswith("service:") else "principal",
            "method": "POST",
            "path": f"/apis/jobs/v2/workspaces/{workspace}/jobs",
        },
    )
    assert result["allowed"] is allowed
