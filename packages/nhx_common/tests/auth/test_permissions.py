# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.auth.permissions import ALL_WORKSPACES, compute_accessible_workspaces


def test_compute_accessible_workspaces_allows_valid_service_principal() -> None:
    assert compute_accessible_workspaces("service:auth", []) == ALL_WORKSPACES


def test_compute_accessible_workspaces_does_not_allow_malformed_service_principal() -> None:
    assert compute_accessible_workspaces("service:", []) == set()


def test_compute_accessible_workspaces_ignores_non_string_workspace_values() -> None:
    accessible = compute_accessible_workspaces(
        "user@example.com",
        [
            {"workspace": "team-a", "role": "Viewer"},
            {"workspace": None, "role": "Viewer"},
            {"workspace": "", "role": "Viewer"},
        ],
    )

    assert accessible == {"team-a"}
