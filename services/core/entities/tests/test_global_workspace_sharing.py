# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for describing the workspaces a refused delete would reach into."""

from nhx.core.entities.api.v2.utils import describe_workspaces


def test_describe_workspaces_counts_inaccessible_ones_without_naming_them() -> None:
    assert describe_workspaces(["team-a", "secret-b", "secret-c"], {"team-a"}) == "team-a, 2 you cannot access"


def test_describe_workspaces_names_all_with_full_access() -> None:
    assert describe_workspaces(["team-a", "team-b"], None) == "team-a, team-b"
