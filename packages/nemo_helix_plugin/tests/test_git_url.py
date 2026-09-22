# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for git_url.git_remote_host."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.git_url import git_remote_host


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # HTTPS
        ("https://github.com/org/repo.git", "github.com"),
        ("https://gitlab.com/group/repo.git", "gitlab.com"),
        ("https://gitlab-master.nvidia.com/team/repo", "gitlab-master.nvidia.com"),
        # SSH (schemed)
        ("ssh://git@github.com/org/repo.git", "github.com"),
        # SCP-style with user@
        ("git@github.com:org/repo.git", "github.com"),
        ("git@gitlab-master.nvidia.com:foo/bar", "gitlab-master.nvidia.com"),
        # SCP-style without user@
        ("github.com:org/repo.git", "github.com"),
        ("gitlab.com:group/repo.git", "gitlab.com"),
        # Lowercases the result
        ("HTTPS://GitHub.com/Org/repo", "github.com"),
        # Negative cases
        ("", ""),
        ("/local/path/to/repo", ""),
        # Windows drive letter must not look like a host
        (r"C:\repo", ""),
        (r"D:\some\folder", ""),
        # Non-host-shaped left side (no dot) shouldn't match
        ("localhost:1234/repo", ""),
    ],
)
def test_git_remote_host(url: str, expected: str) -> None:
    assert git_remote_host(url) == expected
