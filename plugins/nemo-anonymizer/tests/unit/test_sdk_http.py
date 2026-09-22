# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from nemo_anonymizer_plugin.sdk import http


def _platform(*, workspace: str | None = "default") -> http.HelixClient:
    return cast(
        http.HelixClient,
        SimpleNamespace(
            base_url="https://platform.test/",
            workspace=workspace,
            default_headers={"Authorization": "Bearer token", "skip": object()},
        ),
    )


def test_url_encodes_workspace_as_single_path_segment() -> None:
    assert (
        http.url(_platform(), "team/a", "/preview")
        == "https://platform.test/apis/anonymizer/v2/workspaces/team%2Fa/preview"
    )


def test_url_requires_absolute_resource_path() -> None:
    with pytest.raises(ValueError, match="must start with '/'"):
        http.url(_platform(), "team-a", "preview")


def test_path_segment_encodes_reserved_chars() -> None:
    assert http.path_segment("job/a?b") == "job%2Fa%3Fb"


def test_headers_drops_non_string_values() -> None:
    assert http.headers(_platform()) == {"Authorization": "Bearer token"}
