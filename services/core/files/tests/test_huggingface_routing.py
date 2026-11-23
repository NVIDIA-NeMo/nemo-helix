# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF metadata and download routes must agree with authorization contracts."""

import pytest
from nhx.core.files.api.v2.hf.endpoints import router
from starlette.routing import Match


@pytest.mark.parametrize(
    ("method", "suffix", "handler", "workspace"),
    [
        ("GET", "/api/models/resolve/incident", "get_repo_info", "resolve"),
        ("GET", "/api/models/resolve/incident/revision/main", "get_repo_info_at_revision", "resolve"),
        ("GET", "/api/models/resolve/incident/tree/main", "get_tree", "resolve"),
        ("POST", "/api/models/resolve/incident/paths-info/main", "get_paths_info", "resolve"),
        ("HEAD", "/api/models/resolve/incident/revision/main", "head_file", "api"),
        ("GET", "/api/models/resolve/incident/weights/model.bin", "download_file", "api"),
        ("HEAD", "/api/models/resolve/incident/weights/model.bin", "head_file", "api"),
        ("GET", "/resolve/model/resolve/main/config.json", "download_file", "resolve"),
    ],
)
def test_hf_route_selects_expected_operation_and_workspace(
    method: str, suffix: str, handler: str, workspace: str
) -> None:
    scope = {"type": "http", "method": method, "path": f"/v2/hf{suffix}", "root_path": ""}
    for route in router.routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            assert child_scope["endpoint"].__name__ == handler
            assert child_scope["path_params"]["workspace"] == workspace
            return
    pytest.fail(f"No route matched {method} {suffix}")
