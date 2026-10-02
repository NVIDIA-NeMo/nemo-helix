# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_helix_plugin.authz_format import validate_static_authz_data
from nemo_helix_plugin.authz_merge import merge_authz_contributions


def test_merge_adds_endpoints_and_role_permissions() -> None:
    base = {
        "authz": {
            "permissions": {},
            "roles": {"Editor": {"permissions": ["jobs.read"]}, "Viewer": {"permissions": []}},
            "endpoints": {},
        }
    }
    overlay = [
        {
            "permissions": {
                "customization.automodel.jobs.create": "Create automodel jobs",
                "customization.automodel.jobs.read": "Read automodel jobs",
            },
            "endpoints": {
                "/apis/customization/v2/workspaces/{workspace}/automodel/jobs": {
                    "post": {
                        "permissions": ["customization.automodel.jobs.create"],
                        "scopes": ["customization:write", "platform:write"],
                    },
                },
            },
        }
    ]
    merged = merge_authz_contributions(base, overlay)
    validate_static_authz_data(merged)
    endpoints = merged["authz"]["endpoints"]
    assert "post" in endpoints["/apis/customization/v2/workspaces/{workspace}/automodel/jobs"]
    editor_perms = merged["authz"]["roles"]["Editor"]["permissions"]
    assert "customization.automodel.jobs.create" in editor_perms
    viewer_perms = merged["authz"]["roles"]["Viewer"]["permissions"]
    assert "customization.automodel.jobs.read" in viewer_perms
