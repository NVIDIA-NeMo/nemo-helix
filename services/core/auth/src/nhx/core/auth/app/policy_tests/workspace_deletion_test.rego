# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Keep the shipped roles and endpoint mappings so this catches policy-data regressions.
workspace_deletion_principals := {
	"viewer@test.com": {"workspaces": {"test-ws": ["Viewer"]}},
	"editor@test.com": {"workspaces": {"test-ws": ["Editor"]}},
	"admin@test.com": {"workspaces": {"test-ws": ["Admin"]}},
	"platform-admin@test.com": {"workspaces": {"system": ["HelixAdmin"]}},
	"other-admin@test.com": {"workspaces": {"other-ws": ["Admin"]}},
}

test_workspace_deletion_requires_admin[principal_id] if {
	cases := {
		"viewer@test.com": false,
		"editor@test.com": false,
		"admin@test.com": true,
		"platform-admin@test.com": true,
		"other-admin@test.com": false,
		"outsider@test.com": false,
	}
	some principal_id, expected in cases
	result := authz.allow with input as {
		"principal_id": principal_id,
		"method": "DELETE",
		"path": "/apis/entities/v2/workspaces/test-ws",
	}
		with data.authz.principals as workspace_deletion_principals
		with data.authz.workspaces as {"test-ws": {}, "other-ws": {}}

	result.allowed == expected
}

test_editor_retains_workspace_update_and_resource_delete[path] if {
	cases := {
		"/apis/entities/v2/workspaces/test-ws": "PUT",
		"/apis/models/v2/workspaces/test-ws/models/test-model": "DELETE",
	}
	some path, method in cases
	result := authz.allow with input as {
		"principal_id": "editor@test.com",
		"method": method,
		"path": path,
	}
		with data.authz.principals as workspace_deletion_principals
		with data.authz.workspaces as {"test-ws": {}}

	result.allowed == true
}
