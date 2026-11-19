# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if
import future.keywords.in

build_actor := "system:serviceaccount:helix:jobs"

build_principals := {
	build_actor: {"workspaces": {"default": ["TaskImageBuildRunner"]}},
	"creator": {"workspaces": {"default": ["BuildSubmitter"], "other": ["BuildSubmitter"]}},
	"viewer": {"workspaces": {"default": ["Viewer"]}},
}

build_request_allowed(method, path, actor, subject) := allowed if {
	result := authz.allow with input as {
		"method": method,
		"path": path,
		"principal_id": actor,
		"on_behalf_of_principal_id": subject,
		"nhx_workload_job": "task-image-build-ad2465ec926c415d-5",
		"nhx_workload_workspace": "default",
		"scopes": ["platform:read", "platform:write"],
	}
		with data.authz.principals as build_principals
		with data.authz.roles.BuildSubmitter as {
			"permissions": ["evaluator.tasks.read", "evaluator.tasks.image-builds.create"],
		}
		with data.authz.scope_families as {"entities": true, "platform": true}
	allowed := result.allowed
}

test_task_image_build_runner_reads_inputs_and_updates_build if {
	every path in ["task/example", "task_revision", "task_image_build/build-1"] {
		build_request_allowed("GET", sprintf("/apis/entities/v2/workspaces/default/entities/%s", [path]), build_actor, "creator")
	}
	build_request_allowed("PUT", "/apis/entities/v2/workspaces/default/entities/task_image_build/by-id/build-id", build_actor, "creator")
}

test_task_image_build_runner_grant_is_limited if {
	not build_request_allowed("GET", "/apis/entities/v2/workspaces/default/entities/task/example", build_actor, "")
	not build_request_allowed("GET", "/apis/entities/v2/workspaces/default/entities/task/example", build_actor, "viewer")
	not build_request_allowed("GET", "/apis/entities/v2/workspaces/default/entities/task/example", "system:serviceaccount:helix:unbound", "creator")
	not build_request_allowed("GET", "/apis/entities/v2/workspaces/other/entities/task/example", build_actor, "creator")
	not build_request_allowed("GET", "/apis/entities/v2/workspaces/default/entities/role_binding/example", build_actor, "creator")
	not build_request_allowed("PUT", "/apis/entities/v2/workspaces/default/entities/task/by-id/task-id", build_actor, "creator")
	not build_request_allowed("DELETE", "/apis/entities/v2/workspaces/default/entities/task_image_build/by-id/build-id", build_actor, "creator")
}

test_task_image_build_runner_needs_a_build_job_token if {
	result := authz.allow with input as {
		"method": "GET",
		"path": "/apis/entities/v2/workspaces/default/entities/task/example",
		"principal_id": build_actor,
		"on_behalf_of_principal_id": "creator",
		"nhx_workload_job": "some-other-job",
		"nhx_workload_workspace": "default",
		"scopes": ["platform:read"],
	}
		with data.authz.principals as build_principals
		with data.authz.roles.BuildSubmitter as {
			"permissions": ["evaluator.tasks.read", "evaluator.tasks.image-builds.create"],
		}
		with data.authz.scope_families as {"entities": true, "platform": true}
	not result.allowed
}
