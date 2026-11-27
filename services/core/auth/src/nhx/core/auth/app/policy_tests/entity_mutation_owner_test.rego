# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import future.keywords.if
import future.keywords.in

group_entity_target := {"service": "entities", "entity_type": "platform_job_group"}

test_entity_mutations_are_restricted_to_the_declared_owner if {
	requests := [
		{"method": "POST", "suffix": ""},
		{"method": "PUT", "suffix": "/group-name"},
		{"method": "DELETE", "suffix": "/group-name"},
		{"method": "PUT", "suffix": "/by-id/group-id"},
		{"method": "DELETE", "suffix": "/by-id/group-id"},
	]
	every request in requests {
		path := concat("", ["/apis/entities/v2/workspaces/default/entities/platform_job_group", request.suffix])
		job_request := {"principal_id": "service:jobs", "method": request.method, "path": path, "target": group_entity_target}
		other_request := object.union(job_request, {"principal_id": "service:evals"})
		jobs := data.authz.allow with input as job_request
		other := data.authz.allow with input as other_request
		jobs.allowed
		not other.allowed
	}
}

test_entity_mutation_owner_rule_preserves_reads_and_unowned_writes if {
	requests := [
		{"method": "GET", "path": "/apis/entities/v2/workspaces/default/entities/platform_job_group/group-name", "target": group_entity_target},
		{"method": "POST", "path": "/apis/entities/v2/workspaces/default/entities/platform_job", "target": {"service": "entities", "entity_type": "platform_job"}},
	]
	every request in requests {
		result := data.authz.allow with input as object.union(request, {"principal_id": "service:evals"})
		result.allowed
	}
}

test_entity_mutations_deny_when_the_target_is_missing if {
	request := {"principal_id": "service:evals", "method": "POST", "path": "/apis/entities/v2/workspaces/default/entities/platform_job_group"}
	result := data.authz.allow with input as request
	not result.allowed
}
