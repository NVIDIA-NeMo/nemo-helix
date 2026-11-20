# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Who may write a workload delegation record, decided with the roles and endpoints of static-authz.yaml.
#
# The record is a generic entity in the system workspace. What it says about a workload (who it acts for,
# which job it is a step of) is only as trustworthy as entities.create and entities.update there. No user
# role holds them; a platform admin needs none. A workload's delegated token is checked on the workload
# itself, which holds nothing, whoever it acts for. Service principals hold them through the default
# ServiceSystem role.

workload_delegation_record_collection := "/apis/entities/v2/workspaces/system/entities/workload_delegation"

workload_delegation_record_writes := [
	{"method": "POST", "path": workload_delegation_record_collection},
	{"method": "PUT", "path": concat("/", [workload_delegation_record_collection, "some-delegation-name"])},
	{"method": "PUT", "path": concat("/", [workload_delegation_record_collection, "by-id", "some-delegation-id"])},
]

workload_delegation_record_read_by_id := {"method": "GET", "path": "/apis/entities/v2/entities/some-delegation-id"}

workload_delegation_record_principals := {
	"editor@example.com": {"workspaces": {"system": ["Editor"]}},
	"admin@example.com": {"workspaces": {"system": ["Admin"]}},
	"platform-admin@example.com": {"workspaces": {"system": ["HelixAdmin"]}},
}

workload_delegation_record_users := ["editor@example.com", "admin@example.com"]

# A job step's delegated token as the auth middleware presents it: the actor is the workload, and the
# subject is the user who created the job.
workload_delegation_record_delegated_caller(subject) := {
	"principal_id": "system:serviceaccount:nemo-runs:job-runner",
	"actor_groups": ["system:serviceaccounts"],
	"on_behalf_of_principal_id": subject,
	"principal_email": subject,
	"verified_workload_delegation": true,
}

workload_delegation_record_request_allowed(request) := result.allowed if {
	result := authz.allow with input as request
		with data.authz.principals as workload_delegation_record_principals
		with data.authz.workspaces as {"system": {}}
}

# Not even for a platform admin, who may write the record directly, may a workload write it.
test_workload_delegation_record_writes_denied_to_delegated_workload if {
	every write in workload_delegation_record_writes {
		every subject in array.concat(workload_delegation_record_users, ["platform-admin@example.com"]) {
			request := object.union(workload_delegation_record_delegated_caller(subject), write)
			workload_delegation_record_request_allowed(request) == false
		}
	}
}

test_workload_delegation_record_writes_denied_to_workspace_editor_and_admin if {
	every write in workload_delegation_record_writes {
		every user in workload_delegation_record_users {
			request := object.union({"principal_id": user}, write)
			workload_delegation_record_request_allowed(request) == false
		}
	}
}

# The boundary as it stands: every service principal may write the record and read it by ID.
test_workload_delegation_record_writes_and_read_allowed_to_service_principal if {
	every request in array.concat(workload_delegation_record_writes, [workload_delegation_record_read_by_id]) {
		workload_delegation_record_request_allowed(object.union({"principal_id": "service:jobs"}, request)) == true
	}
}

# The other writer: a platform admin, calling directly.
test_workload_delegation_record_writes_allowed_to_platform_admin if {
	every write in workload_delegation_record_writes {
		request := object.union({"principal_id": "platform-admin@example.com"}, write)
		workload_delegation_record_request_allowed(request) == true
	}
}

test_workload_delegation_record_read_by_id_denied_to_delegated_workload if {
	every subject in array.concat(workload_delegation_record_users, ["platform-admin@example.com"]) {
		caller := workload_delegation_record_delegated_caller(subject)
		request := object.union(caller, workload_delegation_record_read_by_id)
		workload_delegation_record_request_allowed(request) == false
	}
}
