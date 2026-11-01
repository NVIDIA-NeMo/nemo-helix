# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import future.keywords.every
import future.keywords.if

job_secret_input := {
	"method": "GET",
	"path": "/apis/secrets/v2/workspaces/target/secrets/key/access",
	"principal_id": "job-runtime",
	"on_behalf_of_principal_id": "creator@example.com",
	"identity_resolution": {"principal": {
		"kind": "workload_access_token",
		"subject": "job-runtime",
		"subject_claim": "act.sub",
		"claims_snapshot": {"nhx_workload": {
			"kind": "job",
			"workspace": "source",
			"job_id": "job-1",
		}},
	}},
}

# object.union recursively merges objects; these negative cases need replacement.
job_secret_replace(base, patch) := object.union(object.remove(base, object.keys(patch)), patch)

# Permission to attempt Secret access is distinct from the handler's mandatory
# creator secrets.read check. No role binding or ServiceSystem promotion needed.
test_verified_job_delegation_can_attempt_secret_access if {
	result := allow with input as job_secret_input with data.authz.principals as {}
	result.allowed
}

test_job_secret_capability_requires_delegation_and_exact_get_route if {
	every patch in [
		{"on_behalf_of_principal_id": ""},
		{"on_behalf_of_principal_id": null},
		{"identity_resolution": {}},
		{"principal_id": "different-actor"},
		{"method": "POST"},
		{"method": "HEAD"},
		{"path": "/apis/secrets/v2/workspaces/target/secrets"},
		{"path": "/apis/secrets/v2/workspaces/target/secrets/key"},
		{"path": "/apis/entities/v2/workspaces/target/projects"},
		{"scopes": ["files:read"]},
	] {
		result := allow with input as job_secret_replace(job_secret_input, patch) with data.authz.principals as {} with data.authz.scope_families as {"files": true, "secrets": true, "platform": true}
		not result.allowed
	}
}

test_job_secret_capability_rejects_other_token_kinds_and_direct_subjects if {
	every patch in [
		{"kind": "oidc_access_token"},
		{"kind": "access_key"},
		{"kind": "trusted_header"},
		{"kind": "workload_subject_token"},
		{"subject_claim": "sub"},
		{"claims_snapshot": {}},
	] {
		descriptor := job_secret_replace(job_secret_input.identity_resolution.principal, patch)
		candidate := job_secret_replace(job_secret_input, {"identity_resolution": {"principal": descriptor}})
		result := allow with input as candidate with data.authz.principals as {}
		not result.allowed
	}
}

test_job_secret_capability_rejects_non_job_or_malformed_attestation if {
	every workload in [
		{"kind": "deployment", "workspace": "source", "job_id": "job-1"},
		{"kind": "job", "workspace": "", "job_id": "job-1"},
		{"kind": "job", "workspace": "source", "job_id": ""},
		{"kind": "job", "workspace": "source"},
	] {
		descriptor := job_secret_replace(job_secret_input.identity_resolution.principal, {"claims_snapshot": {"nhx_workload": workload}})
		candidate := job_secret_replace(job_secret_input, {"identity_resolution": {"principal": descriptor}})
		result := allow with input as candidate with data.authz.principals as {}
		not result.allowed
	}
}
