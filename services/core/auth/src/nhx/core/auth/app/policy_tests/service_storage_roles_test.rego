# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

storage_roles := {
	"ServiceEntityStore": {"service_only": true, "permissions": ["entities.create", "entities.read", "entities.update", "entities.delete"]},
	"Admin": {"permissions": ["iam.list", "workspaces.read"]},
	"ServiceSystem": {"permissions": ["*"]},
	"Combined": {"includes": ["Admin", "ServiceEntityStore"]},
}

storage_principals := {
	"service:scaled-evals": {"workspaces": {"private": ["Admin", "ServiceEntityStore"]}},
	"reader": {"workspaces": {"private": ["Admin"]}},
	"human": {"workspaces": {"private": ["Combined"]}},
	"service:entities-only": {"workspaces": {"private": ["ServiceEntityStore"]}},
	"service:legacy": {"workspaces": {"private": ["Admin"]}},
}

storage_permission(principal, workspace, permission) := result if {
	result := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": principal, "workspace": workspace, "permissions": [permission]}
}

test_admin_and_storage_roles_compose if {
	storage_permission("service:scaled-evals", "private", "entities.read").allowed
	storage_permission("service:scaled-evals", "private", "workspaces.read").allowed
}

test_reader_does_not_get_entity_access if {
	not storage_permission("reader", "private", "entities.read").allowed
	storage_permission("reader", "private", "workspaces.read").allowed
}

test_entity_role_does_not_get_workspace_reads if {
	storage_permission("service:entities-only", "private", "entities.read").allowed
	not storage_permission("service:entities-only", "private", "workspaces.read").allowed
}

test_human_inheritance_does_not_grant_service_permissions if {
	not storage_permission("human", "private", "entities.read").allowed
	storage_permission("human", "private", "workspaces.read").allowed
}

test_creator_admin_still_has_no_entity_access if {
	not storage_permission("service:legacy", "private", "entities.read").allowed
}

test_other_workspaces_retain_the_legacy_fallback if {
	storage_permission("service:scaled-evals", "other", "entities.read").allowed
}
