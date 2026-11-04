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

test_admin_and_storage_roles_compose if {
	result := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "service:scaled-evals", "workspace": "private", "permissions": ["entities.read", "workspaces.read"]}
	result.allowed
}

test_reader_does_not_get_entity_access if {
	entity := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "reader", "workspace": "private", "permissions": ["entities.read"]}
	not entity.allowed
	workspace := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "reader", "workspace": "private", "permissions": ["workspaces.read"]}
	workspace.allowed
}

test_entity_role_does_not_get_workspace_reads if {
	entity := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "service:entities-only", "workspace": "private", "permissions": ["entities.read"]}
	entity.allowed
	workspace := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "service:entities-only", "workspace": "private", "permissions": ["workspaces.read"]}
	not workspace.allowed
}

test_human_inheritance_does_not_grant_service_permissions if {
	entity := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "human", "workspace": "private", "permissions": ["entities.read"]}
	not entity.allowed
	workspace := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "human", "workspace": "private", "permissions": ["workspaces.read"]}
	workspace.allowed
}

test_creator_admin_still_has_no_entity_access if {
	result := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "service:legacy", "workspace": "private", "permissions": ["entities.read"]}
	not result.allowed
}

test_other_workspaces_retain_the_legacy_fallback if {
	result := authz.has_permissions with data.authz.roles as storage_roles
		with data.authz.principals as storage_principals
		with input as {"principal_id": "service:scaled-evals", "workspace": "other", "permissions": ["entities.read"]}
	result.allowed
}
