# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed permission vocabulary for the auditor plugin's CRUD routes.

Two sub-namespaces under ``auditor`` (one per entity collection). Route handlers
reference these constants in their ``@path_rule``; the platform derives the permission
catalog from the routes, so there is no parallel list to keep in sync.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class AuditConfigPerms(PermissionSet, namespace="auditor.configs"):
    CREATE = perm("Create audit configs")
    LIST = perm("List audit configs")
    READ = perm("Read an audit configs entry")
    UPDATE = perm("Update an audit configs entry")
    DELETE = perm("Delete an audit configs entry")


class AuditTargetPerms(PermissionSet, namespace="auditor.targets"):
    CREATE = perm("Create audit targets")
    LIST = perm("List audit targets")
    READ = perm("Read an audit targets entry")
    UPDATE = perm("Update an audit targets entry")
    DELETE = perm("Delete an audit targets entry")
