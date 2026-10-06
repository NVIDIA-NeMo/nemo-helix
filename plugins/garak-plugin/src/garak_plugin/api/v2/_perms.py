# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed permission vocabulary for the garak_plugin plugin's CRUD routes.

Two sub-namespaces under ``garak_plugin`` (one per entity collection). Route handlers
reference these constants in their ``@path_rule``; the platform derives the permission
catalog from the routes, so there is no parallel list to keep in sync.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class ScanConfigPerms(PermissionSet, namespace="garak-plugin.configs"):
    CREATE = perm("Create scan configs")
    LIST = perm("List scan configs")
    READ = perm("Read a scan configs entry")
    UPDATE = perm("Update a scan configs entry")
    DELETE = perm("Delete a scan configs entry")


class ScanTargetPerms(PermissionSet, namespace="garak-plugin.targets"):
    CREATE = perm("Create scan targets")
    LIST = perm("List scan targets")
    READ = perm("Read a scan targets entry")
    UPDATE = perm("Update a scan targets entry")
    DELETE = perm("Delete a scan targets entry")
