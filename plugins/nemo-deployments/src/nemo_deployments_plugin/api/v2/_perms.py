# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed permission vocabulary for the deployments plugin's routes.

Three sub-namespaces under ``deployments`` (one per entity collection). Route handlers
reference these constants in their ``@path_rule``; the platform derives the permission
catalog from the routes, so there is no parallel list to keep in sync. The controller-only
status routes mint a ``status.update`` permission under the collection they project onto.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class DeploymentConfigPerms(PermissionSet, namespace="deployments.deployment-configs"):
    CREATE = perm("Create deployments deployment-configs")
    LIST = perm("List deployments deployment-configs")
    READ = perm("Read deployments deployment-configs")
    DELETE = perm("Delete deployments deployment-configs")


class DeploymentPerms(PermissionSet, namespace="deployments.deployments"):
    CREATE = perm("Create deployments deployments")
    LIST = perm("List deployments deployments")
    READ = perm("Read deployments deployments")
    DELETE = perm("Delete deployments deployments")
    STATUS_UPDATE = perm("Update deployment observed status (controller)", suffix="status.update")


class VolumePerms(PermissionSet, namespace="deployments.volumes"):
    CREATE = perm("Create deployments volumes")
    LIST = perm("List deployments volumes")
    READ = perm("Read deployments volumes")
    DELETE = perm("Delete deployments volumes")
    STATUS_UPDATE = perm("Update volume observed status (controller)", suffix="status.update")
