# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed permission vocabulary for the example plugin's hand-written routes.

Reference these constants in ``@path_rule``; the platform derives the permission catalog
from the routes. The ``greet`` / ``count`` function permissions are stamped onto the
factory routes (see ``get_routers``), so they are not declared here.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class ExampleHelloPerms(PermissionSet, namespace="example.hello"):
    READ = perm("Read the example greeting")


class ExampleItemPerms(PermissionSet, namespace="example.items"):
    CREATE = perm("Create example items")
    LIST = perm("List example items")
    READ = perm("Read an example items entry")
    UPDATE = perm("Update an example items entry")
    DELETE = perm("Delete an example items entry")


class ExampleMiddlewareConfigPerms(PermissionSet, namespace="example.middleware-configs"):
    CREATE = perm("Create example middleware-configs")
    LIST = perm("List example middleware-configs")
    READ = perm("Read an example middleware-configs entry")
    UPDATE = perm("Update an example middleware-configs entry")
    DELETE = perm("Delete an example middleware-configs entry")
