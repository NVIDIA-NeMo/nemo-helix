# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authz scope and permission vocabulary for the notif plugin.

The platform's suffix heuristic grants ``.read`` to Viewer and Editor and everything else to
Editor, so any workspace member can listen and editors (plus service principals) can publish.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope, PermissionSet, perm

scope = AuthzScope("notif")


class NotifEventPerms(PermissionSet, namespace="notif.events"):
    CREATE = perm("Publish notification events to a workspace")
    READ = perm("Listen to a workspace's notification event stream")
