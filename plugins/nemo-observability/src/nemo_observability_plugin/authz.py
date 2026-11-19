# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authz scope and permission vocabulary for the observability plugin."""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope, PermissionSet, perm

scope = AuthzScope("observability")


class ConnectionPerms(PermissionSet, namespace="observability.connections"):
    LIST = perm("List configured trace explorer connections")
    READ = perm("Read a trace explorer connection, its OTLP export snippet, and open its embedded UI")


class UIPerms(PermissionSet, namespace="observability.ui"):
    # One permission for every method: third-party UIs (MLflow trace search) POST to read.
    READ = perm("Browse an embedded trace explorer through the platform proxy")
