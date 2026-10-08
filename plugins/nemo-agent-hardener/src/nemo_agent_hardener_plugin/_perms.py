# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed permission vocabulary for the Agent Hardener plugin routes.

Each ``perm(...)`` member mints a :class:`Permission` whose id is ``<namespace>.<member-lowercased>``
(or ``<namespace>.<suffix>`` when given). Referenced from ``@path_rule(permissions=[...])`` on the
route handlers — never as bare strings.
"""

from __future__ import annotations

from nemo_platform_plugin.authz import PermissionSet, perm


class AgentHardenerRunPerms(PermissionSet, namespace="agent-hardener.runs"):
    LIST = perm("List Agent Hardener runs")
    READ = perm("Read an Agent Hardener run")
    DELETE = perm("Delete an Agent Hardener run record")
    APPLY = perm("Apply a run's hardened workflow to its agent")
    COMPOSE = perm("Compose a chosen subset of a run's recommended defenses")
    EVENTS_READ = perm("Stream an Agent Hardener run's live events", suffix="events.read")
    EVENTS_WRITE = perm("Ingest an Agent Hardener run's live events", suffix="events.write")


class AgentHardenerManifestPerms(PermissionSet, namespace="agent-hardener.manifests"):
    LIST = perm("List Agent Hardener manifests")
    READ = perm("Read an Agent Hardener manifest")
    WRITE = perm("Create, update, or delete an Agent Hardener manifest")
    INSPECT = perm("Inspect projects/agents and validate model config for the create wizard")
