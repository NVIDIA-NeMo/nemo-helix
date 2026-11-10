# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Compose overlays the Harbor runtime adds to every local Docker job for the host.docker.internal alias.

They live in one place because the agent setup egress window has to recognise them. Harbor counts any extra
Compose file as the task running Compose services, and the window keeps the task's restricted policy for those.
These overlays only add ``extra_hosts`` to services Harbor already runs (the task's ``main`` container, or
Harbor's own egress sidecar), so they start no process outside the container the window cleans up.
"""

from pathlib import Path

CONTAINER_HOST_ALIAS_OVERLAY = Path(__file__).with_name("host-gateway.yaml")
"""For a task with public network access: the alias goes on the task's own container."""

SIDECAR_HOST_ALIAS_OVERLAY = Path(__file__).with_name("host-gateway-sidecar.yaml")
"""For a task with restricted network access: Docker refuses the alias on the task's container, which uses the
egress sidecar's network and hosts file, so the alias goes on the sidecar."""

HOST_ALIAS_OVERLAYS: frozenset[Path] = frozenset(
    path.resolve() for path in (CONTAINER_HOST_ALIAS_OVERLAY, SIDECAR_HOST_ALIAS_OVERLAY)
)
"""Both overlays, resolved, for comparing against the paths a Harbor environment was given."""
