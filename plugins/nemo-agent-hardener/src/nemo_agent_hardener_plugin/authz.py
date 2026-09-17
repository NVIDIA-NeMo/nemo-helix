# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one OAuth scope the Agent Hardener plugin owns.

Kept in its own module so the service and every route module share a single ``AuthzScope("agent-hardener")``
without an import cycle. Reads carry ``@scope.read``; mutating routes carry ``@scope.write``.
"""

from __future__ import annotations

from nemo_platform_plugin.authz import AuthzScope

scope = AuthzScope("agent-hardener")
