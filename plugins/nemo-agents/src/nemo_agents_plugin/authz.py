# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agents plugin's authz scope.

Route modules and the service import :data:`scope` so the plugin shares one
``AuthzScope("agents")``.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("agents")
