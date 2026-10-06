# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The garak_plugin plugin's authz scope.

The route modules import :data:`scope` so the plugin shares one ``AuthzScope("garak-plugin")``.
"""

from __future__ import annotations

from garak_plugin._legacy import LEGACY_SCOPE_AREAS
from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("garak-plugin", legacy_scopes=LEGACY_SCOPE_AREAS)
