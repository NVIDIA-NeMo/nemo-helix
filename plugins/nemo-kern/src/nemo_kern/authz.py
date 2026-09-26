# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kern plugin's authz scope.

All service route modules import :data:`scope` so the plugin shares one
``AuthzScope("kern")``.  A dedicated module avoids circular import issues
between service submodules.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("kern")
