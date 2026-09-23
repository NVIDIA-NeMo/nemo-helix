# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The auditor plugin's authz scope.

The route modules import :data:`scope` so the plugin shares one ``AuthzScope("auditor")``.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("auditor")
