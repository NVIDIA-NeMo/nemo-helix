# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder plugin's authz scope. One scope shared by every route module."""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("builder")
