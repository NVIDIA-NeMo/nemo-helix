# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compat re-export shim.

The canonical implementation lives in ``nemo_helix_plugin.api.parsed_filter``.
Existing imports from ``nhx.common.api.parsed_filter`` continue to resolve here.
"""

from nemo_helix_plugin.api.parsed_filter import ParsedFilter as ParsedFilter
from nemo_helix_plugin.api.parsed_filter import make_filter_dep as make_filter_dep
