# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compat re-export shim.

The canonical implementation lives in ``nemo_helix_plugin.api.filter``.
Existing imports from ``nhx.common.api.filter`` continue to resolve here.
"""

from nemo_helix_plugin.api.filter import ComparisonOperation as ComparisonOperation
from nemo_helix_plugin.api.filter import FilterOperation as FilterOperation
from nemo_helix_plugin.api.filter import FilterOperator as FilterOperator
from nemo_helix_plugin.api.filter import FilterRepository as FilterRepository
from nemo_helix_plugin.api.filter import LogicalOperation as LogicalOperation
from nemo_helix_plugin.api.filter import parse_bracket_filter as parse_bracket_filter
from nemo_helix_plugin.api.filter import parse_json_filter as parse_json_filter
