# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compat re-export shim.

The canonical implementation lives in ``nemo_helix_plugin.api.text_filter``.
Existing imports from ``nhx.common.api.text_filter`` continue to resolve here.
"""

from nemo_helix_plugin.api.text_filter import TextFilterParser as TextFilterParser
from nemo_helix_plugin.api.text_filter import parse_text_filter as parse_text_filter
