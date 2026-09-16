# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Customization router plugin for NeMo Helix."""

from nemo_customizer.contributor import CustomizationContributor
from nemo_customizer.discovery import discover_customization_contributors

__all__ = [
    "CustomizationContributor",
    "discover_customization_contributors",
]
