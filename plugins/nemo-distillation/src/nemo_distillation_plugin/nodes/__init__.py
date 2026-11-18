# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node adapters. One per graph kind."""

from nemo_distillation_plugin.nodes.adapters import build_adapters

__all__ = ["build_adapters"]
