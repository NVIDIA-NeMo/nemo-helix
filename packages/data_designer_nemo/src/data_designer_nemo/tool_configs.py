# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import data_designer.config as dd
from data_designer_nemo.errors import NDDInvalidConfigError


def validate_no_tool_configs(config: dd.DataDesignerConfig) -> None:
    if config.tool_configs and len(config.tool_configs) > 0:
        raise NDDInvalidConfigError("Tool configs are not supported in the NeMo Helix Data Designer service.")
