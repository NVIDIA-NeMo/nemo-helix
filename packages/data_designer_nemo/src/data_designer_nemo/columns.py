# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import data_designer.config as dd
from data_designer_nemo.errors import NDDInvalidConfigError

_CUSTOM_COLUMN_TYPE = "custom"
CUSTOM_COLUMNS_UNSUPPORTED_MESSAGE = (
    "Custom columns are not supported by the NeMo Helix Data Designer service. "
    "Replace the custom column with a built-in column type before trying again."
)


def validate_no_custom_columns(data: Any) -> None:
    """Raises if a raw Data Designer request contains a custom column.

    This function is used from "before"-mode Pydantic validators on request models
    carrying a ``config: dd.DataDesignerConfig`` field. At that point the payload
    still has the raw ``column_type`` discriminator, so we can preempt Pydantic's
    generic union/deserialization error with a platform-specific explanation.
    """
    if not _has_custom_column(data):
        return

    raise ValueError(CUSTOM_COLUMNS_UNSUPPORTED_MESSAGE)


def validate_config_has_no_custom_columns(config: dd.DataDesignerConfig) -> None:
    """Raises if a parsed Data Designer config contains a custom column."""
    if any(column.column_type == _CUSTOM_COLUMN_TYPE for column in config.columns):
        raise NDDInvalidConfigError(CUSTOM_COLUMNS_UNSUPPORTED_MESSAGE)


def _has_custom_column(data: Any) -> bool:
    try:
        columns = data["config"]["columns"]
    except Exception:
        return False

    if not isinstance(columns, list):
        return False

    return any(isinstance(column, dict) and column.get("column_type") == _CUSTOM_COLUMN_TYPE for column in columns)
