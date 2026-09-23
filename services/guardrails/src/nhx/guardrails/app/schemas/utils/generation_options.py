# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from nhx.guardrails.entities.values._private import GenerationOptions
from nhx.guardrails.entities.values.common import GuardrailsDataInput


def update_generation_options(
    guardrails_data: GuardrailsDataInput,
    options: Dict[str, Any],
):
    """Helper to update the options of a GuardrailsDataInput object."""

    if hasattr(guardrails_data, "options"):
        for key, value in options.items():
            setattr(guardrails_data.options, key, value)
    else:
        guardrails_data.options = GenerationOptions(**options)

    return guardrails_data


def get_activated_rails_logging_options(guardrails_data: GuardrailsDataInput) -> Dict[str, Any]:
    """Extract activated rails options from GuardrailsDataInput object."""
    if (
        hasattr(guardrails_data, "options")
        and hasattr(guardrails_data.options, "log")
        and hasattr(guardrails_data.options.log, "activated_rails")
    ):
        return {"log": {"activated_rails": guardrails_data.options.log.activated_rails}}
    return {"log": {"activated_rails": False}}


def is_activated_rails_logging_enabled(options_dict: Dict[str, Any]) -> bool:
    """Check if activated_rails logging is enabled in the GenerationOptions.

    Args:
        options_dict (Dict[str, Any]): Options as dictionary (alternative to options parameter)

    Returns:
        bool: True if activated_rails logging is enabled, False otherwise.
    """
    log_config = options_dict.get("log", {})
    return log_config.get("activated_rails", False) is True
