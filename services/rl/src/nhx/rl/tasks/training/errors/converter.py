# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from pathlib import Path

from nhx.common.errors import ExceptionConverter, RulesLoader

from .exceptions import (
    EXCEPTION_REGISTRY,
    CustomizerTrainingError,
    ErrorDetails,
    InternalError,
    default_exception_handler,
)

logger = logging.getLogger(__name__)

# Path to the error rules YAML file (relative to this module)
_ERROR_RULES_PATH = Path(__file__).parent / "error_rules.yaml"

# Additional modules to search for exception types not in the registry
# subprocess.TimeoutExpired is used for training timeout detection
_FALLBACK_MODULES = [subprocess]

# Module-level singleton converter
_converter: ExceptionConverter | None = None


def _load_converter() -> ExceptionConverter:
    """Load the converter from YAML rules."""
    logger.debug(f"Loading Customizer error rules from: {_ERROR_RULES_PATH}")

    converter = RulesLoader.from_yaml(
        _ERROR_RULES_PATH,
        exception_registry=EXCEPTION_REGISTRY,
        default_handler=default_exception_handler,
        fallback_exception=InternalError,
        fallback_modules=_FALLBACK_MODULES,
    )

    logger.info(f"Loaded {converter.rule_count} Customizer error mapping rules")
    return converter


def get_error_converter() -> ExceptionConverter:
    """
    Get the singleton ExceptionConverter for Customizer training errors.

    The converter is created once on first access and reused for the module's lifetime.
    It loads rules from error_rules.yaml and uses InternalError as fallback.

    Returns:
        Configured ExceptionConverter ready to convert exceptions.

    Raises:
        FileNotFoundError: If error_rules.yaml is not found.
        ValueError: If rules file has invalid syntax.
    """
    global _converter
    if _converter is None:
        _converter = _load_converter()
    return _converter


def create_error_details(exception: Exception) -> ErrorDetails:
    """
    Create error_details dict for Jobs service reporting.

    Converts the exception to a CustomizerTrainingError and returns
    a dict suitable for passing to progress_reporter.report_error().

    If the exception is already a CustomizerTrainingError, returns its
    details directly without re-conversion.

    Uses the library's fallback mechanism (InternalError) for unmatched exceptions.

    Args:
        exception: The exception to convert.

    Returns:
        ErrorDetails with 'message', 'type', and 'detail' keys.
    """
    # If already a CustomizerTrainingError, return its details directly
    if isinstance(exception, CustomizerTrainingError):
        return exception.to_error_details()

    # Convert using the library - fallback_exception=InternalError handles unmatched.
    # get_error_converter() is inside the try so a converter-loading failure (e.g. a
    # missing/invalid rules file) still yields a structured InternalError dict rather
    # than propagating out of this last-resort error reporter.
    try:
        converter = get_error_converter()
        converter.raise_converted_or_default(exception)
    except CustomizerTrainingError as converted:
        return converted.to_error_details()
    except Exception as e:  # noqa: BLE001 - intentional last-resort guard to guarantee dict return
        # Unexpected exception type - wrap in InternalError to ensure we always return a dict
        logger.warning(f"Unexpected exception type from converter: {type(e).__name__}: {e}")
        exc = InternalError(
            message=f"An internal error occurred. ({type(exception).__name__}: {exception})",
            detail=str(exception),
        )
        return exc.to_error_details()


__all__ = [
    "get_error_converter",
    "create_error_details",
]
