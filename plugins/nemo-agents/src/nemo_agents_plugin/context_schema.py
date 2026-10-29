# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema checks for an agent's declared ``context_schema``.

An agent declares the per-run ``context`` it accepts as a JSON Schema. The
schema is checked when the agent is defined, and each execute job's context is
checked against it when the job is created, so a misspelled key or a malformed
timestamp fails the request instead of reaching the agent.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from jsonschema import FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.validators import Draft202012Validator, validator_for

# RFC 3339 ``date-time``. jsonschema only checks this format when the optional
# ``rfc3339-validator`` package is installed and otherwise passes every string,
# so the platform checks it itself: this pattern for the shape, then
# ``datetime`` for the ranges (no month 13, no February 30).
_RFC3339_DATE_TIME = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})",
    re.ASCII,
)

_FORMAT_CHECKER = FormatChecker()


@_FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _is_date_time(value: object) -> bool:
    if not isinstance(value, str):
        # Formats constrain only the type they apply to; ``type`` rejects the rest.
        return True
    if not _RFC3339_DATE_TIME.fullmatch(value):
        return False
    datetime.fromisoformat(value.upper())
    return True


def check_context_schema(schema: dict[str, Any]) -> None:
    """Raise ``ValueError`` if *schema* is not a valid JSON Schema.

    Checked against the dialect the schema names in ``$schema``, defaulting to
    Draft 2020-12.
    """
    validator_cls = validator_for(schema, default=Draft202012Validator)
    try:
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"context_schema is not a valid JSON Schema: {exc.message}") from exc


def validate_context(context: dict[str, Any], schema: dict[str, Any]) -> None:
    """Raise ``ValueError`` listing every way *context* violates *schema*."""
    validator_cls = validator_for(schema, default=Draft202012Validator)
    validator = validator_cls(schema, format_checker=_FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(context), key=lambda error: [str(part) for part in error.absolute_path])
    if not errors:
        return
    messages = "; ".join(f"{_location(error.absolute_path)}: {error.message}" for error in errors)
    raise ValueError(f"context does not match the agent's context_schema: {messages}")


def _location(path: Any) -> str:
    return "context" + "".join(f"[{part!r}]" for part in path)
