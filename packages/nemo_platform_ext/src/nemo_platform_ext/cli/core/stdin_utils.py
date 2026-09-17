# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for reading input from stdin."""

from __future__ import annotations

import os
import sys
from collections.abc import Collection, Mapping
from typing import Any, TypeVar

import yaml
from click import UsageError
from pydantic import BaseModel, RootModel

RequestModelT = TypeVar("RequestModelT", bound=BaseModel)


def is_stdin_available() -> bool:
    """
    Check if stdin has data available (i.e., data is being piped in).

    Returns:
        True if stdin is not a TTY (data is being piped), False otherwise
    """
    return not sys.stdin.isatty()


def read_data_from_stdin() -> dict[str, Any]:
    """
    Read and parse JSON from stdin.

    Returns:
        Parsed JSON data as a dictionary

    Raises:
        ValueError: If stdin is empty or contains invalid JSON
    """
    if not is_stdin_available():
        raise ValueError("No input available on stdin")

    stdin_data = sys.stdin.read().strip()

    if not stdin_data:
        raise ValueError("Stdin is empty")

    try:
        data = yaml.safe_load(stdin_data)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid data in stdin: {e}")

    if not isinstance(data, dict):
        raise ValueError("JSON input must be an object/dictionary")

    return data


def merge_stdin_with_options(stdin_data: dict[str, Any], **cli_options: Any) -> dict[str, Any]:
    """
    Merge stdin data with CLI options.

    CLI options take precedence over stdin data.

    Args:
        stdin_data: Data parsed from stdin
        **cli_options: Options provided via CLI flags

    Returns:
        Merged dictionary with CLI options taking precedence
    """
    # Start with stdin data
    result = dict(stdin_data)

    # Override with CLI options (only non-None values)
    for key, value in cli_options.items():
        if value is not None:
            result[key] = value

    return result


def read_data_input_with_flags(
    input_file: str | None = None,
    input_data: str | None = None,
) -> dict[str, Any]:
    """
    Read data from explicit --input-file or --input-json flags.

    Args:
        input_file: Path to data file, or "-" for stdin
        input_data: Inline data (JSON or YAML string)

    Returns:
        Parsed data as dictionary

    Raises:
        ValueError: If both or neither flags are provided, or if input is invalid
    """
    # Validate mutually exclusive flags
    if input_file and input_data:
        raise ValueError("Cannot use both --input-file and --input-data. Choose one.")

    if not input_file and not input_data:
        raise ValueError("Must provide input via --input-file or --input-data")

    # Handle --input-file
    if input_file:
        # Handle stdin (dash convention)
        if input_file == "-":
            if not is_stdin_available():
                raise ValueError("No data available on stdin")
            return read_data_from_stdin()

        # Handle file path
        try:
            # Support both relative and absolute paths
            if not os.path.isabs(input_file):
                input_file = os.path.abspath(input_file)

            with open(input_file) as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise ValueError(f"File {input_file} must contain an object (in the form '{{field: value, ...}}')")

            return data
        except FileNotFoundError:
            raise ValueError(f"File not found: {input_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid data in file {input_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading file {input_file}: {e}")

    # Handle --input-data
    if input_data:
        data = read_payload("input-data", input_data)
        if not isinstance(data, dict):
            raise ValueError("JSON input must be an object (in the form '{field: value, ...}').")
        return data

    # Should never reach here due to validation above
    raise ValueError("No input provided")


def validate_required_fields(
    payload: dict[str, Any],
    required_fields: list[str],
    command_name: str = "this command",
    field_help: dict[str, str] | None = None,
) -> None:
    """
    Validate that all required fields are present in the merged payload.

    Args:
        payload: The merged payload (JSON input + CLI overrides)
        required_fields: List of field names that must be present
        command_name: Name of the command for error messages
        field_help: Optional dict mapping field names to help text

    Raises:
        MissingRequiredFieldsError: If any required field is missing
    """
    from nemo_platform_ext.cli.core.errors import MissingRequiredFieldsError

    missing = [f for f in required_fields if f not in payload or payload[f] is None]

    if missing:
        raise MissingRequiredFieldsError(missing, command_name, field_help)


def _field_model(model_cls: type[BaseModel]) -> type[BaseModel] | None:
    """Return the model whose fields name the accepted payload keys, or None when keys are unconstrained."""
    if issubclass(model_cls, RootModel):
        root_type = model_cls.model_fields["root"].annotation
        if not (isinstance(root_type, type) and issubclass(root_type, BaseModel)):
            return None
        model_cls = root_type
    if model_cls.model_config.get("extra") == "allow":
        return None
    return model_cls


def _accepted_field_names(model_cls: type[BaseModel]) -> list[str]:
    names: list[str] = []
    for name, field in model_cls.model_fields.items():
        names.append(name)
        if isinstance(field.alias, str):
            names.append(field.alias)
        if isinstance(field.validation_alias, str):
            names.append(field.validation_alias)
    return names


def _accepted_input_names(model_cls: type[BaseModel]) -> list[str]:
    """Return each field's accepted input name, preferring its alias.

    The internal Pydantic field name may differ from what a caller types
    (e.g. ``schema_`` with ``alias="schema"``), so the user-facing hint must
    advertise the alias, not the field name.
    """
    return [field.alias or field.validation_alias or name for name, field in model_cls.model_fields.items()]


def build_request_body(
    model_cls: type[RequestModelT],
    payload: Mapping[str, Any],
    *,
    exclude: Collection[str] = (),
    command_name: str = "this command",
) -> RequestModelT:
    """Validate the user-supplied *payload* into a request body of *model_cls*.

    Keys in *exclude* are CLI-only (workspace, exist_ok, ...) and dropped. Any
    remaining key the model does not define is rejected instead of silently
    ignored, so a typo in ``--input-data`` cannot degrade into a no-op. Models
    that declare ``extra="allow"`` (pass-through request shapes) and root models
    wrapping unstructured payloads accept any key. Only the
    keys present in *payload* are marked set on the returned model, so
    ``exclude_unset`` serialization sends exactly what the user provided.
    """
    from nemo_platform_ext.cli.core.errors import UnknownInputFieldsError

    body = {key: value for key, value in payload.items() if key not in exclude}
    field_model = _field_model(model_cls)
    if field_model is not None:
        unknown = sorted(set(body) - set(_accepted_field_names(field_model)))
        if unknown:
            raise UnknownInputFieldsError(unknown, command_name, sorted(_accepted_input_names(field_model)))
    return model_cls.model_validate(body)


def read_payload(field_name: str, field_value: str) -> Any:
    """
    Read payload for a specific field, supporting JSON and YAML strings.

    Args:
        field_name: Name of the field (for error messages)
        field_value: Value provided via CLI flag (JSON string or file path)
    Returns:
        Parsed value (could be dict, list, etc.)
    """

    try:
        return yaml.safe_load(field_value)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid data for field '{field_name}': {e}")


def read_secret_from_file(path: str, option_name: str) -> str:
    if path == "-":
        if not is_stdin_available():
            raise UsageError(f"No input provided on stdin, but required for '--{option_name} -'")
        data = sys.stdin.read().strip()
    else:
        try:
            with open(path) as f:
                data = f.read().strip()
        except FileNotFoundError as e:
            raise UsageError(f"File not found: {path}") from e
        except PermissionError as e:
            raise UsageError(f"Permission denied reading file: {path}") from e

    if not data:
        raise UsageError(
            "Secret value cannot be empty. Provide non-empty content via --from-file "
            "(e.g. a file path or '-' to read from stdin or use --value for inline secret)."
        )
    return data


def resolve_secret_value(
    from_file: str | None,
    value: str | None,
    *,
    required: bool = False,
    command_name: str = "this command",
) -> str | None:
    """Resolve secret value from --from-file or --value.

    Use required=True for create (exactly one source required).
    Use required=False for update (optional; returns None if neither given).
    """
    if from_file is not None and value is not None:
        raise UsageError("Cannot use both --from-file and --value. Choose one.")
    if from_file is None and value is None:
        if required:
            raise UsageError(
                f"Provide secret value via --value or --from-file for {command_name}. "
                'Example: nemo secrets create my-secret --value "abc123"'
            )
        return None
    if from_file is not None:
        return read_secret_from_file(from_file.strip(), "from-file")
    stripped = (value or "").strip()
    if not stripped:
        raise UsageError("Secret value cannot be empty. Provide non-empty content via --value or --from-file.")
    return stripped
