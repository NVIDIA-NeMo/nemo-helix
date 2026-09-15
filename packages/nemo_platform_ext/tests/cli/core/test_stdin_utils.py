# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdin utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click import UsageError
from nemo_platform_ext.cli.core.errors import UnknownInputFieldsError
from nemo_platform_ext.cli.core.stdin_utils import (
    build_request_body,
    is_stdin_available,
    merge_stdin_with_options,
    read_data_from_stdin,
    read_secret_from_file,
    resolve_secret_value,
)
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError


class TestIsStdinAvailable:
    """Test is_stdin_available function."""

    def test_stdin_is_tty(self) -> None:
        """Test when stdin is a TTY (no data piped)."""
        with patch("sys.stdin.isatty", return_value=True):
            assert is_stdin_available() is False

    def test_stdin_not_tty(self) -> None:
        """Test when stdin is not a TTY (data is piped)."""
        with patch("sys.stdin.isatty", return_value=False):
            assert is_stdin_available() is True


class TestReadJsonFromStdin:
    """Test read_json_from_stdin function."""

    def test_read_valid_json(self) -> None:
        """Test reading valid JSON from stdin."""
        json_data = {"id": "test", "description": "Test namespace"}
        stdin_content = json.dumps(json_data)

        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value=stdin_content):
            result = read_data_from_stdin()
            assert result == json_data

    def test_read_json_with_whitespace(self) -> None:
        """Test reading JSON with leading/trailing whitespace."""
        json_data = {"id": "test"}
        stdin_content = f"\n\n  {json.dumps(json_data)}  \n\n"

        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value=stdin_content):
            result = read_data_from_stdin()
            assert result == json_data

    def test_no_stdin_available(self) -> None:
        """Test when no stdin is available."""
        with patch("sys.stdin.isatty", return_value=True):
            with pytest.raises(ValueError, match="No input available on stdin"):
                read_data_from_stdin()

    def test_empty_stdin(self) -> None:
        """Test when stdin is empty."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value="   "):
            with pytest.raises(ValueError, match="Stdin is empty"):
                read_data_from_stdin()

    def test_invalid_data(self) -> None:
        """Test when stdin contains invalid JSON."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value="{{}}invalid<>json}"):
            with pytest.raises(ValueError, match="Invalid data in stdin"):
                read_data_from_stdin()

    def test_json_not_object(self) -> None:
        """Test when JSON is not an object/dictionary."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value='["array"]'):
            with pytest.raises(ValueError, match="JSON input must be an object/dictionary"):
                read_data_from_stdin()


class TestMergeStdinWithOptions:
    """Test merge_stdin_with_options function."""

    def test_merge_empty_options(self) -> None:
        """Test merging with no CLI options."""
        stdin_data = {"id": "test", "description": "From stdin"}
        result = merge_stdin_with_options(stdin_data)
        assert result == stdin_data

    def test_cli_options_override_stdin(self) -> None:
        """Test that CLI options override stdin data."""
        stdin_data = {"id": "test", "description": "From stdin"}
        result = merge_stdin_with_options(stdin_data, description="From CLI")
        assert result == {"id": "test", "description": "From CLI"}

    def test_none_values_dont_override(self) -> None:
        """Test that None values from CLI don't override stdin data."""
        stdin_data = {"id": "test", "description": "From stdin"}
        result = merge_stdin_with_options(stdin_data, description=None, project="proj-123")
        assert result == {"id": "test", "description": "From stdin", "project": "proj-123"}

    def test_cli_adds_new_fields(self) -> None:
        """Test that CLI options can add new fields not in stdin."""
        stdin_data = {"id": "test"}
        result = merge_stdin_with_options(stdin_data, description="New description", project="proj-123")
        assert result == {"id": "test", "description": "New description", "project": "proj-123"}

    def test_empty_stdin_with_options(self) -> None:
        """Test merging empty stdin with CLI options."""
        stdin_data: dict[str, str] = {}
        result = merge_stdin_with_options(stdin_data, id="test", description="From CLI")
        assert result == {"id": "test", "description": "From CLI"}

    def test_complex_merge(self) -> None:
        """Test complex merging scenario."""
        stdin_data = {
            "id": "test",
            "description": "From stdin",
            "custom_fields": {"key1": "value1"},
        }
        result = merge_stdin_with_options(stdin_data, description="From CLI", project="proj-123", extra_field="extra")
        assert result == {
            "id": "test",
            "description": "From CLI",
            "custom_fields": {"key1": "value1"},
            "project": "proj-123",
            "extra_field": "extra",
        }


class TestReadSecretFromFile:
    """Test read_secret_from_file function."""

    def test_read_from_stdin(self) -> None:
        """Test reading secret from stdin."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value="my-secret-value"):
            assert read_secret_from_file("-", "from-file") == "my-secret-value"

    def test_read_from_stdin_strips_whitespace(self) -> None:
        """Test that stdin content is stripped."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value="  secret  \n"):
            assert read_secret_from_file("-", "from-file") == "secret"

    def test_empty_stdin_raises(self) -> None:
        """Test that empty stdin raises a clear error."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value=""):
            with pytest.raises(UsageError, match="Secret value cannot be empty"):
                read_secret_from_file("-", "from-file")

    def test_whitespace_only_stdin_raises(self) -> None:
        """Test that whitespace-only stdin raises a clear error."""
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", return_value="   \n\t  "):
            with pytest.raises(UsageError, match="Secret value cannot be empty"):
                read_secret_from_file("-", "from-file")

    def test_read_from_file(self, tmp_path: Path) -> None:
        """Test reading secret from a file."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("file-secret")
        assert read_secret_from_file(str(secret_file), "from-file") == "file-secret"

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """Test that empty file raises a clear error."""
        secret_file = tmp_path / "empty.txt"
        secret_file.write_text("")
        with pytest.raises(UsageError, match="Secret value cannot be empty"):
            read_secret_from_file(str(secret_file), "from-file")

    def test_file_not_found_raises(self) -> None:
        """Test that missing file raises UsageError."""
        with pytest.raises(UsageError, match="File not found"):
            read_secret_from_file("/nonexistent/path", "from-file")


class TestResolveSecretValue:
    """Test resolve_secret_value function."""

    def test_required_with_data_returns_str(self) -> None:
        """When required=True and --value given, returns value."""
        result = resolve_secret_value(None, "abc", required=True, command_name="create")
        assert result == "abc"

    def test_required_with_from_file(self, tmp_path: Path) -> None:
        """When required=True and path given, returns file content."""
        (tmp_path / "s").write_text("from-file")
        result = resolve_secret_value(str(tmp_path / "s"), None, required=True, command_name="create")
        assert result == "from-file"

    def test_required_neither_raises(self) -> None:
        """When required=True and neither given, raises."""
        with pytest.raises(UsageError, match="Provide secret value via --value or --from-file"):
            resolve_secret_value(None, None, required=True, command_name="secrets create")

    def test_required_both_raises(self) -> None:
        """When both --from-file and --value given, raises."""
        with pytest.raises(UsageError, match="Cannot use both --from-file and --value"):
            resolve_secret_value("/path", "data", required=True, command_name="create")

    def test_optional_neither_returns_none(self) -> None:
        """When required=False and neither given, returns None."""
        assert resolve_secret_value(None, None, required=False) is None

    def test_optional_empty_data_raises(self) -> None:
        """When --value is empty/whitespace, raises."""
        with pytest.raises(UsageError, match="Secret value cannot be empty"):
            resolve_secret_value(None, "  ", required=False)


# ---------------------------------------------------------------------------
# build_request_body
# ---------------------------------------------------------------------------


class _Body(BaseModel):
    name: str
    description: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")


class _PassThroughBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str


class _RootBody(RootModel[dict]):
    pass


def test_build_request_body_marks_only_provided_keys_as_set() -> None:
    body = build_request_body(_Body, {"name": "x", "workspace": "ws"}, exclude={"workspace"})

    assert body.model_dump(exclude_unset=True) == {"name": "x"}


def test_build_request_body_accepts_aliases() -> None:
    body = build_request_body(_Body, {"name": "x", "schema": {"a": 1}})

    assert body.schema_ == {"a": 1}


def test_build_request_body_rejects_unknown_keys_with_accepted_list() -> None:
    with pytest.raises(UnknownInputFieldsError) as excinfo:
        build_request_body(_Body, {"name": "x", "descripton": "typo", "extra": 1}, command_name="things create")

    assert excinfo.value.unknown_fields == ["descripton", "extra"]
    assert excinfo.value.command_name == "things create"
    assert excinfo.value.known_fields == ["description", "name", "schema"]


def test_build_request_body_validates_types() -> None:
    with pytest.raises(ValidationError):
        build_request_body(_Body, {"name": ["not", "a", "string"]})


def test_build_request_body_passes_extra_keys_through_for_extra_allow_models() -> None:
    body = build_request_body(_PassThroughBody, {"model": "m", "seed": 3})

    assert body.model_dump() == {"model": "m", "seed": 3}


def test_build_request_body_leaves_unstructured_root_models_unconstrained() -> None:
    body = build_request_body(_RootBody, {"anything": 1})

    assert body.root == {"anything": 1}
