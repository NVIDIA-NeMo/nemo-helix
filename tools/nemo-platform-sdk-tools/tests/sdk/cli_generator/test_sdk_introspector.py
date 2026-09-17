# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Union

from nemo_platform_sdk_tools.sdk.cli_generator.sdk_introspector import (
    ParsedDocstring,
    SDKIntrospector,
    introspect_typed_dict,
)
from typing_extensions import Literal, TypedDict


class _StringFilter(TypedDict, total=False):
    like: str


class _DatetimeFilter(TypedDict, total=False):
    gte: str
    lte: str


class _ResourceFilterParam(TypedDict, total=False):
    workspace: str
    project: str
    name: Union[_StringFilter, str]
    source: Union[_StringFilter, str]
    status: Union[Literal["active", "completed"], list[Literal["active", "completed"]]]
    created_at: _DatetimeFilter
    updated_at: _DatetimeFilter


def test_introspect_models_resource():
    """Test introspecting a generated SDK resource."""
    introspector = SDKIntrospector()
    methods = introspector.introspect_resource(["models"])

    # Should have found the standard CRUD methods
    assert "list" in methods
    assert "create" in methods
    assert "retrieve" in methods
    assert "delete" in methods

    # Check the create method
    create_method = methods["create"]
    assert create_method.name == "create"

    required_names = {p.name for p in create_method.parameters if p.is_required}
    assert required_names == {"name"}
    name_param = next(p for p in create_method.parameters if p.name == "name")
    assert name_param.python_type_name == "str"

    # Should have optional parameters
    optional_params = [p for p in create_method.parameters if not p.is_path_param and not p.is_required]
    assert len(optional_params) > 0


def test_introspect_models_pagination():
    """Test introspecting pagination on a generated resource."""
    introspector = SDKIntrospector()
    methods = introspector.introspect_resource(["models"])

    assert "list" in methods
    assert "create" in methods
    assert "retrieve" in methods

    # Check the list method
    list_method = methods["list"]
    param_names = [p.name for p in list_method.parameters]
    assert "page" in param_names
    assert "page_size" in param_names


def test_introspect_workspace_path_parameter():
    """Test introspecting a positional resource identifier."""
    introspector = SDKIntrospector()
    methods = introspector.introspect_resource(["workspaces"])

    retrieve_method = methods["retrieve"]
    path_params = retrieve_method.path_parameters
    assert [p.name for p in path_params] == ["name"]


def test_parse_docstring_basic():
    """Test basic docstring parsing."""
    # This matches the format that inspect.getdoc() returns (normalized indentation)
    docstring = (
        "List available customization jobs.\n"
        "\n"
        "Args:\n"
        "  filter: Filter jobs on various criteria.\n"
        "\n"
        "  page: Page number.\n"
        "\n"
        "  page_size: Page size.\n"
        "\n"
        "  sort: The field to sort by. To sort in decreasing order, use `-` in front of the field\n"
        "      name.\n"
        "\n"
        "  extra_headers: Send extra headers\n"
        "\n"
        "  timeout: Override the client-level default timeout for this request, in seconds"
    )
    parsed = ParsedDocstring.parse(docstring)

    assert parsed.description == "List available customization jobs."
    assert parsed.param_descriptions["filter"] == "Filter jobs on various criteria."
    assert parsed.param_descriptions["page"] == "Page number."
    assert parsed.param_descriptions["page_size"] == "Page size."
    assert "The field to sort by" in parsed.param_descriptions["sort"]
    assert "decreasing order" in parsed.param_descriptions["sort"]


def test_parse_docstring_empty():
    """Test parsing empty/None docstrings."""
    parsed = ParsedDocstring.parse(None)
    assert parsed.description == ""
    assert parsed.param_descriptions == {}

    parsed = ParsedDocstring.parse("")
    assert parsed.description == ""
    assert parsed.param_descriptions == {}


def test_parse_docstring_no_args():
    """Test parsing docstring without Args section."""
    docstring = "Get the current user."
    parsed = ParsedDocstring.parse(docstring)
    assert parsed.description == "Get the current user."
    assert parsed.param_descriptions == {}


def test_sdk_method_docstring_parsing():
    """Test that SDKMethod correctly parses docstrings from SDK."""
    introspector = SDKIntrospector()
    methods = introspector.introspect_resource(["models"])

    list_method = methods["list"]

    # Should have parsed docstring
    assert list_method.description.startswith("List Models endpoint with filtering, pagination, and sorting.")
    assert list_method.get_param_description("filter") == (
        "Filter models by name, project, workspace, base_model, adapters, finetuning_type, "
        "prompt, lora_enabled, description, created_at, and updated_at."
    )
    assert "field to sort by" in (list_method.get_param_description("sort") or "")


class TestTypedDictFieldIsListType:
    """Tests for TypedDictField.is_list_type detection."""

    def test_union_with_list_is_a_list_type(self):
        """A filter accepting either one status or a list should be a list type."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_map = {f.name: f for f in fields}

        assert field_map["status"].is_list_type is True

    def test_filter_fields_without_sequence_are_not_list_types(self):
        """Filter fields with simple str type should not be list types."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_map = {f.name: f for f in fields}

        # These should NOT be list types (just str)
        assert field_map["workspace"].is_list_type is False
        assert field_map["project"].is_list_type is False

    def test_string_filter_union_is_not_a_list_type(self):
        """A nested string-filter union is not itself a repeatable scalar option."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_map = {f.name: f for f in fields}

        assert field_map["name"].is_list_type is False
        assert field_map["source"].is_list_type is False


class TestTypedDictFieldIsSimpleCliType:
    """Tests for TypedDictField.is_simple_cli_type detection."""

    def test_str_fields_are_simple(self):
        """String fields should be detected as simple CLI types."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_map = {f.name: f for f in fields}

        assert field_map["workspace"].is_simple_cli_type is True
        assert field_map["project"].is_simple_cli_type is True

    def test_float_fields_are_simple(self):
        """Float fields should be detected as simple CLI types."""
        from nemo_platform.types.evaluations.number_filter_param import NumberFilterParam

        fields = introspect_typed_dict(NumberFilterParam)
        field_map = {f.name: f for f in fields}

        assert field_map["gte"].is_simple_cli_type is True
        assert field_map["lte"].is_simple_cli_type is True

    def test_literal_types_are_simple(self):
        """Literal status values should be detected as simple CLI types."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_map = {f.name: f for f in fields}

        assert field_map["status"].is_simple_cli_type is True

    def test_complex_types_are_not_simple(self):
        """Complex nested types should NOT be detected as simple CLI types."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_map = {f.name: f for f in fields}

        assert field_map["created_at"].is_simple_cli_type is False
        assert field_map["updated_at"].is_simple_cli_type is False


class TestExplodableTypedDict:
    """Tests for detecting explodable TypedDict params (filter/search)."""

    def test_filter_param_is_explodable(self):
        """Filter params should be detected as explodable TypedDicts."""
        introspector = SDKIntrospector()
        methods = introspector.introspect_resource(["models"])
        list_method = methods["list"]

        filter_param = next(p for p in list_method.optional_parameters if p.name == "filter")
        assert filter_param.is_explodable_typed_dict is True

    def test_create_nested_config_param_is_explodable(self):
        """Required TypedDict request parameters should also be explodable."""
        introspector = SDKIntrospector()
        methods = introspector.introspect_resource(["inference", "deployment_configs"])
        create_method = methods["create"]

        executor_config_param = next(p for p in create_method.parameters if p.name == "executor_config")
        assert executor_config_param.is_explodable_typed_dict is True

    def test_sort_param_is_not_explodable(self):
        """Sort params (simple enums) should NOT be detected as explodable."""
        introspector = SDKIntrospector()
        methods = introspector.introspect_resource(["models"])
        list_method = methods["list"]

        sort_param = next(p for p in list_method.optional_parameters if p.name == "sort")
        assert sort_param.is_explodable_typed_dict is False

    def test_page_param_is_not_explodable(self):
        """Simple params like page should NOT be explodable."""
        introspector = SDKIntrospector()
        methods = introspector.introspect_resource(["models"])
        list_method = methods["list"]

        page_param = next(p for p in list_method.optional_parameters if p.name == "page")
        assert page_param.is_explodable_typed_dict is False


class TestTypedDictFieldExtraction:
    """Tests for extracting fields from TypedDict classes."""

    def test_extract_filter_fields(self):
        """Should correctly extract fields from filter TypedDict."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_names = {f.name for f in fields}

        assert "workspace" in field_names
        assert "project" in field_names

    def test_extract_all_job_filter_fields(self):
        """Should correctly extract all resource filter fields."""
        fields = introspect_typed_dict(_ResourceFilterParam)
        field_names = {f.name for f in fields}

        assert field_names == {"created_at", "name", "project", "source", "status", "updated_at", "workspace"}

    def test_all_job_filter_fields_have_evaluated_type(self):
        """All fields should have evaluated_type set for proper type detection."""
        fields = introspect_typed_dict(_ResourceFilterParam)

        for field in fields:
            assert field.evaluated_type is not None, f"Field {field.name} should have evaluated_type"
