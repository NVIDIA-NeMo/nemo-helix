# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
from jinja2 import Environment, FileSystemLoader
from nemo_platform._types import Omit
from nemo_platform_sdk_tools.sdk.cli_generator.config import CLIConfig, get_cli_generator_root, get_templates_dir
from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.base import (
    build_path_params,
    promote_name_to_positional,
)
from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.create_collector import CreateContextCollector
from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.delete_collector import DeleteContextCollector
from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.get_collector import GetContextCollector
from nemo_platform_sdk_tools.sdk.cli_generator.context_collectors.update_collector import UpdateContextCollector
from nemo_platform_sdk_tools.sdk.cli_generator.docstring_parser import transform_query_to_cli
from nemo_platform_sdk_tools.sdk.cli_generator.generator import SimpleGenerator
from nemo_platform_sdk_tools.sdk.cli_generator.models import (
    Parameter,
    clean_type_annotation,
    escape_for_python_string,
    sanitize_help_text,
    strip_api_only_lines,
)
from nemo_platform_sdk_tools.sdk.cli_generator.sdk_introspector import SDKMethod, SDKParameter
from nemo_platform_sdk_tools.sdk.cli_generator.type_formatter import format_type_for_help, get_type_schema


class TestTransformQueryToCli:
    """Tests for transform_query_to_cli function."""

    def test_simple_field_transform(self):
        """Should transform simple query params to CLI dot notation."""
        desc = "`?search[name]=imagenet`"
        result = transform_query_to_cli(desc, "search")
        assert "`--search.name imagenet`" in result

    def test_multiple_fields_transform(self):
        """Should transform multiple query params to multiple CLI options."""
        desc = "`?search[name]=imagenet&search[split]=train`"
        result = transform_query_to_cli(desc, "search")
        assert "--search.name imagenet" in result
        assert "--search.split train" in result

    def test_same_field_multiple_values(self):
        """Should handle same field with multiple values (OR query)."""
        desc = "`?search[name]=imagenet&search[name]=coco`"
        result = transform_query_to_cli(desc, "search")
        assert "--search.name imagenet,coco" in result

    def test_nested_field_to_json(self):
        """Should transform nested fields to JSON format."""
        desc = "`?search[updated_at][start]=2024-01-01`"
        result = transform_query_to_cli(desc, "search")
        assert '--search \'{"updated_at":{"start":"2024-01-01"}}\'' in result

    def test_underscore_to_dash(self):
        """Should convert underscores to dashes in field names."""
        desc = "`?search[files_url]=test`"
        result = transform_query_to_cli(desc, "search")
        assert "--search.files-url test" in result

    def test_preserve_surrounding_text(self):
        """Should preserve text around the query params."""
        desc = "Search by name: `?search[name]=foo` for exact match."
        result = transform_query_to_cli(desc, "search")
        assert "Search by name:" in result
        assert "for exact match." in result
        assert "`--search.name foo`" in result

    def test_no_transform_for_different_param(self):
        """Should not transform params that don't match the param_name."""
        desc = "`?filter[name]=foo`"
        result = transform_query_to_cli(desc, "search")
        # Should remain unchanged since it's filter, not search
        assert "`?filter[name]=foo`" in result

    def test_empty_description(self):
        """Should handle empty description."""
        assert transform_query_to_cli("", "search") == ""
        assert transform_query_to_cli(None, "search") is None


class TestCleanTypeAnnotation:
    """Tests for clean_type_annotation function."""

    def test_replace_omit_with_none(self):
        """Should replace '| Omit' with '| None'."""
        assert clean_type_annotation(str | Omit) == "str | None"

    def test_preserve_existing_none(self):
        """Should preserve existing None in type annotations."""
        assert clean_type_annotation(str | None) == "str | None"

    def test_handle_multiple_union_members(self):
        """Should collapse mixed scalar unions to a Typer-compatible type."""
        result = clean_type_annotation(str | int | Omit)
        assert result == "str | None"

    def test_handle_no_omit(self):
        """Should return unchanged if no Omit present."""
        assert clean_type_annotation(str) == "str"
        assert clean_type_annotation(int | None) == "int | None"

    def test_literal_true_false_simplify_to_bool(self):
        """Should simplify Literal[True] | Literal[False] to bool."""
        from typing import Literal

        result = clean_type_annotation(Literal[True] | Literal[False])
        assert result == "bool"

    def test_literal_true_false_with_none_simplify_to_bool_none(self):
        """Should simplify Literal[True] | Literal[False] | None to bool | None."""
        from typing import Literal

        result = clean_type_annotation(Literal[True] | Literal[False] | None)
        assert result == "bool | None"

    def test_literal_true_false_with_omit_simplify_to_bool_none(self):
        """Should simplify Literal[True] | Literal[False] | Omit to bool | None."""
        from typing import Literal

        result = clean_type_annotation(Literal[True] | Literal[False] | Omit)
        assert result == "bool | None"

    def test_literal_string_not_simplified(self):
        """Should not simplify string Literals to bool."""
        from typing import Literal

        result = clean_type_annotation(Literal["a", "b"])
        assert "bool" not in result
        assert "Literal" in result

    def test_literal_single_bool_not_simplified(self):
        """Should not simplify single Literal[True] or Literal[False] to bool."""
        from typing import Literal

        result = clean_type_annotation(Literal[True] | None)
        assert "bool" not in result  # Only one of True/False, not full bool

    def test_literal_with_other_types(self):
        """Should collapse Literal[True] | Literal[False] combined with other types."""
        from typing import Literal

        result = clean_type_annotation(Literal[True] | Literal[False] | str | None)
        assert result == "str | None"

    def test_literal_string_with_str_and_float_simplifies_to_str(self):
        """Should collapse mixed string/numeric value unions for Typer."""
        from typing import Literal

        result = clean_type_annotation(Literal["positive", "negative"] | str | float | None)
        assert result == "str | None"

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (str | int, "str"),
            (int | float, "float"),
            (int | float | None, "float | None"),
            (int | bool, "str"),
            (list[str] | float, "str"),
        ],
    )
    def test_collapse_mixed_union_types(self, annotation: Any, expected: str):
        """Should collapse mixed unions to one Typer-compatible runtime type."""
        assert clean_type_annotation(annotation) == expected

    def test_deduplicate_types(self):
        """Should deduplicate type names in union."""
        result = clean_type_annotation(str | str | None)
        # Should have exactly one 'str'
        assert result.count("str") == 1


class TestExtractImportsFromTemplateOutput:
    """Tests for SimpleGenerator._extract_imports_from_template_output method."""

    def _extract_imports(self, template_output: str) -> tuple[list[str], str]:
        """Extract imports using the same logic as SimpleGenerator."""
        lines = template_output.split("\n")
        import_lines = []
        code_lines = []
        in_imports = True
        in_multiline_import = False

        for line in lines:
            stripped = line.strip()
            if in_imports:
                if in_multiline_import:
                    import_lines.append(line)
                    if ")" in stripped:
                        in_multiline_import = False
                elif stripped.startswith(("from ", "import ")):
                    import_lines.append(line)
                    if "(" in stripped and ")" not in stripped:
                        in_multiline_import = True
                elif stripped == "":
                    continue
                else:
                    in_imports = False
                    code_lines.append(line)
            else:
                code_lines.append(line)

        return import_lines, "\n".join(code_lines)

    def test_single_line_imports(self):
        """Should extract single line imports."""
        template_output = """from typing import Annotated
import typer

@app.command("test")
def test_func():
    pass
"""
        imports, code = self._extract_imports(template_output)
        assert "from typing import Annotated" in imports
        assert "import typer" in imports
        assert "@app.command" in code
        assert "from typing" not in code

    def test_multiline_imports_with_parentheses(self):
        """Should handle multiline imports with parentheses."""
        template_output = """from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
)
import typer

@app.command("test")
def test_func():
    pass
"""
        imports, code = self._extract_imports(template_output)
        assert len([i for i in imports if "nemo_platform_ext.cli.core.types" in i]) >= 1
        # The multiline import should be fully captured
        full_import = "\n".join(imports)
        assert "EntityOutputFormatOption" in full_import
        assert "ListOutputFormatOption" in full_import
        assert "@app.command" in code

    def test_mixed_single_and_multiline_imports(self):
        """Should handle mix of single and multiline imports."""
        template_output = """from typing import Annotated
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
)
import typer

def test_func():
    pass
"""
        imports, code = self._extract_imports(template_output)
        assert "from typing import Annotated" in imports
        assert "import typer" in imports
        full_import = "\n".join(imports)
        assert "EntityOutputFormatOption" in full_import
        assert "def test_func" in code

    def test_no_imports(self):
        """Should handle code with no imports."""
        template_output = """@app.command("test")
def test_func():
    pass
"""
        imports, code = self._extract_imports(template_output)
        assert imports == []
        assert "@app.command" in code

    def test_blank_lines_between_imports(self):
        """Should handle blank lines within import section."""
        template_output = """from typing import Annotated

import typer

def test_func():
    pass
"""
        imports, code = self._extract_imports(template_output)
        assert "from typing import Annotated" in imports
        assert "import typer" in imports
        assert "def test_func" in code


class TestTypeFormatterForHelpSchemas:
    """Tests for help/schema rendering in type_formatter utilities."""

    def test_format_type_for_help_literal_includes_values(self):
        """Literal help strings should surface the actual allowed values."""
        from typing import Literal

        result = format_type_for_help(Literal["queued", "running", "done"])
        assert result == "'queued' | 'running' | 'done'"

    def test_get_type_schema_list_of_literal_includes_values(self):
        """List[Literal[...]] should preserve allowed values in schema hints."""
        from typing import Literal

        result = get_type_schema(list[Literal["queued", "running", "done"]])
        assert result == "['queued' | 'running' | 'done']"


class TestFilterOverrideSkipLines:
    """Tests for _filter_override_skip_lines functionality."""

    def _filter_skip_lines(self, content: str) -> str:
        """Filter out lines with '# override-skip' comment."""
        lines = content.split("\n")
        filtered = [line for line in lines if "# override-skip" not in line]
        return "\n".join(filtered)

    def test_filter_override_skip_lines(self):
        """Should filter out lines with '# override-skip' comment."""
        content = """from typing import Any
app = cast(Any, None)  # override-skip: provided by generated file

@app.command("test")
def test_func():
    pass
"""
        result = self._filter_skip_lines(content)
        assert "override-skip" not in result
        assert "app = cast" not in result
        assert "@app.command" in result

    def test_no_skip_lines(self):
        """Should return content unchanged if no skip lines."""
        content = """from typing import Annotated

@app.command("test")
def test_func():
    pass
"""
        result = self._filter_skip_lines(content)
        assert "from typing import Annotated" in result
        assert "@app.command" in result


class TestInitFileGeneration:
    def test_imports_subresources_without_annotations_package_attribute_collision(self, tmp_path: Path):
        class _NoResourceConfig:
            def get_resource_config(self, resource_path: list[str]) -> None:
                return None

        def get_sub_resources(resource_path: list[str]) -> list[str]:
            return ["annotations", "traces"]

        package_root = tmp_path
        target_dir = package_root / "nemo_platform_ext" / "cli" / "commands" / "api"
        for package_dir in [
            package_root / "nemo_platform_ext",
            package_root / "nemo_platform_ext" / "cli",
            package_root / "nemo_platform_ext" / "cli" / "commands",
            target_dir,
        ]:
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").write_text("")

        core_dir = package_root / "nemo_platform_ext" / "cli" / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        (core_dir / "__init__.py").write_text("")
        (core_dir / "help_formatter.py").write_text(
            "\n".join(
                [
                    "class _App:",
                    "    def __init__(self):",
                    "        self.children = []",
                    "",
                    "    def add_typer(self, app, name):",
                    "        self.children.append((name, app))",
                    "",
                    "def create_typer_app(*, name: str, help: str):",
                    "    return _App()",
                    "",
                ]
            )
        )

        generator = SimpleGenerator.__new__(SimpleGenerator)
        generator._target_dir = target_dir
        generator._cli_config = _NoResourceConfig()
        generator._get_sub_resources = get_sub_resources

        generator._generate_init_with_commands(["intake"], "", [])

        intake_dir = target_dir / "intake"
        (intake_dir / "annotations.py").write_text('app = "annotations-app"\n')
        (intake_dir / "traces.py").write_text('app = "traces-app"\n')

        content = (intake_dir / "__init__.py").read_text()
        assert "from __future__ import annotations" in content
        assert (
            '_cli_child_annotations = _importlib_import_module("nemo_platform_ext.cli.commands.api.intake.annotations")'
            in content
        )
        assert (
            '_cli_child_traces = _importlib_import_module("nemo_platform_ext.cli.commands.api.intake.traces")'
            in content
        )
        assert 'app.add_typer(_cli_child_annotations.app, name="annotations")' in content
        assert 'app.add_typer(_cli_child_traces.app, name="traces")' in content
        assert "from nemo_platform_ext.cli.commands.api.intake import annotations" not in content
        assert "app.add_typer(annotations.app" not in content

        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "nemo_platform_ext" or name.startswith("nemo_platform_ext.")
        }
        sys.path.insert(0, str(package_root))
        try:
            for name in saved_modules:
                sys.modules.pop(name, None)
            module = importlib.import_module("nemo_platform_ext.cli.commands.api.intake")
            assert module.app.children == [("annotations", "annotations-app"), ("traces", "traces-app")]
        finally:
            sys.path.pop(0)
            for name in list(sys.modules):
                if name == "nemo_platform_ext" or name.startswith("nemo_platform_ext."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


class TestGetApiModules:
    def test_collects_top_level_metadata_by_importing_generated_modules(self, tmp_path: Path):
        generator = SimpleGenerator.__new__(SimpleGenerator)
        target_dir = tmp_path / "nemo_platform_ext" / "cli" / "commands" / "api"
        target_dir.mkdir(parents=True)
        generator._target_dir = target_dir
        generator._cli_config = _make_config("""
top_level:
  evaluation:
    panel: Functional plugins
    help: Evaluation operations.
  files:
    hidden: true
config: []
""")

        for package_dir in [
            tmp_path / "nemo_platform_ext",
            tmp_path / "nemo_platform_ext" / "cli",
            tmp_path / "nemo_platform_ext" / "cli" / "commands",
            target_dir,
        ]:
            (package_dir / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").write_text("")

        core_dir = tmp_path / "nemo_platform_ext" / "cli" / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        (core_dir / "__init__.py").write_text("")
        (core_dir / "help_formatter.py").write_text(
            "\n".join(
                [
                    "class _Info:",
                    "    def __init__(self, help: str):",
                    "        self.help = help",
                    "",
                    "class _App:",
                    "    def __init__(self, help: str):",
                    "        self.info = _Info(help)",
                    "",
                    "def create_typer_app(*, name: str, help: str):",
                    "    return _App(help)",
                    "",
                ]
            )
        )

        (target_dir / "files.py").write_text(
            "\n".join(
                [
                    "# NOTE: This file is auto-generated",
                    "from __future__ import annotations",
                    "",
                    "from nemo_platform_ext.cli.core.help_formatter import create_typer_app",
                    "",
                    'app = create_typer_app(name="files", help="Manage files")',
                    "",
                ]
            )
        )

        evaluation_dir = target_dir / "evaluation"
        evaluation_dir.mkdir()
        (evaluation_dir / "__init__.py").write_text(
            "\n".join(
                [
                    "# NOTE: This file is auto-generated",
                    "from __future__ import annotations",
                    "",
                    "from nemo_platform_ext.cli.core.help_formatter import create_typer_app",
                    "",
                    'app = create_typer_app(name="evaluation", help="Evaluation operations")',
                    "",
                ]
            )
        )

        modules = generator._get_api_modules()

        assert modules == [
            {
                "name": "evaluation",
                "cli_name": "evaluation",
                "help": "Evaluation operations.",
                "panel": "Functional plugins",
                "hidden": False,
            },
            {"name": "files", "cli_name": "files", "help": "Manage files", "panel": "Core plugins", "hidden": True},
        ]


class TestGeneratorIntegration:
    """Integration tests for the generator with real SDK types."""

    def test_jobs_resource_is_skipped_by_real_cli_config(self):
        """Jobs CLI generation should not introspect the deleted Stainless resource."""
        config = CLIConfig(get_cli_generator_root() / "cli_config.yaml")

        assert config.should_skip(["jobs"])


def _make_sdk_method(path_params: list[tuple[str, str | None]]) -> SDKMethod:
    """Build a minimal SDKMethod with the given path parameters.

    Args:
        path_params: List of (name, sdk_description) tuples. Descriptions are embedded
            in the method docstring's Args section so get_param_description can find them.
    """
    parameters = [
        SDKParameter(
            name=name,
            type_annotation=str,
            default=...,
            is_required=True,
            is_positional=True,
        )
        for name, _ in path_params
    ]

    # Build a Google-style docstring so ParsedDocstring.parse picks up the descriptions.
    described = [(name, desc) for name, desc in path_params if desc]
    if described:
        args_lines = "\n".join(f"    {name}: {desc}" for name, desc in described)
        docstring = f"A method.\n\nArgs:\n{args_lines}"
    else:
        docstring = None

    return SDKMethod(
        name="list",
        resource_path=["things"],
        parameters=parameters,
        return_type=type(None),
        docstring=docstring,
    )


class TestBuildPathParams:
    """Tests for build_path_params with param_help_overrides."""

    def test_uses_sdk_description_when_no_override(self):
        method = _make_sdk_method([("entity_type", "SDK description")])
        params = build_path_params(method)
        assert len(params) == 1
        assert params[0].help == "SDK description"

    def test_override_replaces_sdk_description(self):
        method = _make_sdk_method([("entity_type", "SDK description")])
        params = build_path_params(method, param_help_overrides={"entity_type": "Custom help"})
        assert params[0].help == "Custom help"

    def test_override_only_applies_to_named_param(self):
        method = _make_sdk_method([("entity_type", "SDK A"), ("workspace", "SDK B")])
        params = build_path_params(method, param_help_overrides={"entity_type": "Custom A"})
        by_name = {p.var_name: p for p in params}
        assert by_name["entity_type"].help == "Custom A"
        assert by_name["workspace"].help == "SDK B"

    def test_override_with_none_sdk_description(self):
        method = _make_sdk_method([("entity_type", None)])
        params = build_path_params(method, param_help_overrides={"entity_type": "Custom help"})
        assert params[0].help == "Custom help"

    def test_empty_overrides_dict_uses_sdk_description(self):
        method = _make_sdk_method([("entity_type", "SDK description")])
        params = build_path_params(method, param_help_overrides={})
        assert params[0].help == "SDK description"

    def test_no_overrides_uses_sdk_description(self):
        method = _make_sdk_method([("entity_type", "SDK description")])
        params = build_path_params(method, param_help_overrides=None)
        assert params[0].help == "SDK description"


class TestStripApiOnlyLines:
    """Tests for strip_api_only_lines and its integration into help text processing."""

    def test_strips_bracket_notation_line(self):
        text = "CLI help\n- Bracket notation: ?filter[name][$like]=value"
        assert strip_api_only_lines(text) == "CLI help"

    def test_strips_relationship_traversal_line(self):
        text = "CLI help\n- Relationship traversal: ?filter[relationship][$exists]=true"
        assert strip_api_only_lines(text) == "CLI help"

    def test_strips_search_bracket_line(self):
        text = "CLI help\n- Example: ?search[name]=foo"
        assert strip_api_only_lines(text) == "CLI help"

    def test_preserves_non_query_param_lines(self):
        text = '- Text: name:"value" AND status>500\n- Object (JSON): {"name":{"$like":"value"}}'
        assert strip_api_only_lines(text) == text

    def test_no_query_lines_returns_unchanged(self):
        text = "Just regular help text"
        assert strip_api_only_lines(text) == text

    def test_escape_for_python_string_strips_query_lines(self):
        text = "CLI help\n- Bracket: ?filter[name][$like]=value"
        result = escape_for_python_string(text)
        assert "Bracket" not in result

    def test_sanitize_help_text_strips_query_lines(self):
        text = "CLI help\n- Bracket notation: ?filter[name][$like]=value\n- Relationship: ?filter[rel][$exists]=true"
        result = sanitize_help_text(text)
        assert "Bracket" not in result
        assert "Relationship" not in result
        assert result == "CLI help"

    def test_none_inputs(self):
        assert escape_for_python_string(None) is None
        assert sanitize_help_text(None) is None


_EMPTY_CONFIG = "config: []"


def _make_config(config_yaml: str) -> CLIConfig:
    """Create a CLIConfig from an inline YAML string."""
    with NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(config_yaml)
        f.flush()
        return CLIConfig(Path(f.name))


def _make_body_param(name: str, help_text: str | None = None) -> Parameter:
    """Build a minimal body Parameter (option) with the given name."""
    cli_option = f"--{name}"
    option_args = f'"{cli_option}"' if not help_text else f'"{cli_option}", help="{help_text}"'
    return Parameter(
        var_name=name,
        type="str | None",
        option_args=option_args,
        default="None",
        help=help_text,
    )


def _render_create_command(context: dict[str, Any]) -> str:
    env = Environment(  # noqa: S701  # nosec B701
        loader=FileSystemLoader(get_templates_dir()),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["repr"] = repr
    env.filters["to_kebab"] = lambda value: str(value).replace("_", "-")
    return env.get_template("create_command.py.j2").render(**context)


class TestCLIConfig:
    def test_top_level_command_config_ignores_non_mapping_top_level(self):
        config = _make_config("""
top_level: []
config: []
""")

        assert config.get_top_level_command_config("files") == {}


class TestResourceHelp:
    def test_resource_help_uses_configured_value(self):
        generator = SimpleGenerator.__new__(SimpleGenerator)
        generator._cli_config = _make_config("""
config:
- resource: [audit]
  help: Auditor operations.
""")

        assert generator._get_resource_help(["audit"], "Manage audit") == "Auditor operations."
        assert generator._get_resource_help(["files"], "Manage files") == "Manage files"


def _make_sdk_method_with_body(body_params: list[tuple[str, str | None]]) -> SDKMethod:
    """Build an SDKMethod whose optional_parameters come from body_params.

    Each tuple is (param_name, description_or_None).
    """
    parameters = [
        SDKParameter(
            name=name,
            type_annotation=str,
            default=None,
            is_required=False,
            is_positional=False,
            description=desc,
        )
        for name, desc in body_params
    ]
    return SDKMethod(
        name="create",
        resource_path=["things"],
        parameters=parameters,
        return_type=type(None),
        docstring=None,
    )


class TestParameterToTyperArgument:
    """Tests for Parameter.to_typer_argument()."""

    def test_required_argument_has_no_default(self):
        """default=None (Python None) → no '= ...' in output."""
        p = Parameter(var_name="name", type="str", option_args='"--name"', default=None)
        result = p.to_typer_argument()
        assert "= " not in result
        assert "name: Annotated[str, typer.Argument()]," == result

    def test_optional_argument_emits_none_default(self):
        """default='None' (string) → '= None' in output."""
        p = Parameter(var_name="name", type="str | None", option_args='"--name"', default="None")
        result = p.to_typer_argument()
        assert result.endswith("= None,")

    def test_help_text_included(self):
        """Help text is embedded in typer.Argument(help=...)."""
        p = Parameter(
            var_name="name",
            type="str",
            option_args='"--name"',
            default=None,
            help="The resource name",
        )
        result = p.to_typer_argument()
        assert 'help="The resource name"' in result
        assert "typer.Argument(" in result

    def test_no_help_text_uses_bare_argument(self):
        """No help text → typer.Argument() with no args."""
        p = Parameter(var_name="name", type="str", option_args='"--name"', default=None, help=None)
        result = p.to_typer_argument()
        assert "typer.Argument()" in result

    def test_help_text_is_escaped(self):
        """Double quotes in help text are escaped."""
        p = Parameter(
            var_name="name",
            type="str",
            option_args='"--name"',
            default=None,
            help='Say "hello"',
        )
        result = p.to_typer_argument()
        assert '\\"hello\\"' in result

    def test_trailing_comma(self):
        """Output always ends with a comma (for use in function signatures)."""
        p = Parameter(var_name="x", type="str", option_args='"--x"', default=None)
        assert p.to_typer_argument().endswith(",")


class TestPromoteNameToPositional:
    """Tests for the promote_name_to_positional helper."""

    def _config(self, yaml: str = _EMPTY_CONFIG) -> CLIConfig:
        return _make_config(yaml)

    def test_promotes_name_by_default(self):
        params = [_make_body_param("name")]
        promote_name_to_positional(params, ["things"], "create", self._config())
        assert params[0].is_positional is True

    def test_returns_true_when_name_found(self):
        params = [_make_body_param("name")]
        result = promote_name_to_positional(params, ["things"], "create", self._config())
        assert result is True

    def test_returns_false_when_name_not_in_params(self):
        params = [_make_body_param("description")]
        result = promote_name_to_positional(params, ["things"], "create", self._config())
        assert result is False

    def test_suppressed_by_name_positional_false(self):
        config = self._config("""
config:
  - resource: [things]
    methods:
      delete:
        name_positional: false
""")
        params = [_make_body_param("name")]
        result = promote_name_to_positional(params, ["things"], "delete", config)
        assert result is False
        assert params[0].is_positional is False

    def test_applies_type_str_override(self):
        params = [_make_body_param("name")]
        promote_name_to_positional(params, ["things"], "create", self._config(), type_str="str")
        assert params[0].type == "str"

    def test_applies_none_default_for_required(self):
        """Passing default=None (Python None) makes the argument required."""
        params = [_make_body_param("name")]
        promote_name_to_positional(params, ["things"], "update", self._config(), default=None)
        assert params[0].default is None

    def test_applies_string_none_default_for_optional(self):
        """Passing default='None' (string) makes the argument optional."""
        params = [_make_body_param("name")]
        promote_name_to_positional(params, ["things"], "create", self._config(), default="None")
        assert params[0].default == "None"

    def test_does_not_touch_other_params(self):
        params = [_make_body_param("description"), _make_body_param("name")]
        promote_name_to_positional(params, ["things"], "create", self._config())
        assert params[0].is_positional is False
        assert params[1].is_positional is True

    def test_empty_params_list_returns_false(self):
        result = promote_name_to_positional([], ["things"], "create", self._config())
        assert result is False


class TestCollectorNamePromotion:
    """Verify each collector promotes 'name' correctly and respects config."""

    def _config(self, yaml: str = _EMPTY_CONFIG) -> CLIConfig:
        return _make_config(yaml)

    def _suppressed_config(self, method: str) -> CLIConfig:
        return self._config(f"""
config:
  - resource: [things]
    methods:
      {method}:
        name_positional: false
""")

    def test_create_promotes_name_to_optional_positional(self):
        sdk_method = _make_sdk_method_with_body([("name", None), ("description", None)])
        ctx = CreateContextCollector(self._config()).collect(["things"], sdk_method, "create")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is True
        assert name_param.default == "None"
        assert "None" in name_param.type

    def test_update_promotes_name_to_required_positional(self):
        sdk_method = _make_sdk_method_with_body([("name", None), ("description", None)])
        ctx = UpdateContextCollector(self._config()).collect(["things"], sdk_method, "update")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is True
        assert name_param.default is None
        assert name_param.type == "str"

    def test_delete_promotes_name_to_positional(self):
        sdk_method = _make_sdk_method_with_body([("name", None)])
        ctx = DeleteContextCollector(self._config()).collect(["things"], sdk_method, "delete")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is True

    def test_get_promotes_name_to_positional(self):
        sdk_method = _make_sdk_method_with_body([("name", None)])
        ctx = GetContextCollector(self._config()).collect(["things"], sdk_method, "get")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is True

    def test_create_no_name_param_does_not_crash(self):
        sdk_method = _make_sdk_method_with_body([("description", None)])
        ctx = CreateContextCollector(self._config()).collect(["things"], sdk_method, "create")
        assert all(p.var_name != "name" for p in ctx["parameters"])

    def test_create_includes_watch_config(self):
        config = self._config("""
config:
  - resource: [things]
    methods:
      create:
        watch:
          type: platform_job
          resource_label: thing job
""")
        sdk_method = _make_sdk_method_with_body([("name", None), ("spec", None)])
        ctx = CreateContextCollector(config).collect(["things"], sdk_method, "create")

        assert ctx["watch_config"] == {"type": "platform_job", "resource_label": "thing job"}
        assert ctx["wait_config"] is None
        rendered = _render_create_command(ctx)
        assert "from nemo_platform_ext.jobs.watch import watch_job" not in rendered
        assert (
            "from nemo_platform_ext.cli.core.job_watch_renderer import JobWatchRenderResult, render_job_watch_events"
            in rendered
        )
        assert "from nemo_platform_ext.cli.core.waiters import wait_for_platform_job" not in rendered
        assert "from nemo_platform_plugin.client.adapter import client_from_platform" in rendered
        assert "from nemo_platform_plugin.jobs.client import JobsClient" in rendered
        assert "jobs_client.watch_job(" in rendered

    def test_create_platform_job_watch_config_renders_wait_and_watch(self):
        config = self._config("""
config:
  - resource: [things]
    methods:
      create:
        wait:
          type: platform_job
          resource_label: thing job
        watch:
          type: platform_job
          resource_label: thing job
""")
        sdk_method = _make_sdk_method_with_body([("name", None), ("spec", None)])
        ctx = CreateContextCollector(config).collect(["things"], sdk_method, "create")

        rendered = _render_create_command(ctx)

        assert 'typer.Option("--watch"' in rendered
        assert 'typer.Option("--wait"' in rendered
        assert "timeout: Annotated[int | None" in rendered
        assert "] = None" in rendered
        assert 'watch_options={"timeout": timeout, "poll_interval": poll_interval} if watch else None' in rendered
        assert (
            'wait_options={"timeout": timeout if timeout is not None else 1200, "poll_interval": poll_interval} '
            "if wait else None"
        ) in rendered
        assert "wait_config={\"type\": 'platform_job', \"resource_label\": 'thing job'} if wait else None" in rendered
        assert "from nemo_platform_ext.jobs.watch import watch_job" not in rendered
        assert "from nemo_platform_ext.cli.core.waiters import wait_for_platform_job" in rendered
        assert (
            "from nemo_platform_ext.cli.core.job_watch_renderer import JobWatchRenderResult, render_job_watch_events"
            in rendered
        )
        assert "from nemo_platform_plugin.client.adapter import client_from_platform" in rendered
        assert "from nemo_platform_plugin.jobs.client import JobsClient" in rendered
        assert "jobs_client = client_from_platform(client, JobsClient)" in rendered
        assert "client.things.get_status" not in rendered
        assert "wait_for_platform_job(" in rendered
        assert "jobs_client.watch_job(" in rendered
        assert "resource_label='thing job'" in rendered
        assert "watch_result = render_job_watch_events(events, resource_label='thing job')" in rendered
        assert "if watch_result is JobWatchRenderResult.INTERRUPTED:" in rendered
        assert "raise typer.Exit(130)" in rendered
        assert "if watch_result is not JobWatchRenderResult.SUCCEEDED:" in rendered
        assert rendered.index("format_output(") < rendered.index("if wait or watch:")

    def test_create_inference_deployment_config_renders_wait_and_watch(self):
        config = self._config("""
config:
  - resource: [inference, deployments]
    methods:
      create:
        wait:
          type: inference_deployment
          resource_label: deployment
        watch:
          type: inference_deployment
          resource_label: deployment
""")
        sdk_method = _make_sdk_method_with_body([("name", None), ("workspace", None), ("config", None)])
        ctx = CreateContextCollector(config).collect(["inference", "deployments"], sdk_method, "create")

        rendered = _render_create_command(ctx)

        assert 'typer.Option("--wait"' in rendered
        assert 'typer.Option("--watch"' in rendered
        assert "Wait for the created deployment to be up and running" in rendered
        assert "Watch the created deployment until it is stable while streaming status updates" in rendered
        assert "created deployment to reach a terminal state" not in rendered
        assert "timeout if timeout is not None else 1200" not in rendered
        assert "timeout: Annotated[int, typer.Option" in rendered
        assert "] = 1200" in rendered
        assert "from nemo_platform_ext.cli.core.waiters import wait_for_inference_deployment" in rendered
        assert "from nemo_platform_ext.jobs.watch import watch_job" not in rendered
        assert (
            "watch_config={\"type\": 'inference_deployment', \"resource_label\": 'deployment'} if watch else None"
            in rendered
        )
        assert 'watch_options={"timeout": timeout, "poll_interval": poll_interval} if watch else None' in rendered
        assert (
            "wait_config={\"type\": 'inference_deployment', \"resource_label\": 'deployment'} if wait else None"
            in rendered
        )
        assert "if wait or watch:" in rendered
        assert "wait_for_inference_deployment(" in rendered
        assert "verbose=" not in rendered
        assert "Unable to determine created resource name for --wait/--watch" in rendered

    def test_create_inference_deployment_wait_only_does_not_reference_watch(self):
        config = self._config("""
config:
  - resource: [inference, deployments]
    methods:
      create:
        wait:
          type: inference_deployment
          resource_label: deployment
""")
        sdk_method = _make_sdk_method_with_body([("name", None), ("workspace", None), ("config", None)])
        ctx = CreateContextCollector(config).collect(["inference", "deployments"], sdk_method, "create")

        rendered = _render_create_command(ctx)

        assert 'typer.Option("--wait"' in rendered
        assert 'typer.Option("--watch"' not in rendered
        assert "if wait:" in rendered
        assert "verbose=" not in rendered
        assert "verbose=watch" not in rendered

    def test_delete_suppressed_by_config(self):
        sdk_method = _make_sdk_method_with_body([("name", None)])
        ctx = DeleteContextCollector(self._suppressed_config("delete")).collect(["things"], sdk_method, "delete")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is False

    def test_get_suppressed_by_config(self):
        sdk_method = _make_sdk_method_with_body([("name", None)])
        ctx = GetContextCollector(self._suppressed_config("get")).collect(["things"], sdk_method, "get")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is False

    def test_update_suppressed_by_config(self):
        sdk_method = _make_sdk_method_with_body([("name", None)])
        ctx = UpdateContextCollector(self._suppressed_config("update")).collect(["things"], sdk_method, "update")
        name_param = next(p for p in ctx["parameters"] if p.var_name == "name")
        assert name_param.is_positional is False
