# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simplified command-by-command CLI generator (refactored)."""

from __future__ import annotations

import importlib
import logging
import shutil
import sys
from pathlib import Path
from traceback import print_tb
from typing import Any

from caseutil.cases import to_kebab
from jinja2 import Environment, FileSystemLoader
from nemo_helix_sdk_tools.sdk.cli_generator.config import (
    CLIConfig,
    _resolve_placeholders,
    get_overrides_dir,
    get_target_commands_dir,
    get_templates_dir,
)
from nemo_helix_sdk_tools.sdk.cli_generator.context_collectors import ContextCollectorRegistry
from nemo_helix_sdk_tools.sdk.cli_generator.models import escape_for_python_string
from nemo_helix_sdk_tools.sdk.cli_generator.operation_classifier import (
    OperationClassifier,
    OperationInfo,
)
from nemo_helix_sdk_tools.sdk.cli_generator.sdk_introspector import SDKIntrospector
from nemo_helix_sdk_tools.sdk.cli_generator.type_formatter import extract_imports_from_type
from nemo_helix_sdk_tools.sdk.core.common import get_project_dir
from nemo_helix_sdk_tools.sdk.core.stainless import StainlessConfig

AUTO_GENERATED_FILE_HEADER = [
    "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
    "",
    "# NOTE: This file is auto-generated",
    "from __future__ import annotations",
    "",
]

_SUB_RESOURCE_IMPORT_ALIAS_TEMPLATE = "_cli_child_{resource_name}"


logger = logging.getLogger(__name__)


class SimpleGenerator:
    """Generate CLI commands with modular architecture."""

    def __init__(self, stainless_config_path: Path, cli_config_path: Path):
        self._stainless_config = StainlessConfig.from_file(stainless_config_path)
        self._cli_config = CLIConfig(cli_config_path)
        self._introspector = SDKIntrospector()
        self._classifier = OperationClassifier(self._cli_config)
        self._target_dir = get_target_commands_dir()

        # Set up Jinja2
        template_dir = get_templates_dir()
        # Generates Python source, not HTML. Autoescape would corrupt Python
        # string literals / type annotations containing quotes or angle brackets.
        self._jinja_env = Environment(  # noqa: S701  # nosec B701
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        self._jinja_env.filters["repr"] = repr
        self._jinja_env.filters["to_kebab"] = to_kebab

        # Track which resources have sub-resources (need to be directories)
        self._resources_with_children: set[tuple[str, ...]] = set()

    def generate_all(self) -> None:
        """Generate all CLI commands using new modular architecture."""
        print("Generating all CLI commands...")
        print()

        # Step 1: Clear old generated files
        self._clear_generated_files()

        # Step 2: Discover & classify ALL methods
        all_operations = self._discover_and_classify_all()

        # Step 3: Group by resource
        operations_by_resource = self._group_by_resource(all_operations)

        # Step 4: Identify which resources have sub-resources
        self._identify_parent_resources(set(operations_by_resource.keys()))

        # Step 5: Collect contexts for all operations
        contexts_by_resource = self._collect_all_contexts(operations_by_resource)

        # Step 6: Generate files
        generated_count = self._generate_all_files(contexts_by_resource)

        print(f"\n✅ Generated {generated_count} commands")

        # Step 7: Generate api/__init__.py
        print("\nGenerating api/__init__.py...")
        self.generate_api_init_file()

    def _discover_and_classify_all(self) -> list[OperationInfo]:
        """Discover ALL SDK methods and classify them."""
        methods = self._stainless_config.extract_methods()
        operations = []

        # Get unique resource paths (avoid introspecting same resource multiple times)
        unique_resources = set()
        for method_info in methods:
            resource_path = method_info.resource_path
            if not self._cli_config.should_skip(resource_path):
                unique_resources.add(tuple(resource_path))

        # Introspect each unique resource once
        # Sort for deterministic iteration order - prevents Python's Literal type
        # caching from causing non-deterministic output when multiple resources
        # define Literal types with the same values in different orders
        for resource_path in sorted(unique_resources):
            try:
                # Introspect SDK methods for this resource
                sdk_methods = self._introspector.introspect_resource(list(resource_path))

                for _, sdk_method in sdk_methods.items():
                    # Classify this method
                    op_type = self._classifier.classify(sdk_method)

                    operations.append(
                        OperationInfo(
                            operation_type=op_type,
                            sdk_method=sdk_method,
                            method_name=sdk_method.name,
                            resource_path=list(resource_path),
                        )
                    )
            except Exception as e:
                print(f"✗ Error introspecting {'.'.join(resource_path)}: {e}")

        return operations

    def _group_by_resource(self, operations: list[OperationInfo]) -> dict[tuple[str, ...], list[OperationInfo]]:
        """Group operations by resource path.

        Includes both SDK-discovered resources and CLI config resources
        (those with additional_methods but no SDK methods).
        """
        grouped: dict[tuple[str, ...], list[OperationInfo]] = {}

        # From SDK methods
        for op_info in operations:
            resource_key = tuple(op_info.resource_path)
            grouped.setdefault(resource_key, []).append(op_info)

        # From CLI config (resources defined with additional_methods)
        for resource_path in self._cli_config.get_all_resources_with_additional_methods():
            grouped.setdefault(resource_path, [])

        return grouped

    def _identify_parent_resources(self, resource_paths: set[tuple[str, ...]]) -> None:
        """Identify which resources have sub-resources."""
        for path in resource_paths:
            # Check if any other path starts with this one
            for other_path in resource_paths:
                if other_path != path and len(other_path) > len(path):
                    if other_path[: len(path)] == path:
                        self._resources_with_children.add(path)
                        break

    def _collect_all_contexts(
        self, operations_by_resource: dict[tuple[str, ...], list[OperationInfo]]
    ) -> dict[tuple[str, ...], list[dict[str, Any]]]:
        """Collect template contexts for all operations."""
        contexts: dict[tuple[str, ...], list[dict[str, Any]]] = {}

        for resource_path, ops in operations_by_resource.items():
            resource_contexts = []

            for op_info in ops:
                try:
                    # Skip methods marked with skip: true
                    if self._cli_config.should_skip_method(op_info.resource_path, op_info.method_name):
                        continue

                    # Check for override first
                    override_path = self._cli_config.get_method_override(op_info.resource_path, op_info.method_name)
                    if override_path:
                        resource_contexts.append(
                            {
                                "operation_type": "override",
                                "override_path": override_path,
                                "method_name": op_info.method_name,
                                "resource_path": op_info.resource_path,
                            }
                        )
                        continue

                    # Normal template-based generation
                    collector = ContextCollectorRegistry.get_collector(op_info.operation_type, self._cli_config)

                    context = collector.collect(
                        op_info.resource_path,
                        op_info.sdk_method,
                        op_info.method_name,
                    )

                    resource_contexts.append(context)
                except Exception as e:
                    print(
                        f"✗ Error collecting context for {'.'.join(op_info.resource_path)}.{op_info.method_name}: {e}"
                    )
                    print_tb(e.__traceback__)

            # Process additional methods for this resource
            additional_methods = self._cli_config.get_additional_methods(list(resource_path))
            for method_name, method_config in additional_methods.items():
                if override_rel_path := method_config.get("override"):
                    override_path = get_overrides_dir() / str(override_rel_path)
                    resource_contexts.append(
                        {
                            "operation_type": "override",
                            "override_path": override_path,
                            "method_name": method_name,
                            "resource_path": list(resource_path),
                        }
                    )

            if resource_contexts:
                contexts[resource_path] = resource_contexts

        return contexts

    def _generate_all_files(self, contexts_by_resource: dict[tuple[str, ...], list[dict[str, Any]]]) -> int:
        """Generate all CLI files."""
        generated_count = 0

        for resource_path, contexts in contexts_by_resource.items():
            try:
                # Render commands from templates and collect imports
                command_codes = []
                all_imports = []
                all_type_imports = set()

                for context in contexts:
                    try:
                        op_type = context["operation_type"]

                        if op_type == "override":
                            # Load override file instead of rendering template
                            override_path = context["override_path"]
                            if not override_path.exists():
                                raise FileNotFoundError(
                                    f"Override file not found: {override_path}. "
                                    f"Check the 'override' path in cli_config.yaml for method '{context.get('method_name')}'"
                                )
                            override_content = self._filter_override_skip_lines(
                                self._resolve_override_placeholders(override_path.read_text())
                            )
                            imports, code_without_imports = self._extract_imports_from_template_output(override_content)
                            all_imports.extend(imports)
                            command_codes.append(code_without_imports)
                            continue

                        # Normal template rendering
                        template = self._jinja_env.get_template(f"{op_type}_command.py.j2")
                        command_code = template.render(**context)
                        # Extract imports from template output
                        imports, code_without_imports = self._extract_imports_from_template_output(command_code)
                        all_imports.extend(imports)
                        command_codes.append(code_without_imports)

                        # Extract imports needed from parameter types
                        type_imports = self._extract_type_imports_from_context(context)
                        all_type_imports.update(type_imports)
                    except Exception as e:
                        raise RuntimeError(f"Template rendering failed: {e}. Context: {context}") from e

                combined_commands = "\n\n".join(command_codes)

                # Combine template imports and type imports
                all_imports.extend(sorted(all_type_imports))

                # Deduplicate imports while preserving order
                unique_imports = []
                seen = set()
                for imp in all_imports:
                    if imp not in seen:
                        unique_imports.append(imp)
                        seen.add(imp)

                # Determine if this resource has sub-resources
                has_children = resource_path in self._resources_with_children

                if has_children:
                    # Commands go in __init__.py with sub-app registration
                    self._generate_init_with_commands(list(resource_path), combined_commands, unique_imports)
                else:
                    # Commands go in standalone .py file
                    self._generate_standalone_file(list(resource_path), combined_commands, unique_imports)

                # Generate intermediate __init__.py files for nested resources
                if len(resource_path) > 1:
                    self._generate_intermediate_init_files(list(resource_path))

                # Print success
                method_names = ", ".join(c["method_name"] for c in contexts)
                print(f"✓ Generated commands for {'.'.join(resource_path)} ({method_names})")
                generated_count += len(contexts)

            except Exception as e:
                print(f"✗ Error generating {'.'.join(resource_path)}: {e}")
                print_tb(e.__traceback__)

        return generated_count

    def _generate_standalone_file(
        self,
        resource_path: list[str],
        command_code: str,
        imports: list[str],
    ) -> None:
        """Generate a standalone .py file."""
        resource_name = resource_path[-1]

        # Build file with docstring + imports + app + commands
        lines = [
            *AUTO_GENERATED_FILE_HEADER,
            "from nemo_helix_ext.cli.core.help_formatter import create_typer_app",
            "",
        ]

        # Add imports extracted from templates
        lines.extend(imports)
        lines.append("")

        # Add app
        app_help = self._get_resource_help(resource_path, f"Manage {resource_name}")
        lines.append(f'app = create_typer_app(name="{resource_name}", help="{escape_for_python_string(app_help)}")')
        lines.extend(["", ""])

        # Add commands (imports already stripped from templates by render)
        lines.append(command_code)
        lines.append("")

        file_content = "\n".join(lines)

        # Write to disk
        target_file = self._get_target_file(resource_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(file_content)

    def _generate_init_with_commands(
        self,
        resource_path: list[str],
        command_code: str,
        imports: list[str],
    ) -> None:
        """Generate __init__.py with commands and sub-app registrations.

        This is used when a resource has both its own commands AND sub-resources.
        """
        resource_name = resource_path[-1]

        # Determine which sub-resources this parent has
        sub_resources = self._get_sub_resources(resource_path)

        # Build the file content
        import_base = "nemo_helix_ext.cli.commands.api"
        parent_import = f"{import_base}.{'.'.join(resource_path)}"
        sorted_sub_resources = sorted(sub_resources)

        lines = [*AUTO_GENERATED_FILE_HEADER]

        if sub_resources:
            lines.extend(
                [
                    "from importlib import import_module as _importlib_import_module",
                    "",
                ]
            )

        # Add imports extracted from templates
        lines.extend(imports)
        lines.append("")

        lines.extend(
            [
                "from nemo_helix_ext.cli.core.help_formatter import create_typer_app",
                "",
            ]
        )

        # Import child modules after regular imports. import_module forces the
        # submodule load instead of reading same-named package globals.
        for sub in sorted_sub_resources:
            import_alias = _SUB_RESOURCE_IMPORT_ALIAS_TEMPLATE.format(resource_name=sub)
            lines.append(f'{import_alias} = _importlib_import_module("{parent_import}.{sub}")')

        if sub_resources:
            lines.append("")

        # App creation
        app_help = self._get_resource_help(resource_path, f"Manage {resource_name}")
        lines.append(f'app = create_typer_app(name="{resource_name}", help="{escape_for_python_string(app_help)}")')
        lines.append("")

        # Register sub-apps
        for sub in sorted_sub_resources:
            cli_name = sub.replace("_", "-")
            import_alias = _SUB_RESOURCE_IMPORT_ALIAS_TEMPLATE.format(resource_name=sub)
            lines.append(f'app.add_typer({import_alias}.app, name="{cli_name}")')

        if sub_resources:
            lines.append("")

        lines.extend(["", command_code, ""])

        # Write the file
        target_dir = self._target_dir / "/".join(resource_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        init_file = target_dir / "__init__.py"
        init_file.write_text("\n".join(lines))

    def _get_sub_resources(self, parent_path: list[str]) -> list[str]:
        """Get immediate sub-resources of a parent resource.

        Only returns sub-resources that will actually be generated.
        """
        parent_tuple = tuple(parent_path)
        parent_len = len(parent_tuple)
        sub_resources = set()

        # Check all resource paths that will actually be generated
        methods = self._stainless_config.extract_methods()
        resources_with_methods = set()

        for method in methods:
            resource_path = method.resource_path
            if self._cli_config.should_skip(resource_path):
                continue
            # Check if this resource has SDK methods
            try:
                sdk_methods = self._introspector.introspect_resource(list(resource_path))
                if sdk_methods:
                    resources_with_methods.add(tuple(resource_path))
            except Exception:
                pass

        # Find immediate children of the parent
        for path in resources_with_methods:
            if len(path) > parent_len and path[:parent_len] == parent_tuple:
                sub_resources.add(path[parent_len])

        return sorted(sub_resources)

    def _resolve_override_placeholders(self, content: str) -> str:
        """Replace special placeholders in override content with computed values.

        Supported placeholders:
        - {{ENTITY_TYPES}}: comma-separated list of all known entity types, discovered
          by scanning services/, packages/, and plugins/ for __entity_type__ assignments.
        """
        return _resolve_placeholders(content, get_project_dir())

    def _filter_override_skip_lines(self, content: str) -> str:
        """Filter out lines marked with '# override-skip' comment.

        This allows override files to include placeholder definitions (like `app: Any`)
        that satisfy linters but should not be included in the generated output.
        """
        lines = content.split("\n")
        filtered = [line for line in lines if "# override-skip" not in line]
        return "\n".join(filtered)

    def _extract_imports_from_template_output(self, template_output: str) -> tuple[list[str], str]:
        """Extract import statements from rendered template output.

        Handles both single-line and multiline imports (with parentheses).

        Returns:
            Tuple of (imports, code_without_imports)
        """
        lines = template_output.split("\n")
        import_lines = []
        code_lines = []
        in_imports = True
        in_multiline_import = False

        for line in lines:
            stripped = line.strip()
            if in_imports:
                if in_multiline_import:
                    # Continue collecting multiline import
                    import_lines.append(line)
                    if ")" in stripped:
                        in_multiline_import = False
                elif stripped.startswith(("from ", "import ")):
                    import_lines.append(line)
                    # Check if this starts a multiline import
                    if "(" in stripped and ")" not in stripped:
                        in_multiline_import = True
                elif stripped == "" or stripped.startswith("#"):
                    # Blank line or comment (e.g. copyright header) in import section, skip it
                    continue
                else:
                    # First non-import, non-blank line - imports section is done
                    in_imports = False
                    code_lines.append(line)
            else:
                code_lines.append(line)

        filtered_imports = []
        seen_imports = set()
        for line in import_lines:
            stripped = line.strip()
            if stripped == "from __future__ import annotations":
                continue
            if line in seen_imports:
                continue
            seen_imports.add(line)
            filtered_imports.append(line)

        return filtered_imports, "\n".join(code_lines)

    def _extract_type_imports_from_context(self, context: dict[str, Any]) -> set[str]:
        """Extract imports needed from parameter types in context.

        Args:
            context: Template context containing sdk_method reference

        Returns:
            Set of import lines needed for parameter types
        """
        imports = set()

        # We need the actual SDK method to get type annotations
        # The context doesn't have it directly, but we can get it from the resource_path and method_name
        resource_path = context.get("resource_path", [])
        method_name = context.get("method_name", "")

        if resource_path and method_name:
            try:
                sdk_methods = self._introspector.introspect_resource(resource_path)
                if method_name in sdk_methods:
                    sdk_method = sdk_methods[method_name]
                    # Extract imports from all parameter types
                    for param in sdk_method.parameters:
                        imports.update(extract_imports_from_type(param.type_annotation))
            except Exception:
                # If introspection fails, skip type imports
                pass

        return imports

    def _get_target_file(self, resource_path: list[str]) -> Path:
        """Get target file path for a resource."""
        if len(resource_path) == 1:
            return self._target_dir / f"{resource_path[0]}.py"
        else:
            subdir = self._target_dir / "/".join(resource_path[:-1])
            return subdir / f"{resource_path[-1]}.py"

    def _generate_intermediate_init_files(self, resource_path: list[str]) -> None:
        """Generate __init__.py files for all intermediate directories."""
        for i in range(len(resource_path) - 1):
            parent_path = resource_path[: i + 1]
            child_name = resource_path[i + 1]
            self._ensure_init_file(parent_path, child_name)

    def _ensure_init_file(self, parent_path: list[str], child_name: str) -> None:
        """Ensure an __init__.py exists and includes the child sub-app."""
        if len(parent_path) == 0:
            return

        # Skip if this parent already has its own commands
        parent_tuple = tuple(parent_path)
        if parent_tuple in self._resources_with_children:
            # Check if this parent has its own methods
            try:
                sdk_methods = self._introspector.introspect_resource(list(parent_path))
                if sdk_methods:
                    # This parent already has an __init__.py with commands
                    return
            except Exception:
                pass

        init_file = self._target_dir / "/".join(parent_path) / "__init__.py"
        init_file.parent.mkdir(parents=True, exist_ok=True)

        import_base = "nemo_helix_ext.cli.commands.api"
        parent_import = f"{import_base}.{'.'.join(parent_path)}"

        # Get all children for this parent
        all_children = self._get_sub_resources(parent_path)
        if child_name not in all_children:
            all_children.append(child_name)
        all_children = sorted(set(all_children))

        resource_name = parent_path[-1]
        lines = [*AUTO_GENERATED_FILE_HEADER]

        app_help = self._get_resource_help(parent_path, f"{resource_name.replace('_', ' ').title()} operations")
        lines.extend(
            [
                "from importlib import import_module as _importlib_import_module",
                "",
                "from nemo_helix_ext.cli.core.help_formatter import create_typer_app",
                "",
            ]
        )

        for child in all_children:
            import_alias = _SUB_RESOURCE_IMPORT_ALIAS_TEMPLATE.format(resource_name=child)
            lines.append(f'{import_alias} = _importlib_import_module("{parent_import}.{child}")')

        lines.extend(
            [
                "",
                f'app = create_typer_app(name="{resource_name}", help="{escape_for_python_string(app_help)}")',
                "",
            ]
        )

        for child in all_children:
            cli_name = child.replace("_", "-")
            import_alias = _SUB_RESOURCE_IMPORT_ALIAS_TEMPLATE.format(resource_name=child)
            lines.append(f'app.add_typer({import_alias}.app, name="{cli_name}")')

        lines.append("")
        init_file.write_text("\n".join(lines))

    def _clear_generated_files(self) -> None:
        """Clear all generated files in the target directory."""
        if not self._target_dir.exists():
            return

        print(f"Clearing generated files in {self._target_dir}...")

        shutil.rmtree(self._target_dir)
        self._target_dir.mkdir(parents=True, exist_ok=True)

    def _get_resource_help(self, resource_path: list[str], default: str) -> str:
        """Return configured help text for a generated CLI resource."""
        resource_config = self._cli_config.get_resource_config(resource_path)
        if resource_config is None:
            return default

        help_text = resource_config.get("help")
        if isinstance(help_text, str):
            return help_text
        return default

    def generate_api_init_file(self) -> None:
        """Generate the api/__init__.py file with top-level API entries."""
        modules = self._get_api_modules()

        if not modules:
            print("No API modules found to register")
            return

        template = self._jinja_env.get_template("api_init.py.j2")
        content = template.render(modules=modules)

        init_file = self._target_dir / "__init__.py"
        init_file.write_text(content)

        print(f"✓ Generated api/__init__.py with {len(modules)} modules:")
        for module in modules:
            print(f"  - {module['name']} -> {module['cli_name']}")

    def _get_api_modules(self) -> list[dict[str, object]]:
        """Get all top-level API modules."""
        if not self._target_dir.exists():
            return []

        modules = []
        seen = set()

        for item in self._target_dir.iterdir():
            if item.name.startswith("_"):
                continue

            if item.is_dir():
                init_file = item / "__init__.py"
                if init_file.exists():
                    module_name = item.name
                    if module_name not in seen:
                        seen.add(module_name)
                        modules.append(self._top_level_module_metadata(module_name))
            elif item.is_file() and item.suffix == ".py":
                module_name = item.stem
                if module_name not in seen:
                    seen.add(module_name)
                    modules.append(self._top_level_module_metadata(module_name))

        modules.sort(key=lambda m: m["name"])
        return modules

    def _top_level_module_metadata(self, module_name: str) -> dict[str, object]:
        cli_name = to_kebab(module_name)
        top_level_config = self._cli_config.get_top_level_command_config(cli_name)
        return {
            "name": module_name,
            "cli_name": cli_name,
            "help": top_level_config.get("help") or self._get_top_level_module_help(module_name),
            "panel": top_level_config.get("panel") or "Core plugins",
            "hidden": bool(top_level_config.get("hidden", False)),
        }

    def _get_top_level_module_help(self, module_name: str) -> str:
        """Import the generated API module and read its top-level help text."""
        package_root = self._target_dir.parents[3]
        module_path = f"nemo_helix_ext.cli.commands.api.{module_name}"
        generated_package_names = (
            "nemo_helix_ext",
            "nemo_helix_ext.cli",
            "nemo_helix_ext.cli.commands",
            "nemo_helix_ext.cli.commands.api",
        )
        saved_modules = {name: sys.modules.get(name) for name in generated_package_names}

        importlib.invalidate_caches()
        sys.path.insert(0, str(package_root))
        try:
            for name in (*generated_package_names, module_path):
                sys.modules.pop(name, None)
            module = importlib.import_module(module_path)
        finally:
            sys.path.pop(0)
            for name in list(sys.modules):
                if name == module_path or name.startswith(f"{module_path}."):
                    sys.modules.pop(name, None)
            for name in generated_package_names:
                sys.modules.pop(name, None)
                saved_module = saved_modules[name]
                if saved_module is not None:
                    sys.modules[name] = saved_module

        help_text = getattr(module.app, "info", None)
        if help_text is not None:
            help_text = module.app.info.help
        else:
            help_text = getattr(module.app, "help", None)

        if isinstance(help_text, str):
            return help_text
        raise ValueError(f"Could not determine help text for API module {module_name!r}")


def generate_all(
    stainless_config_path: Path,
    cli_config_path: Path,
) -> None:
    """Generate all CLI commands.

    This clears existing generated files and regenerates everything from scratch.

    Args:
        stainless_config_path: Path to the Stainless config file.
        cli_config_path: Path to the CLI config file.
    """
    generator = SimpleGenerator(stainless_config_path, cli_config_path)
    generator.generate_all()
