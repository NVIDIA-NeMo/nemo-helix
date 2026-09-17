# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from datetime import datetime
from fnmatch import fnmatchcase
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, cast

import libcst as cst
import rich
import tomlkit
import tomlkit.items
import typer
from nemo_platform_sdk_tools.sdk.core.common import get_project_dir
from nemo_platform_sdk_tools.sdk.vendor.dependency_utils import merge_dependencies
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import BaseModel

# Logger for verbose output - writes to file only
logger = logging.getLogger(__name__)
logger.propagate = False  # Don't send logs to root logger (prevents console output)
# Marker the vendor flow writes above whole generated tables (e.g.
# `[project.scripts]`).
GENERATED_BUNDLE_TABLE_COMMENT = "# Generated from [tool.bundle-package]; do not edit this table by hand."
# Marker the vendor flow writes above each generated key in
# `[project.optional-dependencies]`. It's load-bearing: the rebuild reads it
# back to decide which existing extras are vendor-owned (refresh or delete)
# vs. hand-written (preserve untouched).
GENERATED_BUNDLE_GROUP_COMMENT = "# Generated from [tool.bundle-package]; do not edit by hand."
GENERATED_PROJECT_COMMENTS = {GENERATED_BUNDLE_GROUP_COMMENT, GENERATED_BUNDLE_TABLE_COMMENT}
VALID_BUNDLE_INHERIT_VALUES = {"entry-points", "optional-dependencies", "scripts"}
GENERATED_FILE_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""

GENERATED_ALIAS_INIT_TEMPLATE = (
    GENERATED_FILE_HEADER
    + """
from nemo_platform._alias import alias_package as _alias_package

_alias_package("{source_module}", globals())
"""
)


def _setup_logging() -> None:
    """Setup logging to write verbose output to a log file."""
    logs_dir = NMP_ROOT_PATH / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"vendor_{timestamp}.log"

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create file handler - only writes to file, not console
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    rich.print(f"📝 Detailed logs: {log_file}")


class ResourceReplacement(BaseModel):
    """Configuration for replacing resource class imports in _client.py."""

    class_names: list[str]  # The class names to replace (e.g., ["FilesetsResource", "AsyncFilesetsResource"])
    original: list[str]  # The original import paths (e.g., [".resources.filesets", ".resources.filesets.filesets"])
    replacement: str  # The replacement import path (e.g., ".filesets")


_CLIENT_CLASS_NAMES = ("NeMoPlatform", "AsyncNeMoPlatform")
_CLIENT_METHOD_NAMES = ("__init__", "__getattr__", "copy", "jobs", "secrets")
_CLIENT_HELPER_FUNCTION_NAMES = ("_should_bootstrap_config", "_copy_requires_bootstrap")
_CLIENT_INIT_REQUIRED_IMPORTS: dict[str, tuple[str, ...]] = {
    "nemo_platform._base_client": ("DefaultAsyncHttpxClient", "DefaultHttpxClient"),
    "nemo_platform_plugin.client.constants": ("WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR",),
    "nemo_platform_plugin.client.tls": ("client_verify_from_env",),
    "nemo_platform_plugin.jobs.client": ("AsyncJobsClient", "JobsClient"),
    "nemo_platform_plugin.secrets.compat": ("AsyncSecretsResource", "SecretsResource"),
    "pathlib": ("Path",),
}
_STALE_CLIENT_INIT_IMPORTS: dict[str, tuple[str, ...]] = {
    "nemo_platform.client.tls": ("client_verify_from_env",),
    "nemo_platform_ext.client.tls": ("client_verify_from_env",),
}


class _ClientMethodCollector(cst.CSTVisitor):
    """Collect methods from NeMo client classes in a source module."""

    def __init__(self, method_names: tuple[str, ...] = _CLIENT_METHOD_NAMES) -> None:
        self._method_names = method_names
        self.class_methods: dict[str, dict[str, cst.FunctionDef]] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        class_name = node.name.value
        if class_name not in _CLIENT_CLASS_NAMES:
            return

        for statement in node.body.body:
            if isinstance(statement, cst.FunctionDef) and statement.name.value in self._method_names:
                self.class_methods.setdefault(class_name, {})[statement.name.value] = statement


class _ClientMethodReplacer(cst.CSTTransformer):
    """Replace methods in SDK _client.py for the NeMo client classes."""

    def __init__(self, methods: dict[str, dict[str, cst.FunctionDef]]) -> None:
        self._methods = methods
        self.replaced_classes: set[str] = set()

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        class_name = original_node.name.value
        if class_name not in _CLIENT_CLASS_NAMES:
            return updated_node

        replacements = self._methods.get(class_name, {})
        if not replacements:
            return updated_node

        replaced_methods: set[str] = set()
        body_items = []
        for statement in updated_node.body.body:
            if isinstance(statement, cst.FunctionDef) and statement.name.value in replacements:
                body_items.append(replacements[statement.name.value])
                replaced_methods.add(statement.name.value)
            else:
                body_items.append(statement)

        for method_name, method_node in replacements.items():
            if method_name not in replaced_methods:
                body_items.append(method_node)

        self.replaced_classes.add(class_name)
        return updated_node.with_changes(body=updated_node.body.with_changes(body=body_items))


def _collect_client_helper_functions(source_module: cst.Module) -> dict[str, cst.FunctionDef]:
    helpers: dict[str, cst.FunctionDef] = {}
    for statement in source_module.body:
        if isinstance(statement, cst.FunctionDef) and statement.name.value in _CLIENT_HELPER_FUNCTION_NAMES:
            helpers[statement.name.value] = statement
    return helpers


def _find_client_helper_insertion_index(module: cst.Module) -> int:
    for index, statement in enumerate(module.body):
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for body_statement in statement.body:
            if not isinstance(body_statement, cst.Assign):
                continue
            for target in body_statement.targets:
                if isinstance(target.target, cst.Name) and target.target.value == "__all__":
                    return index + 1

    return _find_import_insertion_index(module)


def _with_client_helper_functions(
    target_module: cst.Module,
    helper_functions: dict[str, cst.FunctionDef],
) -> cst.Module:
    if not helper_functions:
        return target_module

    helper_names = set(helper_functions)
    body_without_stale_helpers = [
        statement
        for statement in target_module.body
        if not (isinstance(statement, cst.FunctionDef) and statement.name.value in helper_names)
    ]
    target_module = target_module.with_changes(body=body_without_stale_helpers)

    helpers = [helper_functions[name] for name in _CLIENT_HELPER_FUNCTION_NAMES if name in helper_functions]
    insert_at = _find_client_helper_insertion_index(target_module)
    body = list(target_module.body)
    return target_module.with_changes(body=body[:insert_at] + helpers + body[insert_at:])


def _module_name_from_expr(module_expr: cst.BaseExpression | None) -> str | None:
    if module_expr is None:
        return None

    if isinstance(module_expr, cst.Name):
        return module_expr.value

    parts: list[str] = []
    node: cst.BaseExpression = module_expr
    while isinstance(node, cst.Attribute):
        parts.insert(0, node.attr.value)
        node = node.value
    if isinstance(node, cst.Name):
        parts.insert(0, node.value)
        return ".".join(parts)

    return None


def _imported_name(alias: cst.ImportAlias) -> str | None:
    if isinstance(alias.name, cst.Name):
        return alias.name.value

    parts: list[str] = []
    node: cst.BaseExpression = alias.name
    while isinstance(node, cst.Attribute):
        parts.insert(0, node.attr.value)
        node = node.value
    if isinstance(node, cst.Name):
        parts.insert(0, node.value)
        return ".".join(parts)

    return None


def _collect_top_level_imports(module: cst.Module) -> tuple[dict[str, set[str]], set[str], set[str]]:
    from_imports: dict[str, set[str]] = {}
    import_modules: set[str] = set()
    bound_names: set[str] = set()

    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue

        for body_stmt in statement.body:
            if isinstance(body_stmt, cst.ImportFrom):
                module_name = _module_name_from_expr(body_stmt.module)
                if (
                    module_name is None
                    or module_name == "__future__"
                    or body_stmt.relative
                    or isinstance(body_stmt.names, cst.ImportStar)
                ):
                    continue

                imported_names = from_imports.setdefault(module_name, set())
                for alias in body_stmt.names:
                    name = _imported_name(alias)
                    if name is not None:
                        imported_names.add(name)
                        if alias.asname is not None:
                            bound_names.add(alias.asname.name.value)
                        else:
                            bound_names.add(name.split(".")[-1])
            elif isinstance(body_stmt, cst.Import):
                for alias in body_stmt.names:
                    module_name = _imported_name(alias)
                    if module_name is not None:
                        import_modules.add(module_name)
                        if alias.asname is not None:
                            bound_names.add(alias.asname.name.value)
                        else:
                            bound_names.add(module_name.split(".")[0])

    return from_imports, import_modules, bound_names


def _find_import_insertion_index(module: cst.Module) -> int:
    last_import_idx = -1
    for index, statement in enumerate(module.body):
        if isinstance(statement, cst.SimpleStatementLine) and all(
            isinstance(s, (cst.Import, cst.ImportFrom)) for s in statement.body
        ):
            last_import_idx = index
            continue
        if last_import_idx >= 0:
            break

    return last_import_idx + 1 if last_import_idx >= 0 else 0


def _without_stale_client_init_imports(module: cst.Module) -> cst.Module:
    body: list[cst.BaseStatement] = []
    for statement in module.body:
        if isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1:
            body_statement = statement.body[0]
            if isinstance(body_statement, cst.ImportFrom):
                module_name = _module_name_from_expr(body_statement.module)
                stale_names = _STALE_CLIENT_INIT_IMPORTS.get(module_name or "")
                if stale_names is not None and not isinstance(body_statement.names, cst.ImportStar):
                    imported_names = {
                        imported_name
                        for alias in body_statement.names
                        if (imported_name := _imported_name(alias)) is not None
                    }
                    if imported_names <= set(stale_names):
                        continue
        body.append(statement)
    return module.with_changes(body=body)


def _ensure_required_client_init_imports(target_module: cst.Module) -> cst.Module:
    """Ensure statically required imports exist in SDK _client.py."""
    target_module = _without_stale_client_init_imports(target_module)
    target_from_imports, _, _ = _collect_top_level_imports(target_module)
    missing_statements: list[cst.BaseStatement] = []

    for module_name in sorted(_CLIENT_INIT_REQUIRED_IMPORTS):
        required_names = _CLIENT_INIT_REQUIRED_IMPORTS[module_name]
        existing_names = target_from_imports.get(module_name, set())
        missing_names = [name for name in required_names if name not in existing_names]
        if missing_names:
            missing_statements.append(cst.parse_statement(f"from {module_name} import {', '.join(missing_names)}"))

    if not missing_statements:
        return target_module

    insert_at = _find_import_insertion_index(target_module)
    new_body = list(target_module.body[:insert_at]) + missing_statements + list(target_module.body[insert_at:])
    return target_module.with_changes(body=new_body)


NMP_ROOT_PATH = get_project_dir()

# Paths whose pyproject.toml receives dependency/extra metadata during vendoring.
SDK_PATH = NMP_ROOT_PATH / "sdk/python/nemo-platform"
WRAPPER_PATH = NMP_ROOT_PATH / "packages/nemo_platform"

# These module names are already used by the Stainless-generated SDK.
FORBIDDEN_TARGET_MODULES = ["types", "resources", "lib"]


app = typer.Typer(name="vendor", no_args_is_help=True, help="Vendor packages to the SDK")


def _pyproject_target_paths(sdk_path: Path) -> tuple[Path, ...]:
    return (sdk_path,)


def _load_package_config(package: str) -> dict:
    """Load the vendor-package config for a package from its pyproject.toml."""
    search_roots = [NMP_ROOT_PATH / "packages", NMP_ROOT_PATH / "services"]
    matching_configs: list[dict] = []
    matching_paths: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue

        for config_path in root.rglob("pyproject.toml"):
            with open(config_path, "rb") as f:
                config = tomlkit.load(f).get("tool", {}).get("vendor-package")

            if config and config.get("package") == package:
                matching_configs.append(config)
                matching_paths.append(config_path)

    if not matching_configs:
        raise ValueError(f"🛑 No `vendor-package` config found for package `{package}`.")

    if len(matching_configs) > 1:
        matches = ", ".join(str(path.relative_to(NMP_ROOT_PATH)) for path in matching_paths)
        raise ValueError(f"🛑 Multiple `vendor-package` configs found for `{package}`: {matches}")

    return matching_configs[0]


def _vendor_package_files(config: dict) -> None:
    """Copy source files and rewrite imports for one package.

    Writes only to the package's own target directory in the SDK source tree.
    Safe to call in parallel with other packages.
    """
    package = config["package"]
    source_module = config.get("source_module", package)
    target_sdk_module = config.get("target_sdk_module")
    included_paths = list(config.get("included_paths") or [])
    package_root = config.get("package_root")
    top_level = config.get("top_level", False)
    with_src = config.get("with_src", True)
    vendor_tests = config.get("vendor_tests", False)
    tests_path = config.get("tests_path", "tests")
    tests_included_paths = list(config.get("tests_included_paths") or [])
    included_transitive_dependencies = list(config.get("included_transitive_dependencies") or [])
    sdk_path = NMP_ROOT_PATH / "sdk/python/nemo-platform"

    package_root_path, package_path = _build_and_validate_package_path(
        package,
        package_root,
        source_module,
        with_src,
    )

    if config.get("sdk_include_mode") == "source-package":
        _write_source_package_aliases(
            package=package,
            package_path=package_path,
            source_module=source_module,
            target_sdk_module=target_sdk_module,
            top_level=top_level,
        )
        _remove_vendored_tests(sdk_path=sdk_path, package=package, target_sdk_module=target_sdk_module)
        return

    modules_to_vendor = []
    if target_sdk_module is None:
        for d in package_path.iterdir():
            if d.is_dir() and not d.name.startswith("__"):
                modules_to_vendor.append(
                    {"source_path": package_path / d, "source_module": f"{package}.{d.name}", "module_name": d.name}
                )
    else:
        modules_to_vendor = [
            {"source_path": package_path, "source_module": source_module, "module_name": target_sdk_module}
        ]

    module_names = [m["module_name"] for m in modules_to_vendor]
    rich.print(f"📦 Vendoring package `{package}` -> src/{', '.join(module_names)}")

    # Copy all modules to their target paths
    vendored_paths = []
    for module in modules_to_vendor:
        logger.info(f"Vendoring module `{module}`...")
        target_module_name = module["module_name"]
        source_module_path = package_path / module["source_path"]

        target_path = _build_and_validate_target_paths(sdk_path, target_module_name, top_level=top_level)
        vendored_paths.append(target_path)

        logger.info(f"Vendoring package `{source_module_path}` to `{sdk_path}` based on explicit includes.")
        if target_path.exists():
            # Remove existing target path to ensure a clean vendoring (i.e. if files were removed from source package)
            shutil.rmtree(target_path, ignore_errors=True)

        _copy_included_paths(source_module_path, target_path, included_paths)

        if included_transitive_dependencies:
            _vendor_transitive_dependencies(
                sdk_path=sdk_path,
                target_sdk_module=target_module_name,
                included_transitive_dependencies=included_transitive_dependencies,
                top_level=top_level,
            )

    # Build all module rewrites
    all_module_rewrites = []
    for module in modules_to_vendor:
        target_module = module["module_name"] if top_level else f"nemo_platform.{module['module_name']}"
        if module["source_module"] != target_module:
            all_module_rewrites.append((module["source_module"], target_module))

    # Rewrite imports in ALL vendored files using ALL module rewrites
    if all_module_rewrites:
        for target_path, module in zip(vendored_paths, modules_to_vendor, strict=True):
            logger.info(f"Rewriting imports in `{module['module_name']}` module")
            _rewrite_imports_in_vendored_source_multiple(
                target_path=target_path,
                module_rewrites=all_module_rewrites,
                included_transitive_dependencies=included_transitive_dependencies,
            )

    if vendor_tests:
        _vendor_tests_flat(
            source_root_path=package_root_path,
            source_tests_path=tests_path,
            sdk_path=sdk_path,
            tests_target_subdir=_vendored_tests_target_subdir(package=package, target_sdk_module=target_sdk_module),
            tests_included_paths=tests_included_paths,
            module_rewrites=all_module_rewrites,
        )


def _vendor_package_metadata(config: dict) -> None:
    """Apply pyproject.toml and _client.py mutations for one package.

    Writes to the shared pyproject.toml files and _client.py.
    Must be called sequentially across packages to avoid race conditions.
    """
    package = config["package"]
    source_module = config.get("source_module", package)
    package_root = config.get("package_root")
    with_src = config.get("with_src", True)
    sdk_optional_dependencies_name = config.get("sdk_optional_dependencies_name")
    excluded_dependencies = list(config.get("excluded_dependencies") or [])
    dependency_extra_replacements = dict(config.get("dependency_extra_replacements") or {})
    replaces = [ResourceReplacement(**r) for r in (config.get("replaces") or [])]
    replace_client_inits_from = config.get("replace_client_inits_from")
    entrypoints = list(config.get("entrypoints") or [])
    scripts = list(config.get("scripts") or [])

    package_root_path, package_path = _build_and_validate_package_path(package, package_root, source_module, with_src)
    sdk_path = NMP_ROOT_PATH / "sdk/python/nemo-platform"

    # Vendored target pyprojects should never depend on either the wrapper or SDK
    # distribution names; they bundle the code directly.
    for package_name in ("nemo-platform", "nemo-platform-sdk"):
        if package_name not in excluded_dependencies:
            excluded_dependencies.append(package_name)
    if sdk_optional_dependencies_name is not None:
        _update_dependencies_of_sdk_pyproject(
            sdk_path,
            package_root_path,
            excluded_dependencies,
            sdk_optional_dependencies_name,
            dependency_extra_replacements=dependency_extra_replacements,
        )
    else:
        _update_dependencies_of_sdk_pyproject(
            sdk_path,
            package_root_path,
            excluded_dependencies,
            dependency_extra_replacements=dependency_extra_replacements,
        )

    if replaces:
        _apply_resource_replacements(sdk_path, replaces)

    if replace_client_inits_from:
        _replace_client_methods(
            sdk_path=sdk_path,
            source_path=package_path / Path(replace_client_inits_from),
            remove_vendored_source_relative_path=replace_client_inits_from,
        )

    if entrypoints:
        _vendor_entrypoints(sdk_path=sdk_path, entrypoints=entrypoints)

    if scripts:
        _vendor_scripts(sdk_path=sdk_path, scripts=scripts)

    rich.print(f"✅ Vendoring complete for `{package}`")


def _write_source_package_aliases(
    *,
    package: str,
    package_path: Path,
    source_module: str,
    target_sdk_module: str | None,
    top_level: bool,
) -> None:
    """Write tiny SDK alias packages for source packages included at build time."""
    sdk_path = NMP_ROOT_PATH / "sdk/python/nemo-platform"
    modules_to_alias: list[tuple[str, str]] = []

    if target_sdk_module is None:
        for path in package_path.iterdir():
            if path.is_dir() and not path.name.startswith("__"):
                modules_to_alias.append((path.name, f"{source_module}.{path.name}"))
    else:
        modules_to_alias.append((target_sdk_module, source_module))

    module_names = ", ".join(module for module, _ in modules_to_alias)
    rich.print(f"🔗 Linking source package `{package}` through SDK aliases -> src/{module_names}")

    for target_module_name, source_import in modules_to_alias:
        target_path = _build_and_validate_target_paths(sdk_path, target_module_name, top_level=top_level)
        if target_path.exists():
            shutil.rmtree(target_path, ignore_errors=True)
        target_path.mkdir(parents=True, exist_ok=True)
        _ensure_parent_alias_init_files(sdk_path, target_path, top_level=top_level)
        (target_path / "__init__.py").write_text(
            GENERATED_ALIAS_INIT_TEMPLATE.format(source_module=source_import),
            encoding="utf-8",
        )


def _ensure_parent_alias_init_files(sdk_path: Path, target_path: Path, *, top_level: bool) -> None:
    if top_level:
        return

    root = sdk_path / "src/nemo_platform"
    parent = target_path.parent
    while parent != root and root in parent.parents:
        init_file = parent / "__init__.py"
        if not init_file.exists():
            init_file.write_text(GENERATED_FILE_HEADER, encoding="utf-8")
        parent = parent.parent


def _vendored_tests_target_subdir(*, package: str, target_sdk_module: str | None) -> str:
    return target_sdk_module.replace(".", "/") if target_sdk_module else package


def _remove_vendored_tests(*, sdk_path: Path, package: str, target_sdk_module: str | None) -> None:
    """Remove obsolete SDK test copies when a package switches to build-time staging.

    Source-package mode no longer rewrites package tests into the generated SDK
    tree. Removing the stale target prevents SDK test collection from loading an
    old copied snapshot after ``make vendor``.
    """
    tests_target_path = (
        sdk_path
        / "tests"
        / "vendored"
        / _vendored_tests_target_subdir(package=package, target_sdk_module=target_sdk_module)
    )
    if tests_target_path.exists():
        shutil.rmtree(tests_target_path, ignore_errors=True)
        rich.print(f"🧹 Removed stale vendored tests `{tests_target_path.relative_to(sdk_path)}`")


@app.command("from-config")
def vendor_package_from_config(package: str) -> None:
    """Vendor package to target SDK module based on the `vendor-package` config in the package's `pyproject.toml`."""
    config = _load_package_config(package)
    _setup_logging()
    _vendor_package_files(config)
    _vendor_package_metadata(config)


@app.command("all-from-configs")
def vendor_all_packages_from_configs(packages: list[str]) -> None:
    """Vendor all packages: file ops run in parallel, pyproject/client mutations run sequentially.

    File copy and import rewriting are safe to parallelize because each package writes
    to its own isolated target directory. The pyproject.toml and _client.py mutations
    share files across packages and must remain sequential.
    """
    import concurrent.futures

    _setup_logging()
    configs = [_load_package_config(p) for p in packages]

    # Phase 1: file copy + import rewrite — safe to parallelize (no shared writes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(packages)) as executor:
        futures = {executor.submit(_vendor_package_files, config): config["package"] for config in configs}
        for future in concurrent.futures.as_completed(futures):
            pkg = futures[future]
            try:
                future.result()
            except Exception as e:
                rich.print(f"❌ Error vendoring files for `{pkg}`: {e}")
                raise

    # Clear auto-generated fields in the wrapper before re-populating them,
    # so stale entries from previous runs don't persist through merge logic.
    # Only the wrapper is reset — the SDK pyproject is authored by Stainless
    # and post_generation_update; vendor configs add to it but don't own it.
    _reset_generated_pyproject_fields()

    # Phase 2: pyproject.toml + _client.py mutations — must be sequential
    for config in configs:
        _vendor_package_metadata(config)

    # Phase 3: create core-service extra that aggregates all service -service extras
    _create_core_local_extra(configs)

    # Phase 4: process [tool.bundle-package] configs across all workspace packages.
    # This reads each bundled package's deps and writes them into the deps_group
    # specified by the parent's bundle config. It also copies scripts,
    # entry-points, and optional-dependencies from bundled packages.
    _process_bundle_packages()

    # Phase 5: sort auto-generated fields in the wrapper for deterministic output.
    # Without this, ordering depends on package processing order in the Makefile.
    _sort_wrapper_pyproject_fields()

    # Phase 6: rewrite optional-dependencies (manual aliases + generated extras)
    # and annotate generated scripts/entry-point tables.
    _annotate_generated_bundle_groups()

    _normalize_static_force_include_spacing(WRAPPER_PATH / "pyproject.toml")


def _find_workspace_package_dir(package_name: str) -> Path | None:
    """Find a workspace package's directory by reading its pyproject.toml name.

    Scans all workspace members declared in the root pyproject.toml.
    """
    root_pyproject_path = NMP_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return None

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    for member in members:
        member_dir = NMP_ROOT_PATH / member
        member_pyproject = member_dir / "pyproject.toml"
        if not member_pyproject.exists():
            continue
        config = tomlkit.loads(member_pyproject.read_text(encoding="utf-8"))
        name = config.get("project", {}).get("name")
        if name and canonicalize_name(name) == canonicalize_name(package_name):
            return member_dir

    return None


def _find_package_dir(package_name: str, package_config: dict, parent_dir: Path) -> Path | None:
    """Find a bundled package's project directory.

    Prefer workspace membership, then fall back to walking up from the configured
    source path. The fallback supports local vendored packages that are not root
    workspace members, such as nemo-switchyard's vendored switchyard snapshot.
    """
    workspace_dir = _find_workspace_package_dir(package_name)
    if workspace_dir is not None:
        return workspace_dir

    source = package_config.get("source")
    if not source:
        return None

    source_path = (parent_dir / source).resolve()
    for path in (source_path, *source_path.parents):
        pyproject_path = path / "pyproject.toml"
        if not pyproject_path.exists():
            continue
        config = tomlkit.loads(pyproject_path.read_text(encoding="utf-8"))
        name = config.get("project", {}).get("name")
        if name and canonicalize_name(name) == canonicalize_name(package_name):
            return path

    return None


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def _merge_matching_project_entrypoints(target_project: dict, source_project: dict, patterns: list[str]) -> None:
    source_entrypoints = source_project.get("entry-points", {})
    if not source_entrypoints or not patterns:
        return

    target_entrypoints = target_project.setdefault("entry-points", {})
    for group, entries in source_entrypoints.items():
        if not _matches_any_pattern(group, patterns):
            continue
        target_group = target_entrypoints.setdefault(group, {})
        for name, value in entries.items():
            target_group[name] = value


def _merge_matching_project_scripts(target_project: dict, source_project: dict, patterns: list[str]) -> None:
    source_scripts = source_project.get("scripts", {})
    if not source_scripts or not patterns:
        return

    target_scripts = target_project.setdefault("scripts", tomlkit.table())
    for name, value in source_scripts.items():
        if _matches_any_pattern(name, patterns):
            target_scripts[name] = value


def _should_copy_optional_dependency_extra(extra_name: str, target_project_name: str | None) -> bool:
    canonical_extra = canonicalize_name(extra_name)
    if canonical_extra in {"dev", "test", "tests"}:
        return False
    if target_project_name and canonical_extra == canonicalize_name(target_project_name):
        return False
    return True


def _generated_optional_dependency_groups(optional: tomlkit.items.Table) -> set[str]:
    """Return optional-dependency groups owned by the bundle generator."""
    generated: set[str] = set()
    saw_marker = False
    for key, item in optional._value.body:
        if key is None:
            if isinstance(item, tomlkit.items.Comment) and item.as_string().strip() == GENERATED_BUNDLE_GROUP_COMMENT:
                saw_marker = True
            continue
        if saw_marker:
            generated.add(key.key)
        saw_marker = False
    return generated


def _merge_matching_project_optional_dependencies(
    target_project: dict, source_project: dict, patterns: list[str]
) -> None:
    source_optional = source_project.get("optional-dependencies", {})
    if not source_optional or not patterns:
        return

    target_project_name = target_project.get("name")
    target_optional = target_project.setdefault("optional-dependencies", tomlkit.table())
    for extra_name, deps in source_optional.items():
        if not _matches_any_pattern(extra_name, patterns):
            continue
        if not _should_copy_optional_dependency_extra(extra_name, target_project_name):
            continue
        if extra_name not in target_optional:
            target_optional[extra_name] = _build_dependency_array(list(deps))


def _bundle_deps_group(pkg_name: str, pkg_config: dict) -> str:
    return pkg_config.get("deps_group") or pkg_name


def _include_in_services_extra(pkg_config: dict) -> bool:
    return pkg_config.get("include_in_services", True) is not False


def _bundle_inherit_patterns(pkg_config: dict, key: str) -> list[str]:
    inherit = pkg_config.get("inherit", {})
    if not inherit:
        return []
    if not isinstance(inherit, dict):
        raise ValueError("[tool.bundle-package] inherit must be a table")

    unknown = set(inherit.keys()) - VALID_BUNDLE_INHERIT_VALUES
    if unknown:
        values = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown [tool.bundle-package] inherit value(s): {values}")

    value = inherit.get(key, False)
    if value is True:
        return ["*"]
    if value in (False, None):
        return []
    if not isinstance(value, list):
        raise ValueError(f"[tool.bundle-package] inherit.{key} must be a boolean or list of wildcard patterns")
    if not all(isinstance(pattern, str) for pattern in value):
        raise ValueError(f"[tool.bundle-package] inherit.{key} must contain only string wildcard patterns")
    return list(value)


def _is_bundled_plugin_entry(pkg_config: dict) -> bool:
    source = pkg_config.get("source", "")
    return "plugins/" in source and "/src/" in source


def _load_bundle_project(package_name: str, package_config: dict, parent_dir: Path) -> dict:
    pkg_dir = _find_package_dir(package_name, package_config, parent_dir)
    if pkg_dir is None:
        return {}

    pkg_pyproject_path = pkg_dir / "pyproject.toml"
    if not pkg_pyproject_path.exists():
        return {}

    return tomlkit.loads(pkg_pyproject_path.read_text(encoding="utf-8")).get("project", {})


def _process_bundle_packages() -> None:
    """Process [tool.bundle-package] configs across all workspace packages.

    For each package that declares [tool.bundle-package], reads the bundled
    packages' dependencies and writes them into the configured extra group.
    By default, the extra group name is the bundle key. Also copies the
    metadata explicitly listed in the bundle entry's ``inherit`` field, plus any
    scripts declared directly in the bundle config.

    Workspace package dependencies are filtered out (they're not on PyPI).
    If a filtered dep has its own bundle entry, the dependency name is kept in
    source metadata so build hooks can rewrite final wheel metadata if needed.
    """
    root_pyproject_path = NMP_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    # Build a set of all canonicalized workspace package names — these are never
    # installable from PyPI, so they must be excluded from bundled dependency lists.
    # Names are canonicalized (PEP 503) so that nemo-nb, nemo_nb, etc. all match.
    workspace_package_names: set[str] = set()
    for member in members:
        member_dir = NMP_ROOT_PATH / member
        member_pyproject_path = member_dir / "pyproject.toml"
        if not member_pyproject_path.exists():
            continue
        config = tomlkit.loads(member_pyproject_path.read_text(encoding="utf-8"))
        name = config.get("project", {}).get("name")
        if name:
            workspace_package_names.add(canonicalize_name(name))

    for member in members:
        member_dir = NMP_ROOT_PATH / member
        member_pyproject_path = member_dir / "pyproject.toml"
        if not member_pyproject_path.exists():
            continue

        member_config = tomlkit.loads(member_pyproject_path.read_text(encoding="utf-8"))
        bundle_config = member_config.get("tool", {}).get("bundle-package")
        if not bundle_config:
            continue

        parent_name = member_config.get("project", {}).get("name", member)

        rich.print(f"📦 Processing [tool.bundle-package] for `{parent_name}` ({len(bundle_config)} packages)")

        # Build a canonicalized-key version of bundle_config for dep lookups
        canonical_bundle_config = {canonicalize_name(k): v for k, v in bundle_config.items()}

        pyproject = tomlkit.loads(member_pyproject_path.read_text(encoding="utf-8"))
        optional = pyproject["project"].setdefault("optional-dependencies", tomlkit.table())
        generated_optional_groups = _generated_optional_dependency_groups(optional)
        reset_generated_optional_groups: set[str] = set()

        for pkg_name, pkg_config in bundle_config.items():
            deps_group = _bundle_deps_group(pkg_name, pkg_config)
            inherited_script_patterns = _bundle_inherit_patterns(pkg_config, "scripts")
            inherited_entrypoint_patterns = _bundle_inherit_patterns(pkg_config, "entry-points")
            inherited_optional_patterns = _bundle_inherit_patterns(pkg_config, "optional-dependencies")
            pkg_scripts = pkg_config.get("scripts", [])

            # Find the bundled package's pyproject.toml to read its metadata.
            pkg_project = _load_bundle_project(pkg_name, pkg_config, member_dir)
            if not pkg_project:
                rich.print(f"  ⚠️  Could not find bundled package `{pkg_name}`, skipping")
                continue

            _merge_matching_project_scripts(pyproject["project"], pkg_project, inherited_script_patterns)
            _merge_matching_project_entrypoints(pyproject["project"], pkg_project, inherited_entrypoint_patterns)
            _merge_matching_project_optional_dependencies(
                pyproject["project"], pkg_project, inherited_optional_patterns
            )

            pkg_deps = list(pkg_project.get("dependencies", []))

            # Filter out workspace packages. If a workspace dep has a
            # corresponding bundle entry, keep the dependency name readable in
            # source metadata. Wheel builds can rewrite final Requires-Dist
            # metadata to self-referencing extras when needed.
            # All name comparisons use PEP 503 canonicalization.
            canonical_parent = canonicalize_name(parent_name)
            filtered_deps = []
            for dep in pkg_deps:
                dep_name = canonicalize_name(Requirement(dep).name)
                if dep_name == canonical_parent:
                    logger.debug(f"  Filtering self-reference: {dep}")
                    continue
                bundled_dep_config = canonical_bundle_config.get(dep_name)
                if bundled_dep_config is not None:
                    if _bundle_deps_group(dep_name, bundled_dep_config) == deps_group:
                        logger.debug(f"  Filtering same-group bundled dep: {dep}")
                        continue
                    filtered_deps.append(dep)
                    logger.debug(f"  Keeping bundled dep: {dep}")
                    continue
                if dep_name in workspace_package_names:
                    logger.debug(f"  Filtering workspace dep: {dep}")
                    continue
                filtered_deps.append(dep)

            # Separate self-referencing extras from regular deps. Self-refs
            # must stay as individual entries — merge_dependencies would
            # combine them into a single parent[X,Y,Z] line.
            self_ref_prefix = f"{parent_name}["
            existing = list(optional.get(deps_group, []))
            if deps_group in generated_optional_groups and deps_group not in reset_generated_optional_groups:
                existing_regular = []
                existing_self_refs = [d for d in existing if d.startswith(self_ref_prefix)]
                reset_generated_optional_groups.add(deps_group)
                optional[deps_group] = _build_dependency_array(existing_self_refs)
            else:
                existing_regular = [d for d in existing if not d.startswith(self_ref_prefix)]
                existing_self_refs = [d for d in existing if d.startswith(self_ref_prefix)]

            if filtered_deps:
                regular_deps = [d for d in filtered_deps if not d.startswith(self_ref_prefix)]
                self_ref_deps = [d for d in filtered_deps if d.startswith(self_ref_prefix)]

                merged_regular = merge_dependencies(existing_regular, regular_deps)
                # Deduplicate self-refs while preserving order
                all_self_refs = list(dict.fromkeys(existing_self_refs + self_ref_deps))
                optional[deps_group] = _build_dependency_array(merged_regular + all_self_refs)

            rich.print(f"  ✅ `{deps_group}` — {len(filtered_deps)} deps")

            # Write scripts
            if pkg_scripts:
                scripts_table = pyproject["project"].setdefault("scripts", tomlkit.table())
                for script in pkg_scripts:
                    scripts_table[script["name"]] = script["value"]

        member_pyproject_path.write_text(tomlkit.dumps(pyproject), encoding="utf-8")


def _copy_table_without_comments(table: tomlkit.items.Table, comments: set[str]) -> tomlkit.items.Table:
    cleaned = tomlkit.table()
    for key, item in table._value.body:
        if isinstance(item, tomlkit.items.Comment) and item.as_string().strip() in comments:
            continue
        if key is None:
            cleaned.add(item)
        else:
            cleaned.add(key, item)
    while cleaned._value.body and isinstance(cleaned._value.body[0][1], tomlkit.items.Whitespace):
        cleaned._value.body.pop(0)
    while cleaned._value.body and isinstance(cleaned._value.body[-1][1], tomlkit.items.Whitespace):
        cleaned._value.body.pop()
    return cleaned


def _generated_table_comment() -> tomlkit.items.Comment:
    comment = tomlkit.comment(GENERATED_BUNDLE_TABLE_COMMENT.removeprefix("# "))
    comment.trivia.trail = ""
    return comment


def _add_generated_table_comment(table: tomlkit.items.Table) -> None:
    if table._value.body and isinstance(table._value.body[-1][1], tomlkit.items.Whitespace):
        table._value.body.pop()
    if table._value.body:
        table.add(tomlkit.nl())
    table.add(_generated_table_comment())


def _is_generated_project_comment(item: object) -> bool:
    return isinstance(item, tomlkit.items.Comment) and item.as_string().strip() in GENERATED_PROJECT_COMMENTS


def _remove_marked_child_tables(
    table: tomlkit.items.Table,
    child_names: set[str] | None = None,
) -> tuple[tomlkit.items.Table, bool]:
    cleaned = tomlkit.table()
    skip_next_child = False
    changed = False

    for key, item in table._value.body:
        if _is_generated_project_comment(item) and item.as_string().strip() == GENERATED_BUNDLE_TABLE_COMMENT:
            if cleaned._value.body and isinstance(cleaned._value.body[-1][1], tomlkit.items.Whitespace):
                cleaned._value.body.pop()
            skip_next_child = True
            changed = True
            continue

        if skip_next_child and key is None:
            if isinstance(item, tomlkit.items.Whitespace):
                changed = True
                continue
            cleaned.add(item)
            continue

        if skip_next_child and key is not None:
            skip_next_child = False
            if child_names is None or key.key in child_names:
                changed = True
                continue

        if key is None:
            cleaned.add(item)
        else:
            cleaned.add(key, item)

    return cleaned, changed


def _remove_marked_project_table_entries(table: tomlkit.items.Table) -> tuple[tomlkit.items.Table, bool]:
    """Remove generated project metadata from one scripts/entry-points table."""
    cleaned = tomlkit.table()
    skip_next_key = False
    changed = False

    for key, item in table._value.body:
        if _is_generated_project_comment(item):
            if cleaned._value.body and isinstance(cleaned._value.body[-1][1], tomlkit.items.Whitespace):
                cleaned._value.body.pop()
            skip_next_key = True
            changed = True
            continue

        if skip_next_key and key is not None:
            skip_next_key = False
            changed = True
            continue

        if key is None:
            cleaned.add(item)
        else:
            cleaned.add(key, item)

    return cleaned, changed


def _remove_marked_generated_project_tables(content: str) -> str:
    pyproject = tomlkit.loads(content)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return content

    changed = False
    if hasattr(project, "_value"):
        cleaned_project, project_tables_changed = _remove_marked_child_tables(project, {"scripts"})
        if project_tables_changed:
            pyproject["project"] = cleaned_project
            project = pyproject["project"]
            changed = True

    scripts = project.get("scripts")
    if scripts is not None:
        cleaned, scripts_changed = _remove_marked_project_table_entries(scripts)
        if not cleaned:
            del project["scripts"]
            changed = True
        elif scripts_changed:
            project["scripts"] = cleaned
            changed = True

    entrypoints = project.get("entry-points")
    if entrypoints is not None:
        if hasattr(entrypoints, "_value"):
            cleaned_entrypoints, entrypoint_tables_changed = _remove_marked_child_tables(entrypoints)
            if entrypoint_tables_changed:
                project["entry-points"] = cleaned_entrypoints
                entrypoints = project["entry-points"]
                changed = True
        for group in list(entrypoints):
            cleaned, group_changed = _remove_marked_project_table_entries(entrypoints[group])
            if not cleaned:
                del entrypoints[group]
                changed = True
            elif group_changed:
                entrypoints[group] = cleaned
                changed = True
        if not entrypoints:
            del project["entry-points"]
            changed = True

    if not changed:
        return content
    return tomlkit.dumps(pyproject)


def _refresh_bundle_owned_optional_dependencies(content: str, bundle_owned_names: set[str]) -> str:
    """Refresh vendor-owned extras in `[project.optional-dependencies]`.

    The ``# Generated from [tool.bundle-package]; do not edit by hand.``
    marker comment immediately above a key is the load-bearing signal that
    the key is vendor-owned:

    * Marker present, name in ``bundle_owned_names`` → keep, ensure marker.
    * Marker present, name not in ``bundle_owned_names`` → stale, drop.
    * No marker → hand-written, preserve untouched.

    New names in ``bundle_owned_names`` that aren't present yet must already
    have been added to the table by the caller (via
    ``_process_bundle_packages``); this function only handles classification
    and stale-cleanup.
    """
    pyproject = tomlkit.loads(content)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return content

    optional = project.get("optional-dependencies")
    if optional is None:
        return content

    # Find keys preceded by the generated marker. tomlkit attaches comments
    # as `(None, Comment)` body entries; the next `(Key, value)` entry is
    # the key the comment is meant to annotate.
    body = optional._value.body
    marker_for: dict[str, bool] = {}
    saw_marker = False
    for key, item in body:
        if key is None:
            if isinstance(item, tomlkit.items.Comment) and item.as_string().strip() == GENERATED_BUNDLE_GROUP_COMMENT:
                saw_marker = True
            continue
        marker_for[key.key] = saw_marker
        saw_marker = False

    # Hand-written extras keep their original order; vendor-owned extras
    # are emitted alphabetically after them so the generated section is
    # easy to scan and stable across runs (independent of which order
    # earlier phases happened to add new keys in).
    hand_written = [name for name in optional if name not in bundle_owned_names and not marker_for.get(name, False)]
    vendor_owned = sorted(name for name in optional if name in bundle_owned_names)

    rebuilt = tomlkit.table()
    is_first = True
    for name in hand_written:
        if not is_first:
            rebuilt.add(tomlkit.nl())
        rebuilt.add(name, optional[name])
        is_first = False
    for name in vendor_owned:
        if not is_first:
            rebuilt.add(tomlkit.nl())
        rebuilt.add(tomlkit.comment(GENERATED_BUNDLE_GROUP_COMMENT.removeprefix("# ")))
        rebuilt.add(name, optional[name])
        is_first = False

    project["optional-dependencies"] = rebuilt
    return tomlkit.dumps(pyproject)


def _annotate_generated_project_entries(
    content: str,
    script_names: set[str],
    entrypoint_names: dict[str, set[str]],
) -> str:
    if not script_names and not entrypoint_names:
        return content

    pyproject = tomlkit.loads(content)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return content

    scripts = project.get("scripts")
    if scripts is not None:
        if hasattr(project, "_value"):
            annotated_project = tomlkit.table()
            for key, item in project._value.body:
                if key is None:
                    annotated_project.add(item)
                    continue

                if key.key == "scripts":
                    annotated_scripts, whole_table = _annotate_generated_project_table(item, script_names)
                    if whole_table:
                        _add_generated_table_comment(annotated_project)
                    annotated_project.add(key, annotated_scripts)
                    continue

                annotated_project.add(key, item)
            pyproject["project"] = annotated_project
            project = pyproject["project"]
        else:
            project["scripts"], _ = _annotate_generated_project_table(scripts, script_names)

    entrypoints = project.get("entry-points")
    if entrypoints is not None:
        if hasattr(entrypoints, "_value"):
            annotated_entrypoints = tomlkit.table()
            for key, item in entrypoints._value.body:
                if key is None:
                    annotated_entrypoints.add(item)
                    continue

                group_names = entrypoint_names.get(key.key, set())
                annotated_group, whole_table = _annotate_generated_project_table(item, group_names)
                if whole_table:
                    _add_generated_table_comment(annotated_entrypoints)
                annotated_entrypoints.add(key, annotated_group)
            project["entry-points"] = annotated_entrypoints
        else:
            for group, names in entrypoint_names.items():
                if group in entrypoints:
                    entrypoints[group], _ = _annotate_generated_project_table(entrypoints[group], names)

    return tomlkit.dumps(pyproject)


def _annotate_generated_project_table(
    table: tomlkit.items.Table,
    generated_names: set[str],
) -> tuple[tomlkit.items.Table, bool]:
    """Annotate a scripts/entry-points table with the generator marker.

    * Empty ``generated_names`` → no annotation.
    * Every key in the table is bundle-generated → emit a single table-level
      header marker (the second tuple element ``True`` signals the parent
      caller to add the marker above the table header).
    * Mixed table (some generated, some hand-written) → leave the table
      unannotated. We don't add per-key markers; that mode was only used by
      legacy code and the rebuild flow now treats hand-written and generated
      entries equally inside mixed scripts/entry-points tables.

    In all cases, any pre-existing generator marker comments inside the
    table are stripped so we never carry stale markers across runs.
    """
    if not generated_names:
        return table, False

    cleaned = _copy_table_without_comments(table, GENERATED_PROJECT_COMMENTS)
    existing_names = {key.key for key, _ in cleaned._value.body if key is not None}
    is_wholly_generated = bool(existing_names) and existing_names <= generated_names
    return cleaned, is_wholly_generated


def _annotate_generated_bundle_groups() -> None:
    """Annotate metadata generated from [tool.bundle-package]."""
    root_pyproject_path = NMP_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    for member in members:
        member_pyproject_path = NMP_ROOT_PATH / member / "pyproject.toml"
        if not member_pyproject_path.exists():
            continue

        member_config = tomlkit.loads(member_pyproject_path.read_text(encoding="utf-8"))
        bundle_config = member_config.get("tool", {}).get("bundle-package")
        if not bundle_config:
            continue

        member_dir = member_pyproject_path.parent
        member_project_name = member_config.get("project", {}).get("name")
        script_names: set[str] = set()
        entrypoint_names: dict[str, set[str]] = {}
        bundle_owned_names: set[str] = set()
        if member_dir == WRAPPER_PATH:
            # Wrapper-only aggregate extras created by `_create_core_local_extra`.
            bundle_owned_names.update({"core-service", "plugins", "services"})
        for pkg_name, pkg_config in bundle_config.items():
            if not isinstance(pkg_config, dict):
                continue

            inherited_script_patterns = _bundle_inherit_patterns(pkg_config, "scripts")
            inherited_entrypoint_patterns = _bundle_inherit_patterns(pkg_config, "entry-points")
            inherited_optional_patterns = _bundle_inherit_patterns(pkg_config, "optional-dependencies")
            pkg_project = _load_bundle_project(pkg_name, pkg_config, member_dir)
            bundle_owned_names.add(_bundle_deps_group(pkg_name, pkg_config))
            if inherited_optional_patterns:
                bundle_owned_names.update(
                    extra_name
                    for extra_name in pkg_project.get("optional-dependencies", {})
                    if _matches_any_pattern(extra_name, inherited_optional_patterns)
                    if _should_copy_optional_dependency_extra(extra_name, member_project_name)
                )
            if inherited_script_patterns:
                script_names.update(
                    name
                    for name in pkg_project.get("scripts", {})
                    if _matches_any_pattern(name, inherited_script_patterns)
                )
            script_names.update(script["name"] for script in pkg_config.get("scripts", []) if script.get("name"))
            if inherited_entrypoint_patterns:
                for group, entries in pkg_project.get("entry-points", {}).items():
                    if _matches_any_pattern(group, inherited_entrypoint_patterns):
                        entrypoint_names.setdefault(group, set()).update(entries)

        content = member_pyproject_path.read_text(encoding="utf-8")
        annotated = _refresh_bundle_owned_optional_dependencies(content, bundle_owned_names)
        annotated = _annotate_generated_project_entries(annotated, script_names, entrypoint_names)
        if annotated != content:
            member_pyproject_path.write_text(annotated, encoding="utf-8")


def _normalize_static_force_include_spacing(pyproject_path: Path) -> None:
    """Keep repeated vendor runs from accumulating blank lines before static force-include."""
    if not pyproject_path.exists():
        return

    content = pyproject_path.read_text(encoding="utf-8")
    header = "[tool.hatch.build.targets.wheel.force-include]"
    while f"\n\n\n{header}" in content:
        content = content.replace(f"\n\n\n{header}", f"\n\n{header}")

    pyproject_path.write_text(content, encoding="utf-8")


def _reset_generated_pyproject_fields() -> None:
    """Clear auto-generated scripts/entry-point tables so stale ones don't persist.

    `[project.optional-dependencies]` is **not** reset here — vendor-owned
    extras are detected (and stale ones removed) later by
    `_refresh_bundle_owned_optional_dependencies`, which uses the per-key
    `# Generated from [tool.bundle-package]; do not edit by hand.` marker
    to distinguish vendor-owned extras from hand-written ones.

    Only the wrapper and members participating in `[tool.bundle-package]`
    are touched. The SDK pyproject is authored by Stainless and
    post_generation_update.py; the vendor tool adds to it but doesn't own it.
    """
    pyproject_path = WRAPPER_PATH / "pyproject.toml"
    if pyproject_path.exists():
        cleaned = _remove_marked_generated_project_tables(pyproject_path.read_text(encoding="utf-8"))
        if cleaned != pyproject_path.read_text(encoding="utf-8"):
            pyproject_path.write_text(cleaned, encoding="utf-8")
            rich.print("🧹 Reset auto-generated wrapper pyproject fields")

    root_pyproject_path = NMP_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    for member in members:
        member_dir = NMP_ROOT_PATH / member
        member_pyproject_path = member_dir / "pyproject.toml"
        if not member_pyproject_path.exists() or member_dir == WRAPPER_PATH:
            continue

        member_content = member_pyproject_path.read_text(encoding="utf-8")
        cleaned_member_content = _remove_marked_generated_project_tables(member_content)
        if cleaned_member_content != member_content:
            member_pyproject_path.write_text(cleaned_member_content, encoding="utf-8")
            member_name = tomlkit.loads(cleaned_member_content)["project"]["name"]
            rich.print(f"🧹 Reset auto-generated bundle metadata for `{member_name}`")


def _sort_wrapper_pyproject_fields() -> None:
    """Sort auto-generated fields in the wrapper pyproject for deterministic output.

    Without this, ordering depends on package processing order in the Makefile,
    causing noisy diffs when packages are reordered or new ones are added.
    """
    pyproject_path = WRAPPER_PATH / "pyproject.toml"
    if not pyproject_path.exists():
        return

    pyproject = tomlkit.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]

    # Sort main dependencies alphabetically (case-insensitive)
    deps = list(project.get("dependencies", []))
    if deps:
        deps.sort(key=lambda d: Requirement(d).name.lower())
        project["dependencies"] = _build_dependency_array(deps)

    # `[project.optional-dependencies]` is not sorted: hand-written extras
    # keep their original position and `_refresh_bundle_owned_optional_dependencies`
    # only rewrites vendor-owned keys in place.

    # Sort scripts alphabetically
    scripts = project.get("scripts")
    if scripts:
        sorted_scripts = tomlkit.table()
        for key in sorted(scripts.keys(), key=str.lower):
            sorted_scripts[key] = scripts[key]
        project["scripts"] = sorted_scripts

    # Sort entry-point group names alphabetically
    entrypoints = project.get("entry-points")
    if entrypoints:
        sorted_eps = tomlkit.table()
        for key in sorted(entrypoints.keys(), key=str.lower):
            sorted_eps[key] = entrypoints[key]
        project["entry-points"] = sorted_eps

    pyproject_path.write_text(tomlkit.dumps(pyproject), encoding="utf-8")


def _create_core_local_extra(configs: list[dict]) -> None:
    """Create aggregate extras and ensure `services` references them.

    Reads from [tool.bundle-package] on the wrapper to determine which packages
    are core vs non-core services based on their source path, and which bundled
    package extras are first-party plugin extras.
    """
    wrapper_pyproject_path = WRAPPER_PATH / "pyproject.toml"
    if not wrapper_pyproject_path.exists():
        return

    wrapper_config = tomlkit.loads(wrapper_pyproject_path.read_text(encoding="utf-8"))
    bundle_config = wrapper_config.get("tool", {}).get("bundle-package", {})

    # Collect -service extra names from core service bundle entries
    core_local_extras = sorted(
        set(
            _bundle_deps_group(pkg_name, pkg_config)
            for pkg_name, pkg_config in bundle_config.items()
            if isinstance(pkg_config, dict)
            and _bundle_deps_group(pkg_name, pkg_config).endswith("-service")
            and "services/core/" in pkg_config.get("source", "")
        )
    )

    # Collect -service extra names from non-core service bundle entries
    non_core_local_extras = sorted(
        set(
            _bundle_deps_group(pkg_name, pkg_config)
            for pkg_name, pkg_config in bundle_config.items()
            if isinstance(pkg_config, dict)
            and _bundle_deps_group(pkg_name, pkg_config).endswith("-service")
            and _include_in_services_extra(pkg_config)
            and "services/" in pkg_config.get("source", "")
            and "services/core/" not in pkg_config.get("source", "")
        )
    )
    plugin_extras = sorted(
        set(
            _bundle_deps_group(pkg_name, pkg_config)
            for pkg_name, pkg_config in bundle_config.items()
            if isinstance(pkg_config, dict) and _is_bundled_plugin_entry(pkg_config)
        )
    )

    pyproject_path = WRAPPER_PATH / "pyproject.toml"
    if not pyproject_path.exists():
        return

    pyproject = tomlkit.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    optional = project["optional-dependencies"]
    pkg_name = project["name"]

    # Build core-service as self-referencing extras
    core_local = tomlkit.array()
    core_local.multiline(True)
    for extra_name in core_local_extras:
        core_local.append(f"{pkg_name}[{extra_name}]")

    optional["core-service"] = core_local
    plugins = tomlkit.array()
    plugins.multiline(True)
    for extra_name in plugin_extras:
        plugins.append(f"{pkg_name}[{extra_name}]")

    optional["plugins"] = plugins
    _ensure_services_extra_includes_all_locals(project, non_core_local_extras)

    pyproject_path.write_text(tomlkit.dumps(pyproject), encoding="utf-8")

    rich.print(f"✅ Created `core-service` extra with {len(core_local_extras)} service extras")
    rich.print(f"✅ Created `plugins` extra with {len(plugin_extras)} plugin extras")
    rich.print(f"✅ Added {len(non_core_local_extras)} non-core service extras to `services`")


def _ensure_services_extra_includes_all_locals(project: tomlkit.items.Table, non_core_local_extras: list[str]) -> None:
    """Ensure ``services`` references ``core-service`` and all non-core service extras."""
    pkg_name = project.get("name")
    if not pkg_name:
        return

    optional = project.get("optional-dependencies")
    if optional is None or "core-service" not in optional:
        return

    services = optional.get("services")
    deps = [] if services is None else [str(item) for item in services]

    # Ensure aggregate extras are referenced
    core_ref = f"{pkg_name}[core-service]"
    if core_ref not in deps:
        deps.insert(0, core_ref)

    plugins_ref = f"{pkg_name}[plugins]"
    if plugins_ref not in deps:
        deps.insert(1, plugins_ref)

    # Add non-core service extras directly
    for extra_name in non_core_local_extras:
        ref = f"{pkg_name}[{extra_name}]"
        if ref not in deps:
            deps.insert(1, ref)

    optional["services"] = _build_dependency_array(deps)


def _get_transitive_source_module_name(dependency_name: str) -> str:
    """Get the source module name for a transitive dependency."""
    try:
        dist = distribution(dependency_name)
        try:
            top_level_text = dist.read_text("top_level.txt")
            if top_level_text:
                top_level_modules = [m.strip() for m in top_level_text.strip().split("\n") if m.strip()]
                if top_level_modules:
                    return top_level_modules[0]
        except Exception:
            pass
    except Exception:
        pass
    # Fallback to dependency name with hyphens replaced
    return dependency_name.replace("-", "_")


def _copy_included_paths(source_path: Path, destination_path: Path, included_paths: list[str]) -> None:
    """Copy included paths from source to destination, handling glob patterns."""

    def copy_file(source_file: Path, dest_file: Path) -> None:
        if source_file.name == "__init__.py" and source_file.read_text(encoding="utf-8") == "":
            dest_file.write_text(GENERATED_FILE_HEADER, encoding="utf-8")
        else:
            shutil.copy(source_file, dest_file)

    for pattern in included_paths:
        pattern = str(pattern)
        pattern_path = source_path / pattern
        if pattern_path.is_file():
            # If it's a file, copy it directly
            relative_file = pattern_path.relative_to(source_path)
            dest_file = destination_path / relative_file
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Copying {relative_file}")
            copy_file(pattern_path, dest_file)
        elif pattern_path.is_dir():
            # If it's a directory, find all Python files in it
            for py_file in pattern_path.rglob("*.py"):
                relative_file = py_file.relative_to(source_path)
                dest_file = destination_path / relative_file
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Copying {relative_file}")
                copy_file(py_file, dest_file)
        else:
            # Treat as glob pattern
            for matched_file in source_path.glob(pattern):
                if matched_file.is_file():
                    relative_file = matched_file.relative_to(source_path)
                    dest_file = destination_path / relative_file
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Copying {relative_file}")
                    copy_file(matched_file, dest_file)
                elif matched_file.is_dir():
                    # If glob matches a directory, find all Python files in it
                    for py_file in matched_file.rglob("*.py"):
                        relative_file = py_file.relative_to(source_path)
                        dest_file = destination_path / relative_file
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"Copying {relative_file}")
                        copy_file(py_file, dest_file)


def _build_and_validate_package_path(
    package: str, package_root: str | None, source_module: str, with_src: bool = True
) -> tuple[Path, Path]:
    if not package.isidentifier():
        raise ValueError(f"🛑 Package `{package}` is not a valid Python package name.")

    if package_root is None:
        raise ValueError(f"🛑 Package `{package}` must define `package_root` in its vendor config.")

    package_root_path = NMP_ROOT_PATH / Path(package_root)
    package_path = package_root_path / ("src" if with_src else "") / Path(*source_module.split("."))

    if not package_path.exists():
        raise ValueError(f"🛑 Package `{package}` does not exist.")

    return package_root_path, package_path


def _build_and_validate_target_paths(sdk_path: Path, target_sdk_module: str, *, top_level: bool = False) -> Path:
    target_module_path = target_sdk_module.replace(".", "/")
    if top_level:
        target_root = sdk_path / "src"
    else:
        target_root = sdk_path / "src/nemo_platform"
    target_path = target_root / target_module_path

    if not sdk_path.exists():
        raise ValueError(f"🛑 SDK path `{sdk_path}` does not exist.")

    if not all(part.isidentifier() for part in target_sdk_module.split(".")):
        raise ValueError(f"🛑 Target module `{target_sdk_module}` is not a valid Python module name.")

    if not top_level and any(target_sdk_module.startswith(forbidden) for forbidden in FORBIDDEN_TARGET_MODULES):
        raise ValueError(f"🛑 Target module `{target_sdk_module}` is forbidden.")

    target_path.mkdir(parents=True, exist_ok=True)

    if not target_path.is_dir():
        raise ValueError(f"🛑 Target path `{target_path}` is not a directory.")

    return target_path


def _update_dependencies_of_sdk_pyproject(
    sdk_path: Path,
    package_root_path: Path,
    excluded_dependencies: list[str],
    optional_deps_name: str | None = None,
    dependency_extra_replacements: dict[str, str] | None = None,
):
    """
    Update the SDK pyproject.toml to include the given dependencies, excluding specified ones.

    Note: `optional_deps_name` is the name for the `optional-dependencies` section in the pyproject.
    If not provided, the dependencies are added to the main `dependencies` section.

    ``dependency_extra_replacements`` maps excluded dependency names to extra names.
    When an excluded dep has an entry here, it is replaced with a self-referencing
    extra (e.g. ``{"nemo-evaluator-sdk": "evaluator"}`` turns the dep into
    ``nemo-platform[evaluator]``).
    """
    if dependency_extra_replacements is None:
        dependency_extra_replacements = {}

    with open(package_root_path / "pyproject.toml", "rb") as f:
        package_config = tomlkit.load(f)

    # Get dependencies from package config
    raw_package_dependencies = cast(Any, package_config["project"]["dependencies"])
    unwrapped_dependencies = (
        raw_package_dependencies.unwrap() if hasattr(raw_package_dependencies, "unwrap") else raw_package_dependencies
    )
    if not isinstance(unwrapped_dependencies, list):
        raise TypeError(f"Expected project dependencies array in {package_root_path / 'pyproject.toml'}")
    package_dependencies = []
    for dependency in unwrapped_dependencies:
        if not isinstance(dependency, str):
            raise TypeError(f"Expected project dependency string in {package_root_path / 'pyproject.toml'}")
        package_dependencies.append(dependency)

    # Filter out excluded dependencies
    def _get_package_name(dep: str) -> str:
        """Extract the base package name from a dependency string."""
        return Requirement(dep).name

    filtered_dependencies = []
    extra_self_refs = list(dependency_extra_replacements.values())

    for dep in package_dependencies:
        dep_name = _get_package_name(dep)
        if dep_name not in excluded_dependencies:
            filtered_dependencies.append(dep)
        else:
            logger.debug(f"Excluding dependency: {dep}")

    for target_path in _pyproject_target_paths(sdk_path):
        pyproject_path = target_path / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        with open(pyproject_path, "rb") as f:
            config = tomlkit.load(f)

        pkg_name = config["project"]["name"]
        self_ref_deps = [f"{pkg_name}[{extra}]" for extra in extra_self_refs]
        all_deps = filtered_dependencies + self_ref_deps

        if optional_deps_name is not None:
            logger.info(f"Adding `{optional_deps_name}` optional dependencies to {pyproject_path}")
            optional_dependencies = config["project"].setdefault("optional-dependencies", tomlkit.table())
            existing_deps = list(optional_dependencies.get(optional_deps_name, []))
            merged_deps = merge_dependencies(existing_deps, all_deps)
            optional_dependencies[optional_deps_name] = _build_dependency_array(merged_deps)
        else:
            logger.info(f"Adding main dependencies to {pyproject_path}")
            existing_deps = config["project"]["dependencies"]
            merged_deps = merge_dependencies(existing_deps, all_deps)
            config["project"]["dependencies"] = _build_dependency_array(merged_deps)

        logger.info(f"Writing updated pyproject.toml to {pyproject_path}")
        with open(pyproject_path, "w") as f:
            tomlkit.dump(config, f)


def _build_dependency_array(dependencies: list[str]) -> tomlkit.items.Array:
    dep_array = tomlkit.items.Array([], tomlkit.items.Trivia(indent=""))
    for dep in dependencies:
        dep_array.add_line(dep, indent="  ")
    dep_array.add_line(indent="")
    return dep_array


def _rewrite_imports_in_vendored_source_multiple(
    *,
    target_path: Path,
    module_rewrites: list[tuple[str, str]],
    included_transitive_dependencies: list[dict] | None = None,
) -> None:
    """Rewrite imports in vendored source code using pre-computed module rewrites.

    This function applies all module rewrites to all files in target_path,
    which is necessary for flat vendoring where modules may import from each other.
    """
    # Find all Python files that were copied to the target
    copied_files = [f for f in target_path.rglob("*.py") if f.is_file()]

    if not copied_files:
        logger.warning("No Python files found in vendored source code, skipping import rewriting.")
        return

    # Extend module_rewrites with transitive dependency rewrites
    all_rewrites = list(module_rewrites)
    if included_transitive_dependencies:
        for dep_config in included_transitive_dependencies:
            if not (isinstance(dep_config, dict) or hasattr(dep_config, "get")):
                continue

            dependency_name = dep_config.get("dependency")
            if not dependency_name:
                continue

            transitive_source_module = _get_transitive_source_module_name(dependency_name)

            # For flat vendoring, transitive deps are vendored into each target module
            # We need to add rewrites for each target module
            for _, target_module in module_rewrites:
                if (transitive_source_module, target_module) not in all_rewrites:
                    all_rewrites.append((transitive_source_module, target_module))
                    logger.debug(f"Will also rewrite imports: `{transitive_source_module}` -> `{target_module}`")

    if not all_rewrites:
        logger.warning("No module rewrites to apply, skipping import rewriting.")
        return

    for py_file in copied_files:
        relative_file = py_file.relative_to(target_path)
        logger.debug(f"Rewriting imports in {relative_file}")
        _rewrite_imports_in_file_multiple(py_file, all_rewrites)

    logger.info("Source code imports rewritten successfully")


def _create_init_files(target_path: Path) -> None:
    """Create __init__.py files in target directory and all subdirectories.

    This is needed for relative imports to work in vendored tests.
    """
    # Create __init__.py in the target directory itself
    init_file = target_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text(GENERATED_FILE_HEADER, encoding="utf-8")
        logger.debug(f"Created {init_file.relative_to(target_path.parent.parent.parent)}")

    # Create __init__.py in all subdirectories
    for subdir in target_path.rglob("*"):
        if subdir.is_dir():
            init_file = subdir / "__init__.py"
            if not init_file.exists():
                init_file.write_text(GENERATED_FILE_HEADER, encoding="utf-8")
                logger.debug(f"Created {init_file.relative_to(target_path.parent.parent.parent)}")


def _vendor_tests_flat(
    *,
    source_root_path: Path,
    source_tests_path: str,
    sdk_path: Path,
    tests_target_subdir: str,
    tests_included_paths: list[str],
    module_rewrites: list[tuple[str, str]],
) -> None:
    """Vendor tests with import rewriting using pre-computed module rewrites."""
    tests_source = source_root_path / source_tests_path
    if not tests_source.exists():
        logger.warning(f"Tests path `{source_tests_path}` does not exist, skipping.")
        return

    tests_target_path = sdk_path / "tests" / "vendored" / tests_target_subdir
    rich.print(f"🧪 Vendoring tests to `{tests_target_path.relative_to(sdk_path)}`")

    if tests_target_path.exists():
        shutil.rmtree(tests_target_path, ignore_errors=True)

    _copy_included_paths(tests_source, tests_target_path, tests_included_paths)

    # Create __init__.py files in all directories for relative imports to work
    _create_init_files(tests_target_path)

    copied_files = [f for f in tests_target_path.rglob("*.py") if f.is_file()]

    for py_file in copied_files:
        relative_file = py_file.relative_to(tests_target_path)
        logger.debug(f"Rewriting imports in test file {relative_file}")
        _rewrite_imports_in_file_multiple(py_file, module_rewrites)

    logger.info("Tests vendored successfully")


class ImportRewriter(cst.CSTTransformer):
    """Rewrites imports from source module to target module using libcst."""

    def __init__(self, source_module: str, target_module: str):
        self._source_module = source_module
        self._target_module = target_module

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import:
        """Rewrite 'import source_module' to 'import target_module as source_module'."""
        new_names = []
        for name in updated_node.names:
            if isinstance(name, cst.ImportAlias):
                # Get the full module name as a string by walking the Attribute/Name tree
                module_parts = []
                node = name.name
                while isinstance(node, cst.Attribute):
                    module_parts.insert(0, node.attr.value)
                    node = node.value
                if isinstance(node, cst.Name):
                    module_parts.insert(0, node.value)
                module_name = ".".join(module_parts)

                # Skip if the module already starts with the target prefix (already rewritten)
                if module_name.startswith(f"{self._target_module}.") or module_name == self._target_module:
                    new_names.append(name)
                    continue

                if module_name == self._source_module or module_name.startswith(f"{self._source_module}."):
                    # Replace the module name
                    new_module = module_name.replace(self._source_module, self._target_module, 1)
                    new_dotted = cst.helpers.parse_template_expression(new_module)
                    # If there's no alias, add one to preserve the original import name
                    if name.asname is None and module_name == self._source_module:
                        new_names.append(
                            name.with_changes(
                                name=new_dotted,
                                asname=cst.AsName(
                                    name=cst.Name(self._source_module),
                                    whitespace_before_as=cst.SimpleWhitespace(" "),
                                    whitespace_after_as=cst.SimpleWhitespace(" "),
                                ),
                            )
                        )
                    else:
                        new_names.append(name.with_changes(name=new_dotted))
                else:
                    new_names.append(name)
            else:
                new_names.append(name)
        return updated_node.with_changes(names=new_names)

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        """Rewrite 'from source_module import X' to 'from target_module import X'."""
        if updated_node.module is None:
            return updated_node

        # Get the module name as a string by walking the Attribute/Name tree
        module_parts = []
        node = updated_node.module
        while isinstance(node, cst.Attribute):
            module_parts.insert(0, node.attr.value)
            node = node.value
        if isinstance(node, cst.Name):
            module_parts.insert(0, node.value)
        module_name = ".".join(module_parts)

        # Skip if the module already starts with the target prefix (already rewritten)
        if module_name.startswith(f"{self._target_module}.") or module_name == self._target_module:
            return updated_node

        if module_name == self._source_module or module_name.startswith(f"{self._source_module}."):
            # Replace the module name
            new_module_name = module_name.replace(self._source_module, self._target_module, 1)
            # Parse the new module name into a dotted name expression
            # Parse a full import statement to get the module AST node
            import_stmt = f"from {new_module_name} import x"
            parsed = cst.parse_statement(import_stmt)
            if isinstance(parsed, cst.SimpleStatementLine) and parsed.body:
                import_from = parsed.body[0]
                if isinstance(import_from, cst.ImportFrom) and import_from.module:
                    new_module = import_from.module
                    return updated_node.with_changes(module=new_module)

        return updated_node

    def leave_SimpleString(self, original_node: cst.SimpleString, updated_node: cst.SimpleString) -> cst.SimpleString:
        """Rewrite module paths in string literals (e.g., for @patch decorators)."""
        # Get the string value without quotes
        value = updated_node.value
        # Determine the quote style (single, double, or triple quoted)
        if value.startswith('"""') or value.startswith("'''"):
            quote = value[:3]
            string_content = value[3:-3]
        elif value.startswith('"') or value.startswith("'"):
            quote = value[0]
            string_content = value[1:-1]
        else:
            return updated_node

        # Skip if the string already contains the target module path (already rewritten)
        if self._target_module in string_content:
            # Check if it's already rewritten by looking for the target pattern
            import re

            target_pattern = re.escape(self._target_module) + r"(?=\.|$)"
            if re.search(target_pattern, string_content):
                return updated_node

        # Check if the string contains the source module path
        if self._source_module in string_content:
            # Only replace if it's a module path (followed by . or end of string)
            # This avoids replacing the module name in arbitrary text
            import re

            # Match source_module followed by a dot or end of string
            pattern = re.escape(self._source_module) + r"(?=\.|$)"
            if re.search(pattern, string_content):
                new_content = re.sub(pattern, self._target_module, string_content)
                new_value = f"{quote}{new_content}{quote}"
                return updated_node.with_changes(value=new_value)

        return updated_node


class MultiImportRewriter(cst.CSTTransformer):
    """Rewrites all import paths in a single CST traversal."""

    def __init__(self, module_rewrites: list[tuple[str, str]]) -> None:
        self._rewrites = module_rewrites
        self._target_modules = frozenset(target for _, target in module_rewrites)

    def _apply_rewrite(self, module_name: str) -> str | None:
        """Return rewritten module name, or None if no rewrite applies."""
        if any(module_name == t or module_name.startswith(f"{t}.") for t in self._target_modules):
            return None  # Already rewritten
        for source_module, target_module in self._rewrites:
            if module_name == source_module or module_name.startswith(f"{source_module}."):
                return module_name.replace(source_module, target_module, 1)
        return None

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import:
        """Rewrite 'import source_module' to 'import target_module as source_module'."""
        new_names = []
        for name in updated_node.names:
            if isinstance(name, cst.ImportAlias):
                module_parts = []
                node = name.name
                while isinstance(node, cst.Attribute):
                    module_parts.insert(0, node.attr.value)
                    node = node.value
                if isinstance(node, cst.Name):
                    module_parts.insert(0, node.value)
                module_name = ".".join(module_parts)

                new_module_name = self._apply_rewrite(module_name)
                if new_module_name is None:
                    new_names.append(name)
                else:
                    new_dotted = cst.helpers.parse_template_expression(new_module_name)
                    # Find the matched source module for alias preservation on exact match
                    matched_source = next(
                        (s for s, _ in self._rewrites if module_name == s or module_name.startswith(f"{s}.")),
                        None,
                    )
                    if name.asname is None and matched_source is not None and module_name == matched_source:
                        new_names.append(
                            name.with_changes(
                                name=new_dotted,
                                asname=cst.AsName(
                                    name=cst.Name(matched_source),
                                    whitespace_before_as=cst.SimpleWhitespace(" "),
                                    whitespace_after_as=cst.SimpleWhitespace(" "),
                                ),
                            )
                        )
                    else:
                        new_names.append(name.with_changes(name=new_dotted))
            else:
                new_names.append(name)
        return updated_node.with_changes(names=new_names)

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        """Rewrite 'from source_module import X' to 'from target_module import X'."""
        if updated_node.module is None:
            return updated_node

        module_parts = []
        node = updated_node.module
        while isinstance(node, cst.Attribute):
            module_parts.insert(0, node.attr.value)
            node = node.value
        if isinstance(node, cst.Name):
            module_parts.insert(0, node.value)
        module_name = ".".join(module_parts)

        new_module_name = self._apply_rewrite(module_name)
        if new_module_name is None:
            return updated_node

        import_stmt = f"from {new_module_name} import x"
        parsed = cst.parse_statement(import_stmt)
        if isinstance(parsed, cst.SimpleStatementLine) and parsed.body:
            import_from = parsed.body[0]
            if isinstance(import_from, cst.ImportFrom) and import_from.module:
                new_module = import_from.module
                return updated_node.with_changes(module=new_module)

        return updated_node

    def leave_SimpleString(self, original_node: cst.SimpleString, updated_node: cst.SimpleString) -> cst.SimpleString:
        """Rewrite module paths in string literals (e.g., for @patch decorators)."""
        import re

        value = updated_node.value
        if value.startswith('"""') or value.startswith("'''"):
            quote = value[:3]
            string_content = value[3:-3]
        elif value.startswith('"') or value.startswith("'"):
            quote = value[0]
            string_content = value[1:-1]
        else:
            return updated_node

        current_content = string_content
        changed = False
        for source_module, target_module in self._rewrites:
            # Skip if this target module is already present in the string
            if target_module in current_content:
                target_pattern = re.escape(target_module) + r"(?=\.|$)"
                if re.search(target_pattern, current_content):
                    continue

            # Check if source module path is in the string
            if source_module in current_content:
                pattern = re.escape(source_module) + r"(?=\.|$)"
                if re.search(pattern, current_content):
                    current_content = re.sub(pattern, target_module, current_content)
                    changed = True

        if changed:
            return updated_node.with_changes(value=f"{quote}{current_content}{quote}")
        return updated_node


def _find_installed_package_path(package_name: str) -> Path:
    """Find the installed location of a third-party package."""
    import importlib.util

    try:
        for module_name in [package_name.replace("-", "_"), package_name]:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec and spec.origin:
                    module_path = Path(spec.origin).parent
                    if module_path.exists():
                        return module_path
            except Exception:
                continue

        raise ValueError(f"🛑 Could not locate installed package `{package_name}`.")
    except Exception as e:
        raise ValueError(f"🛑 Could not find installed package `{package_name}`: {e}")


def _vendor_transitive_dependencies(
    sdk_path: Path,
    target_sdk_module: str,
    included_transitive_dependencies: list[dict],
    top_level: bool = False,
) -> None:
    """Vendor files from third-party dependencies into the SDK."""
    target_module_path = target_sdk_module.replace(".", "/")
    if top_level:
        target_root = sdk_path / "src"
    else:
        target_root = sdk_path / "src/nemo_platform"
    target_path = target_root / target_module_path

    for dep_config in included_transitive_dependencies:
        # Handle both dict and tomlkit Table objects
        if not (isinstance(dep_config, dict) or hasattr(dep_config, "get")):
            raise ValueError(
                f"🛑 Invalid transitive dependency config: expected dict-like object, got {type(dep_config)}"
            )

        dependency_name = dep_config.get("dependency")
        paths = dep_config.get("paths", [])

        # Convert tomlkit Array to list if needed
        if hasattr(paths, "__iter__") and not isinstance(paths, (str, bytes)):
            paths = list(paths)

        if not dependency_name:
            raise ValueError(f"🛑 Missing 'dependency' key in transitive dependency config: {dep_config}")

        if not paths:
            logger.warning(f"No paths specified for dependency `{dependency_name}`, skipping.")
            continue

        logger.info(f"Vendoring files from third-party dependency `{dependency_name}`")

        # Find the installed package location
        try:
            source_package_path = _find_installed_package_path(dependency_name)
            logger.debug(f"Found at: {source_package_path}")
        except ValueError as e:
            logger.warning(str(e))
            continue

        # Vendor the specified paths
        _copy_included_paths(source_package_path, target_path, paths)

        source_module = _get_transitive_source_module_name(dependency_name)
        target_module_full = target_sdk_module if top_level else f"nemo_platform.{target_sdk_module}"
        logger.debug(f"Rewriting imports: `{source_module}` -> `{target_module_full}`")

        # Find all Python files that were copied and rewrite imports
        copied_files = [f for f in target_path.rglob("*.py") if f.is_file()]

        for target_file in copied_files:
            relative_file = target_file.relative_to(target_path)
            logger.debug(f"Rewriting imports in {relative_file}")
            _rewrite_imports_in_file_multiple(target_file, [(source_module, target_module_full)])

    logger.info("Third-party dependencies vendored successfully")


def _replace_client_methods(
    *,
    sdk_path: Path,
    source_path: Path,
    remove_vendored_source_relative_path: str | None = None,
) -> None:
    """Replace NeMo client methods in SDK _client.py from a source module."""
    target_path = sdk_path / "src" / "nemo_platform" / "_client.py"

    if not source_path.exists():
        logger.warning(f"Client method source not found at {source_path}, skipping replacement.")
        return
    if not target_path.exists():
        logger.warning(f"_client.py not found at {target_path}, skipping replacement.")
        return

    with open(source_path, "r", encoding="utf-8") as f:
        source_content = f.read()

    with open(target_path, "r", encoding="utf-8") as f:
        target_content = f.read()

    # The generated SDK wheel stages the native source package beside
    # ``nemo_platform``. Keep bootstrap imports pointed at that real package so
    # static tooling can resolve them instead of chasing alias-only modules.
    source_tree = cst.parse_module(source_content)
    collector = _ClientMethodCollector()
    source_tree.visit(collector)
    helper_functions = _collect_client_helper_functions(source_tree)

    if not collector.class_methods:
        logger.warning(f"No NeMo client methods found in {source_path}, skipping.")
        return

    target_tree = cst.parse_module(target_content)
    target_tree = _ensure_required_client_init_imports(target_tree)
    target_tree = _with_client_helper_functions(target_tree, helper_functions)
    replacer = _ClientMethodReplacer(collector.class_methods)
    modified_tree = target_tree.visit(replacer)
    modified_content = modified_tree.code

    if modified_content != target_content:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        logger.info(f"Replaced _client.py methods from {source_path}")
    else:
        logger.debug("No changes made while replacing _client.py methods")

    if remove_vendored_source_relative_path:
        vendored_source_path = sdk_path / "src" / "nemo_platform" / Path(remove_vendored_source_relative_path)
        if vendored_source_path.exists():
            vendored_source_path.unlink()
            logger.info(f"Removed vendored client method source file {vendored_source_path}")

    missing = [name for name in _CLIENT_CLASS_NAMES if name not in replacer.replaced_classes]
    if missing:
        logger.warning(f"Did not replace methods for classes in _client.py: {', '.join(missing)}")


def _apply_resource_replacements(sdk_path: Path, replacements: list[ResourceReplacement]) -> None:
    """Replace resource imports in _client.py.

    Uses regex to find imports that ONLY contain the target classes and replaces
    the module path. For example, if we have a replacement with class_names
    ["FilesetsResource", "AsyncFilesetsResource"] from .resources.filesets to .filesets:

    - `from .resources.filesets import FilesetsResource` -> `from .filesets import FilesetsResource`
    - `from .resources.filesets import FilesetsResource, AsyncFilesetsResource` -> `from .filesets import ...`
    - `from .resources.filesets import FilesetsResourceWithRawResponse` -> unchanged (not in class_names)
    """
    import re

    client_path = sdk_path / "src/nemo_platform/_client.py"

    if not client_path.exists():
        logger.warning(f"_client.py not found at {client_path}")
        return

    with open(client_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    for repl in replacements:
        class_names = set(repl.class_names)

        for original_module in repl.original:
            # Escape the module path for regex (dots need escaping)
            escaped_original = re.escape(original_module)

            # Pattern to match: from <original> import <classes>
            # where <classes> is a comma-separated list of class names
            pattern = rf"from {escaped_original} import ([^\n]+)"

            def make_replacer(orig: str):
                def replace_if_only_target_classes(match: re.Match) -> str:
                    imports_str = match.group(1)
                    # Parse the imported names (handle "Name" and "Name as alias")
                    imported = []
                    for part in imports_str.split(","):
                        part = part.strip()
                        if " as " in part:
                            name = part.split(" as ")[0].strip()
                        else:
                            name = part
                        imported.append(name)

                    # Only replace if ALL imported names are in our target set
                    if all(name in class_names for name in imported):
                        logger.debug(
                            f"from {orig} import {imports_str} -> from {repl.replacement} import {imports_str}"
                        )
                        return f"from {repl.replacement} import {imports_str}"
                    return match.group(0)  # No change

                return replace_if_only_target_classes

            content = re.sub(pattern, make_replacer(original_module), content)

    if content != original_content:
        with open(client_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("_client.py imports rewritten")
    else:
        logger.debug("No changes made to _client.py")


def _vendor_entrypoints(
    sdk_path: Path,
    entrypoints: list[dict],
) -> None:
    for target_path in _pyproject_target_paths(sdk_path):
        pyproject_path = target_path / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        with open(pyproject_path, "rb") as f:
            config = tomlkit.load(f)

        for entrypoint_group in entrypoints:
            group = entrypoint_group["group"]
            ep_group = config["project"].setdefault("entry-points", {}).setdefault(group, {})

            for entrypoint in entrypoint_group["entrypoints"]:
                ep_group[entrypoint["name"]] = entrypoint["value"]

        with open(pyproject_path, "w") as f:
            tomlkit.dump(config, f)


def _vendor_scripts(
    sdk_path: Path,
    scripts: list[dict],
) -> None:
    """Add scripts to the SDK pyproject.toml.

    Scripts are defined under [project.scripts] and map command names to module:function paths.
    Example: nmp = "nemo_platform.cli.app:cli"
    """
    for target_path in _pyproject_target_paths(sdk_path):
        pyproject_path = target_path / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        with open(pyproject_path, "rb") as f:
            config = tomlkit.load(f)

        scripts_section = config["project"].setdefault("scripts", {})

        for script in scripts:
            script_name = script["name"]
            script_value = script["value"]
            scripts_section[script_name] = script_value
            logger.info(f"Adding script: {script_name} = {script_value}")

        with open(pyproject_path, "w") as f:
            tomlkit.dump(config, f)


def _rewrite_imports_in_file_multiple(file_path: Path, module_rewrites: list[tuple[str, str]]) -> None:
    """Rewrite imports in a Python file applying multiple module rewrites in sequence."""
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # Fast pre-filter: skip CST parsing if no source modules appear in the file
    if not any(source_mod in source_code for source_mod, _ in module_rewrites):
        return

    original_code = source_code
    try:
        tree = cst.parse_module(source_code)
        tree = tree.visit(MultiImportRewriter(module_rewrites))
        modified_code = tree.code

        # Check if anything actually changed
        if modified_code != original_code:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_code)
            # Show what changed (find import lines that were modified)
            original_lines = original_code.splitlines()
            modified_lines = modified_code.splitlines()
            for i, (orig, mod) in enumerate(zip(original_lines, modified_lines), 1):
                if orig != mod and ("import" in orig or "from" in orig):
                    # Check if any of the source modules are in the original line
                    for source_mod, _ in module_rewrites:
                        if source_mod in orig:
                            logger.debug(f"Line {i}: {orig.strip()} -> {mod.strip()}")
                            break
    except Exception as e:
        logger.error(f"Failed to rewrite imports in {file_path}: {e}")
        raise


if __name__ == "__main__":
    app()
