# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path

import rich
import tomlkit
import tomlkit.items
import typer
from nemo_helix_sdk_tools.sdk.core.common import get_project_dir
from nemo_helix_sdk_tools.sdk.vendor.dependency_utils import merge_dependencies
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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


def _setup_logging() -> None:
    """Setup logging to write verbose output to a log file."""
    logs_dir = NHX_ROOT_PATH / "logs"
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


NHX_ROOT_PATH = get_project_dir()

WRAPPER_PATH = NHX_ROOT_PATH / "packages/nemo_helix"


app = typer.Typer(
    name="vendor", no_args_is_help=True, help="Refresh nemo-helix wheel metadata generated from [tool.bundle-package]"
)


@app.command("bundle-metadata")
def vendor_bundle_metadata() -> None:
    """Refresh wrapper metadata generated from `[tool.bundle-package]`."""
    _setup_logging()
    _reset_generated_pyproject_fields()
    _refresh_bundle_metadata()


def _refresh_bundle_metadata() -> None:
    # Create the core-service extra that aggregates all service -service extras.
    _create_core_local_extra()

    # Process [tool.bundle-package] configs across all workspace packages.
    # This reads each bundled package's deps and writes them into the deps_group
    # specified by the parent's bundle config. It also copies scripts,
    # entry-points, and optional-dependencies from bundled packages.
    _process_bundle_packages()

    # Sort auto-generated fields in the wrapper for deterministic output.
    # Without this, ordering depends on package processing order in the Makefile.
    _sort_wrapper_pyproject_fields()

    # Rewrite optional-dependencies (manual aliases + generated extras)
    # and annotate generated scripts/entry-point tables.
    _annotate_generated_bundle_groups()

    _normalize_static_force_include_spacing(WRAPPER_PATH / "pyproject.toml")


def _find_workspace_package_dir(package_name: str) -> Path | None:
    """Find a workspace package's directory by reading its pyproject.toml name.

    Scans all workspace members declared in the root pyproject.toml.
    """
    root_pyproject_path = NHX_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return None

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    for member in members:
        member_dir = NHX_ROOT_PATH / member
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
    target_project: dict,
    source_project: dict,
    patterns: list[str],
    base_extra: str,
    extra_prefix: str,
) -> None:
    source_optional = source_project.get("optional-dependencies", {})
    if not source_optional or not patterns:
        return

    target_project_name = target_project.get("name")
    target_optional = target_project.setdefault("optional-dependencies", tomlkit.table())
    generated_optional_groups = _generated_optional_dependency_groups(target_optional)
    target_dependencies = {
        canonicalize_name(Requirement(dependency).name) for dependency in target_project.get("dependencies", [])
    }
    source_project_name = source_project.get("name")
    base_requirement = (
        f"{target_project_name}[{base_extra}]"
        if target_project_name
        and source_project_name
        and canonicalize_name(source_project_name) not in target_dependencies
        else None
    )
    for extra_name, deps in source_optional.items():
        if not _matches_any_pattern(extra_name, patterns):
            continue
        if not _should_copy_optional_dependency_extra(extra_name, target_project_name):
            continue
        inherited_extra_name = f"{extra_prefix}{extra_name}"
        if inherited_extra_name not in target_optional or inherited_extra_name in generated_optional_groups:
            inherited = ([base_requirement] if base_requirement else []) + list(deps)
            target_optional[inherited_extra_name] = _build_dependency_array(inherited)


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

    Workspace package dependencies are filtered out (they're not on PyPI),
    except members listed in the parent's ``[tool.bundle-package-published]``
    ``packages``, which release to PyPI on their own and stay regular
    requirements. If a filtered dep has its own bundle entry, the dependency
    name is kept in source metadata so build hooks can rewrite final wheel
    metadata if needed.
    """
    root_pyproject_path = NHX_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    # Build a set of all canonicalized workspace package names — these are never
    # installable from PyPI, so they must be excluded from bundled dependency lists.
    # Names are canonicalized (PEP 503) so that nemo-nb, nemo_nb, etc. all match.
    workspace_package_names: set[str] = set()
    for member in members:
        member_dir = NHX_ROOT_PATH / member
        member_pyproject_path = member_dir / "pyproject.toml"
        if not member_pyproject_path.exists():
            continue
        config = tomlkit.loads(member_pyproject_path.read_text(encoding="utf-8"))
        name = config.get("project", {}).get("name")
        if name:
            workspace_package_names.add(canonicalize_name(name))

    for member in members:
        member_dir = NHX_ROOT_PATH / member
        member_pyproject_path = member_dir / "pyproject.toml"
        if not member_pyproject_path.exists():
            continue

        member_config = tomlkit.loads(member_pyproject_path.read_text(encoding="utf-8"))
        bundle_config = member_config.get("tool", {}).get("bundle-package")
        if not bundle_config:
            continue

        parent_name = member_config.get("project", {}).get("name", member)
        published_package_names = {
            canonicalize_name(name)
            for name in member_config.get("tool", {}).get("bundle-package-published", {}).get("packages", [])
        }

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
                pyproject["project"],
                pkg_project,
                inherited_optional_patterns,
                deps_group,
                pkg_config.get("optional-dependencies-prefix", ""),
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
                if dep_name in workspace_package_names and dep_name not in published_package_names:
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
    root_pyproject_path = NHX_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    for member in members:
        member_pyproject_path = NHX_ROOT_PATH / member / "pyproject.toml"
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
            inherited_optional_prefix = pkg_config.get("optional-dependencies-prefix", "")
            pkg_project = _load_bundle_project(pkg_name, pkg_config, member_dir)
            bundle_owned_names.add(_bundle_deps_group(pkg_name, pkg_config))
            if inherited_optional_patterns:
                bundle_owned_names.update(
                    f"{inherited_optional_prefix}{extra_name}"
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
    are touched.
    """
    pyproject_path = WRAPPER_PATH / "pyproject.toml"
    if pyproject_path.exists():
        cleaned = _remove_marked_generated_project_tables(pyproject_path.read_text(encoding="utf-8"))
        if cleaned != pyproject_path.read_text(encoding="utf-8"):
            pyproject_path.write_text(cleaned, encoding="utf-8")
            rich.print("🧹 Reset auto-generated wrapper pyproject fields")

    root_pyproject_path = NHX_ROOT_PATH / "pyproject.toml"
    if not root_pyproject_path.exists():
        return

    root_config = tomlkit.loads(root_pyproject_path.read_text(encoding="utf-8"))
    members = root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])

    for member in members:
        member_dir = NHX_ROOT_PATH / member
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


def _create_core_local_extra() -> None:
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


def _build_dependency_array(dependencies: list[str]) -> tomlkit.items.Array:
    dep_array = tomlkit.items.Array([], tomlkit.items.Trivia(indent=""))
    for dep in dependencies:
        dep_array.add_line(dep, indent="  ")
    dep_array.add_line(indent="")
    return dep_array


if __name__ == "__main__":
    app()
