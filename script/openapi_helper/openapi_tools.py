# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This file was primarily authored with the assistance of an AI coding assistant (Cursor).
"""

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import typer
import yaml
from nmp.common.api.utils import tweak_spec
from nmp.common.version import OPENAPI_SPEC_VERSION
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

# Global verbose flag
VERBOSE = False


def set_verbose(verbose: bool) -> None:
    """Set the global verbose flag."""
    global VERBOSE
    VERBOSE = verbose


def print_verbose(message: str, style: str = None) -> None:
    """Print message only if verbose mode is enabled."""
    if VERBOSE:
        console.print(message, style=style)


def load_openapi_spec(file_path: str) -> dict:
    """Load OpenAPI specification from YAML or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File {file_path} does not exist")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml", ".txt"]:
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise typer.BadParameter("File must be YAML or JSON format")


def save_openapi_spec(spec: dict, output_path: str) -> None:
    """Save OpenAPI specification to a YAML or JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".tmp"
    existing_mode = output.stat().st_mode & 0o7777 if output.exists() else None
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=output.parent, suffix=suffix, encoding="utf-8") as f:
            temp_path = Path(f.name)
            if output_path.endswith(".json"):
                f.write(json.dumps(spec, indent=2))
            else:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        os.replace(temp_path, output)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    print_verbose(f"Saved specification to {output_path}", style="bold green")


@app.command(name="list-endpoints")
def list_endpoints(spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)")):
    """List all endpoints and schemas from an OpenAPI specification."""
    try:
        spec = load_openapi_spec(spec_file)

        if "paths" not in spec:
            print_verbose("No paths found in the OpenAPI specification")
            return

        # First table with detailed information
        table = Table(title="API Endpoints", show_header=True, header_style="bold magenta")
        table.add_column("Method", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Summary", style="white")

        endpoint_count = 0
        distinct_paths = set()  # Track unique paths
        method_matrix = {}  # Track which methods are supported for each path
        path_tags = {}  # Track tags for each path

        for path, methods in spec["paths"].items():
            method_matrix[path] = set()  # Initialize set for this path
            # Get the first tag from any method in this path
            path_tag = None
            for method_details in methods.values():
                if "tags" in method_details and method_details["tags"]:
                    path_tag = method_details["tags"][0]
                    break
            path_tags[path] = path_tag or "Untagged"

            for method, details in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    summary = details.get("summary", "No summary available")
                    table.add_row(method.upper(), path, summary)
                    endpoint_count += 1
                    distinct_paths.add(path)  # Add path to set of unique paths
                    method_matrix[path].add(method.upper())  # Add method to path's supported methods

        # Add a separator row
        table.add_row("", "", "")
        # Add the count rows
        table.add_row(
            "Total Endpoint-Method Combinations",
            str(endpoint_count),
            "",
            style="bold yellow",
        )
        table.add_row("Distinct Endpoints", str(len(distinct_paths)), "", style="bold yellow")

        print_verbose(table)

        # Second table showing method matrix grouped by tags
        method_table = Table(
            title="Endpoint Method Matrix (Grouped by Tag)",
            show_header=True,
            header_style="bold magenta",
        )
        method_table.add_column("Path", style="green")
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            method_table.add_column(method, style="cyan", justify="center")

        # Group paths by their first tag
        tag_groups = {}
        for path in distinct_paths:
            tag = path_tags[path]
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append(path)

        # Add rows for each path, grouped by tag
        for tag in sorted(tag_groups.keys()):
            # Add tag header
            method_table.add_row(f"[bold blue]{tag}[/bold blue]", "", "", "", "", "")
            # Add paths under this tag
            for path in sorted(tag_groups[tag]):
                row = [path]
                for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    row.append("X" if method in method_matrix[path] else "")
                method_table.add_row(*row)
            # Add separator after each tag group
            method_table.add_row("", "", "", "", "", "")

        print_verbose("\n")  # Add some spacing between tables
        print_verbose(method_table)

        # Third table showing schemas
        if "components" in spec and "schemas" in spec["components"]:
            schemas_table = Table(title="API Schemas", show_header=True, header_style="bold magenta")
            schemas_table.add_column("Schema Name", style="cyan")
            schemas_table.add_column("Type", style="green")
            schemas_table.add_column("Description", style="white")
            schemas_table.add_column("Properties", style="yellow")

            schemas = spec["components"]["schemas"]
            schema_count = len(schemas)

            for schema_name, schema in sorted(schemas.items()):
                schema_type = schema.get("type", "object")
                description = schema.get("description", "No description available")

                # Get properties count and list
                properties = schema.get("properties", {})
                properties_count = len(properties)
                properties_list = ", ".join(sorted(properties.keys())) if properties else "N/A"

                schemas_table.add_row(
                    schema_name,
                    schema_type,
                    description,
                    f"{properties_count} properties: {properties_list}",
                )

            # Add a separator row
            schemas_table.add_row("", "", "", "")
            # Add the count row
            schemas_table.add_row("Total Schemas", str(schema_count), "", "", style="bold yellow")

            print_verbose("\n")  # Add some spacing between tables
            print_verbose(schemas_table)

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


@app.command(name="show-info")
def show_info(spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)")):
    """Display API information and version from the OpenAPI specification."""
    try:
        spec = load_openapi_spec(spec_file)

        info = spec.get("info", {})
        version = spec.get("openapi", "Unknown")

        table = Table(title="API Information", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Title", info.get("title", "N/A"))
        table.add_row("Version", info.get("version", "N/A"))
        table.add_row("Description", info.get("description", "N/A"))
        table.add_row("OpenAPI Version", version)

        print_verbose(table)

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def get_schema_refs(schema: dict) -> set:
    """Extract all schema references from a schema object."""
    refs = set()

    if isinstance(schema, dict):
        # Check for direct reference
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref.startswith("#/components/schemas/"):
                refs.add(ref.replace("#/components/schemas/", ""))

        # Check all values in the dictionary
        for value in schema.values():
            refs.update(get_schema_refs(value))
    elif isinstance(schema, list):
        for item in schema:
            refs.update(get_schema_refs(item))

    return refs


def get_endpoint_schema_refs(spec: dict) -> set:
    """Extract all schema references from endpoints in the OpenAPI specification."""
    endpoint_refs = set()
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            # Check parameters
            for param in details.get("parameters", []):
                if "schema" in param:
                    endpoint_refs.update(get_schema_refs(param["schema"]))
                elif "$ref" in param:
                    ref = param["$ref"]
                    if ref.startswith("#/components/schemas/"):
                        endpoint_refs.add(ref.replace("#/components/schemas/", ""))

            # Check request body
            if "requestBody" in details:
                request_body = details["requestBody"]
                if "$ref" in request_body:
                    ref = request_body["$ref"]
                    if ref.startswith("#/components/requestBodies/"):
                        # Look up the referenced request body
                        request_body_ref = ref.replace("#/components/requestBodies/", "")
                        if "components" in spec and "requestBodies" in spec["components"]:
                            rb = spec["components"]["requestBodies"].get(request_body_ref, {})
                            if "content" in rb:
                                for media_type in rb["content"].values():
                                    if "schema" in media_type:
                                        endpoint_refs.update(get_schema_refs(media_type["schema"]))
                else:
                    content = request_body.get("content", {})
                    for media_type in content.values():
                        if "schema" in media_type:
                            endpoint_refs.update(get_schema_refs(media_type["schema"]))

            # Check responses
            for response in details.get("responses", {}).values():
                if "$ref" in response:
                    ref = response["$ref"]
                    if ref.startswith("#/components/responses/"):
                        # Look up the referenced response
                        response_ref = ref.replace("#/components/responses/", "")
                        if "components" in spec and "responses" in spec["components"]:
                            resp = spec["components"]["responses"].get(response_ref, {})
                            if "content" in resp:
                                for media_type in resp["content"].values():
                                    if "schema" in media_type:
                                        endpoint_refs.update(get_schema_refs(media_type["schema"]))
                elif "content" in response:
                    for media_type in response["content"].values():
                        if "schema" in media_type:
                            endpoint_refs.update(get_schema_refs(media_type["schema"]))

    return endpoint_refs


def build_schema_tree(spec: dict) -> Tuple[dict, set]:
    """Build a tree of schema dependencies."""
    if "components" not in spec or "schemas" not in spec["components"]:
        return {}

    schemas = spec["components"]["schemas"]
    tree = {}

    # First pass: collect all schema references from endpoints
    endpoint_refs = get_endpoint_schema_refs(spec)

    # Second pass: build the dependency tree
    referenced_schemas = set()  # Track all schemas that are referenced somewhere

    def add_to_tree(schema_name: str, visited: set = None):
        if visited is None:
            visited = set()

        if schema_name in visited:
            return

        visited.add(schema_name)

        if schema_name not in schemas:
            print(f"Schema '{schema_name}' not found in the specification")
            return

        schema = schemas[schema_name]
        refs = get_schema_refs(schema)

        # Mark this schema as referenced
        if schema_name in endpoint_refs:
            referenced_schemas.add(schema_name)

        # Mark all referenced schemas as referenced
        referenced_schemas.update(refs)

        tree[schema_name] = {
            "refs": refs,
            "is_top_level": schema_name in endpoint_refs,
            "back_refs": set(),
            "endpoint_reachable": False,
        }

        for ref in refs:
            add_to_tree(ref, visited)

    # Start with all schemas
    for schema_name in schemas:
        add_to_tree(schema_name)

    # Updated all back-refs
    for schema_name, info in tree.items():
        for ref in info["refs"]:
            tree[ref]["back_refs"].add(schema_name)

    reachable_schemas = set()
    changes = True
    while changes:
        changes = False
        for schema_name, info in tree.items():
            reachable = schema_name in endpoint_refs
            for back_ref in info["back_refs"]:
                if back_ref in endpoint_refs or back_ref in reachable_schemas:
                    reachable = True
                    break

            if reachable and not info["endpoint_reachable"]:
                changes = True
                info["endpoint_reachable"] = True
                reachable_schemas.add(schema_name)

    # Find unused schemas (not referenced by endpoints or other schemas)
    unused_schemas = set(schemas.keys()) - reachable_schemas
    if unused_schemas:
        print_verbose("\n[bold yellow]Found unused schemas:[/bold yellow]")
        for schema_name in sorted(unused_schemas):
            print_verbose(f"  - {schema_name}")
            # Mark unused schemas as top-level
            tree[schema_name] = {"refs": set(), "is_top_level": True}

    return tree, unused_schemas


def print_schema_tree(tree: dict, current: str = None, level: int = 0, visited: set = None):
    """Print the schema tree in a tree-like structure."""
    if visited is None:
        visited = set()

    if current is None:
        # Start with top-level schemas
        for schema_name, info in tree.items():
            if info["is_top_level"]:
                print_schema_tree(tree, schema_name, 0, visited)
    else:
        if current in visited:
            return

        visited.add(current)

        # Print current schema
        prefix = "│   " * (level - 1) + "├── " if level > 0 else ""
        print_verbose(f"{prefix}[cyan]{current}[/cyan]")

        # Print dependencies
        for ref in sorted(tree[current]["refs"]):
            print_schema_tree(tree, ref, level + 1, visited)


@app.command(name="schema-tree")
def schema_tree(spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)")):
    """Display schema dependencies in a tree structure."""
    try:
        spec = load_openapi_spec(spec_file)

        if "components" not in spec or "schemas" not in spec["components"]:
            print_verbose("No schemas found in the OpenAPI specification")
            return

        tree, unused_schemas = build_schema_tree(spec)

        if not tree:
            print_verbose("No schema dependencies found")
            return

        print_verbose("\n[bold magenta]Schema Dependency Tree[/bold magenta]")
        print_verbose("Top-level schemas (used directly in endpoints) are shown at the root level")
        print_verbose("Dependent schemas are shown as children\n")
        print_verbose(f"Unused schemas: {', '.join(sorted(unused_schemas))}", style="bold yellow")

        print_schema_tree(tree)

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def remove_unused_schemas(spec: dict) -> dict:
    """Remove schemas that are not used in any endpoints or by other schemas."""
    if "components" not in spec or "schemas" not in spec["components"]:
        return spec

    def remove_schemas(schemas: dict, schemas_to_remove: set, reason: str = "") -> None:
        """Helper function to remove schemas and print information."""
        if schemas_to_remove:
            print_verbose(
                f"Removing {len(schemas_to_remove)} unused schemas:",
                style="bold yellow",
            )
            for schema_name in sorted(schemas_to_remove):
                message = f"  - {schema_name}"
                if reason:
                    message += f" ({reason})"
                print_verbose(message, style="yellow")
                del schemas[schema_name]

    # Build the schema tree to identify used schemas
    tree, unused_schemas = build_schema_tree(spec)
    schemas = spec["components"]["schemas"]
    remove_schemas(schemas, unused_schemas, "only referenced by removed schemas")

    return spec


def remove_invalid_components(spec: dict) -> dict:
    """Remove components (request bodies and responses) that reference non-existent schemas."""
    if "components" not in spec:
        return spec

    schemas = spec["components"].get("schemas", {})
    components_to_check = {"requestBodies": "request body", "responses": "response"}

    def check_schema_refs(obj):
        """Recursively check for invalid schema references."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref.startswith("#/components/schemas/"):
                    schema_name = ref.replace("#/components/schemas/", "")
                    if schema_name not in schemas:
                        return True
            for value in obj.values():
                if check_schema_refs(value):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if check_schema_refs(item):
                    return True
        return False

    # Check each component type
    for component_type, component_name in components_to_check.items():
        if component_type not in spec["components"]:
            continue

        components = spec["components"][component_type]
        invalid_components = set()

        # Check each component
        for comp_name, comp in components.items():
            if check_schema_refs(comp):
                invalid_components.add(comp_name)

        # Remove invalid components
        if invalid_components:
            print_verbose(
                f"Removing {len(invalid_components)} invalid {component_name}s:",
                style="bold yellow",
            )
            for comp_name in sorted(invalid_components):
                print_verbose(f"  - {comp_name} (references non-existent schemas)", style="yellow")
                del components[comp_name]

    return spec


def reorder_spec(spec: dict) -> dict:
    """Reorder the OpenAPI specification with standard ordering and alphabetical sorting.

    - Top-level keys are arranged in the standard OpenAPI order
    - Paths are sorted alphabetically
    - Schemas are sorted alphabetically
    """
    # Define the standard order of top-level keys
    standard_order = [
        "openapi",
        "info",
        "jsonSchemaDialect",
        "servers",
        "paths",
        "webhooks",
        "components",
        "security",
        "tags",
        "externalDocs",
    ]

    # Create a new ordered dictionary
    ordered_spec = {}

    # Add keys in the standard order, if they exist in the original spec
    for key in standard_order:
        if key in spec:
            ordered_spec[key] = spec[key]

    # Add any remaining keys that weren't in the standard order
    for key in spec:
        if key not in ordered_spec:
            ordered_spec[key] = spec[key]

    # Sort paths alphabetically
    if "paths" in ordered_spec:
        ordered_spec["paths"] = {k: ordered_spec["paths"][k] for k in sorted(ordered_spec["paths"].keys())}

    # Sort schemas alphabetically
    if "components" in ordered_spec and "schemas" in ordered_spec["components"]:
        ordered_spec["components"]["schemas"] = {
            k: ordered_spec["components"]["schemas"][k] for k in sorted(ordered_spec["components"]["schemas"].keys())
        }

    return ordered_spec


def hoist_nested_defs(spec: dict) -> dict:
    """Hoist `$defs` nested inside inline schemas into top-level `components.schemas`.

    Pydantic's `model_json_schema(ref_template="#/components/schemas/{model}")` emits
    refs pointing at `#/components/schemas/...` but collects the nested model
    definitions under a `$defs` key on the schema itself. FastAPI splices the result
    into `openapi_extra` as-is, so those `$defs` stay inline even though the refs
    target the global components map.

    In the current spec, every nested filter type (DatetimeFilter, FilesetRef, etc.)
    also lives at the top level via some response_model, so the refs already resolve
    and this pass mostly strips redundant `$defs`. It becomes load-bearing the moment
    a filter introduces a nested type that isn't referenced elsewhere — without the
    hoist those refs would dangle, and the `validate_refs` gate at the end of the
    pipeline would fail the build.

    Collision handling: if a same-named schema already exists at the top level and is
    structurally identical, the nested copy is discarded. If it exists and differs, a
    ValueError is raised so mismatches surface immediately rather than silently
    clobbering one definition with another.
    """
    if "components" not in spec:
        spec["components"] = {}
    if "schemas" not in spec["components"]:
        spec["components"]["schemas"] = {}

    global_defs = spec["components"]["schemas"]
    hoisted = []

    def _strip_null_defaults(schema):
        """Return a copy of *schema* with every ``default: null`` property removed.

        Used only for the hoist equality check, not for the emitted output. See
        the collision branch in ``hoist_from`` for why this exists.
        """
        if isinstance(schema, dict):
            return {k: _strip_null_defaults(v) for k, v in schema.items() if not (k == "default" and v is None)}
        if isinstance(schema, list):
            return [_strip_null_defaults(v) for v in schema]
        return schema

    def hoist_from(local_schema: dict, source: str) -> None:
        """Move any `$defs` inside `local_schema` into `global_defs`.

        Pops the `$defs` block (if present), then for each nested definition:
          - not yet global                 → promote into `global_defs`
          - global and identical           → drop the local copy (already covered)
          - global and differs only in
            ``default: null`` properties   → keep the global copy (see below)
          - global and differs otherwise   → raise ValueError (name collision)

        The ``default: null`` tolerance exists because pydantic emits the same
        nested model (e.g. ``DatetimeFilter``) with or without ``default: null``
        on its properties depending on how it was reached — as a top-level
        ``model_json_schema`` call vs. nested under an ``Optional[X] = None``
        field. Both are semantically identical (required-ness is set by the
        parent's ``required`` array, not by ``default``), but the byte diff
        causes this hoist to falsely collide.

        To revert: delete ``_strip_null_defaults`` and the branch below; the
        hoist will go back to a strict ``!=`` check.

        `source` is a breadcrumb used only in log output and error messages.
        """
        if not isinstance(local_schema, dict):
            return
        local_defs = local_schema.pop("$defs", None)
        if not local_defs:
            return
        for name, local_def in local_defs.items():
            if name in global_defs:
                if global_defs[name] == local_def:
                    continue
                if _strip_null_defaults(global_defs[name]) == _strip_null_defaults(local_def):
                    # Structurally equivalent modulo cosmetic ``default: null`` — keep the
                    # version already in global_defs so emitted schemas stay stable.
                    continue
                raise ValueError(f"Schema '{name}' hoisted from {source} conflicts with existing global definition")
            global_defs[name] = local_def
            hoisted.append((name, source))

    # Walk parameters in path operations — the common case: Pydantic emits
    # $defs on the inline filter schema spliced into openapi_extra.
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            for idx, param in enumerate(operation.get("parameters", [])):
                if isinstance(param, dict) and isinstance(param.get("schema"), dict):
                    hoist_from(param["schema"], f"{method.upper()} {path} param[{idx}]")

    # Also sweep existing global schemas, in case $defs leaked in via another path.
    for schema_name, schema in list(global_defs.items()):
        if isinstance(schema, dict):
            hoist_from(schema, f"components.schemas.{schema_name}")

    if hoisted:
        print_verbose(
            f"Hoisted {len(hoisted)} nested $defs into components.schemas",
            style="bold green",
        )
        for name, source in hoisted:
            print_verbose(f"  - {name} (from {source})", style="green")

    return spec


_INTERNAL_FRAGMENT_PREFIX = "#/"


def _decode_json_pointer_token(token: str) -> str:
    """Unescape a single JSON Pointer reference token per RFC 6901 §4.

    JSON Pointer defines two escape sequences inside a path segment:
      - `~1` stands in for `/` (the segment delimiter, which otherwise can't appear in a token)
      - `~0` stands in for `~` (the escape character itself)

    The `~1` substitution must run before `~0`. Otherwise a literal `~01` —
    which should round-trip to `~1` — would first be expanded to `~1` by the
    `~0` pass and then corrupted into `/` by the `~1` pass.
    """
    return token.replace("~1", "/").replace("~0", "~")


def validate_refs(spec: dict) -> list[str]:
    """Return a list of `$ref` targets in the spec that do not resolve.

    Scans every `$ref` under `paths` and `components` recursively. A ref is considered
    dangling if its target (e.g. `#/components/schemas/Foo`) is not present in the spec.
    Intended to be run at the end of the postprocessing pipeline so that SDK generators
    never see a spec with dangling imports.
    """
    dangling: list[str] = []

    def resolve(ref: str) -> bool:
        # Only internal URI-fragment refs are in scope. Refs without the `#/`
        # prefix (external URLs, opaque fragments, `schemas/foo.yaml#/...`) are
        # treated as valid by default since we can't resolve them from here.
        if not ref.startswith(_INTERNAL_FRAGMENT_PREFIX):
            return True

        # Parse as a JSON Pointer fragment: strip the `#/` marker, split on `/`
        # into reference tokens, and unescape each token per RFC 6901.
        body = ref.removeprefix(_INTERNAL_FRAGMENT_PREFIX)
        tokens = [_decode_json_pointer_token(t) for t in body.split("/")]

        node = spec
        for token in tokens:
            if isinstance(node, dict) and token in node:
                node = node[token]
            else:
                return False
        return True

    def walk(obj, path: str) -> None:
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if isinstance(ref, str) and not resolve(ref):
                dangling.append(f"{path}: {ref}")
            for key, value in obj.items():
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                walk(item, f"{path}[{idx}]")

    walk(spec.get("paths", {}), "paths")
    walk(spec.get("components", {}), "components")

    return dangling


def fix_recursive_schemas(spec: dict) -> dict:
    """Fix schemas that have recursive references in additionalProperties.

    Finds schemas where a property has additionalProperties with a $ref to the same schema type.
    For these cases, removes the additionalProperties and leaves the property as a generic object.
    """
    if "components" not in spec or "schemas" not in spec["components"]:
        return spec

    schemas = spec["components"]["schemas"]
    fixed_schemas = []

    # Helper function to get schema name from a $ref
    def get_schema_name_from_ref(ref):
        if ref.startswith("#/components/schemas/"):
            return ref.split("/")[-1]
        return ref

    for schema_name, schema in schemas.items():
        if "properties" not in schema:
            continue

        for prop_name, prop_def in schema["properties"].items():
            # Case 1: Direct additionalProperties with a $ref to the same schema (self-reference)
            if (
                "additionalProperties" in prop_def
                and isinstance(prop_def["additionalProperties"], dict)
                and "$ref" in prop_def["additionalProperties"]
            ):
                ref_schema_name = get_schema_name_from_ref(prop_def["additionalProperties"]["$ref"])

                # Check if this is a recursive reference to the same schema
                if ref_schema_name == schema_name:
                    # Log the fix
                    print_verbose(
                        f"Fixing recursive schema in '{schema_name}.{prop_name}' by removing additionalProperties",
                        style="bold yellow",
                    )

                    # Remove the additionalProperties and convert to generic object
                    del prop_def["additionalProperties"]
                    prop_def["type"] = "object"
                    fixed_schemas.append(f"{schema_name}.{prop_name}")

            # Dealing with the case where there is an anyOf
            elif "anyOf" in prop_def:
                for any_of in prop_def["anyOf"]:
                    if "$ref" in any_of:
                        ref_schema_name = get_schema_name_from_ref(any_of["$ref"])
                        if ref_schema_name == schema_name:
                            # Log the fix
                            print_verbose(
                                f"Fixing recursive schema in '{schema_name}.{prop_name}' by removing additionalProperties",
                                style="bold yellow",
                            )
                            del any_of["$ref"]
                            any_of["type"] = "object"
                            fixed_schemas.append(f"{schema_name}.{prop_name}")
                            any_of["title"] = schema.get("title", prop_name)
                            any_of["description"] = schema.get("description", "")

    if fixed_schemas:
        print_verbose(
            f"Fixed {len(fixed_schemas)} recursive schema references",
            style="bold green",
        )

    return spec


def update_object_type(spec: dict) -> dict:
    """
    Add `additionalProperties: true` to all schemas with `type: object` that don't have any properties defined.

    The `additionalProperties: true` is default (based on OpenAPI 3.1.0) for object types
    (see https://swagger.io/docs/specification/v3_0/data-models/data-types/#objects).

    However, some tools (e.g. Stainless, which we use for generating client SDKs) uses a different default.
    This leads to issues, where a field with type `dict[str, Any]` in Python code generates as `object` in client SDKs,

    See https://www.stainless.com/docs/reference/diagnostics#Schema/ObjectHasNoProperties
    """
    if "components" not in spec or "schemas" not in spec["components"]:
        return spec

    schemas = spec["components"]["schemas"]
    updated_count = 0

    def update_object_in_schema(obj):
        """Recursively update object types in a schema."""
        nonlocal updated_count

        if not isinstance(obj, dict):
            return obj

        is_object_type = obj.get("type") == "object"
        # Update only if it has no properties defined
        if is_object_type and "additionalProperties" not in obj and "properties" not in obj:
            obj["additionalProperties"] = True
            updated_count += 1

        # Recursively process all properties
        for key, value in obj.items():
            if isinstance(value, dict):
                obj[key] = update_object_in_schema(value)
            elif isinstance(value, list):
                obj[key] = [update_object_in_schema(item) for item in value]

        return obj

    for schema_name, schema in schemas.items():
        if isinstance(schema, dict):
            schemas[schema_name] = update_object_in_schema(schema)

    if updated_count > 0:
        print_verbose(
            f"Added additionalProperties: true to {updated_count} object type schemas",
            style="bold green",
        )
    else:
        print_verbose(
            "No object type schemas found that needed additionalProperties",
            style="bold yellow",
        )

    return spec


def mark_direct_span_json_value_for_stainless(spec: dict) -> dict:
    """Mark the direct-span API's intentionally unconstrained JsonValue for Stainless."""

    schemas = spec.get("components", {}).get("schemas", {})
    if "DirectSpanInput" not in schemas:
        return spec
    json_value = schemas.get("JsonValue")
    if isinstance(json_value, dict):
        json_value["x-stainless-any"] = True
    return spec


@app.command(name="fix-schema")
def fix_schema(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Fix common schema issues in OpenAPI specification."""
    try:
        spec = load_openapi_spec(spec_file)
        spec = tweak_spec(spec)
        spec = remove_unused_schemas(spec)
        spec = remove_invalid_components(spec)
        spec = fix_recursive_schemas(spec)
        spec = update_object_type(spec)
        spec = mark_direct_span_json_value_for_stainless(spec)

        # Make sure the version is set to 3.1.0
        spec["openapi"] = "3.1.0"

        # OpenAPI requires info.version; package versions come from build metadata.
        spec["info"]["version"] = OPENAPI_SPEC_VERSION

        # Reorder the top-level keys in the specification
        spec = reorder_spec(spec)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose("Schema fixes applied", style="bold green")

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1) from e


def rename_schema_references(spec: dict, old_name: str, new_name: str) -> None:
    """Rename all references to a schema throughout the specification."""

    def update_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and obj["$ref"] == f"#/components/schemas/{old_name}":
                obj["$ref"] = f"#/components/schemas/{new_name}"
            for value in obj.values():
                update_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                update_refs(item)

    # Update references in paths
    for path in spec.get("paths", {}).values():
        for method in path.values():
            # Update request body references
            if "requestBody" in method:
                content = method["requestBody"].get("content", {})
                for media_type in content.values():
                    if "schema" in media_type:
                        update_refs(media_type["schema"])

            # Update response references
            for response in method.get("responses", {}).values():
                if "content" in response:
                    for media_type in response["content"].values():
                        if "schema" in media_type:
                            update_refs(media_type["schema"])

    # Update references in components
    if "components" in spec:
        for component_type in spec["components"].values():
            if isinstance(component_type, dict):
                for component in component_type.values():
                    update_refs(component)


@app.command(name="rename-schema")
def rename_schema(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    old_name: str = typer.Argument(..., help="Current name of the schema to rename"),
    new_name: str = typer.Argument(..., help="New name for the schema"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Rename a schema and update all its references in the OpenAPI specification."""
    try:
        spec = load_openapi_spec(spec_file)

        if "components" not in spec or "schemas" not in spec["components"]:
            print_verbose("No schemas found in the OpenAPI specification", style="bold red")
            raise typer.Exit(1)

        schemas = spec["components"]["schemas"]
        if old_name not in schemas:
            print_verbose(f"Schema '{old_name}' not found in the specification", style="bold red")
            raise typer.Exit(1)

        if new_name in schemas:
            print_verbose(
                f"Schema '{new_name}' already exists in the specification",
                style="bold red",
            )
            raise typer.Exit(1)

        # Rename the schema
        schemas[new_name] = schemas.pop(old_name)

        # Update all references
        rename_schema_references(spec, old_name, new_name)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Schema '{old_name}' renamed to '{new_name}' and saved to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


@app.command(name="remove-schema")
def remove_schema_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    name: str = typer.Argument(..., help="Name of the schema to remove (e.g., CustomizationConfigFoo)"),
    replacement_name: str = typer.Argument(default=..., help="Name to update all references to."),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Remove a schema and update all its references in the OpenAPI specification.
    If replacement_name is provided, updates all references to the schema with the new name.
    """
    try:
        spec = load_openapi_spec(spec_file)

        if "components" not in spec or "schemas" not in spec["components"]:
            print_verbose("No schemas found in the OpenAPI specification", style="bold red")
            raise typer.Exit(1)

        schemas = spec["components"]["schemas"]
        if name not in schemas:
            print_verbose(f"Schema '{name}' not found in the specification", style="bold red")
            raise typer.Exit(1)

        # Remove the schema
        del schemas[name]

        # Update all references to the schema
        rename_schema_references(spec, name, replacement_name)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Schema '{name}' removed and references updated to '{replacement_name}'",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def remove_endpoint(spec: dict, path: str, method: Optional[str] = None, prune_unused: bool = True) -> None:
    """Remove an endpoint and its associated schemas from the specification.
    If method is None, removes all methods for the given path."""
    if "paths" not in spec:
        print_verbose("No paths found in the OpenAPI specification", style="bold yellow")
        return

    if path not in spec["paths"]:
        print_verbose(
            f"Warning: Path '{path}' not found in the specification",
            style="bold yellow",
        )
        return

    if method:
        if method.lower() not in spec["paths"][path]:
            print_verbose(
                f"Warning: Method '{method}' not found for path '{path}'",
                style="bold yellow",
            )
            return
        # Remove specific method
        del spec["paths"][path][method.lower()]
    else:
        # Remove all methods for the path
        del spec["paths"][path]

    if prune_unused:
        remove_unused_schemas(spec)


@app.command(name="remove-endpoint")
def remove_endpoint_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    path: str = typer.Argument(..., help="Path of the endpoint to remove (e.g., /api/v1/users)"),
    method: Optional[str] = typer.Option(
        None,
        help="HTTP method to remove (e.g., GET, POST). If not provided, removes all methods for the path.",
    ),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Remove an endpoint and its unused schemas from the OpenAPI specification.
    If no method is specified, removes all methods for the given path."""
    try:
        spec = load_openapi_spec(spec_file)
        remove_endpoint(spec, path, method)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        if method:
            print_verbose(
                f"Endpoint {method} {path} removed and saved to {output_path}",
                style="bold green",
            )
        else:
            print_verbose(
                f"All methods for path {path} removed and saved to {output_path}",
                style="bold green",
            )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def merge_specs(specs_with_files: list[tuple[dict, str]], keep_versions: bool = False) -> dict:
    """Merge multiple OpenAPI specifications in order, preserving the first occurrence of each schema.

    Args:
        specs_with_files: List of tuples containing (spec, filename) for each specification to merge in order
        keep_versions: If True, when encountering duplicate schemas, create a new version with a suffix
                      derived from the source file name instead of skipping it

    Returns:
        Merged OpenAPI specification

    Raises:
        ValueError: If there are conflicting schema definitions
    """
    if not specs_with_files:
        raise ValueError("No specifications provided")

    # Start with the first spec as base
    merged, first_file = specs_with_files[0]
    merged = merged.copy()

    # Initialize components if not present
    if "components" not in merged:
        merged["components"] = {}
    if "schemas" not in merged["components"]:
        merged["components"]["schemas"] = {}
    if "paths" not in merged:
        merged["paths"] = {}

    # Track which schemas we've seen to detect conflicts
    seen_schemas = set(merged["components"]["schemas"].keys())
    schema_sources = {name: first_file for name in seen_schemas}

    # Merge each subsequent spec
    for spec, filename in specs_with_files[1:]:
        # Add visual separator before merging each additional spec
        print_verbose("\n" + "─" * 80, style="bold blue")
        print_verbose(f"Merging specification from: {filename}", style="bold blue")
        print_verbose("─" * 80 + "\n", style="bold blue")

        # Get file prefix for versioning (first two letters)
        file_prefix = Path(filename).stem[:2].upper()

        # Merge paths
        for path, methods in spec.get("paths", {}).items():
            if path not in merged["paths"]:
                merged["paths"][path] = methods
            else:
                # Merge methods for existing paths
                for method, details in methods.items():
                    if method not in merged["paths"][path]:
                        merged["paths"][path][method] = details

        # Merge components
        if "components" in spec:
            for component_type, components in spec["components"].items():
                if component_type not in merged["components"]:
                    merged["components"][component_type] = {}

                # Special handling for schemas to prevent overrides
                if component_type == "schemas":
                    for schema_name, schema in components.items():
                        if schema_name in seen_schemas:
                            if keep_versions:
                                # Compare schemas to see if they're different
                                existing_schema = merged["components"]["schemas"][schema_name]
                                has_differences = compare_schemas(
                                    existing_schema,
                                    schema,
                                    schema_name,
                                    schema_sources[schema_name],
                                    filename,
                                )
                                if has_differences:
                                    # Create new version of the schema
                                    new_schema_name = f"{schema_name}{file_prefix}"
                                    print_verbose(
                                        f"Creating new version of schema '{schema_name}' as '{new_schema_name}' "
                                        f"(from {filename})",
                                        style="bold yellow",
                                    )

                                    # Update all references to this schema in the current spec
                                    rename_schema_references(spec, schema_name, new_schema_name)

                                    # Add the new version
                                    merged["components"]["schemas"][new_schema_name] = schema
                                    seen_schemas.add(new_schema_name)
                                    schema_sources[new_schema_name] = filename
                                else:
                                    print_verbose(
                                        f"Skipping duplicate schema '{schema_name}' from {filename} "
                                        f"(identical to version in {schema_sources[schema_name]})",
                                        style="bold yellow",
                                    )
                            else:
                                print_verbose(
                                    f"Warning: Schema '{schema_name}' is already defined in {schema_sources[schema_name]} "
                                    f"and will be skipped (found in {filename})",
                                    style="bold yellow",
                                )
                                # Compare the schemas and show differences
                                existing_schema = merged["components"]["schemas"][schema_name]
                                compare_schemas(
                                    existing_schema,
                                    schema,
                                    schema_name,
                                    schema_sources[schema_name],
                                    filename,
                                )
                                continue
                        else:
                            merged["components"]["schemas"][schema_name] = schema
                            seen_schemas.add(schema_name)
                            schema_sources[schema_name] = filename
                else:
                    # For other component types, just add if not present
                    for name, component in components.items():
                        if name not in merged["components"][component_type]:
                            merged["components"][component_type][name] = component

    # Display schema source summary table
    if schema_sources:
        table = Table(title="Schema Sources", show_header=True, header_style="bold magenta")
        table.add_column("Schema Name", style="cyan")

        # Group schemas by source file
        schemas_by_source = {}
        for schema_name, source_file in schema_sources.items():
            if source_file not in schemas_by_source:
                schemas_by_source[source_file] = []
            schemas_by_source[source_file].append(schema_name)

        # Add rows grouped by source file
        for source_file in sorted(schemas_by_source.keys()):
            # Add source file header
            table.add_row(f"[bold green]{source_file}[/bold green]", "")
            # Add schemas from this source
            for schema_name in sorted(schemas_by_source[source_file]):
                table.add_row(f"  {schema_name}", "")
            # Add separator after each group
            table.add_row("", "")

        print_verbose("\n")  # Add some spacing before the table
        print_verbose(table)

    return merged


@app.command(name="merge")
def merge_specs_command(
    spec_files: list[str] = typer.Argument(..., help="List of OpenAPI specification files to merge (in order)"),
    output_file: str = typer.Option("openapi.yaml", help="Output file path for the merged specification"),
    keep_versions: bool = typer.Option(False, help="Keep multiple versions of schemas when merging"),
):
    """Merge multiple OpenAPI specifications in order, preserving the first occurrence of each schema.
    If a schema is defined in multiple specs, only the first occurrence is kept unless --keep is specified.
    With --keep, different versions of the same schema will be kept with a suffix derived from the source file name.
    """
    try:
        # Load all specs with their file names
        specs_with_files = [(load_openapi_spec(file), file) for file in spec_files]

        # Merge specs
        merged = merge_specs(specs_with_files, keep_versions)

        # Save the merged spec
        save_openapi_spec(merged, output_file)

        print_verbose(
            f"Successfully merged {len(spec_files)} specifications into {output_file}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def compare_schemas(schema1: dict, schema2: dict, schema_name: str, file1: str, file2: str) -> bool:
    """Compare two schemas and display differences in a table format.

    Args:
        schema1: First schema to compare
        schema2: Second schema to compare
        schema_name: Name of the schema being compared
        file1: Name of the file containing the first schema
        file2: Name of the file containing the second schema

    Returns:
        bool: True if schemas are different, False if they are identical
    """
    table = Table(
        title=f"Schema Differences for '{schema_name}'",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field", style="cyan")
    table.add_column(file1, style="green")
    table.add_column(file2, style="yellow")

    # Get all unique fields from both schemas
    all_fields = set()
    has_differences = False

    # Helper function to get field type
    def get_field_type(schema: dict, field: str) -> str:
        if field not in schema:
            return "MISSING"
        field_schema = schema[field]
        if isinstance(field_schema, dict):
            if "$ref" in field_schema:
                return f"ref: {field_schema['$ref']}"
            if "type" in field_schema:
                type_str = field_schema["type"]
                if "format" in field_schema:
                    type_str += f" ({field_schema['format']})"
                return type_str
            return "object"
        return str(type(field_schema).__name__)

    # Compare properties
    if "properties" in schema1:
        all_fields.update(schema1["properties"].keys())
    if "properties" in schema2:
        all_fields.update(schema2["properties"].keys())

    # Compare required fields
    required1 = set(schema1.get("required", []))
    required2 = set(schema2.get("required", []))
    all_fields.update(required1)
    all_fields.update(required2)

    # Add rows for each field
    for field in sorted(all_fields):
        field_type1 = get_field_type(schema1.get("properties", {}), field)
        field_type2 = get_field_type(schema2.get("properties", {}), field)

        # Add required indicator
        required_indicator1 = " (required)" if field in required1 else ""
        required_indicator2 = " (required)" if field in required2 else ""

        # Only add row if there are differences
        if field_type1 != field_type2 or (field in required1) != (field in required2):
            has_differences = True
            table.add_row(
                field,
                f"{field_type1}{required_indicator1}",
                f"{field_type2}{required_indicator2}",
            )

    # Compare additional schema properties
    schema_properties = {
        "type": "Schema Type",
        "description": "Description",
        "format": "Format",
        "enum": "Enum Values",
        "minimum": "Minimum",
        "maximum": "Maximum",
        "pattern": "Pattern",
        "minLength": "Min Length",
        "maxLength": "Max Length",
        "items": "Items Type",
    }

    for prop, display_name in schema_properties.items():
        value1 = schema1.get(prop)
        value2 = schema2.get(prop)
        if value1 != value2:
            has_differences = True
            table.add_row(
                display_name,
                str(value1) if value1 is not None else "MISSING",
                str(value2) if value2 is not None else "MISSING",
            )

    # Only print table if there are differences
    if table.row_count > 0:
        print_verbose("\n")  # Add spacing before table
        print_verbose(table)
        print_verbose("\n")  # Add spacing after table

    return has_differences


def copy_tags(source_spec: dict, target_spec: dict) -> dict:
    """Copy all tags from source spec to target spec, overwriting existing tags.

    Args:
        source_spec: Source OpenAPI specification to copy tags from
        target_spec: Target OpenAPI specification to copy tags to

    Returns:
        Modified target specification with copied tags
    """
    # Copy global tags list if it exists
    if "tags" in source_spec:
        target_spec["tags"] = source_spec["tags"]
        print_verbose("Copied global tags list", style="bold green")

    # Copy tags from each endpoint
    if "paths" in source_spec and "paths" in target_spec:
        for path, methods in source_spec["paths"].items():
            if path in target_spec["paths"]:
                for method, details in methods.items():
                    if method in target_spec["paths"][path] and "tags" in details:
                        target_spec["paths"][path][method]["tags"] = details["tags"]
                        print_verbose(
                            f"Copied tags for {method.upper()} {path}",
                            style="bold green",
                        )

    return target_spec


@app.command(name="copy-tags")
def copy_tags_command(
    source_file: str = typer.Argument(..., help="Source OpenAPI specification file to copy tags from"),
    target_file: str = typer.Argument(..., help="Target OpenAPI specification file to copy tags to"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the target file."
    ),
):
    """Copy all tags from one OpenAPI specification to another, overwriting existing tags.
    This includes both the global tags list and tags associated with each endpoint."""
    try:
        # Load both specs
        source_spec = load_openapi_spec(source_file)
        target_spec = load_openapi_spec(target_file)

        # Copy tags
        target_spec = copy_tags(source_spec, target_spec)

        # Save the modified spec
        output_path = output_file or target_file
        save_openapi_spec(target_spec, output_path)

        print_verbose(
            f"Successfully copied tags from {source_file} to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def order_endpoints_by_tags(spec: dict) -> dict:
    """Order endpoints in the OpenAPI specification according to the order of tags.

    Args:
        spec: The OpenAPI specification dictionary

    Returns:
        The modified specification with endpoints ordered by tags
    """
    if "tags" not in spec or "paths" not in spec:
        return spec

    # Create a mapping of tag names to their order
    tag_order = {tag["name"]: idx for idx, tag in enumerate(spec["tags"])}

    # Create a list of (path, methods) tuples with their tag order
    path_methods = []
    for path, methods in spec["paths"].items():
        # Find the first tag from any method in this path
        first_tag = None
        for method_details in methods.values():
            if "tags" in method_details and method_details["tags"]:
                first_tag = method_details["tags"][0]
                break

        # If no tag found, use a high number to put untagged paths at the end
        order = tag_order.get(first_tag, float("inf"))
        path_methods.append((order, path, methods))

    # Sort paths by tag order
    path_methods.sort(key=lambda x: x[0])

    # Rebuild the paths dictionary in the new order
    spec["paths"] = {path: methods for _, path, methods in path_methods}

    return spec


@app.command(name="order-endpoints-by-tags")
def order_endpoints_by_tags_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Order endpoints in the OpenAPI specification according to the order of tags.
    Endpoints without tags will be placed at the end."""
    try:
        spec = load_openapi_spec(spec_file)
        spec = order_endpoints_by_tags(spec)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Successfully ordered endpoints by tags and saved to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def load_examples_file(file_path: str) -> list:
    """Load examples from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File {file_path} does not exist")

    with open(path, "r") as f:
        if path.suffix == ".json":
            return json.load(f)
        else:
            raise typer.BadParameter("Examples file must be JSON format")


def include_examples(spec: dict, examples: list) -> dict:
    """Include examples in the OpenAPI specification."""
    for example in examples:
        endpoint = example["endpoint"]
        method = example["method"].lower()

        # Find the path in the spec
        if endpoint not in spec.get("paths", {}):
            print_verbose(
                f"Warning: Endpoint {endpoint} not found in specification",
                style="bold yellow",
            )
            continue

        if method not in spec["paths"][endpoint]:
            print_verbose(
                f"Warning: Method {method} not found for endpoint {endpoint}",
                style="bold yellow",
            )
            continue

        # Get the operation
        operation = spec["paths"][endpoint][method]

        # Add request example if available
        if "request" in example:
            # Ensure requestBody structure exists
            if "requestBody" not in operation:
                operation["requestBody"] = {}
            if "content" not in operation["requestBody"]:
                operation["requestBody"]["content"] = {}
            if "application/json" not in operation["requestBody"]["content"]:
                operation["requestBody"]["content"]["application/json"] = {}
            if "examples" not in operation["requestBody"]["content"]["application/json"]:
                operation["requestBody"]["content"]["application/json"]["examples"] = {}

            # Add request example
            request_example = {
                "summary": example["request"].get("title", ""),
                "description": example["request"].get("description", ""),
                "value": example["request"].get("body", {}),
            }

            # Create a unique example name
            title_suffix = (
                f"_{example['request'].get('title', '').lower().replace(' ', '_')}"
                if example["request"].get("title")
                else ""
            )
            example_name = f"{method}_{endpoint.replace('/', '_')}{title_suffix}_request"
            operation["requestBody"]["content"]["application/json"]["examples"][example_name] = request_example
            print_verbose(
                f"Added request example for {method.upper()} {endpoint}: {example_name}",
                style="bold green",
            )

        # Add response example if available
        if "response" in example:
            # Ensure responses structure exists
            if "responses" not in operation:
                operation["responses"] = {}

            # Use 200 as default status code if not specified
            status_code = "200"
            if status_code not in operation["responses"]:
                # If we have an example for 200, but the endpoint supports 201, we use the same
                if "201" in operation["responses"]:
                    status_code = "201"
                else:
                    operation["responses"][status_code] = {}
            if "content" not in operation["responses"][status_code]:
                operation["responses"][status_code]["content"] = {}

            content_type = example["response"].get("content_type", "application/json")
            if content_type not in operation["responses"][status_code]["content"]:
                operation["responses"][status_code]["content"][content_type] = {}
            if "examples" not in operation["responses"][status_code]["content"][content_type]:
                operation["responses"][status_code]["content"][content_type]["examples"] = {}

            # Add response example
            response_example = {
                "summary": example["response"].get("title", ""),
                "description": example["response"].get("description", ""),
                "value": example["response"].get("body", {}),
            }

            # Create a unique example name
            title_suffix = (
                f"_{example['response'].get('title', '').lower().replace(' ', '_')}"
                if example["response"].get("title")
                else ""
            )
            example_name = f"{method}_{endpoint.replace('/', '_')}{title_suffix}_response"
            operation["responses"][status_code]["content"][content_type]["examples"][example_name] = response_example
            print_verbose(
                f"Added response example for {method.upper()} {endpoint}: {example_name}",
                style="bold green",
            )

    return spec


@app.command(name="include-examples")
def include_examples_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    examples_file: str = typer.Argument(..., help="Path to JSON file containing examples"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Include examples from a JSON file in the OpenAPI specification."""
    try:
        # Load the OpenAPI spec
        spec = load_openapi_spec(spec_file)

        # Load the examples
        examples = load_examples_file(examples_file)

        # Include examples in the spec
        spec = include_examples(spec, examples)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Successfully included examples and saved to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def fix_ref_with_additional_props(spec: dict) -> dict:
    """Fix schemas that have both a $ref and additional properties.

    OpenAPI 3.1 does not allow additional properties alongside $ref.
    This function converts such cases to use allOf instead.
    """
    fixed_count = 0

    # Helper to recursively check and fix all objects in the spec
    def fix_object(obj):
        nonlocal fixed_count

        if not isinstance(obj, dict):
            return obj

        # Check if this is a schema with both $ref and other properties
        if "$ref" in obj and len(obj) > 1:
            ref_value = obj["$ref"]
            # Create a new object with allOf
            new_obj = {"allOf": [{"$ref": ref_value}]}

            # Copy all other properties to the new object
            for key, value in obj.items():
                if key != "$ref":
                    new_obj[key] = value

            fixed_count += 1
            return new_obj

        # Recursively process all properties
        for key, value in list(obj.items()):
            if isinstance(value, dict):
                obj[key] = fix_object(value)
            elif isinstance(value, list):
                obj[key] = [fix_object(item) if isinstance(item, (dict, list)) else item for item in value]

        return obj

    # Fix the entire spec
    fixed_spec = fix_object(spec)

    if fixed_count > 0:
        print_verbose(
            f"Fixed {fixed_count} schemas with $ref + additional properties using allOf",
            style="bold green",
        )

    return fixed_spec


@app.command(name="fix-ref-not-allowed")
def fix_ref_not_allowed_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Fix schemas that have both a $ref and additional properties by converting to allOf.

    In OpenAPI 3.1, a schema object cannot have both a $ref property and other properties.
    This command finds such cases and converts them to use allOf instead, which is allowed.
    """
    try:
        # Load the OpenAPI spec
        spec = load_openapi_spec(spec_file)

        # Fix schemas with $ref and additional properties
        spec = fix_ref_with_additional_props(spec)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Successfully fixed $ref issues and saved to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def fix_openai_streaming_endpoints(spec: dict) -> dict:
    """
    Fixes OpenAI streaming endpoints in the OpenAPI specification.

    The fix is converting this schema (which is what FastAPI generates).
    This is needed due to limitation in FastAPI, where it's not possible to provide the response models in a way that
    it generates schema for different content types correctly.

    Before:
    ```yaml
    /v1/completions:
      post:
        requestBody:
          ...
        responses:
          '200':
            description: Successful Response
            content:
              application/json:
                schema:
                  anyOf:
                  - $ref: '#/components/schemas/ChatCompletionResponse'
                  - $ref: '#/components/schemas/ChatCompletionStreamResponse'
    ```

    After:
    ```yaml
    /v1/completions:
      post:
      requestBody:
        ...
        responses:
          '200':
            description: Successful Response
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/ChatCompletionResponse'
              text/event-stream:
                schema:
                  $ref: '#/components/schemas/ChatCompletionStreamResponse'
    ```
    """

    def fix_response_content(path: str, status_code: str, content: dict) -> dict:
        """Fix response content that has anyOf with streaming schemas."""
        if not isinstance(content, dict):
            return content

        if "application/json" not in content:
            return content

        json_content = deepcopy(content["application/json"])
        if not isinstance(json_content, dict) or "schema" not in json_content:
            return content

        schema = json_content["schema"]

        # Make sure we have a valid anyOf schema with two items
        if (
            not isinstance(schema, dict)
            or "anyOf" not in schema
            or not isinstance(schema["anyOf"], list)
            or len(schema["anyOf"]) != 2
        ):
            return content

        non_streaming_ref = streaming_ref = None
        for ref_schema in schema["anyOf"]:
            if isinstance(ref_schema, dict) and "$ref" in ref_schema:
                ref = ref_schema["$ref"]
                if "stream" in ref.lower():
                    streaming_ref = ref
                else:
                    non_streaming_ref = ref

        if non_streaming_ref and streaming_ref:
            json_content["schema"].pop("anyOf")
            print_verbose(
                f"Fixed {path} {status_code}: {non_streaming_ref} {streaming_ref}",
                style="bold green",
            )
            return {
                "application/json": {
                    **json_content,
                    "schema": {"$ref": non_streaming_ref},
                },
                "text/event-stream": {"schema": {"$ref": streaming_ref}},
            }

        return content

    def fix_responses(path: str, responses: dict) -> dict:
        """Fix responses that contain schemas with anyOf for streaming."""
        if not isinstance(responses, dict):
            return responses

        fixed_responses = {}
        for status_code, response in responses.items():
            if isinstance(response, dict) and "content" in response:
                fixed_responses[status_code] = {
                    **response,
                    "content": fix_response_content(path, status_code, response["content"]),
                }
            else:
                fixed_responses[status_code] = response

        return fixed_responses

    # Process all paths and their operations
    if "paths" in spec:
        for path, path_item in spec["paths"].items():
            if path.endswith("/chat/completions") or path.endswith("/completions"):
                print_verbose(f"Fixing OpenAI streaming endpoint: {path}", style="bold green")

            if isinstance(path_item, dict):
                for method, operation in path_item.items():
                    if isinstance(operation, dict) and "responses" in operation:
                        operation["responses"] = fix_responses(path, operation["responses"])

    return spec


@app.command(name="fix-openai-streaming-endpoints")
def fix_openai_streaming_endpoints_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    try:
        spec = load_openapi_spec(spec_file)
        spec = fix_openai_streaming_endpoints(spec)

        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Successfully fixed OpenAI streaming APIs and saved to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


def remove_response_content_type(
    spec: dict, path: str, method: str, content_type: str, status_code: str = "200"
) -> dict:
    """Remove a specific content type from a response in the OpenAPI specification."""
    try:
        response = spec["paths"][path][method.lower()]["responses"][status_code]
        if content_type in response.get("content", {}):
            del response["content"][content_type]
            print_verbose(
                f"Removed '{content_type}' from {method.upper()} {path} ({status_code})",
                style="bold green",
            )

            # Clean up empty content
            if not response["content"]:
                del response["content"]
        else:
            print_verbose(f"Content type '{content_type}' not found", style="bold yellow")
    except KeyError as exc:
        print_verbose(f"Path, method, or response not found: {exc}", style="bold red")
        raise typer.Exit(1)

    return spec


@app.command(name="remove-response-type")
def remove_response_content_type_command(
    spec_file: str = typer.Argument(..., help="Path to OpenAPI specification file (YAML or JSON)"),
    path: str = typer.Argument(..., help="Path of the endpoint (e.g., /v1/endpoint)"),
    method: str = typer.Argument(..., help="HTTP method (e.g., GET, POST)"),
    content_type: str = typer.Argument(..., help="Content type to remove (e.g., application/json)"),
    status_code: str = typer.Option("200", help="HTTP status code of the response (default: 200)"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file path. If not provided, will overwrite the input file."
    ),
):
    """Remove a specific content type from a response in the OpenAPI specification."""
    try:
        spec = load_openapi_spec(spec_file)
        spec = remove_response_content_type(spec, path, method, content_type, status_code)

        # Save the modified spec
        output_path = output_file or spec_file
        save_openapi_spec(spec, output_path)

        print_verbose(
            f"Successfully removed content type and saved to {output_path}",
            style="bold green",
        )

    except Exception as e:
        print_verbose(f"Error: {str(e)}", style="bold red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
