#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI tool for managing authentication and authorization configuration.

This tool helps ensure that all endpoints defined in the OpenAPI specification
have corresponding entries in the static authorization configuration.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import typer
from rich.console import Console
from rich.table import Table
from ruamel.yaml import YAML

# Initialize YAML with settings to preserve formatting
yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.width = 4096  # Prevent line wrapping

app = typer.Typer(help="Authentication and authorization configuration management tool")
console = Console()

# OPA treats this as "all permissions" for roles such as ServiceSystem; it is not a registry key
# and does not appear on endpoints as a literal permission string.
WILDCARD_PERMISSION = "*"

HTTP_METHODS = frozenset({"get", "post", "put", "delete", "patch", "head", "options"})

# Marker on an endpoint method that means "real endpoint, but intentionally not advertised
# in the public OpenAPI spec" (e.g. internal ops routes). Orphan checks skip these.
OPENAPI_OPT_OUT_KEY = "x-not-in-openapi"

# Maps the URL segment in /apis/<segment>/v<n>/... to the canonical area name used
# in permissions and scopes. Most segments map to themselves; the exception is
# inference-gateway, whose registry/permission name is "inference".
_API_AREA_URL_MAP = {
    "audit": "audit",
    "auth": "auth",
    "customization": "customization",
    "data-designer": "data-designer",
    "entities": "entities",
    "evaluation": "evaluation",
    "files": "files",
    "guardrails": "guardrails",
    "inference-gateway": "inference",
    "intake": "intake",
    "jobs": "jobs",
    "models": "models",
    "safe-synthesizer": "safe-synthesizer",
    "secrets": "secrets",
}

_NESTED_RESOURCE_SEGMENTS = frozenset(
    {"evaluation", "customization", "guardrails", "audit", "data-designer", "safe-synthesizer"}
)


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this script is in services/core/auth/scripts/
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def load_yaml(file_path: Path):
    """Load a YAML file and return its contents, preserving comments."""
    with open(file_path, "r") as f:
        return yaml.load(f)


def save_yaml(file_path: Path, data):
    """Save data to a YAML file, preserving comments and structure."""
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def extract_openapi_endpoints(openapi_path: Path) -> Dict[str, Set[str]]:
    """Extract all endpoints and their methods from the OpenAPI spec."""
    openapi_spec = load_yaml(openapi_path)
    endpoints = {}

    if "paths" not in openapi_spec:
        console.print("[red]Error: No 'paths' section found in OpenAPI spec[/red]")
        return endpoints

    for path, methods in openapi_spec["paths"].items():
        if not isinstance(methods, dict):
            continue

        # Extract HTTP methods (get, post, put, delete, patch, etc.)
        http_methods = {
            method.lower()
            for method in methods.keys()
            if method.lower() in {"get", "post", "put", "delete", "patch", "head", "options"}
        }

        if http_methods:
            endpoints[path] = http_methods

    return endpoints


def extract_auth_endpoints(auth_path: Path) -> Dict[str, Set[str]]:
    """Extract all endpoints and their methods from the auth configuration."""
    auth_config = load_yaml(auth_path)
    endpoints = {}

    if "authz" not in auth_config or "endpoints" not in auth_config["authz"]:
        console.print("[red]Error: No 'authz.endpoints' section found in auth config[/red]")
        return endpoints

    for path, methods in auth_config["authz"]["endpoints"].items():
        if not isinstance(methods, dict):
            continue

        http_methods = {method.lower() for method in methods.keys() if method.lower() in HTTP_METHODS}

        if http_methods:
            endpoints[path] = http_methods

    return endpoints


def extract_openapi_opt_out_endpoints(auth_path: Path) -> Dict[str, Set[str]]:
    """Return endpoints marked ``x-not-in-openapi: true`` in the auth config.

    These are real routes served by the platform but intentionally excluded from the
    public OpenAPI spec (e.g. internal/ops endpoints). They must be skipped from
    orphan-endpoint checks, otherwise the tool would flag or delete them every run.
    """
    auth_config = load_yaml(auth_path)
    opt_outs: Dict[str, Set[str]] = {}

    for path, methods in auth_config.get("authz", {}).get("endpoints", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, config in methods.items():
            if method.lower() not in HTTP_METHODS:
                continue
            if isinstance(config, dict) and config.get(OPENAPI_OPT_OUT_KEY) is True:
                opt_outs.setdefault(path, set()).add(method.lower())

    return opt_outs


def extract_all_permissions_from_endpoints(auth_config: Dict) -> Set[str]:
    """Extract all permissions referenced in endpoints."""
    permissions = set()

    if "authz" not in auth_config or "endpoints" not in auth_config["authz"]:
        return permissions

    for path, methods in auth_config["authz"]["endpoints"].items():
        if not isinstance(methods, dict):
            continue

        for method, config in methods.items():
            if isinstance(config, dict) and "permissions" in config:
                perms = config["permissions"]
                if isinstance(perms, list):
                    permissions.update(perms)

    return permissions


def extract_registered_permissions(auth_config: Dict) -> Dict[str, Dict]:
    """Extract all permissions from the permissions registry.

    Supports both flat keys (``"audit.configs.read": {description: ...}``)
    and nested keys (``audit: configs: read: {description: ...}``).
    A leaf node is any mapping that contains a ``description`` key.
    """
    raw = auth_config.get("authz", {}).get("permissions", {}) or {}

    def _flatten(node: Dict, prefix: str = "") -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for key, value in node.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and "description" in value:
                result[full_key] = value
            elif isinstance(value, dict):
                result.update(_flatten(value, full_key))
            else:
                result[full_key] = value
        return result

    return _flatten(raw)


def get_no_endpoint_permissions(auth_config: Dict) -> Set[str]:
    """Return the set of permissions with has_endpoint: false in the registry."""
    registry = extract_registered_permissions(auth_config)
    return {name for name, meta in registry.items() if meta and meta.get("has_endpoint") is False}


def get_no_role_permissions(auth_config: Dict) -> Set[str]:
    """Return the set of permissions with has_role: false in the registry."""
    registry = extract_registered_permissions(auth_config)
    return {name for name, meta in registry.items() if meta and meta.get("has_role") is False}


def strip_workspace_from_permission(permission: str) -> str:
    """Strip workspace prefix from permission if present.

    Example: 'system/model.create' -> 'model.create'
    """
    if "/" in permission:
        return permission.split("/", 1)[1]
    return permission


def extract_all_permissions_from_roles(auth_config: Dict) -> Set[str]:
    """Extract all permissions defined in roles, including inherited permissions.

    Strips workspace prefixes from permissions for comparison purposes.
    """
    permissions = set()
    roles = auth_config.get("authz", {}).get("roles", {})

    def get_role_permissions(role_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Recursively get all permissions for a role, including inherited ones."""
        if visited is None:
            visited = set()

        if role_name in visited or role_name not in roles:
            return set()

        visited.add(role_name)
        role = roles[role_name]
        # Strip workspace from each permission
        role_perms = {strip_workspace_from_permission(p) for p in role.get("permissions", [])}

        # Process included roles
        for included_role in role.get("includes", []):
            role_perms.update(get_role_permissions(included_role, visited))

        return role_perms

    # Get permissions from all roles
    for role_name in roles:
        permissions.update(get_role_permissions(role_name))

    return permissions


def _strip_apis_shell(path: str) -> Optional[tuple]:
    """Strip the /apis/<area>/v<n>/[workspaces/{workspace}/] shell.

    Returns ``(canonical_area, remaining_parts)`` for paths that match the
    /apis/<area>/... shape (where ``area`` is in ``_API_AREA_URL_MAP``), with
    ``remaining_parts`` being the segments after the version and any
    ``workspaces/{workspace}/`` scope. Returns ``None`` for non-/apis/ paths.

    The canonical area is what appears in permissions/registry (e.g. the
    ``inference-gateway`` URL segment maps to ``inference``).
    """
    parts = path.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "apis" or parts[1] not in _API_AREA_URL_MAP:
        return None
    area = _API_AREA_URL_MAP[parts[1]]
    rest = parts[2:]
    # Drop version segment (v2, v1beta1, ...)
    if rest and rest[0].startswith("v"):
        rest = rest[1:]
    # Drop workspace scope so the next segment is the actual sub-resource
    if len(rest) >= 2 and rest[0] == "workspaces":
        rest = rest[2:]
    return area, rest


def infer_resource_from_path(path: str) -> str:
    """Infer the resource name from an endpoint path."""
    apis = _strip_apis_shell(path)
    if apis is not None:
        return apis[0]

    parts = path.strip("/").split("/")

    # Skip version prefix (v1, v2, v1beta1, etc.)
    if parts and parts[0].startswith("v"):
        parts = parts[1:]

    if not parts:
        return "unknown"

    # Special handling for nested resources
    if len(parts) >= 2:
        # Handle inference endpoints
        if parts[0] == "inference" and len(parts) >= 2:
            if parts[1] == "chat" and len(parts) > 2:
                return "inference"
            return "inference"

        # Handle sub-resources (e.g., /v1/evaluation/configs)
        if parts[0] in _NESTED_RESOURCE_SEGMENTS:
            return parts[0]

        # Handle workspace members
        if len(parts) >= 3 and parts[2] == "members":
            return "workspaces"

    # Default to the first part
    return parts[0] if parts[0] else "unknown"


def infer_permissions(path: str, method: str) -> List[str]:
    """Infer permissions for an endpoint based on its path and method."""
    resource = infer_resource_from_path(path)

    # Special case for inference endpoints
    if resource == "inference":
        if "chat/completions" in path:
            return ["inference.chat.completions"]
        elif "completions" in path:
            return ["inference.completions"]
        elif "embeddings" in path:
            return ["inference.embeddings"]

    # Special case for workspace creation (system-scoped RBAC, not workspace-scoped RBAC)
    if path.endswith("/workspaces") and method == "post":
        return ["workspaces.create"]

    # Special case for workspace members (permission lives under workspaces.* regardless
    # of which API area exposes the route, e.g. /apis/entities/v2/workspaces/{ws}/members)
    if "/workspaces/" in path and "/members" in path:
        method_map = {"get": "list", "post": "create", "put": "update", "delete": "delete"}
        action = method_map.get(method, method)
        return [f"workspaces.members.{action}"]

    # Determine the sub-resource. For /apis/<area>/v<n>/[workspaces/{ws}/]<sub>/...
    # the sub-resource is the first segment after the shell. For legacy paths like
    # /v1/evaluation/configs, it's parts[1] under a known nested resource.
    apis = _strip_apis_shell(path)
    sub_resource = None
    if apis is not None:
        rest = apis[1]
        if rest and not rest[0].startswith("{"):
            sub_resource = rest[0]
    else:
        parts = path.strip("/").split("/")
        if parts and parts[0].startswith("v"):
            parts = parts[1:]
        if len(parts) >= 2 and parts[0] in _NESTED_RESOURCE_SEGMENTS:
            if parts[1] not in ("{id}", "{job_id}", "{config_id}"):
                sub_resource = parts[1]

    permission_prefix = f"{resource}.{sub_resource}" if sub_resource else resource

    # pin/unpin mutate an existing resource in place; both belong under `update`, not the default
    # post→create / delete→delete verb mapping.
    if path.endswith("/pin"):
        return [f"{permission_prefix}.update"]

    # Map HTTP methods to permissions
    method_to_permission = {
        "get": "list" if path.endswith(resource) or path.endswith(f"{resource}s") else "read",
        "post": "create",
        "put": "update",
        "patch": "update",
        "delete": "delete" if "{" in path else "cancel",  # Assume delete on collection means cancel
    }

    permission_type = method_to_permission.get(method, "read")
    return [f"{permission_prefix}.{permission_type}"]


def _get_area_from_api_path(path: str) -> str:
    """Extract the area name from an API path like /apis/<area>/v2/...

    The area is used for per-area scopes (e.g. models:read, models:write).
    Some areas have different names than their URL segments (e.g. inference-gateway -> inference).
    """
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "apis":
        return _API_AREA_URL_MAP.get(parts[1], "")
    return ""


def infer_scopes(path: str, method: str) -> List[str]:
    """Infer scopes for an endpoint based on its path and method.

    Every endpoint gets both an area-specific scope (e.g. models:read) and
    the corresponding platform catch-all scope (platform:read/platform:write).
    """
    # Special case for workspace creation (normal write scopes despite no workspace in path)
    if path.endswith("/workspaces") and method == "post":
        return ["entities:write", "platform:write"]

    # Determine read/write based on method
    is_write = method in ["post", "put", "patch", "delete"]
    scope_type = "write" if is_write else "read"

    # Some POST endpoints are semantically reads (query endpoints)
    read_post_suffixes = ["/query"]
    if method == "post" and any(path.endswith(suffix) for suffix in read_post_suffixes):
        scope_type = "read"

    # Get area-specific scope from the API path
    area = _get_area_from_api_path(path)
    if area:
        return [f"{area}:{scope_type}", f"platform:{scope_type}"]

    # Fallback: try to infer from the resource name for non-/apis/ paths
    resource = infer_resource_from_path(path)
    known_resources = [
        "audit",
        "auth",
        "customization",
        "data-designer",
        "entities",
        "evaluation",
        "files",
        "guardrails",
        "inference",
        "intake",
        "jobs",
        "models",
        "safe-synthesizer",
        "secrets",
    ]
    if resource in known_resources:
        return [f"{resource}:{scope_type}", f"platform:{scope_type}"]

    # Default to platform scope only
    return [f"platform:{scope_type}"]


@app.command()
def check(
    openapi_path: Path = typer.Option(
        None, "--openapi", "-o", help="Path to OpenAPI specification file (relative to project root)"
    ),
    auth_path: Path = typer.Option(
        None, "--auth", "-a", help="Path to static authorization configuration file (relative to project root)"
    ),
):
    """Check if all OpenAPI endpoints have corresponding auth entries."""
    # Use default paths if not provided
    project_root = get_project_root()
    if openapi_path is None:
        openapi_path = project_root / "openapi" / "openapi.yaml"
    else:
        openapi_path = project_root / openapi_path

    if auth_path is None:
        auth_path = (
            project_root
            / "services"
            / "core"
            / "auth"
            / "src"
            / "nhx"
            / "core"
            / "auth"
            / "assets"
            / "static-authz.yaml"
        )
    else:
        auth_path = project_root / auth_path

    console.print("[bold]Checking authentication configuration...[/bold]")

    # Extract endpoints from both sources
    openapi_endpoints = extract_openapi_endpoints(openapi_path)
    auth_endpoints = extract_auth_endpoints(auth_path)

    # Find missing endpoints (in openapi but not in auth config)
    missing = []
    for path, methods in openapi_endpoints.items():
        auth_methods = auth_endpoints.get(path, set())
        missing_methods = methods - auth_methods

        if missing_methods:
            for method in missing_methods:
                missing.append((path, method))

    # Find orphaned endpoints (in auth config but not in openapi, excluding opt-outs)
    opt_out_endpoints = extract_openapi_opt_out_endpoints(auth_path)
    orphaned = []
    for path, methods in auth_endpoints.items():
        openapi_methods = openapi_endpoints.get(path, set())
        opted_out = opt_out_endpoints.get(path, set())
        orphaned_methods = methods - openapi_methods - opted_out
        for method in orphaned_methods:
            orphaned.append((path, method))

    # Find stale opt-outs (marked x-not-in-openapi: true but the endpoint is in openapi)
    stale_opt_outs = []
    for path, methods in opt_out_endpoints.items():
        openapi_methods = openapi_endpoints.get(path, set())
        for method in methods & openapi_methods:
            stale_opt_outs.append((path, method))

    # Load auth config for permission checks
    auth_config = load_yaml(auth_path)

    # Check for permissions not in any role (excluding has_role: false)
    endpoint_permissions = extract_all_permissions_from_endpoints(auth_config)
    endpoint_permissions_stripped = {strip_workspace_from_permission(p) for p in endpoint_permissions}
    role_permissions = extract_all_permissions_from_roles(auth_config)
    no_role_permissions = get_no_role_permissions(auth_config)
    orphaned_permissions = endpoint_permissions_stripped - role_permissions - no_role_permissions

    # Check for stale role permissions (in roles but not in any endpoint, unless has_endpoint: false)
    no_endpoint_permissions = get_no_endpoint_permissions(auth_config)
    stale_permissions = role_permissions - endpoint_permissions_stripped - no_endpoint_permissions
    stale_permissions.discard(WILDCARD_PERMISSION)

    # Check for unregistered permissions (in roles or endpoints but not in the registry)
    registered = set(extract_registered_permissions(auth_config).keys())
    all_used = role_permissions | endpoint_permissions_stripped
    unregistered_permissions = all_used - registered
    unregistered_permissions.discard(WILDCARD_PERMISSION)

    # Display results
    has_issues = False

    if missing:
        has_issues = True
        console.print(f"\n[red]Found {len(missing)} missing endpoint(s) in auth configuration:[/red]")

        table = Table(title="Missing Endpoints", show_header=True, header_style="bold magenta")
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Method", style="yellow", justify="center")
        table.add_column("Suggested Permissions", style="green")
        table.add_column("Suggested Scopes", style="blue")

        for path, method in sorted(missing):
            permissions = ", ".join(infer_permissions(path, method))
            scopes = ", ".join(infer_scopes(path, method))
            table.add_row(path, method.upper(), permissions, scopes)

        console.print(table)
        console.print("\n[bold yellow]⚠ Action Required:[/bold yellow]")
        console.print("  1. Run: [cyan]uv run python services/core/auth/scripts/auth-tools.py update[/cyan]")
        console.print("  2. [bold]Manually review[/bold] the generated permissions and scopes")
        console.print("  3. [bold]Verify[/bold] if new endpoints require additional authorization permissions")
        console.print("  4. Update role definitions if needed to grant appropriate access")
        console.print(
            "  5. Commit the changes to [cyan]services/core/auth/src/nhx/core/auth/assets/static-authz.yaml[/cyan]\n"
        )

    if orphaned:
        has_issues = True
        console.print(f"\n[red]Found {len(orphaned)} orphaned endpoint(s) in auth configuration:[/red]")

        table = Table(title="Orphaned Endpoints", show_header=True, header_style="bold magenta")
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Method", style="yellow", justify="center")

        for path, method in sorted(orphaned):
            table.add_row(path, method.upper())

        console.print(table)
        console.print("\n[yellow]These entries exist in static-authz.yaml but not in openapi.yaml. Either:[/yellow]")
        console.print("  • Remove them via [cyan]uv run python services/core/auth/scripts/auth-tools.py update[/cyan]")
        console.print(
            f"  • Or add [cyan]{OPENAPI_OPT_OUT_KEY}: true[/cyan] on the method if the route exists but is "
            "intentionally excluded from the public OpenAPI spec (e.g. internal/ops endpoints).\n"
        )

    if stale_opt_outs:
        has_issues = True
        console.print(f"\n[red]Found {len(stale_opt_outs)} stale [cyan]{OPENAPI_OPT_OUT_KEY}[/cyan] marker(s):[/red]")

        table = Table(title="Stale Opt-Outs", show_header=True, header_style="bold magenta")
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Method", style="yellow", justify="center")

        for path, method in sorted(stale_opt_outs):
            table.add_row(path, method.upper())

        console.print(table)
        console.print(
            f"\n[yellow]The endpoint is now in openapi.yaml, so [cyan]{OPENAPI_OPT_OUT_KEY}: true[/cyan] "
            "is misleading. Remove the marker (or run [cyan]update[/cyan] to strip it).[/yellow]\n"
        )

    if orphaned_permissions:
        has_issues = True
        console.print(
            f"\nFound [bold]{len(orphaned_permissions)}[/bold] [bold]permission[/bold](s) not included in any role:"
        )

        table = Table(title="Orphaned Permissions", show_header=True, header_style="bold magenta")
        table.add_column("Permission", style="cyan")
        table.add_column("Used in Endpoints", style="yellow")

        for perm in sorted(orphaned_permissions):
            endpoints_using_perm = []
            for path, methods in auth_config.get("authz", {}).get("endpoints", {}).items():
                for method, config in methods.items():
                    for endpoint_perm in config.get("permissions", []):
                        if strip_workspace_from_permission(endpoint_perm) == perm:
                            endpoints_using_perm.append(f"{method.upper()} {path}")

            endpoint_list = "\n".join(endpoints_using_perm[:3])
            if len(endpoints_using_perm) > 3:
                endpoint_list += f"\n... and {len(endpoints_using_perm) - 3} more"

            table.add_row(perm, endpoint_list)

        console.print(table)
        console.print("\n[yellow]Tip: Add these permissions to appropriate roles in the auth configuration.[/yellow]")

    if stale_permissions:
        has_issues = True
        console.print(
            f"\nFound [bold]{len(stale_permissions)}[/bold] stale role permission(s) not used by any endpoint:"
        )

        roles_data = auth_config.get("authz", {}).get("roles", {})
        table = Table(title="Stale Role Permissions", show_header=True, header_style="bold magenta")
        table.add_column("Permission", style="cyan")
        table.add_column("In Roles", style="yellow")

        for perm in sorted(stale_permissions):
            in_roles = [rn for rn in roles_data if perm in list(roles_data[rn].get("permissions", []))]
            table.add_row(perm, ", ".join(in_roles))

        console.print(table)
        console.print(
            "\n[yellow]Tip: Remove stale permissions from roles, or set has_endpoint: false "
            "in the permissions registry if they are checked in policy/code.[/yellow]"
        )

    if unregistered_permissions:
        has_issues = True
        console.print(
            f"\nFound [bold]{len(unregistered_permissions)}[/bold] permission(s) not in the permissions registry:"
        )

        table = Table(title="Unregistered Permissions", show_header=True, header_style="bold magenta")
        table.add_column("Permission", style="cyan")
        table.add_column("Used In", style="yellow")

        for perm in sorted(unregistered_permissions):
            locations = []
            if perm in role_permissions:
                locations.append("roles")
            if perm in endpoint_permissions_stripped:
                locations.append("endpoints")
            table.add_row(perm, ", ".join(locations))

        console.print(table)
        console.print("\n[yellow]Tip: Add these permissions to the permissions section in static-authz.yaml.[/yellow]")

    # Check if permissions reference doc is up to date
    docs_path = project_root / "docs" / "auth" / "authorization" / "permissions-reference.mdx"
    if docs_path.exists():
        expected = _generate_permissions_reference(auth_config)
        actual = docs_path.read_text(encoding="utf-8")
        if actual != expected:
            has_issues = True
            console.print("\n[red]Permissions reference doc is out of date.[/red]")
            console.print(
                "[yellow]Tip: Run [cyan]uv run python services/core/auth/scripts/auth-tools.py "
                "generate-docs[/cyan] to regenerate it.[/yellow]"
            )
        elif not has_issues:
            console.print("[green]✓ Permissions reference doc is up to date![/green]")
    else:
        has_issues = True
        console.print(f"\n[red]Permissions reference doc not found at {docs_path}[/red]")
        console.print(
            "[yellow]Tip: Run [cyan]uv run python services/core/auth/scripts/auth-tools.py "
            "generate-docs[/cyan] to generate it.[/yellow]"
        )

    if not has_issues:
        console.print("[green]✓ All endpoints from OpenAPI spec have corresponding auth entries![/green]")
        console.print("[green]✓ No orphaned endpoints in auth configuration![/green]")
        console.print("[green]✓ All permissions are included in at least one role![/green]")
        console.print("[green]✓ No stale role permissions found![/green]")
        console.print("[green]✓ All permissions are registered![/green]")
        console.print(
            f"[dim]Checked {len(openapi_endpoints)} endpoints, "
            f"{len(endpoint_permissions)} endpoint permissions, "
            f"{len(role_permissions)} role permissions, "
            f"{len(registered)} registered permissions[/dim]"
        )
        sys.exit(0)
    else:
        sys.exit(1)


@app.command()
def update(
    openapi_path: Path = typer.Option(
        None, "--openapi", "-o", help="Path to OpenAPI specification file (relative to project root)"
    ),
    auth_path: Path = typer.Option(
        None, "--auth", "-a", help="Path to static authorization configuration file (relative to project root)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be added without making changes"),
):
    """Synchronize auth configuration with OpenAPI spec.

    This command will:
    - Add missing endpoints found in OpenAPI but not in auth config
    - Remove orphaned endpoints found in auth config but not in OpenAPI
    - Add orphaned permissions to appropriate roles:
      - .read and .list permissions → Viewer role
      - All other permissions → Editor role
    """
    # Use default paths if not provided
    project_root = get_project_root()
    if openapi_path is None:
        openapi_path = project_root / "openapi" / "openapi.yaml"
    else:
        openapi_path = project_root / openapi_path

    if auth_path is None:
        auth_path = (
            project_root
            / "services"
            / "core"
            / "auth"
            / "src"
            / "nhx"
            / "core"
            / "auth"
            / "assets"
            / "static-authz.yaml"
        )
    else:
        auth_path = project_root / auth_path

    console.print("[bold]Updating authentication configuration...[/bold]")

    # Load configurations
    openapi_endpoints = extract_openapi_endpoints(openapi_path)
    auth_config = load_yaml(auth_path)
    auth_endpoints = extract_auth_endpoints(auth_path)

    # Ensure the structure exists
    if "authz" not in auth_config:
        auth_config["authz"] = {}
    if "endpoints" not in auth_config["authz"]:
        auth_config["authz"]["endpoints"] = {}

    # Find and add missing endpoints
    added = []
    for path, methods in sorted(openapi_endpoints.items()):
        auth_methods = auth_endpoints.get(path, set())
        missing_methods = methods - auth_methods

        if missing_methods:
            if path not in auth_config["authz"]["endpoints"]:
                auth_config["authz"]["endpoints"][path] = {}

            for method in sorted(missing_methods):
                permissions = infer_permissions(path, method)
                scopes = infer_scopes(path, method)

                auth_config["authz"]["endpoints"][path][method] = {"permissions": permissions, "scopes": scopes}

                added.append((path, method, permissions, scopes))

    # Find and remove orphaned endpoints (in auth but not in OpenAPI, excluding opt-outs)
    opt_out_endpoints = extract_openapi_opt_out_endpoints(auth_path)
    removed = []
    for path, methods in list(auth_endpoints.items()):
        openapi_methods = openapi_endpoints.get(path, set())
        opted_out = opt_out_endpoints.get(path, set())
        methods_to_remove = methods - openapi_methods - opted_out

        for method in methods_to_remove:
            removed.append((path, method))
            if not dry_run and method in auth_config["authz"]["endpoints"].get(path, {}):
                del auth_config["authz"]["endpoints"][path][method]

        # Drop the path entirely if no HTTP methods remain under it
        if not dry_run and path in auth_config["authz"]["endpoints"]:
            remaining = {key for key in auth_config["authz"]["endpoints"][path].keys() if key.lower() in HTTP_METHODS}
            if not remaining:
                del auth_config["authz"]["endpoints"][path]

    # Strip stale opt-out markers (endpoint is now advertised in openapi → marker lies)
    stale_opt_outs_cleared = []
    for path, methods in opt_out_endpoints.items():
        openapi_methods = openapi_endpoints.get(path, set())
        for method in methods & openapi_methods:
            stale_opt_outs_cleared.append((path, method))
            if not dry_run:
                method_config = auth_config["authz"]["endpoints"].get(path, {}).get(method)
                if isinstance(method_config, dict):
                    method_config.pop(OPENAPI_OPT_OUT_KEY, None)

    # Find and fix orphaned permissions (permissions not in any role)
    endpoint_permissions = extract_all_permissions_from_endpoints(auth_config)
    # Strip workspaces from endpoint permissions for comparison
    endpoint_permissions_stripped = {strip_workspace_from_permission(p) for p in endpoint_permissions}
    role_permissions = extract_all_permissions_from_roles(auth_config)
    no_role_permissions = get_no_role_permissions(auth_config)
    orphaned_permissions = endpoint_permissions_stripped - role_permissions - no_role_permissions

    permissions_added_to_roles = {"Viewer": [], "Editor": []}

    if orphaned_permissions and not dry_run:
        # Ensure roles exist
        if "roles" not in auth_config["authz"]:
            auth_config["authz"]["roles"] = {}

        # Add orphaned permissions to appropriate roles
        for perm in orphaned_permissions:
            # Heuristic: .read and .list go to Viewer, everything else to Editor
            if perm.endswith(".read") or perm.endswith(".list"):
                target_role = "Viewer"
            else:
                target_role = "Editor"

            # Ensure the role exists
            if target_role not in auth_config["authz"]["roles"]:
                auth_config["authz"]["roles"][target_role] = {"description": f"{target_role} role", "permissions": []}

            # Add permission if not already there
            if "permissions" not in auth_config["authz"]["roles"][target_role]:
                auth_config["authz"]["roles"][target_role]["permissions"] = []

            current_perms = auth_config["authz"]["roles"][target_role]["permissions"]
            # Strip workspace when checking if permission already exists
            current_perms_stripped = {strip_workspace_from_permission(p) for p in current_perms}
            if perm not in current_perms_stripped:
                current_perms.append(perm)
                permissions_added_to_roles[target_role].append(perm)

        # Sort permissions in each role for consistency
        for role in ["Viewer", "Editor"]:
            if role in auth_config["authz"]["roles"] and "permissions" in auth_config["authz"]["roles"][role]:
                auth_config["authz"]["roles"][role]["permissions"].sort()
    elif orphaned_permissions and dry_run:
        # For dry run, just categorize what would be added
        for perm in orphaned_permissions:
            if perm.endswith(".read") or perm.endswith(".list"):
                permissions_added_to_roles["Viewer"].append(perm)
            else:
                permissions_added_to_roles["Editor"].append(perm)

    # Display what will be/was added
    if added:
        console.print(
            f"\n[yellow]{'Would add' if dry_run else 'Added'} {len(added)} endpoint(s) to auth configuration:[/yellow]"
        )

        table = Table(
            title=f"{'New' if not dry_run else 'Would Add'} Endpoints", show_header=True, header_style="bold magenta"
        )
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Method", style="yellow", justify="center")
        table.add_column("Permissions", style="green")
        table.add_column("Scopes", style="blue")

        for path, method, permissions, scopes in added:
            perm_str = ", ".join(permissions) if permissions else "(empty)"
            scope_str = ", ".join(scopes) if scopes else "(empty)"
            table.add_row(path, method.upper(), perm_str, scope_str)

        console.print(table)

    # Display what will be/was removed
    if removed:
        console.print(
            f"\n[red]{'Would remove' if dry_run else 'Removed'} {len(removed)} orphaned endpoint(s) from auth configuration:[/red]"
        )

        table = Table(
            title=f"{'Removed' if not dry_run else 'Would Remove'} Endpoints",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Method", style="yellow", justify="center")

        for path, method in sorted(removed):
            table.add_row(path, method.upper())

        console.print(table)

    # Display permissions added to roles
    total_perms_added = sum(len(perms) for perms in permissions_added_to_roles.values())
    if total_perms_added > 0:
        console.print(
            f"\n[green]{'Would add' if dry_run else 'Added'} {total_perms_added} orphaned permission(s) to roles:[/green]"
        )

        for role, perms in permissions_added_to_roles.items():
            if perms:
                console.print(f"\n[bold]{role} role:[/bold] {len(perms)} permission(s)")
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Permission", style="cyan")

                for perm in sorted(perms):
                    table.add_row(f"• {perm}")

                console.print(table)

    # Display stale opt-outs that were (would be) cleared
    if stale_opt_outs_cleared:
        console.print(
            f"\n[yellow]{'Would strip' if dry_run else 'Stripped'} {len(stale_opt_outs_cleared)} "
            f"stale [cyan]{OPENAPI_OPT_OUT_KEY}[/cyan] marker(s) (endpoint is now in openapi.yaml):[/yellow]"
        )
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Path", style="cyan")
        for path, method in sorted(stale_opt_outs_cleared):
            table.add_row(f"• {method.upper()} {path}")
        console.print(table)

    # Save the updated configuration if not dry run
    has_changes = added or removed or total_perms_added > 0 or stale_opt_outs_cleared
    if has_changes:
        if not dry_run:
            # Sort endpoints alphabetically before saving
            if "authz" in auth_config and "endpoints" in auth_config["authz"]:
                sorted_endpoints = {}
                for path in sorted(auth_config["authz"]["endpoints"].keys()):
                    # Sort methods within each endpoint too
                    sorted_methods = dict(sorted(auth_config["authz"]["endpoints"][path].items()))
                    sorted_endpoints[path] = sorted_methods
                auth_config["authz"]["endpoints"] = sorted_endpoints

            save_yaml(auth_path, auth_config)
            console.print(f"\n[green]✓ Updated auth configuration saved to {auth_path}[/green]")
            if added:
                console.print(f"  - Added {len(added)} endpoint(s)")
            if removed:
                console.print(f"  - Removed {len(removed)} orphaned endpoint(s)")
            if total_perms_added > 0:
                console.print(f"  - Added {total_perms_added} permission(s) to roles")
            if stale_opt_outs_cleared:
                console.print(f"  - Stripped {len(stale_opt_outs_cleared)} stale {OPENAPI_OPT_OUT_KEY} marker(s)")
        else:
            console.print(f"\n[dim]Dry run - no changes made to {auth_path}[/dim]")
    else:
        console.print("[green]✓ Auth configuration is fully synchronized with OpenAPI spec![/green]")


@app.command()
def sort(
    auth_path: Path = typer.Option(
        None, "--auth", "-a", help="Path to static authorization configuration file (relative to project root)"
    ),
):
    """Sort endpoints in the auth configuration alphabetically."""
    # Use default path if not provided
    project_root = get_project_root()
    if auth_path is None:
        auth_path = (
            project_root
            / "services"
            / "core"
            / "auth"
            / "src"
            / "nhx"
            / "core"
            / "auth"
            / "assets"
            / "static-authz.yaml"
        )
    else:
        auth_path = project_root / auth_path

    console.print("[bold]Sorting endpoints in auth configuration...[/bold]")

    # Load configuration
    auth_config = load_yaml(auth_path)

    # Sort endpoints alphabetically
    if "authz" in auth_config and "endpoints" in auth_config["authz"]:
        sorted_endpoints = {}
        for path in sorted(auth_config["authz"]["endpoints"].keys()):
            # Sort methods within each endpoint too
            sorted_methods = dict(sorted(auth_config["authz"]["endpoints"][path].items()))
            sorted_endpoints[path] = sorted_methods
        auth_config["authz"]["endpoints"] = sorted_endpoints

        # Save the sorted configuration
        save_yaml(auth_path, auth_config)
        console.print(f"[green]✓ Sorted {len(sorted_endpoints)} endpoints in {auth_path}[/green]")
    else:
        console.print("[yellow]No endpoints found to sort[/yellow]")


@app.command()
def stats(
    auth_path: Path = typer.Option(
        None, "--auth", "-a", help="Path to static authorization configuration file (relative to project root)"
    ),
):
    """Show statistics about the auth configuration."""
    # Use default path if not provided
    project_root = get_project_root()
    if auth_path is None:
        auth_path = (
            project_root
            / "services"
            / "core"
            / "auth"
            / "src"
            / "nhx"
            / "core"
            / "auth"
            / "assets"
            / "static-authz.yaml"
        )
    else:
        auth_path = project_root / auth_path

    console.print("[bold]Authentication Configuration Statistics[/bold]\n")

    auth_config = load_yaml(auth_path)

    # Count endpoints by resource
    resource_counts = {}
    method_counts = {"get": 0, "post": 0, "put": 0, "delete": 0, "patch": 0}
    total_endpoints = 0

    if "authz" in auth_config and "endpoints" in auth_config["authz"]:
        for path, methods in auth_config["authz"]["endpoints"].items():
            resource = infer_resource_from_path(path)
            if resource not in resource_counts:
                resource_counts[resource] = 0

            for method in methods:
                if method.lower() in method_counts:
                    method_counts[method.lower()] += 1
                    resource_counts[resource] += 1
                    total_endpoints += 1

    # Display resource statistics
    table = Table(title="Endpoints by Resource", show_header=True, header_style="bold magenta")
    table.add_column("Resource", style="cyan")
    table.add_column("Count", style="yellow", justify="right")

    for resource, count in sorted(resource_counts.items(), key=lambda x: x[1], reverse=True):
        table.add_row(resource, str(count))

    console.print(table)

    # Display method statistics
    console.print()
    table = Table(title="Endpoints by HTTP Method", show_header=True, header_style="bold magenta")
    table.add_column("Method", style="cyan")
    table.add_column("Count", style="yellow", justify="right")

    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            table.add_row(method.upper(), str(count))

    console.print(table)

    # Display role statistics
    console.print()
    if "authz" in auth_config and "roles" in auth_config["authz"]:
        table = Table(title="Defined Roles", show_header=True, header_style="bold magenta")
        table.add_column("Role", style="cyan")
        table.add_column("Permissions", style="yellow", justify="right")
        table.add_column("Description", style="green")

        for role_name, role_data in auth_config["authz"]["roles"].items():
            perm_count = len(role_data.get("permissions", []))
            description = role_data.get("description", "No description")
            table.add_row(role_name, str(perm_count), description)

        console.print(table)

    console.print(f"\n[bold]Total endpoints configured:[/bold] {total_endpoints}")


AREA_DISPLAY_NAMES = {
    "audit": "Audit API",
    "data-designer": "Data Designer API",
    "datasets": "Datasets API",
    "datastore": "Datastore API",
    "embeddings": "Embeddings API",
    "entities": "Entities API",
    "evaluation": "Evaluation API",
    "filesets": "Files API",
    "guardrails": "Guardrails API",
    "iam": "IAM API",
    "inference": "Inference API",
    "intake": "Intake API",
    "jobs": "Jobs API",
    "models": "Models API",
    "platform": "Platform",
    "projects": "Projects API",
    "safe-synthesizer": "Safe Synthesizer API",
    "secrets": "Secrets API",
    "workspaces": "Workspaces API",
}

DOCS_EXCLUDED_PERMISSION_AREAS = {
    "intake",
}


def _perm_role_signature(perm_name: str, role_perms_map: Dict[str, Set[str]], ordered_roles: List[str]) -> tuple:
    """Return a tuple of booleans indicating which roles have this permission."""
    return tuple(perm_name in role_perms_map.get(role, set()) for role in ordered_roles)


def _perm_prefix(perm_name: str) -> str:
    """Return the prefix before the last dot (e.g. 'audit.configs' from 'audit.configs.read')."""
    parts = perm_name.rsplit(".", 1)
    return parts[0] if len(parts) == 2 else ""


def _perm_action(perm_name: str) -> str:
    """Return the action after the last dot (e.g. 'read' from 'audit.configs.read')."""
    return perm_name.rsplit(".", 1)[-1]


_ACTION_ORDER = {
    "read": 0,
    "list": 1,
    "create": 2,
    "update": 3,
    "delete": 4,
}


def _action_sort_key(action: str) -> tuple:
    """Return a sort key that puts common CRUD actions in natural order."""
    return (_ACTION_ORDER.get(action, 100), action)


def _build_docs_area_groups(registry: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Group registered permissions by area, excluding areas hidden from public docs."""
    area_groups: Dict[str, List[str]] = {}
    for perm_name in sorted(registry.keys()):
        area = perm_name.split(".")[0]
        if area in DOCS_EXCLUDED_PERMISSION_AREAS:
            continue
        area_groups.setdefault(area, []).append(perm_name)
    return area_groups


def _build_grouped_rows(
    perm_names: List[str],
    registry: Dict[str, Dict],
    role_perms_map: Dict[str, Set[str]],
    ordered_roles: List[str],
) -> List[str]:
    """Build table rows, grouping consecutive permissions with the same prefix and role signature."""
    rows: List[str] = []

    # Collect (prefix, role_signature, perm_name) tuples, sorted so that
    # permissions with the same prefix are grouped by role signature.
    entries = []
    for perm_name in perm_names:
        prefix = _perm_prefix(perm_name)
        sig = _perm_role_signature(perm_name, role_perms_map, ordered_roles)
        entries.append((prefix, sig, perm_name))

    # Sort by prefix, then role signature (broader access first), then natural action order.
    # Negate booleans so (True, True, True) sorts before (False, True, True).
    entries.sort(key=lambda e: (e[0], tuple(not v for v in e[1]), _action_sort_key(_perm_action(e[2]))))

    # Group consecutive entries that share the same prefix AND role signature
    i = 0
    while i < len(entries):
        prefix, sig, perm_name = entries[i]
        group = [perm_name]
        j = i + 1
        while j < len(entries) and entries[j][0] == prefix and entries[j][1] == sig and prefix:
            group.append(entries[j][2])
            j += 1

        meta = registry[group[0]]
        has_endpoint = meta.get("has_endpoint", True) if meta else True
        policy_suffix = " *(policy-enforced)*" if not has_endpoint else ""

        role_marks = " | ".join("✓" if v else "" for v in sig)

        if len(group) == 1:
            description = (meta.get("description", "") if meta else "") or ""
            if policy_suffix:
                description += policy_suffix
            rows.append(f"| `{perm_name}` | {description} | {role_marks} |")
        else:
            actions = [_perm_action(p) for p in group]
            actions_str = " &#124; ".join(actions)
            display = f"<code>{prefix}.({actions_str})</code>"
            # Build a combined description from the common prefix
            desc_parts = prefix.split(".")
            area_label = (
                AREA_DISPLAY_NAMES.get(desc_parts[0], desc_parts[0].replace("-", " ")).removesuffix(" API").lower()
            )
            resource = desc_parts[1] if len(desc_parts) > 1 else ""
            action_words = ", ".join(a.replace("_", " ") for a in actions)
            description = f"{action_words.capitalize()} {area_label} {resource}".strip()
            if policy_suffix:
                description += policy_suffix
            rows.append(f"| {display} | {description} | {role_marks} |")

        i = j

    return rows


def _generate_permissions_reference(auth_config: Dict) -> str:
    """Generate the permissions reference markdown content from the auth config."""
    current_year = datetime.now().year
    registry = extract_registered_permissions(auth_config)
    roles_data = auth_config.get("authz", {}).get("roles", {})

    role_perms_map: Dict[str, Set[str]] = {}
    for role_name in roles_data:
        role_perms_map[role_name] = extract_role_permissions_recursive(roles_data, role_name)

    area_groups = _build_docs_area_groups(registry)
    ordered_roles = ["Viewer", "Editor", "Admin", "JobRunner"]

    lines: List[str] = []
    lines.append("---")
    lines.append(
        f"# SPDX-FileCopyrightText: Copyright (c) 2025-{current_year} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    )
    lines.append("# SPDX-License-Identifier: Apache-2.0")
    lines.append('title: "Permissions Reference"')
    lines.append('description: ""')
    lines.append("---")
    lines.append("")
    lines.append(
        "{/* This page is generated from the auth configuration. "
        "Regenerate it with `uv run python services/core/auth/scripts/auth-tools.py generate-docs`. */}"
    )
    lines.append("")
    lines.append(
        "Complete reference of all permissions across the NeMo Helix APIs. "
        "Each permission controls access to a specific operation within an individual API. "
        "Permissions are assigned to users through "
        "[roles](/documentation/access-control/authorization/roles-and-permissions)."
    )
    lines.append("")
    lines.append(
        "For token-level access restrictions, see "
        "[API Scopes](/documentation/access-control/authorization/api-scopes). "
        "For the RBAC model, see [Authorization Concepts](/documentation/access-control/concepts)."
    )
    lines.append("")
    lines.append("<Note>")
    lines.append("")
    lines.append(
        "HelixAdmin is omitted — it bypasses permission checks entirely at the policy level. "
        "JobRunner is intended for workload identities, not interactive users."
    )
    lines.append("")
    lines.append("</Note>")

    for area, perm_names in area_groups.items():
        display_name = AREA_DISPLAY_NAMES.get(area, area.replace("-", " ").title())
        lines.append(f"## {display_name}")
        lines.append("")

        header_cells = ["Permission", "Description", *ordered_roles]
        alignment_cells = ["------------", "-------------", *[":------:" for _ in ordered_roles]]
        lines.append(f"| {' | '.join(header_cells)} |")
        lines.append(f"| {' | '.join(alignment_cells)} |")

        rows = _build_grouped_rows(perm_names, registry, role_perms_map, ordered_roles)
        lines.extend(rows)

        lines.append("")

    lines.append("## Related")
    lines.append("")
    lines.append(
        "- [Roles & Permissions](/documentation/access-control/authorization/roles-and-permissions) "
        "— Role descriptions and hierarchy."
    )
    lines.append(
        "- [API Scopes](/documentation/access-control/authorization/api-scopes) — Token-level scope restrictions."
    )
    lines.append(
        "- [Authorization Concepts](/documentation/access-control/concepts) "
        "— Workspaces, roles, bindings, and the RBAC model."
    )
    lines.append(
        "- [Security Model](/documentation/access-control/security-model) — Trust boundaries and authorization layers."
    )
    lines.append("")

    return "\n".join(lines)


@app.command("generate-docs")
def generate_docs(
    auth_path: Path = typer.Option(
        None, "--auth", "-a", help="Path to static authorization configuration file (relative to project root)"
    ),
    output_path: Path = typer.Option(
        None, "--output", "-o", help="Output path for the generated markdown file (relative to project root)"
    ),
):
    """Generate a permissions reference documentation page from the auth configuration."""
    project_root = get_project_root()
    if auth_path is None:
        auth_path = (
            project_root
            / "services"
            / "core"
            / "auth"
            / "src"
            / "nhx"
            / "core"
            / "auth"
            / "assets"
            / "static-authz.yaml"
        )
    else:
        auth_path = project_root / auth_path

    if output_path is None:
        output_path = project_root / "docs" / "auth" / "authorization" / "permissions-reference.mdx"
    else:
        output_path = project_root / output_path

    console.print("[bold]Generating permissions reference documentation...[/bold]")

    auth_config = load_yaml(auth_path)
    content = _generate_permissions_reference(auth_config)
    registry = extract_registered_permissions(auth_config)
    area_groups = _build_docs_area_groups(registry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    permission_count = sum(len(perm_names) for perm_names in area_groups.values())
    console.print(f"[green]✓ Generated permissions reference at {output_path}[/green]")
    console.print(f"[dim]{permission_count} permissions across {len(area_groups)} areas[/dim]")


def extract_role_permissions_recursive(
    roles_data: Dict, role_name: str, visited: Optional[Set[str]] = None
) -> Set[str]:
    """Recursively get all permissions for a role, including inherited ones."""
    if visited is None:
        visited = set()
    if role_name in visited or role_name not in roles_data:
        return set()
    visited.add(role_name)
    role = roles_data[role_name]
    perms = set(role.get("permissions", []))
    for included_role in role.get("includes", []):
        perms.update(extract_role_permissions_recursive(roles_data, included_role, visited))
    return perms


@app.command("sync-plugins")
def sync_plugins(
    auth_path: Path = typer.Option(
        None, "--auth", "-a", help="Path to static authorization configuration file (relative to project root)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would change without writing"),
):
    """Merge derived plugin authz (``@path_rule``-decorated routes) into static-authz.yaml.

    Run from the repo root with workspace plugins installed (``uv sync``). This
    materializes runtime plugin policy into the committed bundle for CI and
    environments that only load ``static-authz.yaml``.
    """
    project_root = get_project_root()
    if auth_path is None:
        auth_path = (
            project_root
            / "services"
            / "core"
            / "auth"
            / "src"
            / "nhx"
            / "core"
            / "auth"
            / "assets"
            / "static-authz.yaml"
        )
    else:
        auth_path = project_root / auth_path

    try:
        from nhx.core.auth.app.bundle import get_degraded_plugins, merge_plugin_authz_contributions
    except ImportError as exc:
        console.print(f"[red]Cannot import plugin authz merge: {exc}[/red]")
        console.print("[yellow]Run from repo root with workspace packages installed (uv sync).[/yellow]")
        raise typer.Exit(code=1) from exc

    auth_config = load_yaml(auth_path)
    before_endpoints = set(auth_config.get("authz", {}).get("endpoints", {}).keys())
    # Route through the SAME fail-mode merge the running auth service uses (discover →
    # on_invalid_plugin fencing/quarantine → merge), so the committed static-authz.yaml cannot
    # diverge from runtime and fail-open for a plugin that can't be enumerated (a raw
    # discover_authz_contribution_dicts() pass would skip denied_plugin_prefixes).
    merged = merge_plugin_authz_contributions(auth_config)
    after_endpoints = set(merged.get("authz", {}).get("endpoints", {}).keys())
    added_paths = sorted(after_endpoints - before_endpoints)

    degraded = get_degraded_plugins()
    if degraded:
        console.print(f"[yellow]⚠ {len(degraded)} plugin(s) contributed invalid authz (denied / fenced):[/yellow]")
        for key, problems in sorted(degraded.items()):
            console.print(f"  [yellow]![/yellow] {key}: {'; '.join(problems)}")

    console.print("[bold]Merging plugin authz contributions...[/bold]")
    for path in added_paths:
        methods = sorted(merged["authz"]["endpoints"][path].keys())
        console.print(f"  [green]+[/green] {path} ({', '.join(methods)})")

    if not added_paths:
        console.print("[dim]No new endpoint paths (contributions may already be present).[/dim]")

    if dry_run:
        console.print("[dim]Dry run — not writing file.[/dim]")
        return

    sorted_endpoints = {}
    for path in sorted(merged["authz"]["endpoints"].keys()):
        sorted_methods = dict(sorted(merged["authz"]["endpoints"][path].items()))
        sorted_endpoints[path] = sorted_methods
    merged["authz"]["endpoints"] = sorted_endpoints

    save_yaml(auth_path, merged)
    console.print(f"[green]✓ Updated {auth_path}[/green]")


if __name__ == "__main__":
    app()
