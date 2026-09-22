# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPA bundle generation for Auth Service."""

import asyncio
import gzip
import hashlib
import io
import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Optional, Tuple

import yaml
from nmp.common.auth import ALL_WORKSPACES
from nmp.common.auth.authz_format import validate_static_authz_data
from nmp.common.config import get_service_config
from nmp.common.entities import EntityClient
from nmp.core.auth.config import AuthServiceConfig
from nmp.core.auth.entities import RoleBindingEntity

logger = logging.getLogger(__name__)

# Bundle cache configuration
_bundle_cache: Optional[Tuple[bytes, str, float]] = None  # (bundle_bytes, etag, timestamp)
_bundle_lock = asyncio.Lock()


def get_bundle_cache_seconds() -> int:
    """Get bundle cache seconds from config."""
    return get_service_config(AuthServiceConfig).bundle_cache_seconds


def clear_bundle_cache() -> None:
    """Clear the bundle cache.

    This is useful for testing to ensure each test gets a fresh bundle
    generated from its test data instead of a cached bundle from a previous test.
    """
    global _bundle_cache
    _bundle_cache = None


async def get_opa_bundle_with_etag(entities_client: Optional[EntityClient] = None) -> Tuple[bytes, str]:
    """Get OPA bundle with E-Tag support and debouncing.

    This function implements caching and debouncing to avoid regenerating the bundle
    too frequently. The bundle is cached for bundle_cache_seconds seconds.

    Database queries are explicitly ordered by id to ensure stable E-Tag generation -
    without ordering, the database might return results in different orders between
    queries, causing different E-Tags even when the data hasn't changed.

    Returns:
        Tuple[bytes, str]: Bundle bytes and E-Tag value
    """
    global _bundle_cache

    cache_seconds = get_bundle_cache_seconds()

    async with _bundle_lock:
        # Check if we have a cached bundle that's still fresh
        if _bundle_cache is not None:
            bundle_bytes, etag, timestamp = _bundle_cache
            current_time = time.time()

            # If the cache is still fresh, return it
            if current_time - timestamp < cache_seconds:
                return bundle_bytes, etag

        # Generate a new bundle
        bundle_bytes = await _build_opa_bundle_internal(entities_client)

        # Calculate E-Tag (MD5 hash of the bundle content)
        etag = hashlib.md5(bundle_bytes).hexdigest()

        # Update cache
        _bundle_cache = (bundle_bytes, etag, time.time())

        return bundle_bytes, etag


# Plugins whose authz failed validation at the most recent bundle build (key -> problems).
# Surfaced for a status/health endpoint; refreshed on every build.
_degraded_plugins: dict[str, list[str]] = {}


def get_degraded_plugins() -> dict[str, list[str]]:
    """Return plugins with invalid authz at the last bundle build (key -> list of problems)."""
    return dict(_degraded_plugins)


def _quarantine_contribution(contribution_dict: dict) -> dict:
    """Deny every route of a plugin and drop its permissions (quarantine fail-mode)."""
    return {
        "permissions": {},
        "endpoints": {
            path: {method: {"permissions": [], "deny": True} for method in methods}
            for path, methods in contribution_dict.get("endpoints", {}).items()
        },
        "role_permissions": {},
    }


def merge_plugin_authz_contributions(static_data: dict) -> dict:
    """Overlay authorization rules from installed NeMo Platform plugins.

    Applies the configured fail-mode (``authz.on_invalid_plugin``) to any plugin with
    derived authz **errors** (``PluginAuthzResult.problems``: unruled routes, malformed or
    cross-namespace permission ids, duplicate bindings, load failures). The offending routes
    are already explicit denies in the derived contribution; this only controls blast radius
    — ``deny_route`` keeps just those denies, ``quarantine`` denies the whole plugin,
    ``hard_fail`` refuses to build the bundle. Plugins with errors are recorded for the
    status endpoint.

    Plugin *warnings* (``PluginAuthzResult.warnings``: missing or conflicting permission
    descriptions) are metadata-only — the route still requires the right permission — so they
    are logged but never escalate the fail-mode and never mark the plugin degraded. This is
    what keeps a cosmetic description typo from quarantining a whole plugin.
    """
    global _degraded_plugins
    try:
        from nemo_platform_plugin.authz_discovery import discover_plugin_authz
        from nmp.common.auth.authz_merge import merge_authz_contributions
    except ImportError:
        logger.debug("Plugin authz discovery unavailable; using static authz only")
        _degraded_plugins = {}
        return static_data

    results = discover_plugin_authz()
    on_invalid = get_service_config(AuthServiceConfig).on_invalid_plugin
    degraded: dict[str, list[str]] = {}
    contributions: list[dict] = []

    denied_prefixes: list[str] = []
    for result in results:
        contribution_dict = result.contribution.to_dict()
        if result.problems:
            degraded[result.key] = result.problems
            logger.error(
                "Plugin %r contributed invalid authz (%d problem(s)); on_invalid_plugin=%s: %s",
                result.key,
                len(result.problems),
                on_invalid,
                "; ".join(result.problems),
            )
            if on_invalid == "hard_fail":
                raise RuntimeError(
                    f"Plugin {result.key!r} contributed invalid authz and "
                    f"authz.on_invalid_plugin=hard_fail: {'; '.join(result.problems)}"
                )
            # Fence the plugin's whole /apis/<name> namespace (deny-all) whenever per-route
            # coverage can't be trusted:
            #   * no route enumerated at all (load/derivation failure) — the runner may still
            #     mount this plugin via a separate instantiation; OR
            #   * quarantine — _quarantine_contribution only rewrites the routes derivation SAW,
            #     so any runner-mounted-but-unseen route would otherwise stay open.
            # deny_route keeps just the per-route denies already in the contribution. The
            # runner mounts at /apis/<service.name>; the name==key invariant is only warned, not
            # enforced, so fence both the entry-point key and the declared mount name.
            no_endpoints = not contribution_dict.get("endpoints")
            if on_invalid == "quarantine":
                contribution_dict = _quarantine_contribution(contribution_dict)
            if no_endpoints or on_invalid == "quarantine":
                denied_prefixes.append(f"/apis/{result.key}")
                if result.mount_name and result.mount_name != result.key:
                    denied_prefixes.append(f"/apis/{result.mount_name}")
        if result.warnings:
            logger.warning(
                "Plugin %r has %d authz warning(s) (non-deny — e.g. missing or conflicting "
                "permission descriptions): %s",
                result.key,
                len(result.warnings),
                "; ".join(result.warnings),
            )
        contributions.append(contribution_dict)

    _degraded_plugins = degraded
    contributions = [c for c in contributions if c.get("permissions") or c.get("endpoints")]
    if contributions:
        logger.debug("Merging %d plugin authz contribution(s)", len(contributions))
    merged = merge_authz_contributions(static_data, contributions) if contributions else static_data
    if denied_prefixes:
        logger.error("Fencing degraded plugin namespace(s) (deny-all): %s", ", ".join(sorted(set(denied_prefixes))))
        config = merged.setdefault("authz", {}).setdefault("config", {})
        existing = config.get("denied_plugin_prefixes") or []
        existing_prefixes = existing if isinstance(existing, list) else []
        config["denied_plugin_prefixes"] = sorted(set(existing_prefixes) | set(denied_prefixes))
    return merged


async def _build_authorization_data_internal(entities_client: Optional[EntityClient] = None) -> dict:
    """Build authorization data for NeMo Platform.

    This function builds the authorization data structure containing:
    - Static authorization data (roles, permissions, endpoint mappings)
    - Dynamic authorization data (principal role bindings including wildcard "*")

    Note: Role bindings (including wildcard "*" for public access) are seeded as
    entities by the platform-seed task. See nmp.core.auth.app.seeding.

    Returns:
        dict: The authorization data structure
    """
    # Get the app directory path (where policies/data are located)
    app_dir = Path(__file__).parent

    # Read and convert the static authorization data
    # The assets directory is at the package level (nmp/auth/assets)
    static_data_path = app_dir.parent / "assets" / "static-authz.yaml"
    with open(static_data_path, "r") as f:
        static_data = yaml.safe_load(f)

    static_data = merge_plugin_authz_contributions(static_data)
    validate_static_authz_data(static_data)

    # Initialize workspaces and principals if not present
    if "workspaces" not in static_data["authz"]:
        static_data["authz"]["workspaces"] = {}
    if "principals" not in static_data["authz"]:
        static_data["authz"]["principals"] = {}

    # Fetch dynamic data from EntityClient if available
    if entities_client:
        # Fetch all role bindings across ALL workspaces with pagination
        # (role bindings are stored in the workspace they grant access to)
        # Entity store API limits page_size to 1000, so we paginate through all results
        all_bindings = []
        page = 1
        while True:
            page_result = await entities_client.list(
                RoleBindingEntity,
                workspace=ALL_WORKSPACES,  # Query all workspaces
                page=page,
                page_size=1000,
                sort="created_at",
            )
            all_bindings.extend(page_result.data)
            # Check if we've fetched all pages
            if len(page_result.data) < 1000:
                break
            page += 1

        # Filter to only active bindings (revoked_at is None)
        active_bindings = [b for b in all_bindings if b.revoked_at is None]

        # Group role bindings by principal and workspace
        # This includes wildcard principal "*" which grants access to all authenticated users
        for binding in active_bindings:
            principal = binding.principal
            # The workspace field is both where the binding is stored and what it grants access to
            workspace = binding.workspace
            role = binding.role

            # Initialize principal if not exists
            if principal not in static_data["authz"]["principals"]:
                static_data["authz"]["principals"][principal] = {"workspaces": {}}

            # Initialize workspace roles list if not exists
            if workspace not in static_data["authz"]["principals"][principal]["workspaces"]:
                static_data["authz"]["principals"][principal]["workspaces"][workspace] = []

            # Add role if not already present
            if role not in static_data["authz"]["principals"][principal]["workspaces"][workspace]:
                static_data["authz"]["principals"][principal]["workspaces"][workspace].append(role)

    # Return the authorization data structure
    return static_data


async def build_authorization_data(entities_client: Optional[EntityClient] = None) -> dict:
    """Build authorization data for the policy engine.

    This extracts just the data building logic from the bundle generation,
    returning the data structure needed by the embedded WASM policy engine.

    Args:
        entities_client: Optional EntityClient for fetching dynamic data

    Returns:
        dict: Authorization data with authz.principals, authz.roles, etc.
    """
    static_data = await _build_authorization_data_internal(entities_client)
    return static_data


async def _build_opa_bundle_internal(entities_client: Optional[EntityClient] = None) -> bytes:
    """Build an OPA bundle for NeMo Platform authorization.

    This wraps _build_authorization_data_internal and packages it into a tar.gz bundle.
    """
    static_data = await _build_authorization_data_internal(entities_client)

    # Get the app directory path for policy files
    app_dir = Path(__file__).parent
    policies_dir = app_dir / "policies"

    # Auto-discover all .rego policy files
    policy_files = sorted(policies_dir.glob("*.rego"))
    policy_contents = {}
    for policy_path in policy_files:
        with open(policy_path, "r") as f:
            policy_contents[policy_path.name] = f.read()

    # Create the bundle structure
    bundle_data = {"authz": static_data["authz"], "envoy": static_data["authz"], "common": static_data["authz"]}

    # Create a stable revision based on the data content hash
    # This ensures the same data always produces the same revision
    data_content = json.dumps(bundle_data, sort_keys=True)
    revision_hash = hashlib.md5(data_content.encode()).hexdigest()[:8]

    # Create manifest with stable revision
    manifest = {"revision": revision_hash, "roots": ["authz", "envoy", "common"]}

    # Create tar bundle in memory (uncompressed first for deterministic output)
    # Note: This is CPU-intensive (gzip compression) but is mitigated by the E-Tag
    # caching mechanism that prevents regenerating unchanged bundles
    tar_io = io.BytesIO()
    with tarfile.open(fileobj=tar_io, mode="w") as tar:
        # Use a fixed timestamp for all files to ensure stable tar contents
        fixed_time = 0  # Unix epoch

        # Add all policy files (sorted alphabetically for deterministic order)
        for filename in sorted(policy_contents.keys()):
            content = policy_contents[filename]
            policy_info = tarfile.TarInfo(name=filename)
            policy_info.size = len(content.encode())
            policy_info.mtime = fixed_time
            tar.addfile(policy_info, io.BytesIO(content.encode()))

        # Add data.json with deterministic serialization
        data_json_content = json.dumps(bundle_data, sort_keys=True, indent=2)
        data_info = tarfile.TarInfo(name="data.json")
        data_info.size = len(data_json_content.encode())
        data_info.mtime = fixed_time
        tar.addfile(data_info, io.BytesIO(data_json_content.encode()))

        # Add manifest with deterministic serialization
        manifest_content = json.dumps(manifest, sort_keys=True)
        manifest_info = tarfile.TarInfo(name=".manifest")
        manifest_info.size = len(manifest_content.encode())
        manifest_info.mtime = fixed_time
        tar.addfile(manifest_info, io.BytesIO(manifest_content.encode()))

    # Get the tar bytes
    tar_io.seek(0)
    tar_bytes = tar_io.read()

    # Compress with gzip using fixed mtime for deterministic output
    bundle_io = io.BytesIO()
    with gzip.GzipFile(fileobj=bundle_io, mode="wb", mtime=0) as gz:
        gz.write(tar_bytes)

    # Get the final bundle bytes
    bundle_io.seek(0)
    return bundle_io.read()


async def build_opa_bundle_async(entities_client: Optional[EntityClient] = None) -> bytes:
    """Build an OPA bundle for NeMo Platform authorization.

    This is a backward compatibility wrapper that returns just the bundle bytes.
    For E-Tag support, use get_opa_bundle_with_etag() instead.
    """
    bundle_bytes, _ = await get_opa_bundle_with_etag(entities_client)
    return bundle_bytes


def build_opa_bundle() -> bytes:
    """Synchronous wrapper for build_opa_bundle_async without database access.

    This is for backward compatibility and when database is not available.
    """
    return asyncio.run(build_opa_bundle_async(entities_client=None))
