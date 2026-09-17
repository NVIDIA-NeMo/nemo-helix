#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import inspect
import json
import multiprocessing
import os
import queue
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from nmp.common.api.utils import clear_query_param_schemas, register_query_param_schemas
from nmp.common.version import platform_api_version
from uvicorn.importer import import_from_string

from .openapi_helper.openapi_tools import (
    copy_tags,
    fix_openai_streaming_endpoints,
    fix_recursive_schemas,
    fix_ref_with_additional_props,
    hoist_nested_defs,
    include_examples,
    load_openapi_spec,
    mark_direct_span_json_value_for_stainless,
    merge_specs,
    order_endpoints_by_tags,
    remove_endpoint,
    remove_invalid_components,
    remove_unused_schemas,
    rename_schema_references,
    reorder_spec,
    save_openapi_spec,
    set_verbose,
    tweak_spec,
    update_object_type,
    validate_refs,
)
from .openapi_helper.plugin_config import PluginConfig, discover_plugins

# ANSI color codes for better readability
GREEN = "\033[32m"
RED = "\033[31m"
NC = "\033[0m"  # No Color


# Global verbose flag
VERBOSE = False
PLUGIN_WORKER_TIMEOUT_SECONDS = 300


def print_green(message: str, verbose_only: bool = False):
    """Print message in green color."""
    if not verbose_only or VERBOSE:
        print(f"{GREEN}{message}{NC}")


def print_red(message: str, verbose_only: bool = False):
    """Print message in red color."""
    if not verbose_only or VERBOSE:
        print(f"{RED}{message}{NC}")


def print_verbose(message: str):
    """Print message only if verbose mode is enabled."""
    if VERBOSE:
        print(message)


def _emit_worker_output(output: str, force: bool = False) -> None:
    """Print captured subprocess output for verbose runs and failures."""
    if output and (VERBOSE or force):
        print(output, end="" if output.endswith("\n") else "\n")


def _capture_generation_output(fn, item, verbose: bool) -> tuple[str, bool, str, str]:
    """Run an OpenAPI extraction worker while capturing noisy import-time output."""
    global VERBOSE
    VERBOSE = verbose
    set_verbose(verbose)

    output = StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        name, success, error = fn(item)
    return name, success, error, output.getvalue()


def extract_openapi_spec_captured(service: ServiceConfig, verbose: bool) -> tuple[str, bool, str, str]:
    return _capture_generation_output(extract_openapi_spec, service, verbose)


def extract_plugin_openapi_spec_captured(plugin: PluginConfig, verbose: bool) -> tuple[str, bool, str, str]:
    return _capture_generation_output(extract_plugin_openapi_spec, plugin, verbose)


def bounded_worker_count(
    item_count: int,
    requested_workers: int | None = None,
    default_limit: int = 4,
    *,
    cap_by_cpu: bool = True,
) -> int:
    """Return a conservative process count for isolated OpenAPI worker pools."""
    if item_count <= 0:
        return 0
    if requested_workers is not None:
        if requested_workers < 1:
            msg = "--plugin-workers must be at least 1"
            raise ValueError(msg)
        return min(item_count, requested_workers)

    cpu_count = os.cpu_count() or 1
    if cap_by_cpu:
        return min(item_count, default_limit, cpu_count)
    return min(item_count, default_limit)


def plugin_multiprocessing_context():
    """Use the cheapest process start method that preserves plugin isolation.

    macOS is excluded from ``fork``: the plugin imports pull in libraries that
    start threads and touch Objective-C frameworks, and the ObjC runtime aborts
    (SIGABRT) in a forked child when a framework initializer may have been
    running on another thread at fork time. Workers die with exit code -6 after
    printing ``+[NSCharacterSet initialize] may have been in progress in another
    thread when fork() was called``.
    """
    available = multiprocessing.get_all_start_methods()
    start_method = "fork" if ("fork" in available and sys.platform != "darwin") else "spawn"
    return multiprocessing.get_context(start_method)


class SpecType(Enum):
    GA = "ga"
    EA = "ea"
    UTIL = "util"


@dataclass
class ServiceConfig:
    """Configuration for a service's OpenAPI spec generation."""

    name: str
    app_import: str
    output_file: str
    app_dir: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    copy_from: Optional[str] = None  # For deployment-management
    spec_type: SpecType = SpecType.GA

    def is_ga(self) -> bool:
        return self.spec_type == SpecType.GA or self.spec_type == SpecType.UTIL

    def is_ea(self) -> bool:
        return self.spec_type == SpecType.EA

    def temp_output_path(self) -> str:
        return f"openapi/{self.output_file}"

    def final_output_path(self) -> str | None:
        match self.spec_type:
            case SpecType.GA:
                return "openapi/ga/individual/" + self.output_file
            case SpecType.EA:
                return "openapi/ea/individual/" + self.output_file
            case SpecType.UTIL:
                return None


# Define all service configurations
SERVICES = [
    ServiceConfig(
        name="platform",
        app_import="nmp.platform_runner.server:create_platform_openapi_app",
        output_file="platform.openapi.yaml",
        app_dir="/packages/nmp_platform_runner/src",
        # Aggregate platform OpenAPI is generated without plugin services by default.
        env_vars={"NEMO_PLUGIN_SERVICES_ALLOWLIST": ""},
    )
]


FINAL_SPEC_FILES = [
    "openapi/openapi.yaml",
    "openapi/ga/openapi.yaml",
    "openapi/ea/openapi.yaml",
]


def extract_openapi_spec(service: ServiceConfig) -> tuple[str, bool, str]:
    """Extract OpenAPI spec for a service using the extract-openapi.py logic directly.

    Returns:
        tuple: (service_name, success, error_message)
    """
    try:
        print_green(f"=== Generating {service.name} OpenAPI spec... ===")

        # Clear any query-param schema registrations left over from a prior
        # service loaded into this process (no-op under the subprocess-per-service
        # path; load-bearing when services share a process).

        clear_query_param_schemas()

        if service.copy_from:
            temp_path = service.temp_output_path()

            # Ensure directories exist
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            # Copy to both locations
            shutil.copy(service.copy_from, temp_path)
            return service.name, True, ""

        # Set environment variables if specified
        old_env = {}
        if service.env_vars:
            for key, value in service.env_vars.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value

        app_path_to_remove = None
        try:
            # Add app directory to sys.path if specified
            if service.app_dir:
                app_path = service.app_dir
                if app_path.startswith("/"):
                    app_path = app_path[1:]  # Remove leading slash
                sys.path.insert(0, app_path)
                app_path_to_remove = app_path

            # Import the app
            print_verbose(f"importing app from {service.app_import}")
            app = import_from_string(service.app_import)
            # In case we're interacting with something that wraps the app, like a service object
            if hasattr(app, "app"):
                app = app.app
            if inspect.isfunction(app):
                app = app()
            openapi = app.openapi()
            # Inject filter/search schemas registered via generate_openapi_extra_params
            # (their $refs are in query parameters, not walked by pydantic's response model scan).
            openapi = register_query_param_schemas(openapi)
            version = openapi.get("openapi", "unknown version")

            print_verbose(f"writing openapi spec v{version}")

            temp_path = service.temp_output_path()

            # Ensure directories exist
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            # Write to both locations
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write("# This file is generated by extract-openapi.py and should not be changed by hand.\n")
                yaml.dump(openapi, f, sort_keys=False)

            print_verbose(f"spec written to {temp_path}")
            return service.name, True, ""

        finally:
            # Restore environment variables
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

            # Remove app directory from sys.path if it was added
            if app_path_to_remove is not None and app_path_to_remove in sys.path:
                sys.path.remove(app_path_to_remove)

    except Exception as e:
        error_msg = f"Failed to generate {service.name}: {str(e)}\n{traceback.format_exc()}"
        print_red(error_msg)
        return service.name, False, error_msg


def extract_openapi_specs_sequential(services: List[ServiceConfig]) -> None:
    """Extract OpenAPI specs for multiple services sequentially in isolated subprocesses.

    This provides the same isolation as process mode (preventing schema contamination)
    but processes services one-at-a-time in order, making it easier to debug failures.
    Creates a FRESH ProcessPoolExecutor for each service to ensure complete isolation.
    """
    print_green(f"=== Generating OpenAPI specs for {len(services)} services sequentially (isolated) ===")

    if not services:
        return

    # Create a fresh ProcessPoolExecutor for each service to ensure complete isolation
    # This prevents schema cache pollution between services
    failed_services = []
    for service in services:
        # Create a new executor for each service - this ensures a fresh process
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(extract_openapi_spec_captured, service, VERBOSE)
            try:
                name, success, error, output = future.result()
                _emit_worker_output(output, force=not success)
                if success:
                    print_green(f"Completed: {name}")
                else:
                    failed_services.append((name, error))
                    print_red(f"Failed: {name}")
            except Exception as e:
                failed_services.append((service.name, str(e)))
                print_red(f"Exception in {service.name}: {str(e)}")
        # Executor is closed here, ensuring the process is terminated

    if failed_services:
        print_red(f"\n{len(failed_services)} services failed:")
        for name, error in failed_services:
            print_red(f"  - {name}: {error}")
        msg = f"{len(failed_services)} services failed to generate OpenAPI specs"
        raise RuntimeError(msg)
    else:
        print_green(f"All {len(services)} services completed successfully!")


def extract_openapi_specs_with_process_pool(services: List[ServiceConfig]) -> None:
    """Extract OpenAPI specs using ProcessPoolExecutor to avoid import conflicts.

    ProcessPoolExecutor is preferred for this workload because:
    - Complete import isolation prevents SQLAlchemy circular import issues
    - CPU-bound work (module imports and OpenAPI generation) benefits from separate processes
    - Each service is independent with minimal data sharing
    - Crash isolation prevents one service failure from affecting others
    """
    print_green(f"=== Generating OpenAPI specs for {len(services)} services using process pool ===")

    if not services:
        return

    # ProcessPoolExecutor isolates imports in separate processes
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_service = {
            executor.submit(extract_openapi_spec_captured, service, VERBOSE): service for service in services
        }

        # Collect results
        failed_services = []
        for future in as_completed(future_to_service):
            service = future_to_service[future]
            try:
                name, success, error, output = future.result()
                _emit_worker_output(output, force=not success)
                if success:
                    print_green(f"Completed: {name}")
                else:
                    failed_services.append((name, error))
                    print_red(f"Failed: {name}")
            except Exception as e:
                failed_services.append((service.name, str(e)))
                print_red(f"Exception in {service.name}: {str(e)}")

        if failed_services:
            print_red(f"\n{len(failed_services)} services failed:")
            for name, error in failed_services:
                print_red(f"  - {name}: {error}")
            msg = f"{len(failed_services)} services failed to generate OpenAPI specs"
            raise RuntimeError(msg)
        else:
            print_green(f"All {len(services)} services completed successfully!")


def extract_openapi_specs_auto(services: List[ServiceConfig]) -> None:
    """Extract OpenAPI specs for multiple services with ProcessPool and sequential fallback."""
    print_green(f"=== Generating OpenAPI specs for {len(services)} services with robust execution ===")

    if not services:
        return

    # Strategy 1: Try ProcessPoolExecutor first (recommended approach)
    try:
        print_verbose("Attempting parallel execution with ProcessPoolExecutor...")
        extract_openapi_specs_with_process_pool(services)
        print_green("✓ Successfully used ProcessPoolExecutor (recommended for consistent schemas)")
        return
    except Exception as process_error:
        print_red(f"ProcessPoolExecutor failed: {process_error}")

    # Strategy 2: Final fallback to sequential execution
    print_red("⚠️  Using sequential mode - schemas may be incomplete!")
    extract_openapi_specs_sequential(services)


@contextlib.contextmanager
def data_designer_plugin_allowlist(plugin_names: List[str] | None):
    """Limit Data Designer plugin discovery while generating one plugin OpenAPI spec.

    Data Designer builds config unions at import time from the global
    ``data_designer.plugins`` entry-point group. In the monorepo, unrelated
    installed packages can contribute Data Designer plugins, which can leak into
    another plugin's OpenAPI schema. Scope discovery to the plugins explicitly
    needed by the spec being generated.
    """

    if plugin_names is None:
        yield
        return

    allowed = set(plugin_names)
    original_entry_points = importlib.metadata.entry_points

    def scoped_entry_points(*args, **kwargs):
        result = original_entry_points(*args, **kwargs)
        group = kwargs.get("group")
        if group == "data_designer.plugins":
            return [entry_point for entry_point in result if entry_point.name in allowed]
        return result

    importlib.metadata.entry_points = scoped_entry_points
    try:
        yield
    finally:
        importlib.metadata.entry_points = original_entry_points


def extract_plugin_openapi_spec(plugin: PluginConfig) -> tuple[str, bool, str]:
    """Extract a single plugin's OpenAPI spec via the convention loader.

    Runs in a subprocess (ProcessPoolExecutor target). Avoids the
    ``import_from_string`` path used for services because plugin app construction
    needs a runtime argument (the entry-point name) — instead it imports
    ``build_plugin_app`` directly and calls it with the resolved service name.
    Plugins that need special construction set ``factory_override`` and we use
    the standard import path for them.

    Returns ``(plugin_dir, success, error_message)``.
    """
    try:
        print_green(f"=== Generating {plugin.dir} plugin OpenAPI spec... ===")
        clear_query_param_schemas()

        # Set environment variables if specified
        old_env = {}
        if plugin.env_vars:
            for key, value in plugin.env_vars.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value

        try:
            with data_designer_plugin_allowlist(plugin.data_designer_plugin_allowlist):
                return _extract_plugin_openapi_spec(plugin)

        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    except Exception as e:
        error_msg = f"Failed to generate plugin {plugin.dir}: {str(e)}\n{traceback.format_exc()}"
        print_red(error_msg)
        return plugin.dir, False, error_msg


def _extract_plugin_openapi_spec(plugin: PluginConfig) -> tuple[str, bool, str]:
    if plugin.factory_override:
        print_verbose(f"importing plugin app from {plugin.factory_override}")
        app = import_from_string(plugin.factory_override)
        if hasattr(app, "app"):
            app = app.app
        if inspect.isfunction(app):
            app = app()
    else:
        from .openapi_helper.plugin_loader import build_plugin_app

        service_name = plugin.resolve_service_name()
        print_verbose(f"building plugin app for nemo.services entry '{service_name}'")
        app = build_plugin_app(service_name)

    openapi = app.openapi()
    openapi = register_query_param_schemas(openapi)
    version = openapi.get("openapi", "unknown version")
    print_verbose(f"writing plugin openapi spec v{version}")

    output_path = plugin.output_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# This file is generated by extract-openapi.py and should not be changed by hand.\n")
        yaml.dump(openapi, f, sort_keys=False)

    print_verbose(f"plugin spec written to {output_path}")
    return plugin.dir, True, ""


def extract_plugin_openapi_spec_to_queue(plugin: PluginConfig, verbose: bool, result_queue) -> None:
    """Extract one plugin spec in a child process and return a captured result."""
    try:
        result_queue.put(extract_plugin_openapi_spec_captured(plugin, verbose))
    except BaseException as exc:
        error_msg = f"Plugin worker for {plugin.dir} crashed: {exc}\n{traceback.format_exc()}"
        result_queue.put((plugin.dir, False, error_msg, ""))


def extract_plugin_spec_batch(plugin_batch: list[PluginConfig]) -> list[tuple[str, bool, str, str]]:
    """Extract a batch with exactly one child process per plugin."""
    context = plugin_multiprocessing_context()
    processes = []
    for plugin in plugin_batch:
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=extract_plugin_openapi_spec_to_queue,
            args=(plugin, VERBOSE, result_queue),
        )
        process.start()
        deadline = time.monotonic() + PLUGIN_WORKER_TIMEOUT_SECONDS
        processes.append((plugin, process, result_queue, deadline))

    results = []
    for plugin, process, result_queue, deadline in processes:
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if process.is_alive():
                        process.kill()
                        process.join()
                        name = plugin.dir
                        success = False
                        output = ""
                        error = f"Plugin worker for {plugin.dir} timed out after {PLUGIN_WORKER_TIMEOUT_SECONDS}s"
                    else:
                        process.join()
                        try:
                            name, success, error, output = result_queue.get(timeout=1.0)
                        except queue.Empty:
                            name = plugin.dir
                            success = False
                            output = ""
                            error = (
                                f"Plugin worker for {plugin.dir} exited with code {process.exitcode} "
                                "without returning a result"
                            )
                    break

                try:
                    name, success, error, output = result_queue.get(timeout=min(0.1, remaining))
                    break
                except queue.Empty:
                    if not process.is_alive():
                        process.join()
                        try:
                            name, success, error, output = result_queue.get(timeout=1.0)
                        except queue.Empty:
                            name = plugin.dir
                            success = False
                            output = ""
                            error = (
                                f"Plugin worker for {plugin.dir} exited with code {process.exitcode} "
                                "without returning a result"
                            )
                        break
            process.join()
        except Exception as e:
            error = f"Plugin worker failed during collection {e}"
        finally:
            result_queue.close()
            result_queue.join_thread()

        if process.exitcode not in (0, None) and success:
            success = False
            error = f"Plugin worker for {plugin.dir} exited with code {process.exitcode}"
        results.append((name, success, error, output))

    return results


def extract_plugin_specs_with_process_pool(plugins: List[PluginConfig], max_workers: int | None = None) -> None:
    """Extract OpenAPI specs for plugins via isolated subprocesses.

    Each plugin runs in a process that handles at most one plugin. That matters
    on Linux: discovering services for plugin A can import route modules for
    plugin B, which registers query-param filter schemas at import time; plugin
    B's extraction then calls clear_query_param_schemas() and openapi() without
    re-importing those routes, leaving dangling ``#/components/schemas/*Filter``
    refs (e.g. MetricFilter). ``max_tasks_per_child=1`` preserves that isolation
    while still allowing multiple plugins to generate concurrently.
    """
    if not plugins:
        return

    worker_count = bounded_worker_count(len(plugins), max_workers, default_limit=12, cap_by_cpu=False)
    print_green(
        f"=== Generating OpenAPI specs for {len(plugins)} plugin(s) using {worker_count} isolated worker(s) ==="
    )

    failed_plugins = []
    for start in range(0, len(plugins), worker_count):
        plugin_batch = plugins[start : start + worker_count]
        for name, success, error, output in extract_plugin_spec_batch(plugin_batch):
            _emit_worker_output(output, force=not success)
            if success:
                print_green(f"Completed: {name}")
            else:
                failed_plugins.append((name, error))
                print_red(f"Failed: {name}")

    if failed_plugins:
        print_red(f"\n{len(failed_plugins)} plugins failed:")
        for name, error in failed_plugins:
            print_red(f"  - {name}: {error}")
        raise RuntimeError(f"{len(failed_plugins)} plugins failed to generate OpenAPI specs")

    print_green(f"All {len(plugins)} plugin(s) completed successfully!")


def apply_standard_schema_fixes(spec: dict, apply_reorder: bool = True) -> dict:
    """Apply common schema normalization after path-specific edits."""
    spec = tweak_spec(spec)
    spec = hoist_nested_defs(spec)
    spec = remove_unused_schemas(spec)
    spec = remove_invalid_components(spec)
    spec = fix_recursive_schemas(spec)
    spec = update_object_type(spec)
    spec = mark_direct_span_json_value_for_stainless(spec)
    spec["openapi"] = "3.1.0"
    spec["info"]["version"] = platform_api_version

    if apply_reorder:
        spec = reorder_spec(spec)

    return spec


def apply_schema_fixes_to_spec(spec: dict, spec_file: str, apply_reorder: bool = True) -> dict:
    """Apply schema fixes to one OpenAPI spec."""
    # Endpoints stripped from the public OpenAPI spec (not exposed in SDK).
    # Includes health/status and internal endpoints.
    health_endpoints = [
        "/status",
        "/metrics",
        "/cluster-info",
        "/health/live",
        "/health/ready",
    ]

    # Hoist nested `$defs` up front so every downstream pass (including
    # remove_endpoint → remove_unused_schemas → build_schema_tree) can resolve
    # refs like ``#/components/schemas/DatetimeFilter`` against top-level
    # components instead of hunting through inline ``$defs``.
    spec = hoist_nested_defs(spec)
    for endpoint in health_endpoints:
        remove_endpoint(spec, endpoint, prune_unused=False)

    # Special handling for deployment management
    if "platform" in spec_file:
        print_verbose(f"Applying deployment management specific fixes to {spec_file}")
        # Rename schema
        if "components" in spec and "schemas" in spec["components"]:
            schemas = spec["components"]["schemas"]
            if "PageResponse" in schemas:
                schemas["DeploymentsPage"] = schemas.pop("PageResponse")
                rename_schema_references(spec, "PageResponse", "DeploymentsPage")

        # Remove endpoints
        remove_endpoint(spec, "/v1/deployments", prune_unused=False)
        remove_endpoint(spec, "/v1/deployments/{deploymentId}", prune_unused=False)

    if "platform" in spec_file:
        # Remove internal endpoints (not part of public API)
        internal_endpoints = [p for p in spec.get("paths", {}).keys() if p.startswith("/internal/")]
        for endpoint in internal_endpoints:
            remove_endpoint(spec, endpoint, prune_unused=False)

    # Apply streaming fixes for specific files
    if any(name in spec_file for name in ["platform"]):
        print_verbose(f"Applying streaming fixes to {spec_file}")
        spec = fix_openai_streaming_endpoints(spec)

    return apply_standard_schema_fixes(spec, apply_reorder=apply_reorder)


def apply_schema_fixes(spec_files: List[str], apply_reorder: bool = True) -> None:
    """Apply schema fixes to a list of OpenAPI spec files."""
    print_green("=== Applying fixes to OpenAPI schemas ===")

    for spec_file in spec_files:
        if os.path.exists(spec_file):
            print_verbose(f"Fixing schema for {spec_file}")
            spec = load_openapi_spec(spec_file)
            spec = apply_schema_fixes_to_spec(spec, spec_file, apply_reorder=apply_reorder)
            save_openapi_spec(spec, spec_file)


def merge_and_process_specs() -> None:
    """Merge OpenAPI specs and apply final processing."""
    print_green("=== STEP 3: Merging all OpenAPI specs into a single file ===")

    # Create directories
    os.makedirs("openapi/ga", exist_ok=True)
    os.makedirs("openapi/ea", exist_ok=True)

    # GA specs
    ga_specs = [service.temp_output_path() for service in SERVICES if service.is_ga()]

    # Load and merge GA specs
    ga_specs_with_files = []
    for spec_file in ga_specs:
        if os.path.exists(spec_file):
            ga_specs_with_files.append((load_openapi_spec(spec_file), spec_file))

    if ga_specs_with_files:
        merged_ga = merge_specs(ga_specs_with_files, keep_versions=True)
        save_openapi_spec(merged_ga, "openapi/ga/openapi.yaml")

    # EA specs
    ea_specs = [service.temp_output_path() for service in SERVICES if service.is_ea()]

    # Load and merge EA specs
    ea_specs_with_files = []
    for spec_file in ea_specs:
        if os.path.exists(spec_file):
            ea_specs_with_files.append((load_openapi_spec(spec_file), spec_file))

    if ea_specs_with_files:
        merged_ea = merge_specs(ea_specs_with_files, keep_versions=True)
        save_openapi_spec(merged_ea, "openapi/ea/openapi.yaml")


def apply_schema_removals_to_spec(spec: dict) -> dict:
    """Apply schema removals to fix inconsistencies."""
    # Remove schemas and update references
    schema_removals = [
        # ("DeploymentConfigOutput", "DeploymentConfig"),
        # ("GuardrailConfigOutput", "GuardrailConfig"),
        # ("EvaluationConfig", "EvaluationConfigOutput"),
        # ("EvaluationTarget", "EvaluationTargetOutput"),
        # ("CustomizationTarget", "CustomizationTargetOutput"),
    ]

    for old_name, new_name in schema_removals:
        if "components" in spec and "schemas" in spec["components"]:
            schemas = spec["components"]["schemas"]
            if old_name in schemas:
                del schemas[old_name]
                rename_schema_references(spec, old_name, new_name)

    return spec


def apply_schema_removals() -> None:
    """Apply schema removals to fix inconsistencies."""
    ga_spec_file = "openapi/ga/openapi.yaml"
    if os.path.exists(ga_spec_file):
        spec = load_openapi_spec(ga_spec_file)
        spec = apply_schema_removals_to_spec(spec)
        save_openapi_spec(spec, ga_spec_file)


def merge_final_specs() -> None:
    """Merge GA and EA specs into final spec."""
    print_green("=== Merging EA and GA specs ===")

    final_specs = []
    for spec_file in ["openapi/ga/openapi.yaml", "openapi/ea/openapi.yaml"]:
        if os.path.exists(spec_file):
            final_specs.append((load_openapi_spec(spec_file), spec_file))

    if final_specs:
        merged_final = merge_specs(final_specs, keep_versions=True)
        save_openapi_spec(merged_final, "openapi/openapi.yaml")


def apply_final_fixes() -> None:
    """Apply final schema fixes and cleanup."""
    print_green("=== Final removing of unused schemas ===")

    apply_schema_fixes(FINAL_SPEC_FILES)


def remove_guardrail_endpoints_from_spec(spec: dict) -> dict:
    """Remove guardrail models endpoints from all final specs."""

    guardrail_endpoints = [
        ("/v2/guardrail/models", None),
        ("/v2/guardrail/models/{model_id}", None),
    ]

    # Remove guardrail endpoints
    for endpoint, method in guardrail_endpoints:
        remove_endpoint(spec, endpoint, method, prune_unused=False)

    # Apply schema fixes after removals, but do not reorder to preserve tag ordering.
    return apply_standard_schema_fixes(spec, apply_reorder=False)


def remove_guardrail_endpoints() -> None:
    """Remove guardrail models endpoints from all final specs."""

    for spec_file in FINAL_SPEC_FILES:
        if os.path.exists(spec_file):
            spec = load_openapi_spec(spec_file)
            spec = remove_guardrail_endpoints_from_spec(spec)
            save_openapi_spec(spec, spec_file)


def add_examples_and_finalize() -> None:
    """Add examples, copy tags, and order endpoints for all final specs in one pass."""
    print_green("=== Adding examples and finalizing specs ===")

    example_files = list(sorted(Path("openapi/api-examples").glob("*.json")))
    source_file = "openapi/nmp-common.openapi.yaml"

    # Load source spec once for tag copying
    source_spec = None
    if os.path.exists(source_file):
        source_spec = load_openapi_spec(source_file)

    for spec_file in FINAL_SPEC_FILES:
        if os.path.exists(spec_file):
            print_verbose(f"Finalizing {spec_file}")
            spec = load_openapi_spec(spec_file)

            # Add examples
            for example_file in example_files:
                print_verbose(f"  Adding examples from {example_file}...")
                with open(example_file, encoding="utf-8") as f:
                    examples = json.load(f)
                spec = include_examples(spec, examples)

            # Copy tags and order endpoints
            if source_spec:
                print_verbose(f"  Copying tags and ordering endpoints for {spec_file}")
                spec = copy_tags(source_spec, spec)
                spec = order_endpoints_by_tags(spec)

            save_openapi_spec(spec, spec_file)


def move_individual_specs() -> None:
    """Move individual OpenAPI specs to separate folders."""
    print_green("=== STEP 4: Moving individual specs to separate folder ===")

    # GA individual specs
    os.makedirs("openapi/ga/individual", exist_ok=True)
    os.makedirs("openapi/ea/individual", exist_ok=True)
    file_paths = [(service.temp_output_path(), service.final_output_path()) for service in SERVICES]

    for start, end in file_paths:
        if not end:  # Don't copy util files
            os.remove(start)
        elif os.path.exists(start):
            shutil.move(start, end)


def fix_ref_not_allowed_errors(spec_files: List[str]) -> None:
    """Fix ref not allowed errors in the given spec files."""
    print_green("=== Fixing ref not allowed errors ===")

    for spec_file in spec_files:
        if os.path.exists(spec_file):
            spec = load_openapi_spec(spec_file)
            spec = fix_ref_with_additional_props(spec)
            save_openapi_spec(spec, spec_file)


def validate_final_specs(spec_files: List[str]) -> None:
    """Fail loudly if any of the given specs contains a dangling `$ref`.

    Must run before SDK generation — Stainless and Orval produce broken imports
    when they encounter refs that don't resolve, so we gate spec publishing on
    this check.
    """
    print_green("=== Validating specs for dangling $refs ===")

    all_dangling: list[tuple[str, list[str]]] = []
    for spec_file in spec_files:
        if os.path.exists(spec_file):
            spec = load_openapi_spec(spec_file)
            dangling = validate_refs(spec)
            if dangling:
                all_dangling.append((spec_file, dangling))

    if all_dangling:
        print_red("Found dangling $refs in the following spec files:")
        for spec_file, dangling in all_dangling:
            print_red(f"  {spec_file}")
            for ref in dangling:
                print_red(f"    - {ref}")
        raise RuntimeError(f"{sum(len(d) for _, d in all_dangling)} dangling $refs detected")


def can_process_single_platform_spec_in_memory(services: list[ServiceConfig]) -> bool:
    """Return true when platform outputs are known to be identical."""
    if len(services) != 1:
        return False

    service = services[0]
    if not service.is_ga() or service.final_output_path() is None or service.copy_from:
        return False

    # Tags/examples are final-spec-only transformations in the generic path.
    # Keep that path if those inputs exist so individual and aggregate outputs
    # retain their existing semantics.
    return (
        not Path("openapi/ea/openapi.yaml").exists()
        and not Path("openapi/nmp-common.openapi.yaml").exists()
        and not any(Path("openapi/api-examples").glob("*.json"))
    )


def process_single_platform_spec_in_memory(services: list[ServiceConfig]) -> bool:
    """Process the current single-platform OpenAPI layout without repeated file passes."""
    if not can_process_single_platform_spec_in_memory(services):
        return False

    service = services[0]
    temp_path = service.temp_output_path()
    final_path = service.final_output_path()
    if final_path is None or not os.path.exists(temp_path):
        return False

    print_green("=== Processing single platform OpenAPI spec in memory ===")
    spec = load_openapi_spec(temp_path)
    spec = apply_schema_fixes_to_spec(spec, temp_path)
    individual_spec = fix_ref_with_additional_props(copy.deepcopy(spec))

    final_spec = apply_schema_removals_to_spec(spec)
    final_spec = apply_schema_fixes_to_spec(final_spec, "openapi/openapi.yaml")
    final_spec = remove_guardrail_endpoints_from_spec(final_spec)
    final_spec = fix_ref_with_additional_props(final_spec)

    dangling_specs = [
        ("platform individual OpenAPI spec", validate_refs(individual_spec)),
        ("platform OpenAPI spec", validate_refs(final_spec)),
    ]
    for spec_name, dangling in dangling_specs:
        if dangling:
            print_red(f"Found dangling $refs in the {spec_name}:")
            for ref in dangling:
                print_red(f"  - {ref}")
            raise RuntimeError(f"{len(dangling)} dangling $refs detected")

    for output_path in ["openapi/openapi.yaml", "openapi/ga/openapi.yaml"]:
        save_openapi_spec(final_spec, output_path)
    save_openapi_spec(individual_spec, final_path)

    os.remove(temp_path)
    return True


def process_plugin_specs(plugin_workers: int | None = None) -> None:
    """Generate, fix, and validate OpenAPI specs for all opted-in plugins.

    Plugin specs land in ``plugins/<dir>/openapi/`` and are never merged into
    the platform spec.
    """
    plugins = discover_plugins()
    if not plugins:
        return

    print_green(f"=== STEP 5: Generating OpenAPI specs for {len(plugins)} plugin(s) ===")
    extract_plugin_specs_with_process_pool(plugins, max_workers=plugin_workers)

    plugin_spec_files = [p.output_path() for p in plugins]

    # Reuse the platform schema-fix pipeline. Branches in apply_schema_fixes
    # gated on `"platform" in spec_file` are no-ops because plugin paths live
    # under plugins/<dir>/openapi/ — no false positives by inspection.
    #
    # Schema-name collisions fail the build unconditionally: two distinct models
    # sharing a schema name with differing content is always a bug.
    apply_schema_fixes(plugin_spec_files)
    fix_ref_not_allowed_errors(plugin_spec_files)
    validate_final_specs(plugin_spec_files)


def main():
    """Main function that orchestrates the OpenAPI spec generation."""
    global VERBOSE

    parser = argparse.ArgumentParser(description="Generate OpenAPI specifications for all services")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output showing detailed progress",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["process", "auto", "sequential"],
        default="process",
        help="Execution mode: process (ProcessPoolExecutor - recommended), "
        "auto (ProcessPool with sequential fallback), sequential (no parallelism)",
    )
    parser.add_argument(
        "--only-gen-schema", action="store_true", help="Only generate the schema, then exist, for debugging purposes"
    )
    parser.add_argument(
        "--plugin-workers",
        type=int,
        default=None,
        help="Maximum isolated worker processes for plugin OpenAPI generation (default: min(12, plugins))",
    )
    args = parser.parse_args()

    # Set global verbose flag
    VERBOSE = args.verbose
    # Also set verbose flag in openapi_tools
    set_verbose(args.verbose)

    try:
        # Load environment variables from dummy.env
        env_file = Path("script/dummy.env")
        if env_file.exists():
            print_green("Loading environment variables from script/dummy.env", verbose_only=True)
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        # Handle both 'export KEY=value' and 'KEY=value' formats
                        if line.startswith("export "):
                            line = line[7:]  # Remove 'export ' prefix
                        key, value = line.split("=", 1)
                        os.environ[key] = value
                        print_verbose(f"  Set {key}={value}")
        else:
            print_green(
                "No script/dummy.env file found, skipping environment setup",
                verbose_only=True,
            )

        # Determine which services to generate specs for
        services_to_generate = SERVICES
        print_green("=== STEP 1: Generating OpenAPI specs for all services ===")

        # Generate OpenAPI specs based on execution mode
        if args.execution_mode == "process":
            extract_openapi_specs_with_process_pool(services_to_generate)
        elif args.execution_mode == "auto":
            extract_openapi_specs_auto(services_to_generate)  # Uses ProcessPool with sequential fallback
        elif args.execution_mode == "sequential":
            extract_openapi_specs_sequential(services_to_generate)
        # Apply all schema fixes in one consolidated pass
        all_spec_files = [service.temp_output_path() for service in SERVICES]
        if args.only_gen_schema:
            return

        if not process_single_platform_spec_in_memory(services_to_generate):
            apply_schema_fixes(all_spec_files)

            # Merge and process specs - use same logic for both modes
            merge_and_process_specs()

            apply_schema_removals()

            merge_final_specs()

            # Apply final fixes - use same logic for consistency
            apply_final_fixes()

            add_examples_and_finalize()

            # Remove health endpoints and apply final processing (after tag ordering)
            remove_guardrail_endpoints()

            move_individual_specs()

            platform_spec_files = [
                path for path in (service.final_output_path() for service in SERVICES) if path is not None
            ] + FINAL_SPEC_FILES
            fix_ref_not_allowed_errors(platform_spec_files)
            validate_final_specs(platform_spec_files)

        process_plugin_specs(plugin_workers=args.plugin_workers)

        print_green("=== OpenAPI spec generation completed successfully! ===")

    except Exception as e:
        traceback.print_exc()
        print_red(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
