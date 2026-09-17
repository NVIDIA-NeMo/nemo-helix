#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenAPI spec to Stainless config model mapper.

See README.md(https://nv/nmp/tools/nemo-platform-sdk-tools/src/nemo_platform_sdk_tools/sdk/README.md) for more details.
"""

import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from caseutil import to_snake
from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPI, OpenAPIEndpoint
from nemo_platform_sdk_tools.sdk.core.stainless import StainlessConfig, StainlessMethod, StainlessModel
from nemo_platform_sdk_tools.sdk.source_owned_resources import (
    SOURCE_OWNED_RESOURCE_EXCLUSIONS,
    SourceOwnedResource,
    active_source_owned_resources,
    derive_source_owned_schema_names,
    endpoint_is_source_owned,
    filter_schema_usage_for_generated_resources,
)
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="openapi-stainless",
    help="Tools for keeping the Stainless config in sync with OpenAPI spec.",
    no_args_is_help=True,
)


@app.command("sync-methods")
def sync_stainless_methods(
    openapi_spec_path: Annotated[Path, typer.Option(help="Path to OpenAPI specification file", exists=True)],
    stainless_config_path: Annotated[Path, typer.Option(help="Path to Stainless configuration file", exists=True)],
    output_path: Annotated[Path, typer.Option(help="Output path for updated Stainless configuration")],
):
    """
    Sync OpenAPI endpoints with Stainless methods.

    This operation will:
    - Add methods for new endpoints (i.e. endpoints that exist in OpenAPI but not in Stainless)
    - Remove methods for deleted endpoints (i.e. endpoints that exist in Stainless but not in OpenAPI)
    """

    # Setup YAML loaders
    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    yaml_loader.width = 120

    # Load files
    openapi_spec = OpenAPI.from_file(openapi_spec_path)
    stainless_config = StainlessConfig.from_file(stainless_config_path)

    mapper = SchemaMapper(openapi_spec, stainless_config)

    methods_were_updated = mapper.sync_endpoints_with_methods()
    if methods_were_updated:
        _write_outputs(
            updated_resources_type="methods",
            report=mapper.generate_report(),
            output_path=output_path,
            updated_stainless_config=stainless_config,
        )
        sys.exit(1)


@app.command("sync-models")
def sync_stainless_models(
    openapi_spec_path: Annotated[Path, typer.Option(help="Path to OpenAPI specification file", exists=True)],
    stainless_config_path: Annotated[Path, typer.Option(help="Path to Stainless configuration file", exists=True)],
    output_path: Annotated[Path, typer.Option(help="Output path for updated Stainless configuration")],
    refresh_all_models: Annotated[
        bool,
        typer.Option(
            help="Refresh all models. This will recalculate and update where each model is defined. This will lead to backward incompatible changes, if models are imported in the user's code."
        ),
    ] = False,
):
    """
    Sync OpenAPI schemas with Stainless models.

    This operation will:
    - Add models for new schemas (i.e. schemas that exist in OpenAPI but not in Stainless)
    - Remove models for deleted schemas (i.e. schemas that exist in Stainless but not in OpenAPI)
    """

    # Setup YAML loaders
    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    yaml_loader.width = 120

    # Load files
    openapi_spec = OpenAPI.from_file(openapi_spec_path)
    stainless_config = StainlessConfig.from_file(stainless_config_path)

    mapper = SchemaMapper(openapi_spec, stainless_config)
    if refresh_all_models:
        mapper.clear_models()

    methods_were_updated = mapper.sync_endpoints_with_methods()
    if methods_were_updated:
        # We need to have endpoints in sync first, so we know where to put the models
        raise RuntimeError(
            "Methods in Stainless config are out-of-sync with OpenAPI spec. "
            "Please run the 'sync-stainless-methods' command first, review and update the result and run this command after."
        )

    models_were_updated = mapper.sync_schemas_with_models()
    if models_were_updated:
        _write_outputs(
            updated_resources_type="models",
            report=mapper.generate_report(),
            output_path=output_path,
            updated_stainless_config=stainless_config,
        )
        sys.exit(1)


def _write_outputs(
    *, updated_resources_type: str, report: str, output_path: Path, updated_stainless_config: StainlessConfig
):
    updated_stainless_config.write(output_path)
    with open(_report_path(output_path), "w") as f:
        f.write(report)

    print(
        f"""\033[91m
⚠️  There were changes to {updated_resources_type} in the Stainless config and manual review is required! ⚠️
Please review:
- the report at {_report_path(output_path)}
- the new config at {output_path}.
\033[0m

➡️  IMPORTANT! Once reviewed, you need to rerun the stainless.sh command!
For more information see https://nv/nmp/tools/nemo-platform-sdk-tools/src/nemo_platform_sdk_tools/sdk/README.md
"""
    )


class MappingReport:
    def __init__(self):
        self._warnings = []

    def add_warning(self, message: str):
        logger.warning(message)
        self._warnings.append(message)


class SchemaMapper:
    """
    Maps OpenAPI schemas to Stainless model locations.
    """

    def __init__(
        self,
        openapi_spec: OpenAPI,
        stainless_config: StainlessConfig,
        source_owned_resources: Sequence[SourceOwnedResource] = SOURCE_OWNED_RESOURCE_EXCLUSIONS,
    ):
        self._openapi_spec = openapi_spec
        self._stainless_config = stainless_config
        self._active_source_owned_resources = active_source_owned_resources(
            configured_resource_names=self._stainless_config.top_level_resource_names(),
            source_owned_resources=source_owned_resources,
        )

        schema_usage = self._openapi_spec.calculate_schema_to_endpoints()
        excluded_schema_names = derive_source_owned_schema_names(
            schema_usage,
            self._active_source_owned_resources,
        )
        excluded_schema_names.update(self._openapi_spec.calculate_sdk_excluded_schema_names())
        self._schemas = set(openapi_spec.schemas()) - excluded_schema_names

        # Extract existing models from Stainless config
        self._existing_models = self._stainless_config.extract_models()
        self._existing_methods = self._stainless_config.extract_methods()

        # Analyze schema usage
        self._schema_usage = filter_schema_usage_for_generated_resources(
            schema_usage,
            self._active_source_owned_resources,
        )

        # Report
        self._report = MappingReport()

    def _endpoint_is_excluded(self, endpoint: OpenAPIEndpoint) -> bool:
        return endpoint_is_source_owned(endpoint, self._active_source_owned_resources)

    def determine_schema_locations(self, *, skip_existing: bool = True) -> dict[str, list[str]]:
        """
        Determine the best Stainless location for each schema.

        Args:
            skip_existing: If True, schemas already mapped in the config are not changed.
        """
        existing_endpoints = {m.endpoint: m for m in self._existing_methods}
        existing_schemas = {}
        if skip_existing:
            existing_schemas = {m.schema_name: m for m in self._existing_models}

        schema_locations = {}

        # First, map schemas that are used in endpoints (excluding already mapped ones)
        for schema_name, endpoints in self._schema_usage.items():
            # Skip schemas that are already mapped in the config
            if schema_name in existing_schemas:
                continue

            if not endpoints:
                self._report.add_warning(
                    f"{schema_name!r} has no endpoint. These should be dropped by the OpenAPI build process."
                )
                schema_locations[schema_name] = ["$shared"]
                continue

            logger.info(
                f"Mapping schema {schema_name!r} used in {len(endpoints)} endpoints: {[str(e) for e in endpoints]}"
            )
            # Find all Stainless locations where this schema is used
            stainless_locations = []
            for endpoint in endpoints:
                if method := existing_endpoints.get(endpoint, None):
                    stainless_locations.append(method)

            if not stainless_locations:
                # No mapped endpoints - put in $shared
                schema_locations[schema_name] = ["$shared"]
                continue

            # Determine best location
            location = self._find_best_location(stainless_locations)
            schema_locations[schema_name] = location

        # Ensure all schemas are mapped (including unused ones), but skip existing
        for schema_name in self._schemas:
            if schema_name not in schema_locations and schema_name not in existing_schemas:
                schema_locations[schema_name] = ["$shared"]

        return schema_locations

    def _find_best_location(self, locations: list[StainlessMethod]) -> list[str] | None:
        """
        Find the best common location for a set of resource locations.
        Basically find the longest common prefix of the resource paths.
        """
        if len(locations) == 1:
            return locations[0].resource_path

        common_parent = []
        while True:
            if len(common_parent) >= min(len(loc.resource_path) for loc in locations):
                break

            next_candidate = locations[0].resource_path[len(common_parent)]
            if not all(loc.resource_path[len(common_parent)] == next_candidate for loc in locations):
                break

            common_parent.append(next_candidate)

        return common_parent if common_parent else ["$shared"]

    def generate_model_names(self, schema_locations: dict[str, list[str]]) -> dict[str, str]:
        """
        Generate model names for schemas with conflict resolution.
        """
        model_names = {}
        used_names = {m.model_name for m in self._existing_models}

        for schema_name, resource_path in schema_locations.items():
            # Generate preferred model name
            preferred_name = self._schema_to_model_name(schema_name)

            if preferred_name not in used_names:
                model_names[schema_name] = preferred_name
                used_names.add(preferred_name)
            else:
                # Handle conflict with semantic resolution
                resolved_name = self._resolve_naming_conflict(preferred_name, schema_name, resource_path, used_names)
                # Make sure this name gets reviewed
                resolved_name = "reviewme_" + resolved_name

                model_names[schema_name] = resolved_name
                used_names.add(resolved_name)
                self._report.add_warning(
                    f"Naming conflict for schema {schema_name!r}: preferred name {preferred_name!r} is already used. "
                    f"Resolved to {resolved_name!r}."
                )

        return model_names

    def _schema_to_model_name(self, schema_name: str) -> str:
        """
        Convert schema name to model name using naming rules.
        """
        # Special rule: types ending with 2 uppercase letters (e.g. GuardrailConfigInputGU -> guardrail_config_gu_param)
        disambiguating_suffix = re.search(r"([A-Z]{2})$", schema_name)
        two_uppercase_suffix = ""
        base_name = schema_name

        if disambiguating_suffix:
            two_uppercase_suffix = disambiguating_suffix.group(1).lower()
            base_name = schema_name[:-2]  # Remove the 2 uppercase letters

        name = to_snake(base_name)

        # Apply transformation rules
        if name.endswith("_input") and not name.startswith("input"):
            name = name[:-6]
            if two_uppercase_suffix:
                name = name + "_" + two_uppercase_suffix + "_param"
            else:
                name = name + "_param"
        elif name.endswith("_output") and not name.startswith("output"):
            name = name[:-7]
            if two_uppercase_suffix:
                name = name + "_" + two_uppercase_suffix
        else:
            if two_uppercase_suffix:
                name = name + "_" + two_uppercase_suffix

        return name

    def _resolve_naming_conflict(
        self, preferred_name: str, schema_name: str, resource_path: list[str], used_names: set[str]
    ) -> str:
        """
        Resolve naming conflicts.
        """

        # Try adding resource prefix
        if resource_path[0] != "$shared":
            resource_prefixed = f"{'_'.join(resource_path)}_{preferred_name}"
            if resource_prefixed not in used_names:
                return resource_prefixed

        # Final fallback: add numeric suffix
        counter = 2
        while f"{schema_name}_{counter}" in used_names:
            counter += 1

        return f"{schema_name}_{counter}"

    def sync_endpoints_with_methods(self) -> bool:
        endpoint_to_stainless = {m.endpoint: m for m in self._existing_methods}

        updates = False
        openapi_endpoints = [
            endpoint for endpoint in self._openapi_spec.extract_endpoints() if not self._endpoint_is_excluded(endpoint)
        ]
        for endpoint in openapi_endpoints:
            if endpoint not in endpoint_to_stainless:
                print(endpoint)
                method_name = f"reviewme_{endpoint.approx_method_name()}"
                resource_path = endpoint.approx_resource_path()

                # Check to see if there's any methods that conflict with this new method. Two potential example conflicts are
                # `list` and `reviewme_list`. `list` might already be taken by a method added in a prior commit, while `reviewme_list`
                # could've been added by an earlier iteration of this loop.
                existing_methods = [
                    self._stainless_config.find_method_for_resource_path(resource_path, mn)
                    for mn in [method_name, endpoint.approx_method_name()]
                ]
                if any(existing_methods):
                    # Try another name to disambiguate
                    method_name += "-" + endpoint.last_path_component()
                    self._report.add_warning(f"Naming conflict for endpoint {endpoint!r}, renamed to {method_name!r}")

                method = StainlessMethod(
                    method_name=method_name,
                    endpoint=endpoint,
                    resource_path=resource_path,
                )
                self._report.add_warning(
                    f"New endpoint detected: {method}. Please review the method name and update the config!"
                )

                self._stainless_config.add_method(method)
                updates = True

        openapi_endpoints_set = set(openapi_endpoints)
        for method in self._existing_methods:
            if method.endpoint not in openapi_endpoints_set:
                logger.info(f"Removing stale endpoint: {method.endpoint}")
                self._stainless_config.remove_method(method)
                updates = True

        self._existing_methods = self._stainless_config.extract_methods()

        return updates

    def sync_schemas_with_models(self) -> bool:
        new_schema_locations = self.determine_schema_locations()
        model_names = self.generate_model_names(new_schema_locations)

        updates = False
        for schema_name, location in new_schema_locations.items():
            model_name = model_names[schema_name]
            model = StainlessModel(model_name=model_name, schema_name=schema_name, resource_path=location)
            logger.info(f"Adding new schema as model: {model}")
            self._stainless_config.add_model(model)
            updates = True

        # Remove models for schemas that no longer exist
        existing_schema_names = {m.schema_name for m in self._existing_models}
        for schema_name in existing_schema_names:
            if schema_name not in self._schemas:
                model = next(m for m in self._existing_models if m.schema_name == schema_name)
                logger.info(f"Removing model for deleted schema: {model}")
                self._stainless_config.remove_model(model)
                updates = True

        return updates

    def generate_report(self) -> str:
        """
        Generate a summary report of the mapping.
        """
        lines = []
        lines.append("=== OpenAPI to Stainless Model Mapping Report ===\n")
        lines.append("=" * 80 + "\n")

        lines.append("Warnings:\n")
        lines.append("=" * 80 + "\n")
        lines.append("IMPORTANT! Make sure to review all of these and apply manual updates where necessary!\n")
        if not self._report._warnings:
            lines.append("  None\n")
        else:
            for warning in self._report._warnings:
                lines.append(f"  - {warning}\n")

        return "".join(lines)

    def clear_models(self):
        self._existing_models = []
        self._stainless_config.clear_models()


def _report_path(output_path: Path) -> Path:
    return output_path.parent / "openapi_schema_mapping_report.txt"
