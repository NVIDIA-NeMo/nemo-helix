# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Self

from caseutil import to_snake
from ruamel.yaml import YAML

# https://www.stainless.com/docs/reference/diagnostics/#endpoint shows the supported HTTP methods, this is created explicitly to avoid HEAD endpoints as they are not supported by Stainless.
SUPPORTED_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "query"}
SDK_EXCLUDED_PATHS = {
    # Auth bootstrap and utility endpoints do not follow the generated SDK's
    # versioned resource path convention. They are used directly by client auth
    # code instead of through generated resource methods.
    "/apis/auth/discovery",
    "/apis/auth/jwks",
    "/apis/auth/token",
}

logger = logging.getLogger(__name__)


@dataclass(unsafe_hash=True)
class OpenAPIEndpoint:
    method: str
    path: str

    def __post_init__(self):
        self.method = self.method.lower()
        self.path = self.path.lower()

    def approx_method_name(self) -> str:
        """
        Convert OpenAPI method and path to Stainless method name.
        """
        if self.method == "get":
            if self.path.endswith("}"):
                return "retrieve"
            else:
                return "list"
        elif self.method == "post":
            return "create"
        elif self.method == "put":
            return "update"
        elif self.method == "delete":
            return "delete"
        elif self.method == "patch":
            return "patch"
        elif self.method == "query":
            return "query"
        else:
            raise ValueError(f"Invalid HTTP method {self.method!r}")

    def approx_resource_path(self) -> list[str]:
        """
        Convert OpenAPI path to Stainless resource path.
        """
        # Remove leading slash and split by slash
        parts = self.path.lstrip("/").split("/")
        if parts[0] == "apis":
            parts = parts[2:]
        allowed_prefixes = ["v1", "v2"]
        if parts[0] not in allowed_prefixes:
            raise ValueError(f"Invalid OpenAPI path {self.path!r} - must start with {allowed_prefixes}")
        path = []
        # Remove anything after the first parameter (e.g. {job_id})
        for part in parts[1:]:
            if part.startswith("{") and part.endswith("}"):
                # This is a parameter, we can skip it
                continue
            else:
                snake_part = to_snake(part)
                if snake_part:  # Only add non-empty parts
                    path.append(snake_part)

        if path and path[-1] and path[-1][-1] != "s":
            # not plural - should be in the parent resource
            path = path[:-1]

        if len(path) >= 2 and path[0] == "workspaces":
            # Special case: workspaces, resources, ... -> resources, ...
            path = path[1:]

        return path

    def last_path_component(self) -> str:
        """
        Get the last component of the path, ignoring parameters.
        E.g. /v1/customization/configs/{namespace}/{config_name} -> configs
        """
        parts = self.path.lstrip("/").split("/")
        for part in reversed(parts):
            if not (part.startswith("{") and part.endswith("}")):
                component = to_snake(part)
                if component:
                    return component

        raise ValueError(f"Invalid OpenAPI path {self.path!r} - no non-parameter components found")

    def __str__(self) -> str:
        return f"{self.method} {self.path}"


class OpenAPI:
    def __init__(self, spec: dict):
        self._spec = spec

    @classmethod
    def from_file(cls, config_path: Path) -> Self:
        """
        Load Stainless config from a file.
        """
        yaml_loader = YAML()
        yaml_loader.preserve_quotes = True
        yaml_loader.width = 120

        return cls(yaml_loader.load(config_path.open("r")))

    def schemas(self) -> list[str]:
        return list(self._spec.get("components", {}).get("schemas", {}).keys())

    def calculate_schema_to_endpoints(self) -> dict[str, list[OpenAPIEndpoint]]:
        """Analyze OpenAPI spec and return schema usage mapping."""
        logger.info("Calculating endpoint mappings for each OpenAPI component...")

        schemas = self.schemas()
        if not schemas:
            raise ValueError("No schemas found in OpenAPI spec")

        # Track which endpoints use each schema
        schema_usage = defaultdict(list)

        # Analyze paths
        for path, path_item in self._spec.get("paths", {}).items():
            if _should_skip_sdk_endpoint(path):
                continue

            for method, spec in path_item.items():
                method_lower = method.lower()
                if method_lower not in SUPPORTED_HTTP_METHODS:
                    continue

                endpoint = OpenAPIEndpoint(method_lower, path)

                logger.debug(f"Extracting schemas for endpoint {endpoint}...")
                # Find all schema references, i.e. all the schemas that are used by inputs/outputs for this endpoint
                used_schemas = self._extract_schema_refs(spec)

                # Add this endpoint to each schema's usage list
                for schema_name in used_schemas:
                    if schema_name in schemas:
                        schema_usage[schema_name].append(endpoint)

        return dict(schema_usage)

    def calculate_sdk_excluded_schema_names(self) -> set[str]:
        """Return schemas used only by endpoint paths omitted from the generated SDK."""
        schemas = set(self.schemas())
        if not schemas:
            raise ValueError("No schemas found in OpenAPI spec")

        excluded_schema_names: set[str] = set()
        for path, path_item in self._spec.get("paths", {}).items():
            if not _should_skip_sdk_endpoint(path):
                continue

            for method, spec in path_item.items():
                if method.lower() not in SUPPORTED_HTTP_METHODS:
                    continue
                excluded_schema_names.update(schema for schema in self._extract_schema_refs(spec) if schema in schemas)

        generated_schema_names = set(self.calculate_schema_to_endpoints())
        return excluded_schema_names - generated_schema_names

    def _extract_schema_refs(self, obj: object) -> set[str]:
        """
        Iteratively extract all '$ref' references to schemas from an object.
        Detects and handles cycles in schema references.
        """
        from collections import deque

        refs = set()
        visited_schemas = set()
        # Queue contains: (object_to_process, path_of_schemas_that_led_here)
        queue: deque[tuple[object, list[str]]] = deque([(obj, [])])

        while queue:
            current, schema_path = queue.popleft()

            if isinstance(current, list):
                for item in current:
                    queue.append((item, schema_path))

            elif isinstance(current, dict):
                for key, value in current.items():
                    if key == "$ref" and isinstance(value, str):
                        # Extract schema name from reference like "#/components/schemas/User"
                        if value.startswith("#/components/schemas/"):
                            schema_name = value.split("/")[-1]
                            refs.add(schema_name)

                            # Check if this schema is already in the current path (this means we have a cycle)
                            if schema_name in schema_path:
                                cycle_start_idx = schema_path.index(schema_name)
                                cycle = schema_path[cycle_start_idx:] + [schema_name]
                                logger.warning(f"Cycle detected in schema references: {' -> '.join(cycle)}")

                            elif schema_name not in visited_schemas:
                                # New schema - add to visited and extract any schemas that it references
                                visited_schemas.add(schema_name)
                                schema_def = self._spec.get("components", {}).get("schemas", {}).get(schema_name, {})
                                if schema_def:
                                    queue.append((schema_def, schema_path + [schema_name]))

                    else:
                        queue.append((value, schema_path))

        return refs

    def extract_endpoints(self) -> Iterable[OpenAPIEndpoint]:
        for path, path_item in self._spec.get("paths", {}).items():
            for method, _ in path_item.items():
                method_lower = method.lower()
                if method_lower not in SUPPORTED_HTTP_METHODS:
                    continue
                if path.startswith("/v1/jobs"):
                    continue
                if _should_skip_sdk_endpoint(path):
                    continue

                yield OpenAPIEndpoint(method_lower, path)


def _should_skip_sdk_endpoint(path: str) -> bool:
    return path.lower() in SDK_EXCLUDED_PATHS
