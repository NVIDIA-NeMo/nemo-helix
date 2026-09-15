# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Self

from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPIEndpoint
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


@dataclass
class StainlessModel:
    model_name: str  # name of the Stainless model
    schema_name: str  # name of the schema in the OpenAPI spec
    resource_path: list[str]  # path to that model (e.g. ["customization", "jobs"])


@dataclass
class StainlessMethod:
    method_name: str  # name of the Stainless method
    endpoint: OpenAPIEndpoint
    resource_path: list[str]  # path to that method (e.g. ["customization", "jobs"])


class StainlessConfig:
    """
    Documentation: https://www.stainless.com/docs/reference/config
    """

    def __init__(self, stainless_config: dict) -> None:
        self._stainless_config = stainless_config

    @classmethod
    def from_file(cls, config_path: Path) -> Self:
        """
        Load Stainless config from a file.
        """
        yaml_loader = YAML()
        yaml_loader.preserve_quotes = True
        yaml_loader.width = 120

        return cls(yaml_loader.load(config_path.open("r")))

    def extract_models(self) -> list[StainlessModel]:
        """
        Extract all schema names that are already mapped as models in the config.
        """
        logger.info("Extracting existing models from Stainless config.")
        existing_models = []

        # Extract from all resources
        resources = self._stainless_config.get("resources", {})
        for resource_name, resource_config in resources.items():
            existing_models += list(_extract_models_from_resource_config(resource_config, [resource_name]))

        logger.info(f"Extracted {len(existing_models)} models from Stainless config.")
        return existing_models

    def top_level_resource_names(self) -> set[str]:
        resources = self._stainless_config.get("resources", {})
        if not isinstance(resources, dict):
            return set()

        return {
            resource_name
            for resource_name in resources
            if isinstance(resource_name, str) and resource_name != "$shared"
        }

    def clear_models(self) -> None:
        """
        Clear all models from the Stainless config.
        """

        def _clear_models_recursively(resources: dict[str, object]) -> None:
            if not resources:
                return
            for resource_name, resource_config in resources.items():
                resource_config.pop("models", None)

                _clear_models_recursively(resource_config.get("subresources", {}))

        resources = self._stainless_config.get("resources", {})
        _clear_models_recursively(resources)

    def extract_methods(self) -> list[StainlessMethod]:
        """
        Extract all methods defined in the Stainless config.
        """
        logger.info("Extracting existing methods from Stainless config.")
        existing_methods = []

        # Extract from all resources
        resources = self._stainless_config.get("resources", {})
        for resource_name, resource_config in resources.items():
            existing_methods += list(_extract_methods_from_resource_config(resource_config, [resource_name]))

        logger.info(f"Extracted {len(existing_methods)} methods from Stainless config.")
        return existing_methods

    def find_method_for_resource_path(self, resource_path: list[str], name: str) -> StainlessMethod | None:
        """
        Find a method in the Stainless config by its endpoint.
        """
        for method in self.extract_methods():
            if method.resource_path == resource_path and method.method_name == name:
                return method

        return None

    def find_method_for_endpoint(self, endpoint: OpenAPIEndpoint) -> StainlessMethod | None:
        """
        Find a method in the Stainless config by its endpoint.
        """
        for method in self.extract_methods():
            if method.endpoint == endpoint:
                return method

        return None

    def add_method(self, method: StainlessMethod) -> None:
        """
        Add a new method to the Stainless config.
        """
        config_path = _resource_path_to_stainless_config(method.resource_path)

        # Traverse/create the resource path
        current_level = self._stainless_config.setdefault("resources", {})
        for i, part in enumerate(config_path):
            current_level = current_level.setdefault(part, {})

            # Add standalone_api flag for top-level resources and beta subresources
            if i == 0:
                current_level.setdefault("standalone_api", True)
            elif i >= 2 and config_path[i - 1] == "subresources" and config_path[i - 2] in {"beta", "v2"}:
                current_level.setdefault("standalone_api", True)

        # Add the method
        methods = current_level.setdefault("methods", {})
        if method.method_name in methods:
            method.method_name = f"{method.endpoint.method}_{uuid.uuid4()}"

        # if it's a list method, add pagination
        if method.method_name == "list":
            methods[method.method_name] = {
                "endpoint": f"{method.endpoint.method} {method.endpoint.path}",
                "paginated": "default_pagination",
            }

        else:
            methods[method.method_name] = f"{method.endpoint.method} {method.endpoint.path}"

    def remove_method(self, method: StainlessMethod) -> None:
        """
        Remove a method from the Stainless config.
        """
        current_level = self._stainless_config.get("resources", {})

        # Traverse the resource path
        config_path = _resource_path_to_stainless_config(method.resource_path)
        for part in config_path:
            if part in current_level:
                current_level = current_level[part]
            else:
                raise ValueError(f"Resource path '{'/'.join(method.resource_path)}' does not exist.")

        # Remove the method
        methods = current_level.get("methods", {})
        if method.method_name in methods:
            del methods[method.method_name]
        else:
            raise ValueError(
                f"Method '{method.method_name}' does not exist in resource path '{'/'.join(method.resource_path)}'"
            )

    def add_model(self, model: StainlessModel) -> None:
        """
        Add a new model to the Stainless config.
        """
        config_path = _resource_path_to_stainless_config(model.resource_path)

        # Traverse/create the resource path
        current_level = self._stainless_config.setdefault("resources", {})
        for part in config_path:
            current_level = current_level.setdefault(part, {})

        # Add the model
        models = current_level.setdefault("models", {})
        if model.model_name in models:
            raise ValueError(
                f"Model '{model.model_name}' already exists in resource path '{'/'.join(model.resource_path)}'."
            )

        models[model.model_name] = model.schema_name

    def remove_model(self, model: StainlessModel) -> None:
        """
        Remove a model from the Stainless config.
        """
        current_level = self._stainless_config.get("resources", {})

        # Traverse the resource path
        config_path = _resource_path_to_stainless_config(model.resource_path)
        for part in config_path:
            if part in current_level:
                current_level = current_level[part]
            else:
                raise ValueError(f"Resource path '{'/'.join(model.resource_path)}' does not exist.")

        # Remove the model
        models = current_level.get("models", {})
        if model.model_name in models:
            del models[model.model_name]
        else:
            raise ValueError(
                f"Model '{model.model_name}' does not exist in resource path '{'/'.join(model.resource_path)}'."
            )

    def _sort_entities(self):
        """
        Sort models and methods alphabetically in the Stainless config for easier readability.
        """

        def _sort_entities_recursive(resource_config: dict):
            if "models" in resource_config and isinstance(resource_config["models"], dict):
                resource_config["models"] = dict(sorted(resource_config["models"].items(), key=lambda item: item[0]))

            # NOTE: we don't sort methods, they follow a logical order (list, retrieve, create, update, delete)

            if "subresources" in resource_config and isinstance(resource_config["subresources"], dict):
                # Recurse into subresources
                for subresource_config in resource_config["subresources"].values():
                    _sort_entities_recursive(subresource_config)

            # apply a consistent key order
            key_order = ["standalone_api", "models", "methods", "subresources"]
            sorted_resource_config = {k: resource_config[k] for k in key_order if k in resource_config}
            sorted_resource_config.update({k: v for k, v in resource_config.items() if k not in key_order})
            resource_config.clear()
            resource_config.update(sorted_resource_config)

        resources = self._stainless_config.get("resources", {})
        for resource_name, resource_config in resources.items():
            if resource_name == "$shared":
                continue

            _sort_entities_recursive(resource_config)

    def write(self, output_path: Path) -> None:
        self._sort_entities()

        yaml_writer = YAML()
        yaml_writer.preserve_quotes = False
        yaml_writer.width = 120
        yaml_writer.default_flow_style = False

        with output_path.open("w") as f:
            yaml_writer.dump(self._stainless_config, f)


def _resource_path_to_stainless_config(resource_path: list[str]) -> list[str]:
    """
    Convert a resource path to the corresponding Stainless config path.

    E.g. ["customization", "configs"] -> ["customization", "subresources", "configs"]
    """
    if len(resource_path) < 2:
        return resource_path

    stainless_path = []
    for part in resource_path[:-1]:
        stainless_path.append(part)
        stainless_path.append("subresources")
    stainless_path.append(resource_path[-1])

    return stainless_path


def _extract_models_from_resource_config(config: object, current_path: list[str]) -> Iterator[StainlessModel]:
    if not isinstance(config, dict):
        return

    # Check for models at this level
    models = config.get("models", {})
    if isinstance(models, dict):
        for model_name, schema_ref in models.items():
            if isinstance(schema_ref, str):
                yield StainlessModel(model_name, _clean_schema_name(schema_ref), current_path)
            else:
                raise ValueError(
                    f"Invalid model definition '{model_name}', only a direct string reference is supported."
                )

    # Recurse into subresources
    subresources = config.get("subresources", {})
    if isinstance(subresources, dict):
        for sub_name, sub_config in subresources.items():
            yield from _extract_models_from_resource_config(sub_config, current_path + [sub_name])


def _clean_schema_name(schema_name: str) -> str:
    if schema_name.startswith("#/components/schemas/"):
        return schema_name.split("/")[-1]

    return schema_name


def _extract_methods_from_resource_config(config: object, current_path: list[str]) -> Iterator[StainlessMethod]:
    if not isinstance(config, dict):
        return

    methods = config.get("methods", {})
    if isinstance(methods, dict):
        for method_name, method_config in methods.items():
            http_verb, http_endpoint = _extract_endpoint(method_config)
            yield StainlessMethod(method_name, OpenAPIEndpoint(http_verb, http_endpoint), current_path)

    # Recurse into subresources
    subresources = config.get("subresources", {})
    if isinstance(subresources, dict):
        for sub_name, sub_config in subresources.items():
            if sub_name != "$shared":
                # $shared is only for models
                yield from _extract_methods_from_resource_config(sub_config, current_path + [sub_name])


def _extract_endpoint(method_config: str | dict) -> tuple[str, str]:
    """
    Extract endpoint path from method configuration.

    Examples:

      list:
        endpoint: get /v1/deployment/configs
        paginated: default_pagination
      create: post /v1/deployment/configs
    """
    if isinstance(method_config, str):
        return tuple(method_config.split(" ", 1))

    elif isinstance(method_config, dict) and "endpoint" in method_config:
        return tuple(method_config["endpoint"].split(" ", 1))

    raise ValueError(f"Invalid method configuration: {method_config}")
