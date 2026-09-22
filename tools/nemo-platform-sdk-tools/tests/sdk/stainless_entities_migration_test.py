# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPI, OpenAPIEndpoint
from nemo_platform_sdk_tools.sdk.core.stainless import StainlessConfig
from nemo_platform_sdk_tools.sdk.openapi_stainless_mapper import SchemaMapper
from nemo_platform_sdk_tools.sdk.source_owned_resources import (
    SOURCE_OWNED_RESOURCE_EXCLUSIONS,
    SourceOwnedResource,
    endpoint_is_source_owned,
)
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[4]
ENTITIES_RESOURCE = SourceOwnedResource(
    resource_name="entities",
    path_prefixes=(
        "/apis/entities/v2/workspaces/{workspace}/entities",
        "/apis/entities/v2/entities",
    ),
)
REMOVED_GENERATED_PATHS = (
    REPO_ROOT / "sdk/python/nemo-platform/src/nemo_platform/resources/entities",
    REPO_ROOT / "sdk/python/nemo-platform/src/nemo_platform/types/entities",
    REPO_ROOT / "sdk/python/nemo-platform/tests/api_resources/entities",
    REPO_ROOT / "sdk/python/nemo-platform/tests/api_resources/test_entities.py",
)


def _string_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)

    result: dict[str, object] = {}
    for key, nested_value in value.items():
        assert isinstance(key, str)
        result[key] = nested_value
    return result


def _load_stainless_config() -> dict[str, object]:
    yaml = YAML(typ="safe")
    config = yaml.load((REPO_ROOT / "sdk" / "stainless.yaml").read_text())
    return _string_mapping(config)


def _schema_ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_response(schema_name: str) -> dict[str, object]:
    return {
        "description": "OK",
        "content": {
            "application/json": {
                "schema": _schema_ref(schema_name),
            }
        },
    }


def _openapi_spec_with_entities_and_workspaces() -> OpenAPI:
    return OpenAPI(
        {
            "openapi": "3.1.0",
            "paths": {
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}": {
                    "get": {"responses": {"200": _json_response("EntitiesPage")}},
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": _schema_ref("EntityCreateInput"),
                                }
                            }
                        },
                        "responses": {"200": _json_response("Entity")},
                    },
                },
                "/apis/entities/v2/entities/{id}": {
                    "get": {"responses": {"200": _json_response("Entity")}},
                },
                "/apis/entities/v2/workspaces": {
                    "get": {"responses": {"200": _json_response("WorkspacesPage")}},
                },
            },
            "components": {
                "schemas": {
                    "EntitiesPage": {"type": "object"},
                    "EntityCreateInput": {"type": "object"},
                    "Entity": {"type": "object"},
                    "WorkspacesPage": {"type": "object"},
                }
            },
        }
    )


def test_stainless_config_has_no_entities_resource() -> None:
    resources = _string_mapping(_load_stainless_config()["resources"])

    assert "entities" not in resources


def test_entities_source_owned_exclusion_is_registered_narrowly() -> None:
    assert ENTITIES_RESOURCE in SOURCE_OWNED_RESOURCE_EXCLUSIONS
    assert endpoint_is_source_owned(
        OpenAPIEndpoint(method="get", path="/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}"),
        (ENTITIES_RESOURCE,),
    )
    assert not endpoint_is_source_owned(
        OpenAPIEndpoint(method="get", path="/apis/entities/v2/workspaces"),
        (ENTITIES_RESOURCE,),
    )


def test_generated_entities_sdk_output_is_removed() -> None:
    generated_files = [
        generated_file
        for path in REMOVED_GENERATED_PATHS
        if path.exists()
        for generated_file in (path.rglob("*") if path.is_dir() else (path,))
        if generated_file.is_file()
    ]

    assert generated_files == []


def test_mapper_does_not_recreate_source_owned_entities_resource() -> None:
    stainless_config = StainlessConfig(
        {
            "resources": {
                "workspaces": {
                    "methods": {
                        "list": "get /apis/entities/v2/workspaces",
                    },
                    "models": {},
                }
            }
        }
    )
    mapper = SchemaMapper(_openapi_spec_with_entities_and_workspaces(), stainless_config)

    assert mapper.sync_endpoints_with_methods() is False

    methods = stainless_config.extract_methods()
    assert all(method.resource_path[0] != "entities" for method in methods)
    assert any(method.endpoint.path == "/apis/entities/v2/workspaces" for method in methods)
