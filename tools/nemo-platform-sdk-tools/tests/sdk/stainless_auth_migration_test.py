# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPI, OpenAPIEndpoint
from nemo_platform_sdk_tools.sdk.core.stainless import StainlessConfig
from nemo_platform_sdk_tools.sdk.openapi_stainless_mapper import SchemaMapper
from nemo_platform_sdk_tools.sdk.source_owned_resources import (
    SourceOwnedResource,
    derive_generated_schema_names,
    derive_source_owned_schema_names,
)
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[4]
AUTH_RESOURCE = SourceOwnedResource(resource_name="auth", path_prefixes=("/apis/auth/authenticate",))
ACCESS_KEYS_RESOURCE = SourceOwnedResource(resource_name="access_keys", path_prefixes=("/apis/auth/v2/access-keys",))
IAM_RESOURCE = SourceOwnedResource(resource_name="iam", path_prefixes=("/apis/auth/v2/iam", "/apis/auth/v2/authz"))
AUTH_RESOURCES = (AUTH_RESOURCE, ACCESS_KEYS_RESOURCE, IAM_RESOURCE)
PRESERVED_SHARED_SCHEMA_NAMES = {"AuthContext"}
REMOVED_SDK_API_DOC_TYPE_NAMES = {
    "AuthDiscoveryResponse",
    "JsonWebKey",
    "JsonWebKeySetResponse",
    "OidcDiscoveryResponse",
    "WorkloadTokenExchangeErrorResponse",
    "WorkloadTokenExchangeResponse",
}
REMOVED_SDK_API_DOC_RESOURCE_LINKS = {
    "src/nemo_platform/resources/access_keys/api.md",
    "src/nemo_platform/resources/auth/api.md",
    "src/nemo_platform/resources/iam/api.md",
}


def _schema_usage() -> dict[str, list[OpenAPIEndpoint]]:
    return OpenAPI.from_file(REPO_ROOT / "openapi" / "openapi.yaml").calculate_schema_to_endpoints()


def _schema_ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_response(schema_name: str) -> dict[str, object]:
    return {
        "description": "OK",
        "content": {"application/json": {"schema": _schema_ref(schema_name)}},
    }


def _openapi_spec_with_source_owned_auth() -> OpenAPI:
    return OpenAPI(
        {
            "openapi": "3.1.0",
            "paths": {
                "/apis/auth/authenticate": {
                    "get": {"responses": {"200": _json_response("AuthenticateResponse")}},
                    "post": {"responses": {"200": _json_response("AuthenticateResponse")}},
                },
                "/apis/auth/discovery": {
                    "get": {"responses": {"200": _json_response("AuthDiscoveryResponse")}},
                },
                "/apis/auth/jwks": {
                    "get": {"responses": {"200": _json_response("JsonWebKeySetResponse")}},
                },
                "/apis/auth/token": {
                    "post": {
                        "responses": {
                            "200": _json_response("WorkloadTokenExchangeResponse"),
                            "400": _json_response("WorkloadTokenExchangeErrorResponse"),
                        }
                    },
                },
                "/apis/auth/v2/access-keys": {
                    "get": {"responses": {"200": _json_response("AccessKeyListResponse")}},
                    "post": {
                        "requestBody": {
                            "content": {"application/json": {"schema": _schema_ref("AccessKeyCreateRequest")}}
                        },
                        "responses": {"200": _json_response("AccessKeyCreateResponse")},
                    },
                },
                "/apis/auth/v2/access-keys/{jti}/rotate": {
                    "post": {
                        "requestBody": {
                            "content": {"application/json": {"schema": _schema_ref("AccessKeyRotateRequest")}}
                        },
                        "responses": {"200": _json_response("AccessKeyRotateResponse")},
                    },
                },
                "/apis/auth/v2/iam/role-bindings": {
                    "get": {"responses": {"200": _json_response("RoleBindingsPage")}},
                    "post": {
                        "requestBody": {"content": {"application/json": {"schema": _schema_ref("RoleBindingInput")}}},
                        "responses": {"200": _json_response("RoleBinding")},
                    },
                },
                "/apis/auth/v2/authz/{entrypoint}": {
                    "post": {
                        "requestBody": {"content": {"application/json": {"schema": _schema_ref("AuthzRequest")}}},
                        "responses": {"200": _json_response("AuthzResponse")},
                    },
                },
                "/apis/gadgets/v2/workspaces/{workspace}/gadgets": {
                    "get": {"responses": {"200": _json_response("GadgetPage")}},
                },
            },
            "components": {
                "schemas": {
                    "AccessKeyCreateRequest": {"type": "object"},
                    "AccessKeyCreateResponse": {
                        "type": "object",
                        "properties": {"metadata": _schema_ref("AccessKeyMetadataResponse")},
                    },
                    "AccessKeyListResponse": {
                        "type": "object",
                        "properties": {"data": {"type": "array", "items": _schema_ref("AccessKeyMetadataResponse")}},
                    },
                    "AccessKeyMetadataResponse": {"type": "object"},
                    "AccessKeyRotateRequest": {"type": "object"},
                    "AccessKeyRotateResponse": {
                        "type": "object",
                        "properties": {"new_key": _schema_ref("AccessKeyCreateResponse")},
                    },
                    "AuthenticateResponse": {"type": "object"},
                    "AuthDiscoveryResponse": {"type": "object"},
                    "AuthzRequest": {"type": "object"},
                    "AuthzResponse": {"type": "object"},
                    "GadgetPage": {
                        "type": "object",
                        "properties": {"auth_context": _schema_ref("AuthContext")},
                    },
                    "AuthContext": {"type": "object"},
                    "JsonWebKey": {"type": "object"},
                    "JsonWebKeySetResponse": {
                        "type": "object",
                        "properties": {"keys": {"type": "array", "items": _schema_ref("JsonWebKey")}},
                    },
                    "RoleBinding": {"type": "object"},
                    "RoleBindingInput": {"type": "object"},
                    "RoleBindingsPage": {
                        "type": "object",
                        "properties": {"data": {"type": "array", "items": _schema_ref("RoleBinding")}},
                    },
                    "WorkloadTokenExchangeErrorResponse": {"type": "object"},
                    "WorkloadTokenExchangeResponse": {"type": "object"},
                }
            },
        }
    )


def _stainless_config() -> StainlessConfig:
    return StainlessConfig.from_file(REPO_ROOT / "sdk" / "stainless.yaml")


def _stainless_config_without_auth() -> StainlessConfig:
    return StainlessConfig(
        {
            "resources": {
                "$shared": {
                    "models": {
                        "auth_context": "AuthContext",
                    }
                }
            }
        }
    )


def _load_stainless_config() -> dict[object, object]:
    yaml = YAML(typ="safe")
    config = yaml.load((REPO_ROOT / "sdk" / "stainless.yaml").read_text())
    assert isinstance(config, dict)
    return config


def _mapped_schema_names() -> set[str]:
    return {model.schema_name for model in _stainless_config().extract_models()}


def test_stainless_config_has_no_top_level_auth_resources() -> None:
    resources = _load_stainless_config()["resources"]
    assert isinstance(resources, dict)

    assert {"auth", "access_keys", "iam"}.isdisjoint(resources)


def test_stainless_config_has_no_auth_owned_schema_mappings() -> None:
    auth_owned_schema_names = derive_source_owned_schema_names(_schema_usage(), AUTH_RESOURCES)
    auth_utility_schema_names = OpenAPI.from_file(
        REPO_ROOT / "openapi" / "openapi.yaml"
    ).calculate_sdk_excluded_schema_names()

    assert (auth_owned_schema_names | auth_utility_schema_names).isdisjoint(_mapped_schema_names())


def test_stainless_config_keeps_auth_context_for_generated_resources() -> None:
    generated_schema_names = derive_generated_schema_names(_schema_usage(), AUTH_RESOURCES)
    mapped_schema_names = _mapped_schema_names()

    assert PRESERVED_SHARED_SCHEMA_NAMES.issubset(generated_schema_names)
    assert PRESERVED_SHARED_SCHEMA_NAMES.issubset(mapped_schema_names)


def test_sdk_api_index_has_no_source_owned_auth_artifacts() -> None:
    api_index = (REPO_ROOT / "sdk" / "python" / "nemo-platform" / "api.md").read_text()

    assert all(type_name not in api_index for type_name in REMOVED_SDK_API_DOC_TYPE_NAMES)
    assert all(link not in api_index for link in REMOVED_SDK_API_DOC_RESOURCE_LINKS)


def test_schema_mapper_skips_source_owned_auth_methods_when_resources_are_absent() -> None:
    stainless_config = _stainless_config_without_auth()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_auth(), stainless_config, source_owned_resources=AUTH_RESOURCES
    )

    assert mapper.sync_endpoints_with_methods() is True

    methods = stainless_config.extract_methods()
    assert all(not method.endpoint.path.startswith("/apis/auth/") for method in methods)
    assert any(
        method.endpoint.path == "/apis/gadgets/v2/workspaces/{workspace}/gadgets"
        and method.method_name.startswith("reviewme_")
        for method in methods
    )


def test_schema_mapper_excludes_source_owned_and_auth_utility_schemas() -> None:
    stainless_config = _stainless_config_without_auth()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_auth(), stainless_config, source_owned_resources=AUTH_RESOURCES
    )

    assert mapper.sync_schemas_with_models() is True

    schema_names = {model.schema_name for model in stainless_config.extract_models()}
    assert "AccessKeyCreateRequest" not in schema_names
    assert "AccessKeyCreateResponse" not in schema_names
    assert "AccessKeyMetadataResponse" not in schema_names
    assert "AuthenticateResponse" not in schema_names
    assert "AuthDiscoveryResponse" not in schema_names
    assert "AuthzRequest" not in schema_names
    assert "AuthzResponse" not in schema_names
    assert "JsonWebKey" not in schema_names
    assert "JsonWebKeySetResponse" not in schema_names
    assert "RoleBinding" not in schema_names
    assert "RoleBindingInput" not in schema_names
    assert "RoleBindingsPage" not in schema_names
    assert "WorkloadTokenExchangeErrorResponse" not in schema_names
    assert "WorkloadTokenExchangeResponse" not in schema_names
    assert "AuthContext" in schema_names
    assert "GadgetPage" in schema_names
