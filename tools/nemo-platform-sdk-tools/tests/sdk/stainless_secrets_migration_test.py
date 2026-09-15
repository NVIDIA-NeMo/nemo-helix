# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SECRETS_RESOURCE = SourceOwnedResource(resource_name="secrets", path_prefixes=("/apis/secrets/v2",))
SECRETS_ONLY_SCHEMA_NAMES = {
    "PlatformSecretAccessResponse",
    "PlatformSecretAdminRotationResponse",
    "PlatformSecretCreateRequest",
    "PlatformSecretResponse",
    "PlatformSecretResponsesPage",
    "PlatformSecretUpdateRequest",
}
PRESERVED_SHARED_SCHEMA_NAMES = {
    "SecretRef",
}


def _schema_usage() -> dict[str, list[OpenAPIEndpoint]]:
    return OpenAPI.from_file(REPO_ROOT / "openapi" / "openapi.yaml").calculate_schema_to_endpoints()


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


def _openapi_spec_with_source_owned_secrets() -> OpenAPI:
    return OpenAPI(
        {
            "openapi": "3.1.0",
            "paths": {
                "/apis/secrets/v2/workspaces/{workspace}/secrets": {
                    "get": {
                        "responses": {
                            "200": _json_response("PlatformSecretResponsesPage"),
                        }
                    },
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": _schema_ref("PlatformSecretCreateRequest"),
                                }
                            }
                        },
                        "responses": {
                            "201": _json_response("PlatformSecretResponse"),
                        },
                    },
                },
                "/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access": {
                    "get": {
                        "responses": {
                            "200": _json_response("PlatformSecretAccessResponse"),
                        }
                    }
                },
                "/apis/secrets/v2/workspaces/{workspace}/secrets/{name}": {
                    "patch": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": _schema_ref("PlatformSecretUpdateRequest"),
                                }
                            }
                        },
                        "responses": {
                            "200": _json_response("PlatformSecretResponse"),
                        },
                    }
                },
                "/apis/secrets/v2/rotate-encryption-keys": {
                    "post": {
                        "responses": {
                            "202": _json_response("PlatformSecretAdminRotationResponse"),
                        }
                    }
                },
                "/apis/files/v2/workspaces/{workspace}/filesets": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": _schema_ref("FileStorageRequest"),
                                }
                            }
                        },
                        "responses": {
                            "200": _json_response("FileStorageResponse"),
                        },
                    }
                },
                "/apis/gadgets/v2/workspaces/{workspace}/gadgets": {
                    "get": {
                        "responses": {
                            "200": _json_response("GadgetPage"),
                        }
                    }
                },
            },
            "components": {
                "schemas": {
                    "PlatformSecretAccessResponse": {"type": "object"},
                    "PlatformSecretAdminRotationResponse": {"type": "object"},
                    "PlatformSecretCreateRequest": {"type": "object"},
                    "PlatformSecretResponse": {"type": "object"},
                    "PlatformSecretResponsesPage": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": _schema_ref("PlatformSecretResponse"),
                            }
                        },
                    },
                    "PlatformSecretUpdateRequest": {"type": "object"},
                    "SecretRef": {"type": "string"},
                    "FileStorageRequest": {
                        "type": "object",
                        "properties": {
                            "token_secret": _schema_ref("SecretRef"),
                        },
                    },
                    "FileStorageResponse": {"type": "object"},
                    "GadgetPage": {"type": "object"},
                }
            },
        }
    )


def _stainless_config() -> StainlessConfig:
    return StainlessConfig.from_file(REPO_ROOT / "sdk" / "stainless.yaml")


def _stainless_config_without_secrets() -> StainlessConfig:
    return StainlessConfig(
        {
            "resources": {
                "$shared": {
                    "models": {
                        "secret_ref": "SecretRef",
                    }
                }
            }
        }
    )


def _stainless_config_with_secrets() -> StainlessConfig:
    return StainlessConfig(
        {
            "resources": {
                "secrets": {
                    "standalone_api": True,
                    "methods": {},
                },
                "$shared": {
                    "models": {
                        "secret_ref": "SecretRef",
                    }
                },
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


def test_stainless_config_has_no_top_level_secrets_resource() -> None:
    resources = _load_stainless_config()["resources"]
    assert isinstance(resources, dict)

    assert "secrets" not in resources


def test_stainless_config_has_no_secrets_only_schema_mappings() -> None:
    secrets_only_schema_names = derive_source_owned_schema_names(_schema_usage(), (SECRETS_RESOURCE,))

    assert SECRETS_ONLY_SCHEMA_NAMES.issubset(secrets_only_schema_names)
    assert secrets_only_schema_names.isdisjoint(_mapped_schema_names())


def test_stainless_config_keeps_shared_secret_ref_for_non_secrets_endpoints() -> None:
    generated_schema_names = derive_generated_schema_names(_schema_usage(), (SECRETS_RESOURCE,))
    mapped_schema_names = _mapped_schema_names()

    assert PRESERVED_SHARED_SCHEMA_NAMES.issubset(generated_schema_names)
    assert PRESERVED_SHARED_SCHEMA_NAMES.issubset(mapped_schema_names)


def test_schema_mapper_skips_source_owned_secrets_methods_when_secrets_resource_is_absent() -> None:
    stainless_config = _stainless_config_without_secrets()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_secrets(), stainless_config, source_owned_resources=(SECRETS_RESOURCE,)
    )

    assert mapper.sync_endpoints_with_methods() is True

    methods = stainless_config.extract_methods()
    assert all(not method.endpoint.path.startswith("/apis/secrets/v2") for method in methods)
    assert any(
        method.endpoint.path == "/apis/gadgets/v2/workspaces/{workspace}/gadgets"
        and method.method_name.startswith("reviewme_")
        for method in methods
    )


def test_schema_mapper_excludes_secrets_only_schemas_but_keeps_shared_schemas() -> None:
    stainless_config = _stainless_config_without_secrets()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_secrets(), stainless_config, source_owned_resources=(SECRETS_RESOURCE,)
    )

    assert mapper.sync_schemas_with_models() is True

    schema_names = {model.schema_name for model in stainless_config.extract_models()}
    assert SECRETS_ONLY_SCHEMA_NAMES.isdisjoint(schema_names)
    assert "SecretRef" in schema_names
    assert "FileStorageRequest" in schema_names
    assert "FileStorageResponse" in schema_names
    assert "GadgetPage" in schema_names


def test_schema_mapper_includes_secrets_methods_when_secrets_resource_exists() -> None:
    stainless_config = _stainless_config_with_secrets()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_secrets(), stainless_config, source_owned_resources=(SECRETS_RESOURCE,)
    )

    assert mapper.sync_endpoints_with_methods() is True

    methods = stainless_config.extract_methods()
    assert any(method.endpoint.path.startswith("/apis/secrets/v2") for method in methods)
