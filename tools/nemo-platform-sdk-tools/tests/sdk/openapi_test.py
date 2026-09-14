# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPI, OpenAPIEndpoint


def test_stainless_resource_path():
    endpoint = OpenAPIEndpoint(method="get", path="/v2/data-designer/jobs/{job_id}/results/{result_id}")
    assert endpoint.approx_resource_path() == ["data_designer", "jobs", "results"]

    endpoint = OpenAPIEndpoint(method="post", path="/v2/data-designer/jobs")
    assert endpoint.approx_resource_path() == ["data_designer", "jobs"]

    endpoint = OpenAPIEndpoint(method="get", path="/v2/data-designer/preview")
    assert endpoint.approx_resource_path() == ["data_designer"]

    endpoint = OpenAPIEndpoint(
        method="get", path="/v2/workspaces/{workspace}/customization/configs/{namespace}/{config_name}"
    )
    assert endpoint.approx_resource_path() == ["customization", "configs"]


def test_extract_endpoints_skips_auth_utility_routes():
    spec = {
        "paths": {
            "/apis/auth/discovery": {"get": {}},
            "/apis/auth/jwks": {"get": {}},
            "/apis/auth/token": {"post": {}},
            "/apis/auth/v2/iam/role-bindings": {"get": {}},
            "/apis/entities/v2/workspaces": {"get": {}},
        },
        "components": {"schemas": {"Workspace": {"type": "object"}}},
    }

    endpoints = list(OpenAPI(spec).extract_endpoints())

    assert endpoints == [
        OpenAPIEndpoint(method="get", path="/apis/auth/v2/iam/role-bindings"),
        OpenAPIEndpoint(method="get", path="/apis/entities/v2/workspaces"),
    ]


def test_schema_usage_skips_auth_utility_routes():
    spec = {
        "paths": {
            "/apis/auth/token": {
                "post": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/TokenExchangeResponse"}}
                            }
                        }
                    }
                }
            },
            "/apis/entities/v2/workspaces": {
                "get": {
                    "responses": {
                        "200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Workspace"}}}}
                    }
                }
            },
        },
        "components": {
            "schemas": {
                "TokenExchangeResponse": {"type": "object"},
                "Workspace": {"type": "object"},
            }
        },
    }

    schema_usage = OpenAPI(spec).calculate_schema_to_endpoints()

    assert schema_usage == {
        "Workspace": [OpenAPIEndpoint(method="get", path="/apis/entities/v2/workspaces")],
    }


def test_sdk_excluded_schema_names_include_nested_utility_route_refs() -> None:
    spec = {
        "paths": {
            "/apis/auth/token": {
                "post": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/TokenExchangeResponse"}}
                            }
                        }
                    }
                }
            },
            "/apis/entities/v2/workspaces": {
                "get": {
                    "responses": {
                        "200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Workspace"}}}}
                    }
                }
            },
        },
        "components": {
            "schemas": {
                "TokenExchangeResponse": {
                    "type": "object",
                    "properties": {"subject": {"$ref": "#/components/schemas/TokenSubject"}},
                },
                "TokenSubject": {"type": "object"},
                "Workspace": {"type": "object"},
            }
        },
    }

    schema_names = OpenAPI(spec).calculate_sdk_excluded_schema_names()

    assert schema_names == {"TokenExchangeResponse", "TokenSubject"}


def test_sdk_excluded_schema_names_preserve_schemas_used_by_generated_paths() -> None:
    spec = {
        "paths": {
            "/apis/auth/token": {
                "post": {
                    "responses": {
                        "200": {
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SharedResponse"}}}
                        }
                    },
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TokenOnlyRequest"}}}
                    },
                }
            },
            "/apis/entities/v2/workspaces": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SharedResponse"}}}
                        }
                    }
                }
            },
        },
        "components": {
            "schemas": {
                "SharedResponse": {
                    "type": "object",
                    "properties": {"principal": {"$ref": "#/components/schemas/SharedPrincipal"}},
                },
                "SharedPrincipal": {"type": "object"},
                "TokenOnlyRequest": {"type": "object"},
            }
        },
    }

    schema_names = OpenAPI(spec).calculate_sdk_excluded_schema_names()

    assert schema_names == {"TokenOnlyRequest"}


def test_extract_schema_refs_with_cycle(caplog):
    """Test that _extract_schema_refs handles cycles correctly."""
    # Create a mock OpenAPI spec with circular references
    spec = {
        "components": {
            "schemas": {
                "SchemaA": {"type": "object", "properties": {"b": {"$ref": "#/components/schemas/SchemaB"}}},
                "SchemaB": {"type": "object", "properties": {"c": {"$ref": "#/components/schemas/SchemaC"}}},
                "SchemaC": {
                    "type": "object",
                    "properties": {
                        "a": {"$ref": "#/components/schemas/SchemaA"}  # Creates cycle: A -> B -> C -> A
                    },
                },
            }
        }
    }

    openapi = OpenAPI(spec)

    # Test with an endpoint that uses SchemaA
    endpoint_spec = {
        "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/SchemaA"}}}}
    }

    with caplog.at_level(logging.WARNING):
        refs = openapi._extract_schema_refs(endpoint_spec)

    # Should find all three schemas despite the cycle
    assert refs == {"SchemaA", "SchemaB", "SchemaC"}

    # Should log a warning about the cycle
    assert any("Cycle detected" in record.message for record in caplog.records)
    assert any(
        "SchemaA" in record.message and "SchemaB" in record.message and "SchemaC" in record.message
        for record in caplog.records
        if "Cycle detected" in record.message
    )


def test_extract_schema_refs_diamond_pattern():
    """Test that diamond patterns (not cycles) are handled correctly without warnings."""
    # Create a mock OpenAPI spec with a diamond dependency pattern
    spec = {
        "components": {
            "schemas": {
                "SchemaA": {
                    "type": "object",
                    "properties": {
                        "b": {"$ref": "#/components/schemas/SchemaB"},
                        "c": {"$ref": "#/components/schemas/SchemaC"},
                    },
                },
                "SchemaB": {"type": "object", "properties": {"d": {"$ref": "#/components/schemas/SchemaD"}}},
                "SchemaC": {
                    "type": "object",
                    "properties": {
                        "d": {"$ref": "#/components/schemas/SchemaD"}  # Both B and C reference D
                    },
                },
                "SchemaD": {"type": "object", "properties": {"value": {"type": "string"}}},
            }
        }
    }

    openapi = OpenAPI(spec)

    endpoint_spec = {
        "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/SchemaA"}}}}
    }

    refs = openapi._extract_schema_refs(endpoint_spec)

    # Should find all four schemas
    assert refs == {"SchemaA", "SchemaB", "SchemaC", "SchemaD"}
