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
JOBS_RESOURCE = SourceOwnedResource(resource_name="jobs", path_prefixes=("/apis/jobs/v2",))
PRESERVED_SHARED_SCHEMA_NAMES = {
    "PlatformJobLog",
    "PlatformJobLogPage",
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


def _openapi_spec_with_source_owned_jobs() -> OpenAPI:
    return OpenAPI(
        {
            "openapi": "3.1.0",
            "paths": {
                "/apis/jobs/v2/workspaces/{workspace}/jobs": {
                    "get": {
                        "responses": {
                            "200": _json_response("JobPage"),
                        }
                    },
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": _schema_ref("JobCreateRequest"),
                                }
                            }
                        },
                        "responses": {
                            "200": _json_response("JobResponse"),
                        },
                    },
                },
                "/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/logs": {
                    "get": {
                        "responses": {
                            "200": _json_response("SharedLogPage"),
                        }
                    }
                },
                "/apis/files/v2/workspaces/{workspace}/files/{name}/logs": {
                    "get": {
                        "responses": {
                            "200": _json_response("SharedLogPage"),
                        }
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
                    "JobCreateRequest": {
                        "type": "object",
                        "properties": {
                            "runtime": _schema_ref("JobRuntime"),
                        },
                    },
                    "JobRuntime": {"type": "object"},
                    "JobResponse": {"type": "object"},
                    "JobPage": {"type": "object"},
                    "SharedLog": {"type": "object"},
                    "SharedLogPage": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": _schema_ref("SharedLog"),
                            }
                        },
                    },
                    "GadgetPage": {"type": "object"},
                }
            },
        }
    )


def _stainless_config() -> StainlessConfig:
    return StainlessConfig.from_file(REPO_ROOT / "sdk" / "stainless.yaml")


def _stainless_config_without_jobs() -> StainlessConfig:
    return StainlessConfig(
        {
            "resources": {
                "$shared": {
                    "models": {
                        "shared_log": "SharedLog",
                        "shared_log_page": "SharedLogPage",
                    }
                }
            }
        }
    )


def _stainless_config_with_jobs() -> StainlessConfig:
    return StainlessConfig(
        {
            "resources": {
                "jobs": {
                    "standalone_api": True,
                    "methods": {},
                },
                "$shared": {
                    "models": {
                        "shared_log": "SharedLog",
                        "shared_log_page": "SharedLogPage",
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


def test_stainless_config_has_no_top_level_jobs_resource() -> None:
    resources = _load_stainless_config()["resources"]
    assert isinstance(resources, dict)

    assert "jobs" not in resources


def test_stainless_config_has_no_jobs_only_schema_mappings() -> None:
    jobs_only_schema_names = derive_source_owned_schema_names(_schema_usage(), (JOBS_RESOURCE,))

    assert jobs_only_schema_names.isdisjoint(_mapped_schema_names())


def test_stainless_config_keeps_shared_job_log_models_for_non_jobs_endpoints() -> None:
    generated_schema_names = derive_generated_schema_names(_schema_usage(), (JOBS_RESOURCE,))
    mapped_schema_names = _mapped_schema_names()

    assert PRESERVED_SHARED_SCHEMA_NAMES.issubset(generated_schema_names)
    assert PRESERVED_SHARED_SCHEMA_NAMES.issubset(mapped_schema_names)


def test_schema_mapper_skips_source_owned_jobs_methods_when_jobs_resource_is_absent() -> None:
    stainless_config = _stainless_config_without_jobs()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_jobs(), stainless_config, source_owned_resources=(JOBS_RESOURCE,)
    )

    assert mapper.sync_endpoints_with_methods() is True

    methods = stainless_config.extract_methods()
    assert all(not method.endpoint.path.startswith("/apis/jobs/v2") for method in methods)
    assert any(
        method.endpoint.path == "/apis/gadgets/v2/workspaces/{workspace}/gadgets"
        and method.method_name.startswith("reviewme_")
        for method in methods
    )


def test_schema_mapper_excludes_jobs_only_schemas_but_keeps_shared_schemas() -> None:
    stainless_config = _stainless_config_without_jobs()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_jobs(), stainless_config, source_owned_resources=(JOBS_RESOURCE,)
    )

    assert mapper.sync_schemas_with_models() is True

    schema_names = {model.schema_name for model in stainless_config.extract_models()}
    assert "JobCreateRequest" not in schema_names
    assert "JobRuntime" not in schema_names
    assert "JobResponse" not in schema_names
    assert "JobPage" not in schema_names
    assert "SharedLog" in schema_names
    assert "SharedLogPage" in schema_names
    assert "GadgetPage" in schema_names


def test_schema_mapper_includes_jobs_methods_when_jobs_resource_exists() -> None:
    stainless_config = _stainless_config_with_jobs()
    mapper = SchemaMapper(
        _openapi_spec_with_source_owned_jobs(), stainless_config, source_owned_resources=(JOBS_RESOURCE,)
    )

    assert mapper.sync_endpoints_with_methods() is True

    methods = stainless_config.extract_methods()
    assert any(method.endpoint.path.startswith("/apis/jobs/v2") for method in methods)
