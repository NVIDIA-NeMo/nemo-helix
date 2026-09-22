# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_helix_sdk_tools.sdk.core.openapi import OpenAPI
from nemo_helix_sdk_tools.sdk.core.stainless import StainlessConfig
from nemo_helix_sdk_tools.sdk.openapi_stainless_mapper import SchemaMapper
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[4]
REMOVED_RESOURCE_NAMES = {"audit", "safe_synthesizer"}
REMOVED_PATH_PREFIXES = ("/apis/audit", "/apis/safe-synthesizer", "/apis/safe_synthesizer")


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


def _stainless_config() -> StainlessConfig:
    return StainlessConfig.from_file(REPO_ROOT / "sdk" / "stainless.yaml")


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


def _openapi_spec_with_unrelated_generated_resource() -> OpenAPI:
    return OpenAPI(
        {
            "openapi": "3.1.0",
            "paths": {
                "/apis/widgets/v2/workspaces/{workspace}/widgets": {
                    "get": {
                        "responses": {
                            "200": _json_response("WidgetPage"),
                        }
                    }
                }
            },
            "components": {
                "schemas": {
                    "WidgetPage": {"type": "object"},
                }
            },
        }
    )


def test_stainless_config_has_no_removed_empty_shell_resources() -> None:
    resources = _string_mapping(_load_stainless_config()["resources"])

    assert REMOVED_RESOURCE_NAMES.isdisjoint(resources)


def test_readme_examples_do_not_reference_removed_shell_paths() -> None:
    readme = _string_mapping(_load_stainless_config()["readme"])
    example_requests = _string_mapping(readme["example_requests"])

    endpoints: list[str] = []
    for example in example_requests.values():
        if not isinstance(example, dict):
            continue
        endpoint = _string_mapping(example).get("endpoint")
        if isinstance(endpoint, str):
            endpoints.append(endpoint)

    assert all(not endpoint.startswith(REMOVED_PATH_PREFIXES) for endpoint in endpoints)


def test_aggregate_openapi_has_no_removed_shell_paths() -> None:
    paths = OpenAPI.from_file(REPO_ROOT / "openapi" / "openapi.yaml").extract_endpoints()

    assert all(not endpoint.path.startswith(REMOVED_PATH_PREFIXES) for endpoint in paths)


def test_schema_mapper_still_discovers_unrelated_generated_resources_without_shells() -> None:
    stainless_config = StainlessConfig({"resources": {}})
    mapper = SchemaMapper(
        _openapi_spec_with_unrelated_generated_resource(), stainless_config, source_owned_resources=()
    )

    assert mapper.sync_endpoints_with_methods() is True

    methods = stainless_config.extract_methods()
    assert any(method.endpoint.path == "/apis/widgets/v2/workspaces/{workspace}/widgets" for method in methods)


def test_real_stainless_config_methods_avoid_removed_shell_paths() -> None:
    methods = _stainless_config().extract_methods()

    assert all(not method.endpoint.path.startswith(REMOVED_PATH_PREFIXES) for method in methods)
