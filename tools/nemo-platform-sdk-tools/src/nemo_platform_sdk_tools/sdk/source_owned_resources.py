# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPIEndpoint


@dataclass(frozen=True)
class SourceOwnedResource:
    resource_name: str
    path_prefixes: tuple[str, ...]


SOURCE_OWNED_RESOURCE_EXCLUSIONS = (
    SourceOwnedResource(
        resource_name="auth",
        path_prefixes=("/apis/auth/authenticate",),
    ),
    SourceOwnedResource(
        resource_name="access_keys",
        path_prefixes=("/apis/auth/v2/access-keys",),
    ),
    SourceOwnedResource(
        resource_name="iam",
        path_prefixes=("/apis/auth/v2/iam", "/apis/auth/v2/authz"),
    ),
    SourceOwnedResource(
        resource_name="jobs",
        path_prefixes=("/apis/jobs/v2",),
    ),
    SourceOwnedResource(
        resource_name="secrets",
        path_prefixes=("/apis/secrets/v2",),
    ),
)


def active_source_owned_resources(
    configured_resource_names: Collection[str],
    source_owned_resources: Sequence[SourceOwnedResource] = SOURCE_OWNED_RESOURCE_EXCLUSIONS,
) -> tuple[SourceOwnedResource, ...]:
    return tuple(
        resource for resource in source_owned_resources if resource.resource_name not in configured_resource_names
    )


def endpoint_is_source_owned(
    endpoint: OpenAPIEndpoint,
    source_owned_resources: Sequence[SourceOwnedResource],
) -> bool:
    return any(
        _path_matches_prefix(endpoint.path, path_prefix)
        for resource in source_owned_resources
        for path_prefix in resource.path_prefixes
    )


def derive_source_owned_schema_names(
    schema_usage: Mapping[str, Sequence[OpenAPIEndpoint]],
    source_owned_resources: Sequence[SourceOwnedResource],
) -> set[str]:
    return {
        schema_name
        for schema_name, endpoints in schema_usage.items()
        if endpoints and all(endpoint_is_source_owned(endpoint, source_owned_resources) for endpoint in endpoints)
    }


def derive_generated_schema_names(
    schema_usage: Mapping[str, Sequence[OpenAPIEndpoint]],
    source_owned_resources: Sequence[SourceOwnedResource],
) -> set[str]:
    return {
        schema_name
        for schema_name, endpoints in schema_usage.items()
        if any(not endpoint_is_source_owned(endpoint, source_owned_resources) for endpoint in endpoints)
    }


def filter_schema_usage_for_generated_resources(
    schema_usage: Mapping[str, Sequence[OpenAPIEndpoint]],
    source_owned_resources: Sequence[SourceOwnedResource],
) -> dict[str, list[OpenAPIEndpoint]]:
    source_owned_schema_names = derive_source_owned_schema_names(schema_usage, source_owned_resources)
    return {
        schema_name: [
            endpoint for endpoint in endpoints if not endpoint_is_source_owned(endpoint, source_owned_resources)
        ]
        for schema_name, endpoints in schema_usage.items()
        if schema_name not in source_owned_schema_names
    }


def _path_matches_prefix(path: str, path_prefix: str) -> bool:
    normalized_path = path.lower().rstrip("/")
    normalized_prefix = path_prefix.lower().rstrip("/")
    return normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}/")
