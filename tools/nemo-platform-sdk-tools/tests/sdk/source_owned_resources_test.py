# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_sdk_tools.sdk.core.openapi import OpenAPIEndpoint
from nemo_platform_sdk_tools.sdk.source_owned_resources import (
    SOURCE_OWNED_RESOURCE_EXCLUSIONS,
    SourceOwnedResource,
    active_source_owned_resources,
    derive_generated_schema_names,
    derive_source_owned_schema_names,
    endpoint_is_source_owned,
    filter_schema_usage_for_generated_resources,
)

JOBS = SourceOwnedResource(resource_name="jobs", path_prefixes=("/apis/jobs/v2",))
AUTH = SourceOwnedResource(resource_name="auth", path_prefixes=("/apis/auth/authenticate",))
ACCESS_KEYS = SourceOwnedResource(resource_name="access_keys", path_prefixes=("/apis/auth/v2/access-keys",))
IAM = SourceOwnedResource(resource_name="iam", path_prefixes=("/apis/auth/v2/iam", "/apis/auth/v2/authz"))
SECRETS = SourceOwnedResource(resource_name="secrets", path_prefixes=("/apis/secrets/v2",))
WIDGETS = SourceOwnedResource(resource_name="widgets", path_prefixes=("/apis/widgets/v2",))


def _endpoint(path: str) -> OpenAPIEndpoint:
    return OpenAPIEndpoint(method="get", path=path)


def test_default_source_owned_resource_registry_contains_migrated_resources() -> None:
    assert AUTH in SOURCE_OWNED_RESOURCE_EXCLUSIONS
    assert ACCESS_KEYS in SOURCE_OWNED_RESOURCE_EXCLUSIONS
    assert IAM in SOURCE_OWNED_RESOURCE_EXCLUSIONS
    assert JOBS in SOURCE_OWNED_RESOURCE_EXCLUSIONS
    assert SECRETS in SOURCE_OWNED_RESOURCE_EXCLUSIONS


def test_active_source_owned_resources_only_includes_absent_resources() -> None:
    active_resources = active_source_owned_resources(
        configured_resource_names={"files"},
        source_owned_resources=(
            WIDGETS,
            SourceOwnedResource(resource_name="files", path_prefixes=("/apis/files/v2",)),
        ),
    )

    assert active_resources == (WIDGETS,)


def test_endpoint_is_source_owned_uses_path_boundaries() -> None:
    assert endpoint_is_source_owned(_endpoint("/apis/widgets/v2/workspaces/default/widgets"), (WIDGETS,))
    assert endpoint_is_source_owned(_endpoint("/apis/widgets/v2"), (WIDGETS,))
    assert not endpoint_is_source_owned(_endpoint("/apis/widgets/v20/workspaces/default/widgets"), (WIDGETS,))
    assert not endpoint_is_source_owned(_endpoint("/apis/files/v2/workspaces/default/files"), (WIDGETS,))


def test_derive_source_owned_schema_names_excludes_shared_schemas() -> None:
    schema_usage = {
        "WidgetCreateRequest": [_endpoint("/apis/widgets/v2/workspaces/default/widgets")],
        "SharedLogPage": [
            _endpoint("/apis/widgets/v2/workspaces/default/widgets/widget/logs"),
            _endpoint("/apis/files/v2/workspaces/default/files/file/logs"),
        ],
        "FilePage": [_endpoint("/apis/files/v2/workspaces/default/files")],
    }

    assert derive_source_owned_schema_names(schema_usage, (WIDGETS,)) == {"WidgetCreateRequest"}
    assert derive_generated_schema_names(schema_usage, (WIDGETS,)) == {"SharedLogPage", "FilePage"}


def test_filter_schema_usage_for_generated_resources_removes_source_owned_endpoints() -> None:
    schema_usage = {
        "WidgetCreateRequest": [_endpoint("/apis/widgets/v2/workspaces/default/widgets")],
        "SharedLogPage": [
            _endpoint("/apis/widgets/v2/workspaces/default/widgets/widget/logs"),
            _endpoint("/apis/files/v2/workspaces/default/files/file/logs"),
        ],
    }

    assert filter_schema_usage_for_generated_resources(schema_usage, (WIDGETS,)) == {
        "SharedLogPage": [_endpoint("/apis/files/v2/workspaces/default/files/file/logs")]
    }
