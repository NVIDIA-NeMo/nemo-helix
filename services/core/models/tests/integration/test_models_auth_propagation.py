# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for auth context propagation in models service.

Verifies GitLab issue #3390: models and providers capture the creating
user's auth context for delegated access when the controller accesses secrets.

Auth context is only visible to service principals (principal ID starting with "service:")
and when auth is disabled. Regular users see auth_context stripped from responses.

Note: These tests verify creation with authenticated users works correctly.
Full verification of auth propagation to secret access is done in unit tests
(test_docker_backend.py, test_k8s_nim_operator_backend.py).
"""

from typing import Generator

import pytest
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.models.client import ModelsClient
from nmp.core.models.service import ModelsService
from nmp.testing import as_user, create_test_client, short_unique_name, unique_email

from .conftest import (
    create_deployment,
    create_deployment_config,
    create_provider,
    get_deployment,
    get_provider,
    list_deployments,
    list_providers,
    models_client_from_sdk,
    upsert_provider,
)


@pytest.fixture(scope="module")
def sdk() -> Generator[NeMoPlatform, None, None]:
    """SDK client with ModelsService (auth enabled)."""
    with create_test_client(
        ModelsService,
        auth_enabled=True,
    ) as sdk:
        yield sdk


def _as_service_principal(sdk: NeMoPlatform, service_name: str = "models-controller") -> ModelsClient:
    """Create an SDK client authenticated as a service principal."""
    return models_client_from_sdk(as_user(sdk, f"service:{service_name}"))


def _create_deployment(
    user_sdk: NeMoPlatform,
    workspace: str = "default",
    prefix: str = "test",
):
    """Create a deployment config + deployment, returning the deployment name."""
    client = models_client_from_sdk(user_sdk)
    config_name = short_unique_name(f"{prefix}-config")
    deployment_name = short_unique_name(f"{prefix}-deploy")
    create_deployment_config(
        client,
        workspace=workspace,
        name=config_name,
        engine="nim",
        model_spec={"model_type": "llm", "model_namespace": "nvidia", "model_name": "test-model"},
        executor_config={"image_name": "nvcr.io/nvidia/nim/llm", "image_tag": "latest", "gpu": 0},
    )
    return create_deployment(
        client,
        workspace=workspace,
        name=deployment_name,
        config=config_name,
        config_version=1,
    )


class TestDeploymentAuthPropagation:
    def test_auth_context_sanitized_for_regular_user(self, sdk: NeMoPlatform):
        """Regular users should not see auth_context on create, retrieve, or list."""
        creator_sdk = as_user(sdk, unique_email("creator"), groups=["team-alpha"])
        deployment = _create_deployment(creator_sdk, prefix="sanitize")

        # Create response
        assert deployment.auth_context is None, "create: regular user should not see auth_context"
        creator_client = models_client_from_sdk(creator_sdk)

        # Retrieve response
        retrieved = get_deployment(
            creator_client,
            workspace="default",
            name=deployment.name,
        )
        assert retrieved.auth_context is None, "retrieve: regular user should not see auth_context"

        # List response
        matching = [d for d in list_deployments(creator_client, workspace="default") if d.name == deployment.name]
        assert len(matching) == 1
        assert matching[0].auth_context is None, "list: regular user should not see auth_context"

    def test_auth_context_visible_to_service_principal(self, sdk: NeMoPlatform):
        """Service principals should see auth_context on retrieve and list."""
        creator_email = unique_email("creator")
        creator_groups = ["team-alpha", "ml-engineers"]
        creator_sdk = as_user(sdk, creator_email, groups=creator_groups)
        deployment = _create_deployment(creator_sdk, prefix="svc")

        service_sdk = _as_service_principal(sdk)

        # Retrieve response
        retrieved = get_deployment(
            service_sdk,
            workspace="default",
            name=deployment.name,
        )
        assert retrieved.auth_context is not None, "retrieve: service principal should see auth_context"
        assert retrieved.auth_context.principal_id == creator_email
        assert retrieved.auth_context.principal_email == creator_email
        assert retrieved.auth_context.principal_groups == creator_groups

        # List response
        matching = [d for d in list_deployments(service_sdk, workspace="default") if d.name == deployment.name]
        assert len(matching) == 1
        assert matching[0].auth_context is not None, "list: service principal should see auth_context"
        assert matching[0].auth_context.principal_id == creator_email
        assert matching[0].auth_context.principal_groups == creator_groups

    def test_auth_context_persisted_across_users(self, sdk: NeMoPlatform):
        """Auth context persists the original creator's identity, invisible to other users."""
        creator_email = unique_email("creator")
        creator_groups = ["admins"]
        creator_sdk = as_user(sdk, creator_email, groups=creator_groups)
        deployment = _create_deployment(creator_sdk, prefix="persist")

        # Different regular user should not see auth_context
        other_user = as_user(sdk, unique_email("admin"), groups=["admins"])
        retrieved_by_user = get_deployment(
            models_client_from_sdk(other_user),
            workspace="default",
            name=deployment.name,
        )
        assert retrieved_by_user.auth_context is None, "Regular user should not see auth_context"

        # Service principal should see the original creator's auth_context
        service_sdk = _as_service_principal(sdk)
        retrieved_by_service = get_deployment(
            service_sdk,
            workspace="default",
            name=deployment.name,
        )
        assert retrieved_by_service.auth_context is not None
        assert retrieved_by_service.auth_context.principal_id == creator_email
        assert retrieved_by_service.auth_context.principal_groups == creator_groups


class TestProviderAuthPropagation:
    def test_auth_context_captured_at_creation(self, sdk: NeMoPlatform):
        """Auth context is captured when provider is created, visible to service principals."""
        creator_email = unique_email("creator")
        creator_groups = ["team-beta"]
        provider_name = short_unique_name("auth-prov")

        creator_sdk = as_user(sdk, creator_email, groups=creator_groups)
        creator_client = models_client_from_sdk(creator_sdk)

        # Regular user should not see auth_context in the create response
        provider = create_provider(
            creator_client,
            workspace="default",
            name=provider_name,
            host_url="http://test.local:8000",
        )
        assert provider.auth_context is None, "Regular user should not see auth_context"

        # Service principal should see it
        service_sdk = _as_service_principal(sdk)
        retrieved = get_provider(
            service_sdk,
            workspace="default",
            name=provider_name,
        )
        assert retrieved.auth_context is not None, "Service principal should see auth_context"
        assert retrieved.auth_context.principal_id == creator_email
        assert retrieved.auth_context.principal_email == creator_email
        assert retrieved.auth_context.principal_groups == creator_groups

    def test_auth_context_stripped_for_regular_user_on_list(self, sdk: NeMoPlatform):
        """Auth context should be stripped from list responses for regular users."""
        provider_name = short_unique_name("list-prov")

        creator_sdk = as_user(sdk, unique_email("creator"), groups=["team-gamma"])
        creator_client = models_client_from_sdk(creator_sdk)

        create_provider(
            creator_client,
            workspace="default",
            name=provider_name,
            host_url="http://test.local:8000",
        )

        # List as regular user — auth_context should be stripped
        matching = [p for p in list_providers(creator_client, workspace="default") if p.name == provider_name]
        assert len(matching) == 1
        assert matching[0].auth_context is None, "Regular user should not see auth_context in list"

    def test_auth_context_on_upsert(self, sdk: NeMoPlatform):
        """Auth context is captured on upsert (create and update paths), visible to service principals."""
        creator_email = unique_email("creator")
        creator_groups = ["ml-ops"]
        provider_name = short_unique_name("upsert-prov")

        creator_sdk = as_user(sdk, creator_email, groups=creator_groups)
        creator_client = models_client_from_sdk(creator_sdk)

        # Upsert creates a new provider
        provider = upsert_provider(
            creator_client,
            workspace="default",
            name=provider_name,
            host_url="http://upsert.local:8000",
        )
        assert provider.auth_context is None, "Regular user should not see auth_context"

        # Service principal should see auth_context after create
        service_sdk = _as_service_principal(sdk)
        retrieved = get_provider(
            service_sdk,
            workspace="default",
            name=provider_name,
        )
        assert retrieved.auth_context is not None
        assert retrieved.auth_context.principal_id == creator_email
        assert retrieved.auth_context.principal_groups == creator_groups

        # Upsert updates the existing provider
        updated = upsert_provider(
            creator_client,
            workspace="default",
            name=provider_name,
            host_url="http://upsert-updated.local:9000",
        )
        assert updated.auth_context is None, "Regular user should not see auth_context after update"

        # Service principal should still see auth_context after update
        retrieved_after_update = get_provider(
            service_sdk,
            workspace="default",
            name=provider_name,
        )
        assert retrieved_after_update.auth_context is not None, "Service principal should see auth_context after update"
        assert retrieved_after_update.auth_context.principal_id == creator_email
        assert retrieved_after_update.auth_context.principal_groups == creator_groups
