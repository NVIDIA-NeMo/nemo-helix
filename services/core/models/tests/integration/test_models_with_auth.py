# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Models service AuthZ.

These tests verify:
- Unauthenticated requests are rejected (401)
- Viewer can list/read but not create/update/delete
- Editor can create/update/delete resources
- ModelProvider create/upsert with api_key_secret_name requires secrets.read
- ModelProvider create/upsert with model_deployment_id requires inference.deployments.read
- ModelDeployment create/update requires inference.deployment-configs.read for config reference
- ModelDeploymentConfig create/update with model_entity_id requires models.read
- Model/adapter create/update with fileset requires filesets.read
- trust_remote_code set requires models.trust-remote-code.set when repo not on allow list

Uses the create_test_client pattern with auth_enabled=True for fast in-memory testing.
"""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

import pytest
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import PermissionDeniedError
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    CreateModelAdapterRequest,
    CreateModelEntityRequest,
    FinetuningType,
    UpdateModelEntityRequest,
)
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.core.auth.app.bundle import build_authorization_data as _real_build_authorization_data
from nmp.core.files.service import FilesService
from nmp.core.models.config import config as models_config
from nmp.core.models.service import ModelsService
from nmp.core.secrets.service import SecretsService
from nmp.testing import (
    TEST_ADMIN_EMAIL,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)
from pydantic import SecretStr

from .conftest import (
    create_deployment,
    create_deployment_config,
    create_provider,
    delete_deployment,
    delete_deployment_config,
    delete_provider,
    get_deployment,
    get_deployment_config,
    get_provider,
    list_deployment_configs,
    list_deployments,
    list_providers,
    models_client_from_sdk,
    update_deployment,
    update_deployment_config,
    upsert_provider,
)


async def _build_authorization_data_without_secrets(entities_client=None):
    """Wraps the real build_authorization_data to inject an EditorNoSecrets role.

    The custom role has all Editor write permissions but inherits from a
    ViewerNoSecrets role that strips secrets.read and secrets.list.
    """
    data = await _real_build_authorization_data(entities_client)
    data["authz"]["roles"]["ViewerNoSecrets"] = {
        "description": "Viewer without secrets (test-only)",
        "permissions": [
            p for p in data["authz"]["roles"]["Viewer"]["permissions"] if p not in ("secrets.read", "secrets.list")
        ],
    }
    data["authz"]["roles"]["EditorNoSecrets"] = {
        "description": "Editor without secrets access (test-only)",
        "includes": ["ViewerNoSecrets"],
        "permissions": list(data["authz"]["roles"]["Editor"]["permissions"]),
    }
    return data


async def _build_authorization_data_without_deployment_read(entities_client=None):
    """Inject EditorNoDeploymentRead: Editor without inference.deployments.read.

    Used to test that provider model_deployment_id requires inference.deployments.read.
    """
    data = await _real_build_authorization_data(entities_client)
    strip = ("inference.deployments.read", "inference.deployments.list")
    data["authz"]["roles"]["ViewerNoDeploymentRead"] = {
        "description": "Viewer without deployment read (test-only)",
        "permissions": [p for p in data["authz"]["roles"]["Viewer"]["permissions"] if p not in strip],
    }
    data["authz"]["roles"]["EditorNoDeploymentRead"] = {
        "description": "Editor without deployment read (test-only)",
        "includes": ["ViewerNoDeploymentRead"],
        "permissions": [p for p in data["authz"]["roles"]["Editor"]["permissions"] if p not in strip],
    }
    return data


async def _build_authorization_data_without_deployment_config_read(entities_client=None):
    """Inject EditorNoDeploymentConfigRead: Editor without inference.deployment-configs.read.

    Used to test that deployment config reference requires inference.deployment-configs.read.
    """
    data = await _real_build_authorization_data(entities_client)
    strip = ("inference.deployment-configs.read", "inference.deployment-configs.list")
    data["authz"]["roles"]["ViewerNoDeploymentConfigRead"] = {
        "description": "Viewer without deployment config read (test-only)",
        "permissions": [p for p in data["authz"]["roles"]["Viewer"]["permissions"] if p not in strip],
    }
    data["authz"]["roles"]["EditorNoDeploymentConfigRead"] = {
        "description": "Editor without deployment config read (test-only)",
        "includes": ["ViewerNoDeploymentConfigRead"],
        "permissions": [p for p in data["authz"]["roles"]["Editor"]["permissions"] if p not in strip],
    }
    return data


async def _build_authorization_data_without_model_read(entities_client=None):
    """Inject EditorNoModelRead: Editor without models.read.

    Used to test that deployment config model_entity_id requires models.read.
    """
    data = await _real_build_authorization_data(entities_client)
    strip = ("models.read", "models.list")
    data["authz"]["roles"]["ViewerNoModelRead"] = {
        "description": "Viewer without model read (test-only)",
        "permissions": [p for p in data["authz"]["roles"]["Viewer"]["permissions"] if p not in strip],
    }
    data["authz"]["roles"]["EditorNoModelRead"] = {
        "description": "Editor without model read (test-only)",
        "includes": ["ViewerNoModelRead"],
        "permissions": [p for p in data["authz"]["roles"]["Editor"]["permissions"] if p not in strip],
    }
    return data


async def _build_authorization_data_without_fileset_read(entities_client=None):
    """Inject EditorNoFilesetRead: Editor without filesets.read.

    Used to test that model/adapter fileset reference requires filesets.read.
    """
    data = await _real_build_authorization_data(entities_client)
    strip = ("filesets.read", "filesets.list")
    data["authz"]["roles"]["ViewerNoFilesetRead"] = {
        "description": "Viewer without fileset read (test-only)",
        "permissions": [p for p in data["authz"]["roles"]["Viewer"]["permissions"] if p not in strip],
    }
    data["authz"]["roles"]["EditorNoFilesetRead"] = {
        "description": "Editor without fileset read (test-only)",
        "includes": ["ViewerNoFilesetRead"],
        "permissions": [p for p in data["authz"]["roles"]["Editor"]["permissions"] if p not in strip],
    }
    return data


@contextmanager
def patched_authz_data(build_fn):
    """Patch build_authorization_data in both the bundle and embedded PDP modules."""
    with (
        patch("nmp.core.auth.app.bundle.build_authorization_data", side_effect=build_fn),
        patch("nmp.core.auth.app.embedded_pdp.data.build_authorization_data", side_effect=build_fn),
    ):
        yield


@pytest.fixture(scope="module")
def sdk() -> Generator[NeMoPlatform, None, None]:
    """SDK client with ModelsService, FilesService, and SecretsService (auth enabled).

    FilesService is needed for model/adapter create with fileset (validate_fileset_ref_exists).
    SecretsService is needed for provider create with api_key_secret_name (check_secret_access).
    """
    with create_test_client(
        ModelsService,
        FilesService,
        SecretsService,
        auth_enabled=True,
    ) as sdk:
        yield sdk


@pytest.mark.integration
class TestModelsUnauthenticated:
    """Unauthenticated requests should be rejected (401) for all Models API endpoints."""

    # -- Models --

    def test_list_models_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/models")
        assert response.status_code == 401

    def test_get_model_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/models/any-model")
        assert response.status_code == 401

    def test_create_model_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.post(
            "/apis/models/v2/workspaces/default/models",
            json={"name": "test-model"},
        )
        assert response.status_code == 401

    def test_update_model_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.patch(
            "/apis/models/v2/workspaces/default/models/any-model",
            json={"description": "updated"},
        )
        assert response.status_code == 401

    def test_delete_model_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.delete("/apis/models/v2/workspaces/default/models/any-model")
        assert response.status_code == 401

    # -- Providers --

    def test_list_providers_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/providers")
        assert response.status_code == 401

    def test_get_provider_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/providers/any-provider")
        assert response.status_code == 401

    def test_create_provider_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.post(
            "/apis/models/v2/workspaces/default/providers",
            json={"name": "test-provider", "host_url": "http://example.com"},
        )
        assert response.status_code == 401

    def test_upsert_provider_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.put(
            "/apis/models/v2/workspaces/default/providers/any-provider",
            json={"host_url": "http://example.com"},
        )
        assert response.status_code == 401

    def test_delete_provider_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.delete("/apis/models/v2/workspaces/default/providers/any-provider")
        assert response.status_code == 401

    # -- Deployments --

    def test_list_deployments_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/deployments")
        assert response.status_code == 401

    def test_get_deployment_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/deployments/any-deployment")
        assert response.status_code == 401

    def test_create_deployment_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.post(
            "/apis/models/v2/workspaces/default/deployments",
            json={"name": "test-deploy", "config": "some-config"},
        )
        assert response.status_code == 401

    def test_update_deployment_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.post(
            "/apis/models/v2/workspaces/default/deployments/any-deployment",
            json={"config": "some-config"},
        )
        assert response.status_code == 401

    def test_delete_deployment_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.delete("/apis/models/v2/workspaces/default/deployments/any-deployment")
        assert response.status_code == 401

    # -- Deployment Configs --

    def test_list_deployment_configs_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/deployment-configs")
        assert response.status_code == 401

    def test_get_deployment_config_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.get("/apis/models/v2/workspaces/default/deployment-configs/any-config")
        assert response.status_code == 401

    def test_create_deployment_config_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.post(
            "/apis/models/v2/workspaces/default/deployment-configs",
            json={
                "name": "test-config",
                "engine": "nim",
                "model_spec": {"model_name": "test"},
                "executor_config": {"gpu": 1},
            },
        )
        assert response.status_code == 401

    def test_update_deployment_config_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.post(
            "/apis/models/v2/workspaces/default/deployment-configs/any-config",
            json={"engine": "nim", "model_spec": {"model_name": "test"}, "executor_config": {"gpu": 1}},
        )
        assert response.status_code == 401

    def test_delete_deployment_config_without_auth_fails(self, sdk: NeMoPlatform):
        response = sdk._client.delete("/apis/models/v2/workspaces/default/deployment-configs/any-config")
        assert response.status_code == 401


@pytest.fixture()
def viewer_workspace(sdk: NeMoPlatform):
    """Create a workspace with a Viewer user and pre-populated resources for read tests.

    Returns (workspace, viewer_sdk, admin_sdk, resource_names) where resource_names
    contains the names of each resource type created by the admin.
    """
    workspace = short_unique_name("vw")
    viewer_email = unique_email("viewer")

    admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
    client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
        body=CreateWorkspaceRequest(name=workspace)
    ).data()
    grant_workspace_role(
        admin_sdk,
        workspace=workspace,
        principal=viewer_email,
        roles=["Viewer"],
    )

    model_name = short_unique_name("mdl")
    provider_name = short_unique_name("prv")
    config_name = short_unique_name("cfg")
    deployment_name = short_unique_name("dep")

    client_from_platform(admin_sdk, ModelsClient).create_model(
        workspace=workspace, body=CreateModelEntityRequest(name=model_name)
    ).data()
    create_provider(
        models_client_from_sdk(admin_sdk),
        workspace=workspace,
        name=provider_name,
        host_url="http://example.com",
    )
    create_deployment_config(
        models_client_from_sdk(admin_sdk),
        workspace=workspace,
        name=config_name,
        engine="nim",
        model_spec={"model_name": "test"},
        executor_config={"gpu": 1},
    )
    create_deployment(
        models_client_from_sdk(admin_sdk),
        workspace=workspace,
        name=deployment_name,
        config=config_name,
    )

    viewer_sdk = as_user(sdk, viewer_email)
    return (
        workspace,
        viewer_sdk,
        admin_sdk,
        {
            "model": model_name,
            "provider": provider_name,
            "config": config_name,
            "deployment": deployment_name,
        },
    )


@pytest.mark.integration
class TestViewerModelsAccess:
    """Test that Viewer role can list/read but not create/update/delete."""

    # -- Models: allowed --

    def test_viewer_can_list_models(self, viewer_workspace):
        workspace, viewer_sdk, _, _ = viewer_workspace
        result = client_from_platform(viewer_sdk, ModelsClient).list_models(workspace=workspace)
        assert list(result.items()) is not None

    def test_viewer_can_get_model(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        model = (
            client_from_platform(viewer_sdk, ModelsClient).get_model(name=names["model"], workspace=workspace).data()
        )
        assert model.name == names["model"]

    # -- Models: denied --

    def test_viewer_cannot_create_model(self, viewer_workspace):
        workspace, viewer_sdk, _, _ = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            client_from_platform(viewer_sdk, ModelsClient).create_model(
                workspace=workspace, body=CreateModelEntityRequest(name="should-fail")
            ).data()

    def test_viewer_cannot_update_model(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            client_from_platform(viewer_sdk, ModelsClient).update_model(
                name=names["model"], workspace=workspace, body=UpdateModelEntityRequest(description="nope")
            ).data()

    def test_viewer_cannot_delete_model(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            client_from_platform(viewer_sdk, ModelsClient).delete_model(name=names["model"], workspace=workspace)

    # -- Providers: allowed --

    def test_viewer_can_list_providers(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        result = list_providers(models_client_from_sdk(viewer_sdk), workspace=workspace)
        assert any(provider.name == names["provider"] for provider in result)

    def test_viewer_can_get_provider(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        provider = get_provider(models_client_from_sdk(viewer_sdk), name=names["provider"], workspace=workspace)
        assert provider.name == names["provider"]

    # -- Providers: denied --

    def test_viewer_cannot_create_provider(self, viewer_workspace):
        workspace, viewer_sdk, _, _ = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            create_provider(
                models_client_from_sdk(viewer_sdk),
                workspace=workspace,
                name="should-fail",
                host_url="http://example.com",
            )

    def test_viewer_cannot_upsert_provider(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            upsert_provider(
                models_client_from_sdk(viewer_sdk),
                name=names["provider"],
                workspace=workspace,
                host_url="http://updated.com",
            )

    def test_viewer_cannot_delete_provider(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            delete_provider(models_client_from_sdk(viewer_sdk), name=names["provider"], workspace=workspace)

    # -- Deployment Configs: allowed --

    def test_viewer_can_list_deployment_configs(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        result = list_deployment_configs(models_client_from_sdk(viewer_sdk), workspace=workspace)
        assert any(config.name == names["config"] for config in result)

    def test_viewer_can_get_deployment_config(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        config = get_deployment_config(models_client_from_sdk(viewer_sdk), name=names["config"], workspace=workspace)
        assert config.name == names["config"]

    # -- Deployment Configs: denied --

    def test_viewer_cannot_create_deployment_config(self, viewer_workspace):
        workspace, viewer_sdk, _, _ = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            create_deployment_config(
                models_client_from_sdk(viewer_sdk),
                workspace=workspace,
                name="should-fail",
                engine="nim",
                model_spec={"model_name": "test-model"},
                executor_config={"gpu": 1},
            )

    def test_viewer_cannot_update_deployment_config(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            update_deployment_config(
                models_client_from_sdk(viewer_sdk),
                name=names["config"],
                workspace=workspace,
                engine="nim",
                model_spec={"model_name": "updated"},
                executor_config={"gpu": 2},
            )

    def test_viewer_cannot_delete_deployment_config(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            delete_deployment_config(models_client_from_sdk(viewer_sdk), name=names["config"], workspace=workspace)

    # -- Deployments: allowed --

    def test_viewer_can_list_deployments(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        result = list_deployments(models_client_from_sdk(viewer_sdk), workspace=workspace)
        assert any(deployment.name == names["deployment"] for deployment in result)

    def test_viewer_can_get_deployment(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        deployment = get_deployment(models_client_from_sdk(viewer_sdk), name=names["deployment"], workspace=workspace)
        assert deployment.name == names["deployment"]

    # -- Deployments: denied --

    def test_viewer_cannot_create_deployment(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            create_deployment(
                models_client_from_sdk(viewer_sdk),
                workspace=workspace,
                name="should-fail",
                config=names["config"],
            )

    def test_viewer_cannot_update_deployment(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            update_deployment(
                models_client_from_sdk(viewer_sdk),
                name=names["deployment"],
                workspace=workspace,
                config=names["config"],
            )

    def test_viewer_cannot_delete_deployment(self, viewer_workspace):
        workspace, viewer_sdk, _, names = viewer_workspace
        with pytest.raises(PermissionDeniedError):
            delete_deployment(models_client_from_sdk(viewer_sdk), name=names["deployment"], workspace=workspace)


@pytest.mark.integration
class TestEditorModelsAccess:
    """Test that Editor role can create, read, update, and delete resources."""

    def test_editor_can_create_and_read_model(self, sdk: NeMoPlatform):
        workspace = short_unique_name("ed-mdl")
        editor_email = unique_email("editor")
        model_name = short_unique_name("model")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        models = client_from_platform(editor_sdk, ModelsClient)
        created = models.create_model(workspace=workspace, body=CreateModelEntityRequest(name=model_name)).data()
        assert created.name == model_name

        retrieved = models.get_model(name=model_name, workspace=workspace).data()
        assert retrieved.name == model_name

    def test_editor_can_delete_model(self, sdk: NeMoPlatform):
        workspace = short_unique_name("ed-del")
        editor_email = unique_email("editor")
        model_name = short_unique_name("model")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        models = client_from_platform(editor_sdk, ModelsClient)
        models.create_model(workspace=workspace, body=CreateModelEntityRequest(name=model_name)).data()
        models.delete_model(name=model_name, workspace=workspace)

    def test_editor_can_create_provider(self, sdk: NeMoPlatform):
        workspace = short_unique_name("ed-prv")
        editor_email = unique_email("editor")
        provider_name = short_unique_name("prov")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        created = create_provider(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=provider_name,
            host_url="http://example.com",
        )
        assert created.name == provider_name

    def test_editor_can_create_deployment_config(self, sdk: NeMoPlatform):
        workspace = short_unique_name("ed-cfg")
        editor_email = unique_email("editor")
        config_name = short_unique_name("cfg")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        created = create_deployment_config(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=config_name,
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 1},
        )
        assert created.name == config_name


@pytest.mark.integration
class TestProviderSecretPermissions:
    """Test that creating/upserting a provider with api_key_secret_name requires secrets.read."""

    def test_editor_can_create_provider_without_secret(self, sdk: NeMoPlatform):
        workspace = short_unique_name("ps-nos")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        provider = create_provider(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=short_unique_name("prov"),
            host_url="http://example.com",
        )
        assert provider.api_key_secret_name is None

    def test_editor_can_create_provider_with_secret(self, sdk: NeMoPlatform):
        """Editor has secrets.read via Viewer inheritance, so this should succeed."""
        workspace = short_unique_name("ps-sec")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, SecretsClient).create_secret(
            body=PlatformSecretCreateRequest(name="my-api-key", value=SecretStr("test-value")),
            workspace=workspace,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        provider = create_provider(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=short_unique_name("prov"),
            host_url="http://example.com",
            api_key_secret_name="my-api-key",
        )
        assert provider.api_key_secret_name == "my-api-key"

    def test_editor_can_upsert_provider_with_secret(self, sdk: NeMoPlatform):
        """Editor has secrets.read via Viewer inheritance, so upsert with secret should succeed."""
        workspace = short_unique_name("ps-ups")
        editor_email = unique_email("editor")
        provider_name = short_unique_name("prov")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, SecretsClient).create_secret(
            body=PlatformSecretCreateRequest(name="my-api-key", value=SecretStr("test-value")),
            workspace=workspace,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        provider = upsert_provider(
            models_client_from_sdk(editor_sdk),
            name=provider_name,
            workspace=workspace,
            host_url="http://example.com",
            api_key_secret_name="my-api-key",
        )
        assert provider.api_key_secret_name == "my-api-key"

    def test_custom_role_denied_create_provider_with_secret(self, sdk: NeMoPlatform):
        """A role with provider write but no secrets.read should be denied on create."""
        with patched_authz_data(_build_authorization_data_without_secrets):
            workspace = short_unique_name("ns-crt")
            user_email = unique_email("nosecrets")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            client_from_platform(admin_sdk, SecretsClient).create_secret(
                body=PlatformSecretCreateRequest(name="should-be-denied", value=SecretStr("test")),
                workspace=workspace,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoSecrets"],
            )

            user_sdk = as_user(sdk, user_email)

            provider_ok = create_provider(
                models_client_from_sdk(user_sdk),
                workspace=workspace,
                name=short_unique_name("prov"),
                host_url="http://example.com",
            )
            assert provider_ok.api_key_secret_name is None

            with pytest.raises(PermissionDeniedError):
                create_provider(
                    models_client_from_sdk(user_sdk),
                    workspace=workspace,
                    name=short_unique_name("prov"),
                    host_url="http://example.com",
                    api_key_secret_name="should-be-denied",
                )

    def test_custom_role_denied_upsert_provider_with_secret(self, sdk: NeMoPlatform):
        """A role with provider write but no secrets.read should be denied on upsert."""
        with patched_authz_data(_build_authorization_data_without_secrets):
            workspace = short_unique_name("ns-ups")
            user_email = unique_email("nosecrets")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            client_from_platform(admin_sdk, SecretsClient).create_secret(
                body=PlatformSecretCreateRequest(name="should-be-denied", value=SecretStr("test")),
                workspace=workspace,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoSecrets"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                upsert_provider(
                    models_client_from_sdk(user_sdk),
                    name=short_unique_name("prov"),
                    workspace=workspace,
                    host_url="http://example.com",
                    api_key_secret_name="should-be-denied",
                )


@pytest.mark.integration
class TestProviderDeploymentRefPermissions:
    """Test that creating/upserting a provider with model_deployment_id requires inference.deployments.read."""

    def test_editor_can_create_provider_with_deployment_ref(self, sdk: NeMoPlatform):
        """Editor has inference.deployments.read, so referencing a deployment should succeed."""
        workspace = short_unique_name("pd-ok")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        provider = create_provider(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=short_unique_name("prov"),
            host_url="http://example.com",
            model_deployment_id=f"{workspace}/some-deployment",
        )
        assert provider.model_deployment_id == f"{workspace}/some-deployment"

    def test_editor_can_upsert_provider_with_deployment_ref(self, sdk: NeMoPlatform):
        """Editor has inference.deployments.read, so upsert with deployment ref should succeed."""
        workspace = short_unique_name("pd-ups")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        provider = upsert_provider(
            models_client_from_sdk(editor_sdk),
            name=short_unique_name("prov"),
            workspace=workspace,
            host_url="http://example.com",
            model_deployment_id=f"{workspace}/some-deployment",
        )
        assert provider.model_deployment_id == f"{workspace}/some-deployment"

    def test_custom_role_denied_create_provider_with_deployment_ref(self, sdk: NeMoPlatform):
        """A role without inference.deployments.read should be denied when referencing a deployment."""
        with patched_authz_data(_build_authorization_data_without_deployment_read):
            workspace = short_unique_name("pd-ncr")
            user_email = unique_email("noread")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoDeploymentRead"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                create_provider(
                    models_client_from_sdk(user_sdk),
                    workspace=workspace,
                    name=short_unique_name("prov"),
                    host_url="http://example.com",
                    model_deployment_id=f"{workspace}/some-deployment",
                )

    def test_custom_role_denied_upsert_provider_with_deployment_ref(self, sdk: NeMoPlatform):
        """A role without inference.deployments.read should be denied when upserting with a deployment ref."""
        with patched_authz_data(_build_authorization_data_without_deployment_read):
            workspace = short_unique_name("pd-nup")
            user_email = unique_email("noread")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoDeploymentRead"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                upsert_provider(
                    models_client_from_sdk(user_sdk),
                    name=short_unique_name("prov"),
                    workspace=workspace,
                    host_url="http://example.com",
                    model_deployment_id=f"{workspace}/some-deployment",
                )


@pytest.mark.integration
class TestDeploymentConfigPermissions:
    """Test that creating/updating a deployment config with model_entity_id requires models.read."""

    def test_editor_can_create_config_with_model_entity_id(self, sdk: NeMoPlatform):
        """Editor has models.read, so referencing a model_entity_id should succeed."""
        workspace = short_unique_name("dc-mei")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        config = create_deployment_config(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=short_unique_name("cfg"),
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 1},
            model_entity_id=f"{workspace}/my-model",
        )
        assert config.model_entity_id == f"{workspace}/my-model"

    def test_editor_cannot_reference_model_in_inaccessible_workspace(self, sdk: NeMoPlatform):
        """Editor should be denied when referencing a model_entity_id in a workspace they can't access."""
        workspace = short_unique_name("dc-nop")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        with pytest.raises(PermissionDeniedError):
            create_deployment_config(
                models_client_from_sdk(editor_sdk),
                workspace=workspace,
                name=short_unique_name("cfg"),
                engine="nim",
                model_spec={"model_name": "test-model"},
                executor_config={"gpu": 1},
                model_entity_id="inaccessible-workspace/some-model",
            )

    def test_editor_can_update_config_with_model_entity_id(self, sdk: NeMoPlatform):
        """Editor has models.read, so updating a config with model_entity_id should succeed."""
        workspace = short_unique_name("dc-upd")
        editor_email = unique_email("editor")
        config_name = short_unique_name("cfg")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        create_deployment_config(
            models_client_from_sdk(admin_sdk),
            workspace=workspace,
            name=config_name,
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 1},
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        updated = update_deployment_config(
            models_client_from_sdk(editor_sdk),
            name=config_name,
            workspace=workspace,
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 2},
            model_entity_id=f"{workspace}/my-model",
        )
        assert updated.model_entity_id == f"{workspace}/my-model"

    def test_editor_cannot_update_config_with_inaccessible_model(self, sdk: NeMoPlatform):
        """Editor should be denied when updating a config with a model_entity_id in an inaccessible workspace."""
        workspace = short_unique_name("dc-unp")
        editor_email = unique_email("editor")
        config_name = short_unique_name("cfg")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        create_deployment_config(
            models_client_from_sdk(admin_sdk),
            workspace=workspace,
            name=config_name,
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 1},
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        with pytest.raises(PermissionDeniedError):
            update_deployment_config(
                models_client_from_sdk(editor_sdk),
                name=config_name,
                workspace=workspace,
                engine="nim",
                model_spec={"model_name": "test-model"},
                executor_config={"gpu": 2},
                model_entity_id="inaccessible-workspace/some-model",
            )

    def test_custom_role_denied_create_config_with_model_entity_id_without_read(self, sdk: NeMoPlatform):
        """A role without models.read should be denied when creating a config with model_entity_id."""
        with patched_authz_data(_build_authorization_data_without_model_read):
            workspace = short_unique_name("dc-ncr")
            user_email = unique_email("noread")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoModelRead"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                create_deployment_config(
                    models_client_from_sdk(user_sdk),
                    workspace=workspace,
                    name=short_unique_name("cfg"),
                    engine="nim",
                    model_spec={"model_name": "test-model"},
                    executor_config={"gpu": 1},
                    model_entity_id=f"{workspace}/some-model",
                )

    def test_custom_role_denied_update_config_with_model_entity_id_without_read(self, sdk: NeMoPlatform):
        """A role without models.read should be denied when updating a config with model_entity_id."""
        with patched_authz_data(_build_authorization_data_without_model_read):
            workspace = short_unique_name("dc-nup")
            user_email = unique_email("noread")
            config_name = short_unique_name("cfg")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            create_deployment_config(
                models_client_from_sdk(admin_sdk),
                workspace=workspace,
                name=config_name,
                engine="nim",
                model_spec={"model_name": "test-model"},
                executor_config={"gpu": 1},
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoModelRead"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                update_deployment_config(
                    models_client_from_sdk(user_sdk),
                    name=config_name,
                    workspace=workspace,
                    engine="nim",
                    model_spec={"model_name": "test-model"},
                    executor_config={"gpu": 2},
                    model_entity_id=f"{workspace}/some-model",
                )


@pytest.mark.integration
class TestDeploymentPermissions:
    """Test that creating/updating a deployment with a config reference requires inference.deployment-configs.read."""

    def test_editor_can_create_deployment_with_config(self, sdk: NeMoPlatform):
        """Editor has inference.deployment-configs.read, so referencing a config should succeed."""
        workspace = short_unique_name("dp-ok")
        editor_email = unique_email("editor")
        config_name = short_unique_name("cfg")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        create_deployment_config(
            models_client_from_sdk(admin_sdk),
            workspace=workspace,
            name=config_name,
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 1},
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        deployment = create_deployment(
            models_client_from_sdk(editor_sdk),
            workspace=workspace,
            name=short_unique_name("dep"),
            config=config_name,
        )
        assert deployment.config == config_name

    def test_editor_can_update_deployment_with_config(self, sdk: NeMoPlatform):
        """Editor has inference.deployment-configs.read, so updating a deployment with a config ref should succeed."""
        workspace = short_unique_name("dp-upd")
        editor_email = unique_email("editor")
        config_name = short_unique_name("cfg")
        deploy_name = short_unique_name("dep")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        create_deployment_config(
            models_client_from_sdk(admin_sdk),
            workspace=workspace,
            name=config_name,
            engine="nim",
            model_spec={"model_name": "test-model"},
            executor_config={"gpu": 1},
        )
        create_deployment(
            models_client_from_sdk(admin_sdk),
            workspace=workspace,
            name=deploy_name,
            config=config_name,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        updated = update_deployment(
            models_client_from_sdk(editor_sdk),
            name=deploy_name,
            workspace=workspace,
            config=config_name,
        )
        assert updated.config == config_name

    def test_custom_role_denied_create_deployment_without_read(self, sdk: NeMoPlatform):
        """A role with deployment write but no inference.deployment-configs.read should be denied on create."""
        with patched_authz_data(_build_authorization_data_without_deployment_config_read):
            workspace = short_unique_name("dp-ncr")
            user_email = unique_email("noread")
            config_name = short_unique_name("cfg")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            create_deployment_config(
                models_client_from_sdk(admin_sdk),
                workspace=workspace,
                name=config_name,
                engine="nim",
                model_spec={"model_name": "test-model"},
                executor_config={"gpu": 1},
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoDeploymentConfigRead"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                create_deployment(
                    models_client_from_sdk(user_sdk),
                    workspace=workspace,
                    name=short_unique_name("dep"),
                    config=config_name,
                )

    def test_custom_role_denied_update_deployment_without_read(self, sdk: NeMoPlatform):
        """A role with deployment write but no inference.deployment-configs.read should be denied on update."""
        with patched_authz_data(_build_authorization_data_without_deployment_config_read):
            workspace = short_unique_name("dp-nup")
            user_email = unique_email("noread")
            config_name = short_unique_name("cfg")
            deploy_name = short_unique_name("dep")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            create_deployment_config(
                models_client_from_sdk(admin_sdk),
                workspace=workspace,
                name=config_name,
                engine="nim",
                model_spec={"model_name": "test-model"},
                executor_config={"gpu": 1},
            )
            create_deployment(
                models_client_from_sdk(admin_sdk),
                workspace=workspace,
                name=deploy_name,
                config=config_name,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoDeploymentConfigRead"],
            )

            user_sdk = as_user(sdk, user_email)

            with pytest.raises(PermissionDeniedError):
                update_deployment(
                    models_client_from_sdk(user_sdk),
                    name=deploy_name,
                    workspace=workspace,
                    config=config_name,
                )


@pytest.mark.integration
class TestFilesetPermissions:
    """Test fileset permission checks (filesets.read) on model/adapter create/update."""

    # -- Write: editor allowed (same workspace fileset) --

    def test_editor_can_create_model_with_fileset(self, sdk: NeMoPlatform):
        workspace = short_unique_name("fs-crt")
        editor_email = unique_email("editor")
        fileset_name = short_unique_name("fs")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, FilesClient).create_fileset(
            workspace=workspace, body=CreateFilesetRequest(name=fileset_name)
        )
        admin_sdk.files.upload_content(
            content=b"x", remote_path="placeholder.txt", fileset=fileset_name, workspace=workspace
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        model = (
            client_from_platform(editor_sdk, ModelsClient)
            .create_model(
                workspace=workspace,
                body=CreateModelEntityRequest(name=short_unique_name("mdl"), fileset=f"{workspace}/{fileset_name}"),
            )
            .data()
        )
        assert model.fileset == f"{workspace}/{fileset_name}"

    def test_editor_can_update_model_with_fileset(self, sdk: NeMoPlatform):
        workspace = short_unique_name("fs-upd")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")
        fileset_name = short_unique_name("fs")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, ModelsClient).create_model(
            workspace=workspace, body=CreateModelEntityRequest(name=model_name)
        ).data()
        client_from_platform(admin_sdk, FilesClient).create_fileset(
            workspace=workspace, body=CreateFilesetRequest(name=fileset_name)
        )
        admin_sdk.files.upload_content(
            content=b"x", remote_path="placeholder.txt", fileset=fileset_name, workspace=workspace
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        updated = (
            client_from_platform(editor_sdk, ModelsClient)
            .update_model(
                name=model_name,
                workspace=workspace,
                body=UpdateModelEntityRequest(fileset=f"{workspace}/{fileset_name}"),
            )
            .data()
        )
        assert updated.fileset == f"{workspace}/{fileset_name}"

    def test_editor_can_create_adapter_with_fileset(self, sdk: NeMoPlatform):
        workspace = short_unique_name("fs-adp")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")
        fileset_name = short_unique_name("adpfs")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, ModelsClient).create_model(
            workspace=workspace, body=CreateModelEntityRequest(name=model_name)
        ).data()
        client_from_platform(admin_sdk, FilesClient).create_fileset(
            workspace=workspace, body=CreateFilesetRequest(name=fileset_name)
        )
        admin_sdk.files.upload_content(
            content=b"x", remote_path="placeholder.txt", fileset=fileset_name, workspace=workspace
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        adapter = (
            client_from_platform(editor_sdk, ModelsClient)
            .create_model_adapter(
                workspace=workspace,
                model_name=model_name,
                body=CreateModelAdapterRequest(
                    name=short_unique_name("adp"),
                    fileset=f"{workspace}/{fileset_name}",
                    finetuning_type=FinetuningType.LORA,
                ),
            )
            .data()
        )
        assert adapter.fileset == f"{workspace}/{fileset_name}"

    # -- Write: denied (cross-workspace fileset) --

    def test_editor_denied_create_model_with_inaccessible_fileset(self, sdk: NeMoPlatform):
        workspace = short_unique_name("fs-dnc")
        editor_email = unique_email("editor")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        with pytest.raises(PermissionDeniedError):
            client_from_platform(editor_sdk, ModelsClient).create_model(
                workspace=workspace,
                body=CreateModelEntityRequest(name=short_unique_name("mdl"), fileset="inaccessible-ws/some-fileset"),
            ).data()

    def test_editor_denied_update_model_with_inaccessible_fileset(self, sdk: NeMoPlatform):
        workspace = short_unique_name("fs-dnu")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, ModelsClient).create_model(
            workspace=workspace, body=CreateModelEntityRequest(name=model_name)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        with pytest.raises(PermissionDeniedError):
            client_from_platform(editor_sdk, ModelsClient).update_model(
                name=model_name,
                workspace=workspace,
                body=UpdateModelEntityRequest(fileset="inaccessible-ws/some-fileset"),
            ).data()

    def test_editor_denied_create_adapter_with_inaccessible_fileset(self, sdk: NeMoPlatform):
        workspace = short_unique_name("fs-dna")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, ModelsClient).create_model(
            workspace=workspace, body=CreateModelEntityRequest(name=model_name)
        ).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_sdk = as_user(sdk, editor_email)
        with pytest.raises(PermissionDeniedError):
            client_from_platform(editor_sdk, ModelsClient).create_model_adapter(
                workspace=workspace,
                model_name=model_name,
                body=CreateModelAdapterRequest(
                    name=short_unique_name("adp"),
                    fileset="inaccessible-ws/adapter-fileset",
                    finetuning_type=FinetuningType.LORA,
                ),
            ).data()

    def test_custom_role_denied_create_model_with_fileset_without_fileset_read(self, sdk: NeMoPlatform):
        """A role without filesets.read should be denied when creating a model with a fileset."""
        with patched_authz_data(_build_authorization_data_without_fileset_read):
            workspace = short_unique_name("fs-nfr")
            user_email = unique_email("nofileset")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoFilesetRead"],
            )

            user_sdk = as_user(sdk, user_email)
            with pytest.raises(PermissionDeniedError):
                client_from_platform(user_sdk, ModelsClient).create_model(
                    workspace=workspace,
                    body=CreateModelEntityRequest(name=short_unique_name("mdl"), fileset=f"{workspace}/some-fileset"),
                ).data()


@pytest.mark.integration
@pytest.mark.usefixtures("no_hf_network")
class TestTrustRemoteCodePermission:
    """Test trust_remote_code permission (models.trust-remote-code.set) at the API layer.

    When a model fileset resolves to a repo not on the allow list, setting
    trust_remote_code=True requires models.trust-remote-code.set.
    """

    @pytest.fixture(autouse=True)
    def _no_hf(self, no_hf_network):
        """These tests verify authorization logic, not HF connectivity."""

    def test_create_model_trust_remote_code_true_has_permission_succeeds(self, sdk: NeMoPlatform):
        """Create with trust_remote_code=True succeeds when principal has models.trust-remote-code.set (repo not on allow list)."""
        workspace = short_unique_name("trc-has")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")
        fileset_name = short_unique_name("fs")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, FilesClient).create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(name=fileset_name, storage=HuggingfaceStorageConfig(repo_id="Qwen/Qwen3-0.6B")),
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Admin"],
        )

        with patch.object(models_config.trust_remote_code, "hf_allow_list", ["nvidia/*"]):
            editor_sdk = as_user(sdk, editor_email)
            created = (
                client_from_platform(editor_sdk, ModelsClient)
                .create_model(
                    workspace=workspace,
                    body=CreateModelEntityRequest(
                        name=model_name, fileset=f"{workspace}/{fileset_name}", trust_remote_code=True
                    ),
                )
                .data()
            )
        assert created.trust_remote_code is True

    def test_create_model_trust_remote_code_true_without_permission_raises(self, sdk: NeMoPlatform):
        """Create with trust_remote_code=True returns 403 when repo not on allow list and principal lacks models.trust-remote-code.set."""
        with patched_authz_data(_real_build_authorization_data):
            workspace = short_unique_name("trc-no")
            user_email = unique_email("editor")
            fileset_name = short_unique_name("fs")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            client_from_platform(admin_sdk, FilesClient).create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=fileset_name, storage=HuggingfaceStorageConfig(repo_id="Qwen/Qwen3-0.6B")
                ),
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["Editor"],
            )

            with patch.object(models_config.trust_remote_code, "hf_allow_list", ["nvidia/*"]):
                user_sdk = as_user(sdk, user_email)
                with pytest.raises(PermissionDeniedError) as exc_info:
                    client_from_platform(user_sdk, ModelsClient).create_model(
                        workspace=workspace,
                        body=CreateModelEntityRequest(
                            name=short_unique_name("mdl"), fileset=f"{workspace}/{fileset_name}", trust_remote_code=True
                        ),
                    ).data()
                assert "Insufficient permissions to set the trust_remote_code" in str(exc_info.value)

    def test_update_model_trust_remote_code_true_has_permission_succeeds(self, sdk: NeMoPlatform):
        """Update with trust_remote_code=True succeeds when principal has models.trust-remote-code.set (repo not on allow list)."""
        workspace = short_unique_name("trc-upd-has")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")
        fileset_name = short_unique_name("fs")

        admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
        client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        ).data()
        client_from_platform(admin_sdk, ModelsClient).create_model(
            workspace=workspace, body=CreateModelEntityRequest(name=model_name)
        ).data()
        client_from_platform(admin_sdk, FilesClient).create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(name=fileset_name, storage=HuggingfaceStorageConfig(repo_id="Qwen/Qwen3-0.6B")),
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Admin"],
        )

        with patch.object(models_config.trust_remote_code, "hf_allow_list", ["nvidia/*"]):
            editor_sdk = as_user(sdk, editor_email)
            updated = (
                client_from_platform(editor_sdk, ModelsClient)
                .update_model(
                    name=model_name,
                    workspace=workspace,
                    body=UpdateModelEntityRequest(fileset=f"{workspace}/{fileset_name}", trust_remote_code=True),
                )
                .data()
            )
        assert updated.trust_remote_code is True

    def test_update_model_trust_remote_code_true_without_permission_raises(self, sdk: NeMoPlatform):
        """Update with trust_remote_code=True returns 403 when repo not on allow list and principal lacks models.trust-remote-code.set."""
        with patched_authz_data(_real_build_authorization_data):
            workspace = short_unique_name("trc-upd-no")
            user_email = unique_email("editor")
            model_name = short_unique_name("mdl")
            fileset_name = short_unique_name("fs")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            client_from_platform(admin_sdk, ModelsClient).create_model(
                workspace=workspace, body=CreateModelEntityRequest(name=model_name)
            ).data()
            client_from_platform(admin_sdk, FilesClient).create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=fileset_name, storage=HuggingfaceStorageConfig(repo_id="Qwen/Qwen3-0.6B")
                ),
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["Editor"],
            )

            with patch.object(
                models_config.trust_remote_code, "hf_allow_list", ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]
            ):
                user_sdk = as_user(sdk, user_email)
                with pytest.raises(PermissionDeniedError) as exc_info:
                    client_from_platform(user_sdk, ModelsClient).update_model(
                        name=model_name,
                        workspace=workspace,
                        body=UpdateModelEntityRequest(fileset=f"{workspace}/{fileset_name}", trust_remote_code=True),
                    ).data()
                assert "Insufficient permissions to set the trust_remote_code" in str(exc_info.value)

    def test_update_model_new_fileset_not_trusted_raises_permission_error(self, sdk: NeMoPlatform):
        """Update model (created with valid trust_remote_code) to a new fileset not on allow list returns 403 when principal lacks models.trust-remote-code.set."""
        with patched_authz_data(_real_build_authorization_data):
            workspace = short_unique_name("trc-newfs")
            user_email = unique_email("editor")
            model_name = short_unique_name("mdl")
            trusted_fs = short_unique_name("fs1")
            new_fs = short_unique_name("fs2")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            files = client_from_platform(admin_sdk, FilesClient)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            # Model created with a trusted fileset (on allow list) so it has trust_remote_code=True.
            files = client_from_platform(admin_sdk, FilesClient)
            files.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=trusted_fs,
                    storage=HuggingfaceStorageConfig(repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
                ),
            )
            client_from_platform(admin_sdk, ModelsClient).create_model(
                workspace=workspace,
                body=CreateModelEntityRequest(
                    name=model_name, fileset=f"{workspace}/{trusted_fs}", trust_remote_code=True
                ),
            ).data()
            # New fileset resolves to a repo not on the allow list.
            files.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(name=new_fs, storage=HuggingfaceStorageConfig(repo_id="Qwen/Qwen3-0.6B")),
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["Editor"],
            )

            with patch.object(models_config.trust_remote_code, "hf_allow_list", ["nvidia/*"]):
                user_sdk = as_user(sdk, user_email)
                with pytest.raises(PermissionDeniedError) as exc_info:
                    client_from_platform(user_sdk, ModelsClient).update_model(
                        name=model_name,
                        workspace=workspace,
                        body=UpdateModelEntityRequest(fileset=f"{workspace}/{new_fs}"),
                    ).data()
                assert "Insufficient permissions to set the trust_remote_code" in str(exc_info.value)

    def test_exact_match_on_allow_list_succeeds(self, sdk: NeMoPlatform):
        """Update with trust_remote_code=True succeeds when repo matches exactly, not via regex."""
        with patched_authz_data(_real_build_authorization_data):
            workspace = short_unique_name("trc-upd-no")
            user_email = unique_email("editor")
            model_name = short_unique_name("mdl")
            fileset_name = short_unique_name("fs")

            admin_sdk = as_user(sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, WorkspacesClient).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            client_from_platform(admin_sdk, FilesClient).create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=fileset_name,
                    storage=HuggingfaceStorageConfig(repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
                ),
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=user_email,
                roles=["Editor"],
            )

            with patch.object(
                models_config.trust_remote_code, "hf_allow_list", ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]
            ):
                user_sdk = as_user(sdk, user_email)
                created = (
                    client_from_platform(user_sdk, ModelsClient)
                    .create_model(
                        workspace=workspace,
                        body=CreateModelEntityRequest(
                            name=model_name, fileset=f"{workspace}/{fileset_name}", trust_remote_code=True
                        ),
                    )
                    .data()
                )
                assert created.trust_remote_code is True
