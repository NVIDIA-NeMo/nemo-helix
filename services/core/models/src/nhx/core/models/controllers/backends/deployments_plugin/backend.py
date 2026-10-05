# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Models ServiceBackend backed by nemo-deployments plugin entities."""

import logging
from typing import Any

from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.auth import AuthContext as DeploymentAuthContext
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.deployments.client import AsyncDeploymentsClient
from nemo_helix_plugin.deployments.types import (
    Container as ContainerDTO,
)
from nemo_helix_plugin.deployments.types import (
    CreateDeploymentRequest,
)
from nemo_helix_plugin.deployments.types import (
    Deployment as DeploymentDTO,
)
from nemo_helix_plugin.deployments.types import (
    DeploymentConfig as DeploymentConfigDTO,
)
from nemo_helix_plugin.deployments.types import (
    Prerequisite as RequestPrerequisite,
)
from nemo_helix_plugin.deployments.types import (
    Volume as VolumeDTO,
)
from nemo_helix_plugin.models.types import ModelDeployment, ModelDeploymentStatus
from nemo_helix_plugin.sdk_provider import get_async_platform_sdk
from nhx.common.config import Runtime
from nhx.core.models.app.constants import MODEL_MANAGED_BY_LABEL, MODEL_MANAGED_BY_MODELS_CONTROLLER
from nhx.core.models.controllers.backends.backends import DeploymentStatusUpdate, ServiceBackend
from nhx.core.models.controllers.backends.common import deployment_elapsed_seconds
from nhx.core.models.controllers.backends.deployments_plugin.compiler import compile_model_deployment
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nhx.core.models.controllers.backends.deployments_plugin.executor import executor_for_runtime
from nhx.core.models.controllers.backends.deployments_plugin.naming import entity_names
from nhx.core.models.controllers.backends.deployments_plugin.request_converters import (
    deployment_config_create_request,
    volume_create_request,
)
from nhx.core.models.controllers.backends.deployments_plugin.resolve import resolve_plugin_deployment
from nhx.core.models.controllers.backends.deployments_plugin.status import (
    aggregate_status,
    apply_pending_timeout,
)
from nhx.core.models.controllers.context import ModelContext

logger = logging.getLogger(__name__)

_DEPLOYMENT_WORKSPACE_LABEL = "nhx.nvidia.com/deployment-workspace"
_DEPLOYMENT_NAME_LABEL = "nhx.nvidia.com/deployment-name"
_MODELS_ROLE_LABEL = "nhx.nvidia.com/models-role"


def _config_references_volume(config: DeploymentConfigDTO, volume_name: str) -> bool:
    """Whether a deployment config mounts the named volume (config- or container-level)."""
    if any(mount.name == volume_name for mount in config.volume_mounts):
        return True
    containers: list[ContainerDTO] = [*config.containers, *config.init_containers]
    return any(mount.name == volume_name for container in containers for mount in container.volume_mounts)


def _deployment_auth_context(source: ModelDeployment) -> DeploymentAuthContext | None:
    auth_context = source.auth_context
    if auth_context is None:
        return None
    if isinstance(auth_context, DeploymentAuthContext):
        return auth_context
    return DeploymentAuthContext.model_validate(auth_context)


def _on_behalf_of_headers(auth_context: DeploymentAuthContext | None) -> dict[str, str]:
    """Headers so the models service principal acts on behalf of the requester.

    The deployment must be owned by the user who requested the model, not the
    models service. The service SDK already sends its own principal id; these add
    the requester as the on-behalf-of principal so the plugin stamps the entity's
    auth context to the requester (collapsing any nested delegation to the acting
    identity).
    """
    if auth_context is None:
        return {}
    principal = auth_context.to_principal().effective_principal
    headers = {"X-NHX-Principal-On-Behalf-Of": principal.id}
    if principal.groups:
        headers["X-NHX-Principal-On-Behalf-Of-Groups"] = ",".join(principal.groups)
    if principal.email:
        headers["X-NHX-Principal-On-Behalf-Of-Email"] = principal.email
    # The subject's stable account id + authz aliases must travel too, or the auth
    # layer re-resolves the forwarded identity as a fresh trusted-header account
    # and account-bound permissions drift. Mirror the platform's delegation headers.
    if principal.account_id:
        headers["X-NHX-Subject-Account-Id"] = principal.account_id
    if principal.authz_aliases:
        headers["X-NHX-Subject-Aliases"] = ",".join(principal.authz_aliases)
    return headers


class DeploymentsPluginServiceBackend(ServiceBackend):
    """Compile model deployments into Volume and Deployment plugin entities."""

    def __init__(self, nhx_sdk: AsyncNeMoHelix, config: dict[str, Any], huggingface_model_puller: str) -> None:
        self._backend_config: DeploymentsPluginConfig | None = None
        self._deployments: AsyncDeploymentsClient | None = None
        self._huggingface_model_puller = huggingface_model_puller
        super().__init__(nhx_sdk, config)

    def init(self) -> None:
        self._backend_config = DeploymentsPluginConfig(**self._config)

    def shutdown(self) -> None:
        self._deployments = None

    def _deployments_client(self) -> AsyncDeploymentsClient:
        # The models controller is a pure HTTP consumer of the deployments-plugin
        # API (as the "models" service principal); it never touches the entity store
        # or backend substrate directly.
        if self._deployments is None:
            sdk = get_async_platform_sdk(as_service="models", internal=True)
            self._deployments = client_from_platform(sdk, AsyncDeploymentsClient)
        return self._deployments

    @property
    def _cfg(self) -> DeploymentsPluginConfig:
        assert self._backend_config is not None
        return self._backend_config

    async def create_model_deployment(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Create plugin substrate entities in volume, puller, server order.

        Tears down any leftover substrate first so drift recovery (LOST → recreate)
        does not collide with orphaned Volume / DeploymentConfig / Deployment entities.
        """
        resolved = resolve_plugin_deployment(ctx, self._huggingface_model_puller)
        if resolved.runtime == Runtime.NONE:
            return DeploymentStatusUpdate(
                status=ModelDeploymentStatus.UNKNOWN,
                status_message="Deployments plugin is unavailable for runtime none.",
            )
        teardown = await self.delete_model_deployment(resolved.deployment.workspace, resolved.deployment.name)
        if teardown.status == ModelDeploymentStatus.DELETING:
            return DeploymentStatusUpdate(
                status=ModelDeploymentStatus.CREATED,
                status_message="Waiting for prior deployments-plugin substrate teardown before recreate.",
            )
        executor = executor_for_runtime(self._cfg, resolved.runtime)
        if executor is None:
            return DeploymentStatusUpdate(
                status=ModelDeploymentStatus.ERROR,
                status_message=(
                    "No deployments-plugin executor configured for the current runtime. "
                    "Set docker_executor, k8s_executor, or default_executor under "
                    "models.controller.backends.deployments_plugin."
                ),
                error_details={
                    "reason": "executor_not_configured",
                    "runtime": resolved.runtime.value,
                },
            )
        try:
            compiled = compile_model_deployment(resolved, self._cfg)
            auth_context = _deployment_auth_context(resolved.deployment)
            workspace = resolved.deployment.workspace
            # Create every entity on behalf of the requester so the deployment is
            # owned by the user, not the models service principal. When there is no
            # requester context, use the service client as-is.
            obo_headers = _on_behalf_of_headers(auth_context)
            client = self._deployments_client()
            if obo_headers:
                client = client.with_headers(obo_headers)
            if compiled.volume is not None:
                await client.create_volume(workspace=workspace, body=volume_create_request(compiled.volume))
            if compiled.scratch_volume is not None:
                await client.create_volume(workspace=workspace, body=volume_create_request(compiled.scratch_volume))
            if compiled.puller_config is not None:
                await client.create_deployment_config(
                    workspace=workspace, body=deployment_config_create_request(compiled.puller_config)
                )
                await client.create_deployment(
                    workspace=workspace,
                    body=CreateDeploymentRequest(
                        name=compiled.names.puller,
                        deployment_config=compiled.names.puller,
                        executor=executor,
                        desired_state="READY",
                    ),
                )
            await client.create_deployment_config(
                workspace=workspace, body=deployment_config_create_request(compiled.server_config)
            )
            await client.create_deployment(
                workspace=workspace,
                body=CreateDeploymentRequest(
                    name=compiled.names.server,
                    deployment_config=compiled.names.server,
                    executor=executor,
                    desired_state="READY",
                    prerequisites=(
                        [RequestPrerequisite(deployment_name=compiled.names.puller, condition="succeeded")]
                        if compiled.puller_prerequisite
                        else []
                    ),
                ),
            )
        except Exception as exc:
            await self._rollback_create(ctx)
            return DeploymentStatusUpdate(
                status=ModelDeploymentStatus.ERROR,
                status_message=f"Unable to create deployments-plugin entities: {exc}",
                error_details={"error": str(exc)},
            )
        return DeploymentStatusUpdate(
            status=ModelDeploymentStatus.PENDING,
            status_message="Created deployments-plugin entities.",
        )

    async def _rollback_create(self, ctx: ModelContext) -> None:
        """Best-effort controlled teardown after a partial create failure."""
        if ctx.model_deployment is None:
            return
        try:
            await self.delete_model_deployment(ctx.model_deployment.workspace, ctx.model_deployment.name)
        except Exception:
            logger.warning(
                "Failed to roll back deployments-plugin substrate after a create failure; orphaned entities may remain",
                extra={
                    "workspace": ctx.model_deployment.workspace,
                    "deployment_name": ctx.model_deployment.name,
                },
                exc_info=True,
            )

    async def get_model_deployment_status(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Project plugin entity health into models deployment status.

        Aggregates Volume, puller, and server Deployment entities, then applies
        ``pending_timeout_seconds`` when the deployment remains PENDING too long.
        """
        if ctx.model_deployment is None:
            return DeploymentStatusUpdate(
                status=ModelDeploymentStatus.UNKNOWN,
                status_message="Model deployment unavailable.",
            )
        names = entity_names(ctx.model_deployment.name)
        server = await self._get_optional_deployment(ctx.model_deployment.workspace, names.server)
        puller = await self._get_optional_deployment(ctx.model_deployment.workspace, names.puller)
        volume = await self._get_optional_volume(ctx.model_deployment.workspace, names.volume)
        result = aggregate_status(
            volume,
            puller,
            server,
            previously_ready=ctx.model_deployment.status == ModelDeploymentStatus.READY,
        )
        elapsed = deployment_elapsed_seconds(ctx.model_deployment)
        return apply_pending_timeout(
            result,
            elapsed_seconds=elapsed,
            timeout_seconds=self._cfg.pending_timeout_seconds,
            deployment_name=ctx.model_deployment.name,
        )

    async def update_model_deployment(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        del ctx
        return DeploymentStatusUpdate(
            status=ModelDeploymentStatus.ERROR,
            status_message="Update via recreate not yet supported.",
        )

    async def delete_model_deployment(
        self,
        workspace: str,
        name: str,
        *,
        deleting_elapsed_seconds: float | None = None,
    ) -> DeploymentStatusUpdate:
        """Stop deployments, then remove configs and volumes once substrate is gone.

        Reports the plugin's teardown progress upward rather than inventing a
        models-side terminal state: while any substrate is still present the model
        deployment stays DELETING. The deployments plugin owns the terminal outcome
        (its reconciler retries substrate teardown and, if it gives up, leaves the
        deployment FAILED); the models controller never force-removes an entity, so
        it cannot orphan backend resources. A stuck teardown therefore surfaces as a
        model deployment that stays DELETING until the plugin-side issue is resolved.
        """
        names = entity_names(name)
        for deployment_name, config_name in ((names.server, names.server), (names.puller, names.puller)):
            if not await self._complete_deployment_delete(workspace, deployment_name, config_name):
                return DeploymentStatusUpdate(
                    status=ModelDeploymentStatus.DELETING,
                    status_message="Waiting for plugin deployment teardown.",
                )
        volumes_removed = True
        for volume_name in (names.scratch, names.volume):
            try:
                volume_removed = await self._complete_volume_delete(workspace, volume_name)
            except Exception:
                logger.exception("Failed to complete volume teardown for %s/%s", workspace, volume_name)
                volume_removed = False
            if not volume_removed:
                volumes_removed = False
        if not volumes_removed:
            return DeploymentStatusUpdate(
                status=ModelDeploymentStatus.DELETING, status_message="Waiting for plugin volume teardown."
            )
        return DeploymentStatusUpdate(
            status=ModelDeploymentStatus.DELETED, status_message="Deleted deployments-plugin entities."
        )

    async def _complete_deployment_delete(self, workspace: str, deployment_name: str, config_name: str) -> bool:
        """Initiate plugin deployment stop and return True once config can be removed.

        Issues an idempotent delete against the plugin (a soft-delete: the plugin
        sets the deployment DELETING and its reconciler tears down the substrate).
        A deployment the plugin left stuck (e.g. FAILED after a deleting-timeout)
        stays present, so this keeps returning False and the models-side
        deleting-timeout escalates — the plugin owns that state and a human re-issues
        delete once resolved, rather than the models controller force-removing the
        entity and orphaning backend substrate.
        """
        deployment = await self._get_optional_deployment(workspace, deployment_name)
        if deployment is not None:
            # Idempotent soft-delete: a 404 means it's already gone, which is fine.
            try:
                await self._deployments_client().delete_deployment(name=deployment_name, workspace=workspace)
            except NotFoundError:
                pass
            return False
        # The deployment is gone; remove its config. An already-absent config (404)
        # is success — this delete runs on a puller-less deploy, on repeat reconcile
        # passes, and on the teardown-before-create of a fresh deployment.
        try:
            await self._deployments_client().delete_deployment_config(name=config_name, workspace=workspace)
        except NotFoundError:
            pass
        return True

    async def _complete_volume_delete(self, workspace: str, volume_name: str) -> bool:
        """Request plugin volume teardown and return True once its entity is gone.

        A volume still referenced by another deployment config is not owned
        exclusively by this model deployment and must be preserved.
        """
        volume = await self._get_optional_volume(workspace, volume_name)
        if volume is None:
            return True
        if volume.status == "DELETING":
            return False

        referencing = await self._deployment_config_names_referencing_volume(workspace, volume_name)
        if referencing:
            logger.info(
                "Preserving deployments-plugin volume %s/%s referenced by deployment configs: %s",
                workspace,
                volume_name,
                ", ".join(referencing),
            )
            return True

        await self._deployments_client().delete_volume(name=volume_name, workspace=workspace)
        return False

    async def _deployment_config_names_referencing_volume(self, workspace: str, volume_name: str) -> list[str]:
        """Names of deployment configs in the workspace whose mounts reference the volume."""
        response = await self._deployments_client().list_deployment_configs(workspace=workspace)
        return [config.name async for config in response.items() if _config_references_volume(config, volume_name)]

    async def _get_optional_deployment(self, workspace: str, name: str) -> DeploymentDTO | None:
        try:
            return (await self._deployments_client().get_deployment(name=name, workspace=workspace)).data()
        except NotFoundError:
            return None

    async def _get_optional_volume(self, workspace: str, name: str) -> VolumeDTO | None:
        try:
            return (await self._deployments_client().get_volume(name=name, workspace=workspace)).data()
        except NotFoundError:
            return None

    async def list_managed_deployment_names(self) -> list[str]:
        """List workspace/name IDs for model deployments managed by this backend.

        Discovers server-role DeploymentConfig entities stamped with models-controller
        ownership labels (deployments-plugin does not mirror labels onto Deployment).
        The cross-workspace query (``workspace="-"``) relies on the models service
        principal's cross-workspace list permission.
        """
        response = await self._deployments_client().list_deployment_configs(workspace="-")
        names = {
            f"{config.labels[_DEPLOYMENT_WORKSPACE_LABEL]}/{config.labels[_DEPLOYMENT_NAME_LABEL]}"
            async for config in response.items()
            if config.labels.get(MODEL_MANAGED_BY_LABEL) == MODEL_MANAGED_BY_MODELS_CONTROLLER
            and config.labels.get(_MODELS_ROLE_LABEL) == "server"
            and _DEPLOYMENT_WORKSPACE_LABEL in config.labels
            and _DEPLOYMENT_NAME_LABEL in config.labels
        }
        return sorted(names)
