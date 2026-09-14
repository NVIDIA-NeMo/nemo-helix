# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model provider reconciliation logic for Models Controller."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger
from typing import Callable, TypedDict

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError, NemoHTTPError, NotFoundError
from nemo_platform_plugin.inference_gateway.client import AsyncInferenceGatewayProviderClient
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.types import (
    ModelDeployment,
    ModelDeploymentConfig,
    ModelEntity,
    ModelProvider,
    ModelProviderStatus,
    ServedModelMapping,
    UpdateModelProviderStatusRequest,
)
from nemo_platform_plugin.virtual_models.client import AsyncVirtualModelsClient
from nemo_platform_plugin.virtual_models.types import CreateVirtualModelRequest, VirtualModel
from nmp.common.datetime_utils import ensure_utc
from nmp.common.entities.constants import NAME_PATTERN
from nmp.common.entities.utils import parse_entity_ref
from nmp.core.models.app import (
    ModelWeightsType,
    get_model_weights_type,
    normalize_model_entity_name,
    parse_model_name_revision,
)
from nmp.core.models.config import ControllerConfig
from nmp.core.models.controllers.context import ModelContext
from nmp.core.models.controllers.entity_cache import ModelEntityCache
from nmp.core.models.schemas import BackendFormat

logger = getLogger(__name__)

# The VirtualModel list endpoint caps page_size at 200.
_VIRTUAL_MODEL_PAGE_SIZE = 200

# Use entity store's NAME_PATTERN so we only create/link names the entity store accepts
_VALID_MODEL_ENTITY_NAME_PATTERN = re.compile(NAME_PATTERN)

# When the backend returns 404, the gateway rewrites it to 502 with this phrase in the detail.
_GATEWAY_BACKEND_404_DETAIL = "Backend returned 404"

# Seconds a provider can stay in CREATED with transient failures before escalating to ERROR (~6 cycles at 5s)
PROVIDER_ERROR_THRESHOLD_SECONDS = 30
# Seconds between discovery retry attempts while a provider is in ERROR
PROVIDER_ERROR_RETRY_INTERVAL_SECONDS = 60
# Seconds from provider creation before a persistently-failing provider is marked LOST
PROVIDER_LOST_THRESHOLD_SECONDS = 900


def _infer_backend_format(model_name: str) -> str:
    """Infer backend_format from a model name. Defaults to OPENAI_CHAT."""
    lower = model_name.lower()
    if lower.startswith("claude-") or lower.startswith(("anthropic.", "anthropic/")):
        return BackendFormat.ANTHROPIC_MESSAGES.value
    return BackendFormat.OPENAI_CHAT.value


def _has_backend_format(model_entity: ModelEntity) -> bool:
    value = model_entity.backend_format
    return isinstance(value, str) and bool(value)


def _get_virtual_model_db_version(virtual_model: VirtualModel) -> int | None:
    db_version = virtual_model.db_version
    if isinstance(db_version, bool):
        return None
    if isinstance(db_version, int):
        return db_version
    return None


def _get_virtual_model_observed_at(virtual_model: VirtualModel) -> datetime | None:
    return ensure_utc(virtual_model.updated_at) or ensure_utc(virtual_model.created_at)


# ---------------------------------------------------------------------------
# Discovery result types
# ---------------------------------------------------------------------------


class DiscoveryResult:
    """Result of querying a provider for available models."""


class DiscoveredModel(TypedDict, total=False):
    """Typed shape of an entry in GET /v1/models ``data[]`` we care about.

    ``id`` is always present (we filter out entries missing it in :meth:`_discover_models`).
    ``root`` and ``parent`` come from OpenAI-compatible LoRA/prompt-tuned metadata and may
    be missing or None.
    """

    id: str
    root: str | None
    parent: str | None


class ApiEndpointDict(TypedDict):
    """Shape of :attr:`ArtifactDetails.api_endpoint`, stored on Model Entities."""

    url: str
    model_id: str
    format: str


class DiscoverySuccess(DiscoveryResult):
    """Provider responded with valid OpenAI model list.

    Carries full model objects (id, root, parent) for deployment-backed classification.
    model_ids is derived from models for external provider path.
    """

    def __init__(self, models: list[DiscoveredModel]) -> None:
        """Initialize with list of model objects from GET /v1/models data[].

        Each dict has an ``id`` (str); ``root`` and ``parent`` may be str or None.
        """
        self.models = models

    @property
    def model_ids(self) -> list[str]:
        """List of model id strings for external provider path."""
        return [m["id"] for m in self.models if isinstance(m, dict) and "id" in m]


class DiscoveryNonCompliant(DiscoveryResult):
    """Provider responded (or has no URL) but is not OpenAI-compliant."""


class DiscoveryTransientError(DiscoveryResult):
    """Query failed due to a transient error (network, 404, timeout)."""

    def __init__(self, message: str = "") -> None:
        self.message = message


# ---------------------------------------------------------------------------
# Value type for artifact details
# ---------------------------------------------------------------------------


@dataclass
class ArtifactDetails:
    """Artifact and API endpoint details resolved for a model entity."""

    fileset_url: str | None = None
    api_endpoint: ApiEndpointDict | None = field(default=None)


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _is_valid_model_entity_name(name: str) -> bool:
    """Return True if name matches the entity store's NAME_PATTERN.

    Uses the same pattern as the entity store so names we accept here will be
    accepted by the entity store on create.
    """
    return bool(_VALID_MODEL_ENTITY_NAME_PATTERN.match(name))


def _is_valid_served_model_entity_id(model_entity_id: str) -> bool:
    """Return True iff ``model_entity_id`` is a routable IGW served-model id.

    Accepts two shapes — matching the IGW-side ``validate_model_entity_name`` contract:

    - ``{workspace}/{name}`` (plain ModelEntity routing).
    - ``{workspace}/{base}&adapters/{adapter_workspace}/{adapter_name}`` (LoRA composite).

    Each NAME_PATTERN-bearing segment (workspace, plain name, LoRA base, adapter workspace,
    adapter name) is checked against :func:`_is_valid_model_entity_name` so mappings accepted
    here are guaranteed to be accepted by IGW's ``validate_model_entity_name`` at proxy time
    and by the entity store for any plain-name routes.
    """
    if "/" not in model_entity_id:
        return False
    workspace, remainder = model_entity_id.split("/", 1)
    if not _is_valid_model_entity_name(workspace):
        return False
    if "&adapters/" in remainder:
        base, _, adapter_part = remainder.partition("&adapters/")
        if not base or not adapter_part or "/" not in adapter_part:
            return False
        adapter_ws, _, adapter_name = adapter_part.partition("/")
        if not adapter_ws or not adapter_name:
            return False
        return all(_is_valid_model_entity_name(s) for s in (base, adapter_ws, adapter_name))
    return _is_valid_model_entity_name(remainder)


def _partition_valid_external_models(
    provider: ModelProvider,
    result: DiscoverySuccess,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split discovered models into ``(valid_pairs, skipped_ids)`` for external providers.

    ``valid_pairs`` are ``(model_id, normalized_entity_name)`` tuples. ``skipped_ids`` are
    original ids that failed name normalization or NAME_PATTERN validation.

    Filtering stages, applied in order:

    1. ``provider.enabled_models`` allowlist (no-op when ``None``).
    2. ``_entity_name_from_discovered_model`` normalization — ``ValueError`` → skipped.
    3. ``_is_valid_model_entity_name`` check against entity-store ``NAME_PATTERN`` — fail → skipped.

    Keeping both the "create Model Entity" and "emit ServedModelMapping" paths wired through
    this single helper guarantees they always see the same set of (id, name) pairs, preventing
    drift where one path registers entities the other never exposes (or vice versa).
    """
    model_ids = result.model_ids
    if provider.enabled_models is not None:
        enabled = set(provider.enabled_models)
        model_ids = [m for m in model_ids if m in enabled]

    valid: list[tuple[str, str]] = []
    skipped: list[str] = []
    for model_id in model_ids:
        try:
            normalized = _entity_name_from_discovered_model(model_id, provider.workspace)
        except ValueError:
            skipped.append(model_id)
            continue
        if _is_valid_model_entity_name(normalized):
            valid.append((model_id, normalized))
        else:
            skipped.append(model_id)
    return valid, skipped


def _entity_name_from_discovered_model(model_id: str, provider_workspace: str) -> str:
    """Derive the model entity name from a discovered model ID.

    If the model ID is prefixed with the provider's workspace (e.g. 'my-ws/my-model'),
    strip that prefix and normalize only the remainder so we align with existing
    entities in that workspace. Otherwise normalize the whole ID (e.g. 'meta/llama-3.2-1b').

    Args:
        model_id: Raw model ID from the backend's /v1/models (e.g. 'workspace/qwen-2-5-1-5b').
        provider_workspace: The provider's workspace (same as deployment workspace).

    Returns:
        Normalized name to use as the model entity name and in model_entity_id.
    """
    return normalize_model_entity_name(model_id.removeprefix(f"{provider_workspace}/"))


def _resolve_base_backend_model_id(
    config: ModelDeploymentConfig | None,
    base_model_entity: ModelEntity | None,
) -> str | None:
    """Resolve the NIM base model id ``workspace/base_name`` used in GET /v1/models parent/root/id.

    Matches ModelsController._retrieve_model_entity_for_config precedence:
    ``model_entity_id`` if set; otherwise the model entity from ``model_spec`` (exposed here
    as ``base_model_entity`` when the controller prefetched it); otherwise parse
    ``model_spec.model_namespace`` / ``model_name`` / ``model_revision`` (same as config
    create when ``model_entity_id`` is inferred from the entity store).
    """
    if config and config.model_entity_id:
        model_workspace, model_name, _revision = parse_model_name_revision(model_name=config.model_entity_id)
        if model_workspace and model_name:
            return f"{model_workspace}/{model_name}"
        return config.model_entity_id

    if base_model_entity is not None:
        return f"{base_model_entity.workspace}/{base_model_entity.name}"

    model_spec = config.model_spec if config else None
    if model_spec is not None:
        model_workspace, model_name, _revision = parse_model_name_revision(
            model_namespace=model_spec.model_namespace,
            model_name=model_spec.model_name,
            model_revision=model_spec.model_revision,
        )
        if model_workspace and model_name:
            return f"{model_workspace}/{model_name}"

    return None


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class ModelProviderReconciler:
    """
    Handles reconciliation of model providers and their served models.

    This class manages the autodiscovery of models from providers and ensures
    Model Entities are created and linked appropriately.
    """

    def __init__(
        self,
        models_sdk: AsyncNeMoPlatform,
        controller_config: ControllerConfig,
        entity_cache: ModelEntityCache,
        emit_heartbeat: Callable[[], None],
    ) -> None:
        """Initialize the provider reconciler.

        Args:
            models_sdk: SDK client for Models API interactions
            controller_config: Models controller configuration (discovery timeout/retry policy)
            entity_cache: Model Entity reads and staged writes for the current phase
            emit_heartbeat: Called as each unit of work finishes so a long pass is
                distinguishable from a stalled one
        """
        self._models_sdk = models_sdk
        self._controller_config = controller_config
        self._entity_cache = entity_cache
        self._emit_heartbeat = emit_heartbeat
        self._models_client = client_from_platform(models_sdk, AsyncModelsClient)
        self._virtual_models_client = client_from_platform(models_sdk, AsyncVirtualModelsClient)
        discovery_sdk = models_sdk.with_options(
            max_retries=controller_config.provider_discovery_max_retries,
        )
        self._gateway_provider_client = client_from_platform(discovery_sdk, AsyncInferenceGatewayProviderClient)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    async def reconcile_model_providers(self, provider_contexts: list[ModelContext]) -> None:
        """Reconcile the model providers by autodiscovering their served models.

        For each provider, calls GET /v1/models to discover available models,
        then updates the provider's served_models field with the discovered mappings.
        Uses pre-fetched data from ModelContext to avoid repeated retrieval calls.

        Providers in CREATED that fail discovery persistently are escalated to ERROR,
        then to LOST after PROVIDER_LOST_THRESHOLD_SECONDS from creation. ERROR
        providers use a slower retry cadence (PROVIDER_ERROR_RETRY_INTERVAL_SECONDS)
        to avoid hammering unreachable endpoints. All state is DB-backed via the
        provider entity's status, status_message, created_at, and updated_at fields.

        Args:
            provider_contexts: List of provider contexts with pre-fetched related data
        """
        now = datetime.now(timezone.utc)

        # Read the VirtualModels once for the whole pass. ``existing_vm_names``
        # answers "does this one already exist" without a create-and-catch-conflict
        # per served model, and grows as VirtualModels are created so providers
        # sharing a workspace do not each try the same name. ``vm_snapshot`` stays
        # as read: orphan cleanup only ever considers VirtualModels that predate
        # the pass, and keeping it separate means a VirtualModel created here can
        # never be mistaken for a deletion candidate.
        # A listing failure must not cost us discovery, provider status, or entity
        # linking; only the VirtualModel work depends on it.
        virtual_models = await self._load_virtual_models()
        existing_vm_names = virtual_models[1] if virtual_models is not None else None

        for ctx in provider_contexts:
            provider = ctx.model_provider
            if provider is None:
                logger.warning("Skipping provider reconciliation for context with no model_provider")
                continue
            provider_id = f"{provider.workspace}/{provider.name}"
            try:
                await self._reconcile_single_provider(ctx, provider, provider_id, now, existing_vm_names)
            except Exception:
                logger.exception(
                    "Unexpected error reconciling provider",
                    extra={"provider": provider_id},
                )
            finally:
                self._emit_heartbeat()

        # Runs after every provider so the active set reflects the served_models
        # resolved during this pass rather than the previously persisted values.
        if virtual_models is None:
            logger.warning("Skipping orphaned VirtualModel cleanup because the VirtualModel listing failed")
            return
        try:
            await self._cleanup_orphaned_virtual_models(provider_contexts, virtual_models[0], snapshot_taken_at=now)
        except Exception:
            logger.exception("Unexpected error cleaning up orphaned autoprovisioned VirtualModels")

    async def _load_virtual_models(self) -> tuple[list[VirtualModel], set[tuple[str, str]]] | None:
        """Read every VirtualModel once, returning the list and an existence set.

        The existence set covers all VirtualModels, not just autoprovisioned ones,
        so a name already taken by a user-managed VirtualModel is left untouched.

        Returns ``None`` if the listing failed, which callers read as "the
        VirtualModel state is unknown this pass" and skip VirtualModel work rather
        than acting on a partial view.
        """
        vm_snapshot: list[VirtualModel] = []
        try:
            response = await self._virtual_models_client.list_virtual_models(
                workspace="-",
                query_params={"page_size": _VIRTUAL_MODEL_PAGE_SIZE},
            )
            async for virtual_model in response.items():
                vm_snapshot.append(virtual_model)
                self._emit_heartbeat()
        except Exception:
            logger.warning("Failed to list VirtualModels for provider reconciliation", exc_info=True)
            return None
        existing_vm_names = {(vm.workspace, vm.name) for vm in vm_snapshot}
        logger.debug("Loaded %d VirtualModel(s) for provider reconciliation", len(vm_snapshot))
        return vm_snapshot, existing_vm_names

    async def _reconcile_single_provider(
        self,
        ctx: ModelContext,
        provider: ModelProvider,
        provider_id: str,
        now: datetime,
        existing_vm_names: set[tuple[str, str]] | None,
    ) -> None:
        """Reconcile a single provider. Extracted for per-provider exception isolation.

        Leaving ``ctx.served_models`` as ``None`` on any early return is deliberate:
        orphan cleanup reads it as "this provider was not resolved this pass" and
        falls back to the persisted served_models. Setting it to an empty list
        instead would present every one of the provider's models as inactive.
        """
        ctx.served_models = None

        # LOST providers are permanently failed; skip until user deletes and recreates.
        if provider.status == ModelProviderStatus.LOST:
            return

        # ERROR providers: check LOST escalation, then enforce slow retry cadence.
        if provider.status == ModelProviderStatus.ERROR:
            if self._is_past_lost_threshold(provider, provider_id, now):
                await self._mark_lost(ctx, provider, provider_id)
                return
            if self._is_within_retry_cooldown(provider, now):
                return

        result = await self._discover_models(provider)

        match result:
            case DiscoveryTransientError() as err:
                await self._on_transient_failure(provider, provider_id, err, now)
                return
            case DiscoveryNonCompliant():
                logger.info(
                    "Non-OpenAI compliant response, disabling model entity routing",
                    extra={"provider": provider_id},
                )
                ctx.served_models = []
                try:
                    ctx.model_provider = (
                        await self._models_client.update_provider_status(
                            name=provider.name,
                            workspace=provider.workspace,
                            body=UpdateModelProviderStatusRequest(
                                served_models=[],
                                status=ModelProviderStatus.READY,
                                status_message="Non-OpenAI compliant endpoint, model entity routing disabled",
                            ),
                        )
                    ).data()
                except Exception as e:
                    logger.error(f"Failed to update provider {provider_id} status: {e}")
                return
            case DiscoverySuccess() as success:
                pass
            case _:  # pragma: no cover - exhaustive over DiscoveryResult subclasses
                logger.error("Unexpected DiscoveryResult subclass %s", type(result).__name__)
                return

        if provider.model_deployment_id is None:
            await self._ensure_external_entities(provider, provider_id, success, ctx)
            served_models = self._generate_external_served_model_mappings(provider, success)
        else:
            served_models = self._generate_deployment_served_model_mappings(provider, provider_id, success, ctx)
            # `None` signals "transient: base model's backend id couldn't be resolved this tick"
            # Bail without touching served_models so we preserve any previously-published routes.
            if served_models is None:
                logger.warning(
                    "Skipping update for deployment-backed provider %s: base model id "
                    "unresolvable this tick (likely transient prefetch miss); preserving existing served_models",
                    provider_id,
                )
                return

        # Drop any mapping whose model_entity_id wouldn't survive IGW's
        # validate_model_entity_name at proxy time. Both generators should already emit valid
        # shapes, but a bad backend response (e.g. NIM returning a prompt-tuned id with ":" or
        # an empty adapter name) would otherwise be written verbatim and 422 at every request.
        # The external path is a no-op here by construction; the deployment path benefits most.
        validated: list[ServedModelMapping] = []
        for m in served_models:
            if _is_valid_served_model_entity_id(m.model_entity_id):
                validated.append(m)
            else:
                logger.warning(
                    "Dropping invalid served_model for provider %s: model_entity_id=%r served_model_name=%r",
                    provider_id,
                    m.model_entity_id,
                    m.served_model_name,
                )
        served_models = validated
        ctx.served_models = served_models

        logger.debug(f"Provider {provider_id}: serving {len(served_models)} model(s)")

        try:
            ctx.model_provider = (
                await self._models_client.update_provider_status(
                    name=provider.name,
                    workspace=provider.workspace,
                    body=UpdateModelProviderStatusRequest(
                        served_models=served_models,
                        status=ModelProviderStatus.READY,
                    ),
                )
            ).data()
            if served_models:
                logger.debug(f"Updated provider {provider_id} with {len(served_models)} served model(s)")
            else:
                logger.warning(f"No valid models discovered for provider {provider_id}")
        except Exception as e:
            logger.error(f"Failed to update provider {provider_id} with served models: {e}")

        # Ensure a passthrough VirtualModel exists for every served model entity.
        # Runs regardless of whether update_status succeeded — VirtualModel creation is
        # independent of provider status and back-fills any models already in served_models
        # before this feature was deployed.  Empty served_models (DiscoveryNonCompliant)
        # is naturally a no-op.
        #
        # LoRA composite ids are skipped because their "&" / multi-segment form has no standalone ModelEntity;
        # base entities carry the passthrough VirtualModel that fronts the full LoRA id via IGW's
        # OpenAI route (see ModelCache.rebuild_model_entity_map which splits on the first "/").
        for served_model in served_models:
            if "&adapters/" in served_model.model_entity_id:
                continue
            ref = parse_entity_ref(served_model.model_entity_id)
            # Back-reference this provider on the entity it serves. The external path
            # already does this via _ensure_model_entity_for_provider; deployment-backed
            # providers had no equivalent, so their base entity's model_providers stayed
            # empty and every Studio surface gating on hasModelProvider hid a model that
            # was in fact serving traffic.
            #
            # Safe for both branches: staging a link is idempotent (the external path's
            # earlier stage_provider_link for the same entity merges into one write), and
            # it never creates an entity, so the deployment-backed "no autocreation"
            # contract is preserved. An entity that does not exist yet (a prompt-tuned
            # variant still being registered by its training job) has its link dropped by
            # ModelEntityCache._create and re-staged on the next tick, which is the
            # intended self-healing behaviour.
            self._entity_cache.stage_provider_link(ref.workspace, ref.name, provider_id)
            await self._ensure_passthrough_virtual_model(ref.workspace, ref.name, existing_vm_names)
            self._emit_heartbeat()

    # -------------------------------------------------------------------------
    # Status machine
    # -------------------------------------------------------------------------

    def _is_past_lost_threshold(self, provider: ModelProvider, provider_id: str, now: datetime) -> bool:
        """Return True if this ERROR provider has been alive longer than PROVIDER_LOST_THRESHOLD_SECONDS."""
        created_at = ensure_utc(provider.created_at)
        if created_at is None:
            return False
        elapsed = (now - created_at).total_seconds()
        if elapsed > PROVIDER_LOST_THRESHOLD_SECONDS:
            logger.warning(
                "Provider exceeded LOST threshold",
                extra={
                    "provider": provider_id,
                    "elapsed_s": elapsed,
                    "threshold_s": PROVIDER_LOST_THRESHOLD_SECONDS,
                },
            )
            return True
        return False

    async def _mark_lost(self, ctx: ModelContext, provider: ModelProvider, provider_id: str) -> None:
        """Write LOST status for a provider that has permanently failed discovery."""
        try:
            ctx.model_provider = (
                await self._models_client.update_provider_status(
                    name=provider.name,
                    workspace=provider.workspace,
                    body=UpdateModelProviderStatusRequest(
                        status=ModelProviderStatus.LOST,
                        status_message="Provider discovery permanently failed. Delete and recreate to retry.",
                    ),
                )
            ).data()
        except Exception:
            logger.exception(
                "Failed to transition provider to LOST",
                extra={"provider": provider_id},
            )

    def _is_within_retry_cooldown(self, provider: ModelProvider, now: datetime) -> bool:
        """Return True if this ERROR provider last attempted discovery within the retry interval."""
        updated_at = ensure_utc(provider.updated_at)
        if updated_at is None:
            return False
        return (now - updated_at).total_seconds() < PROVIDER_ERROR_RETRY_INTERVAL_SECONDS

    async def _on_transient_failure(
        self,
        provider: ModelProvider,
        provider_id: str,
        err: DiscoveryTransientError,
        now: datetime,
    ) -> None:
        """React to a transient discovery failure with status-aware escalation.

        - CREATED/UNKNOWN: escalate to ERROR after PROVIDER_ERROR_THRESHOLD_SECONDS.
        - ERROR: bump status_message (resets the slow retry timer via updated_at).
        - READY/other: preserve served_models (existing behavior).
        """
        if provider.status in (ModelProviderStatus.CREATED, ModelProviderStatus.UNKNOWN):
            updated_at = ensure_utc(provider.updated_at)
            if updated_at and (now - updated_at).total_seconds() > PROVIDER_ERROR_THRESHOLD_SECONDS:
                try:
                    await self._models_client.update_provider_status(
                        name=provider.name,
                        workspace=provider.workspace,
                        body=UpdateModelProviderStatusRequest(
                            status=ModelProviderStatus.ERROR,
                            status_message=f"Provider discovery failed: {err.message}"
                            if err.message
                            else "Provider discovery failed: unable to reach GET /v1/models",
                        ),
                    )
                    logger.warning(
                        "Provider escalated to ERROR after persistent discovery failures",
                        extra={"provider": provider_id},
                    )
                except Exception:
                    logger.exception(
                        "Failed to escalate provider to ERROR",
                        extra={"provider": provider_id},
                    )
            else:
                logger.info(
                    "Transient error querying provider, preserving served_models",
                    extra={
                        "provider": provider_id,
                        "served_models_count": len(provider.served_models or ()),
                    },
                )
        elif provider.status == ModelProviderStatus.ERROR:
            # Bump updated_at to pace the next retry
            try:
                await self._models_client.update_provider_status(
                    name=provider.name,
                    workspace=provider.workspace,
                    body=UpdateModelProviderStatusRequest(
                        status=ModelProviderStatus.ERROR,
                        status_message=f"Discovery retry failed: {err.message}"
                        if err.message
                        else "Discovery retry failed: still unable to reach GET /v1/models",
                    ),
                )
            except Exception:
                logger.exception(
                    "Failed to update provider after retry failure",
                    extra={"provider": provider_id},
                )
        else:
            logger.info(
                "Transient error querying provider, preserving served_models",
                extra={
                    "provider": provider_id,
                    "served_models_count": len(provider.served_models or ()),
                },
            )

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    async def _discover_models(self, provider: ModelProvider) -> DiscoveryResult:
        """Call GET /v1/models through the IGW provider route and return a typed result.

        Routes through the Inference Gateway to maintain IGW as the single source of
        truth for provider endpoint access. Provider routing in IGW works without
        served_models being populated (unlike model routing which needs model awareness).

        Returns:
            DiscoverySuccess with model objects on success,
            DiscoveryNonCompliant if the provider gave a valid response but wrong format,
            DiscoveryTransientError if the query failed due to a transient error.
        """
        provider_id = f"{provider.workspace}/{provider.name}"

        if not provider.host_url:
            logger.warning(f"Provider {provider_id} has no host_url, skipping autodiscovery")
            return DiscoveryNonCompliant()

        try:
            # This hits: /v2/workspaces/{workspace}/inference/gateway/provider/{name}/-/v1/models
            # This intentionally uses the controller's service principal to perform
            # infrastructure reconciliation. User-level secret access remains guarded
            # at provider create/upsert validation and by IGW when proxying requests.
            models_response = await self._gateway_provider_client.get_provider_models(
                workspace=provider.workspace,
                name=provider.name,
                timeout=self._controller_config.provider_discovery_timeout_seconds,
            )

            if isinstance(models_response, str):
                # Attempt to parse JSON, in case the value is valid JSON but the
                # content-type header was not sent.
                try:
                    models_response = json.loads(models_response)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Non-OpenAI compliant response format from {provider_id}")
                    return DiscoveryNonCompliant()

            if not isinstance(models_response, Mapping) or "data" not in models_response:
                logger.warning(f"Non-OpenAI compliant response format from {provider_id}")
                return DiscoveryNonCompliant()

            discovered_models = models_response.get("data")
            if not isinstance(discovered_models, list):
                logger.warning(f"Non-OpenAI compliant data field from {provider_id}")
                return DiscoveryNonCompliant()

            models: list[DiscoveredModel] = []
            for model in discovered_models:
                if not isinstance(model, Mapping):
                    logger.warning(f"Skipping invalid model entry in {provider_id}: {model}")
                    continue
                model_id = model.get("id")
                if not isinstance(model_id, str):
                    logger.warning(f"Skipping invalid model entry in {provider_id}: {model}")
                    continue
                root = model.get("root")
                parent = model.get("parent")
                discovered_model: DiscoveredModel = {
                    "id": model_id,
                    "root": root if isinstance(root, str) else None,
                    "parent": parent if isinstance(parent, str) else None,
                }
                models.append(discovered_model)

            return DiscoverySuccess(models)

        except NemoHTTPError as e:
            # 404 from the provider proxy is only returned when the provider is not in the gateway
            # cache yet (single code path in IGW). Preserve served_models.
            if e.status_code == 404:
                logger.warning(f"Provider {provider_id} not yet in gateway cache (404), preserving served_models")
                return DiscoveryTransientError("Provider not yet in gateway cache (404)")
            # IGW (FastAPI) returns 502 with body {"detail": "Backend returned 404: ..."} when backend has no /v1/models.
            if e.status_code == 502 and _GATEWAY_BACKEND_404_DETAIL in e.detail:
                # Backend (NIM) returned 404 — no GET /v1/models or similar. Mark non-compliant.
                logger.info(
                    f"Backend for {provider_id} returned 404 for GET /v1/models, disabling model entity routing"
                )
                return DiscoveryNonCompliant()
            # Other 4xx/5xx (e.g. 502 with other detail, 429, 5xx) — treat as transient
            logger.debug(
                "Failed to get models from provider via gateway",
                extra={"provider": provider_id, "error": str(e), "status_code": e.status_code},
            )
            return DiscoveryTransientError(f"Gateway error (HTTP {e.status_code})")
        except Exception as e:
            # Network errors, timeouts, etc. — do NOT clear served_models
            logger.debug(
                "Failed to get models from provider via gateway",
                extra={"provider": provider_id, "error": str(e)},
            )
            return DiscoveryTransientError(f"Network error: {e}")

    # -------------------------------------------------------------------------
    # Success path — external providers
    # -------------------------------------------------------------------------

    async def _ensure_external_entities(
        self,
        provider: ModelProvider,
        provider_id: str,
        result: DiscoverySuccess,
        ctx: ModelContext,
    ) -> None:
        """Ensure Model Entities exist and are linked for every valid discovered model.

        Uses :func:`_partition_valid_external_models` for filtering/validation so the set of
        entities initialized here matches the set later returned by
        :meth:`_generate_external_served_model_mappings`.

        Invokes :meth:`_ensure_model_entity_for_provider` for **every** valid pair on
        every reconcile cycle, not just the newly discovered ones. The helper is
        idempotent — it short-circuits cheaply when the entity already exists with
        this provider linked — and re-running it lets transient retrieve/create
        failures recover on a later tick. Gating by the provider's current
        ``served_models`` would make those failures permanent: the mapping is
        published by :meth:`_reconcile_single_provider` regardless of whether entity
        init succeeded, so the served_model_name would be treated as "already
        initialized" on every subsequent cycle and the retry would never happen.
        That left providers routable while their backing Model Entity / passthrough
        VirtualModel were never created.
        """
        valid_pairs, skipped_models = _partition_valid_external_models(provider, result)
        existing = {mapping.served_model_name for mapping in (provider.served_models or ())}
        newly_discovered = [(mid, name) for mid, name in valid_pairs if mid not in existing]

        logger.debug(
            f"Provider {provider_id}: valid={len(valid_pairs)}, existing={len(existing)}, "
            f"newly_discovered={len(newly_discovered)}, skipped={len(skipped_models)}"
        )

        if newly_discovered:
            logger.info(
                f"Autodiscovered {len(newly_discovered)} new model(s) from {provider_id}: "
                f"{[mid for mid, _ in newly_discovered]}"
            )
        if skipped_models:
            logger.info(
                f"Skipped {len(skipped_models)} model(s) with invalid names for provider {provider_id}: "
                f"{skipped_models}"
            )

        for _model_id, normalized in valid_pairs:
            await self._ensure_model_entity_for_provider(
                model_workspace=provider.workspace,
                model_name=normalized,
                provider=provider,
                provider_id=provider_id,
                ctx=ctx,
            )
            self._emit_heartbeat()

    def _generate_external_served_model_mappings(
        self,
        provider: ModelProvider,
        result: DiscoverySuccess,
    ) -> list[ServedModelMapping]:
        """Generate served_model mappings for an external provider from discovered model ids.

        Uses the same :func:`_partition_valid_external_models` helper as
        :meth:`_ensure_external_entities` so the returned mappings always match the set of
        initialized entities.
        """
        valid_pairs, _ = _partition_valid_external_models(provider, result)
        return [
            ServedModelMapping(
                model_entity_id=f"{provider.workspace}/{normalized}",
                served_model_name=model_id,  # Keep original unnormalized name for backend
            )
            for model_id, normalized in valid_pairs
        ]

    # -------------------------------------------------------------------------
    # Success path — deployment-backed providers
    # -------------------------------------------------------------------------

    def _generate_deployment_served_model_mappings(
        self,
        provider: ModelProvider,
        provider_id: str,
        result: DiscoverySuccess,
        ctx: ModelContext,
    ) -> list[ServedModelMapping] | None:
        """Generate served_model mappings for a deployment-backed provider.

        No entity autocreation happens here, by design — each kind of discovered entry has a
        different owner for its Model Entity:

        - **Base**: the Model Entity is the input to the deployment, so it already exists
          before we ever reach this method (the controller prefetches it as ``ctx.model_entity``).
        - **LoRA adapter**: has no standalone Model Entity. Routing is handled via the
          composite ``{base_id}&adapters/{adapter_workspace}/{adapter_name}`` id, which the
          IGW ``ModelCache`` splits on the first ``/`` only. The discovered NIM id is the
          flat ``{adapter_workspace}--{adapter_name}`` directory name written by the
          sidecar; we recover the two segments with ``mid.partition("--")``.
          For backward compatibility, ids without ``"--"`` (older sidecars that emitted
          bare adapter names) are assumed to live in the **base model's** workspace —
          legacy sidecars nested adapters under the model entity, and the new sidecar's
          ``_resolve_adapter_workspace`` fallback collapses onto the same workspace, so
          ``base_ws`` is the correct anchor regardless of whether the provider's own
          workspace matches.
        - **Prompt-tuned**: the Model Entity is expected to be registered out-of-band by the
          prompt-tuning training job that produced the tuned variant (e.g. the customizer's
          post-training step). The ``ServedModelMapping`` we emit here assumes that entity
          exists; if it doesn't, OpenAI-compatible routes still work (IGW builds ``ModelCache``
          from ``served_models`` alone) but the Model Entity route returns 404 until the
          training job finishes registration.

        Filters by ``enabled_models``, then classifies each discovered model (id, root, parent)
        as base, LoRA, or prompt-tuned using the resolved base backend id (see
        :func:`_resolve_base_backend_model_id`).

        Returns ``None`` when the base backend model id cannot be resolved this tick.
        This signals to the caller that there is a transient issue, and should skip ``update_status``
        so existing ``served_models`` are preserved.
        """
        discovered_models = result.models
        if provider.enabled_models is not None:
            enabled = set(provider.enabled_models)
            discovered_models = [m for m in discovered_models if m.get("id") in enabled]
            logger.debug(
                f"Filtered to {len(discovered_models)} model(s) based on enabled_models for provider {provider_id}"
            )

        workspace = provider.workspace
        served: list[ServedModelMapping] = []
        base_id = _resolve_base_backend_model_id(ctx.model_deployment_config, ctx.model_entity)

        if not base_id:
            logger.debug(
                f"Deployment-backed provider {provider_id} could not resolve base backend model id "
                "(no model_entity_id, model_spec model fields, or prefetched model entity); "
                "skipping base/LoRA/prompt-tuned classification"
            )
            return None

        base_ws, _, _ = base_id.partition("/")

        for obj in discovered_models:
            mid = obj.get("id")
            root = obj.get("root")
            parent = obj.get("parent")
            if not mid:
                continue

            # Base: parent is null and id equals our base
            if parent is None and mid == base_id:
                served.append(ServedModelMapping(model_entity_id=base_id, served_model_name=mid))
                continue

            # LoRA: parent equals our base. The sidecar writes adapter directories
            # as ``{adapter_ws}--{adapter_name}``, and NIM's flat scanner echoes that exact
            # string back as the model id. Note that ``--`` is a lossless delimiter because the entity-store
            # NAME_PATTERN forbids consecutive hyphens in any workspace or entity name.
            if parent == base_id:
                # NIM may emit ids qualified with the served-model namespace (the base
                # model's workspace), giving four shapes:
                #   1. ``{adapter_ws}--{name}``
                #   2. ``{base_ws}/{adapter_ws}--{name}``
                #   3. ``{name}`` or ``{base_ws}/{name}`` (legacy)
                # Strategy is to strip the ``{base_ws}/`` qualifier once upfront
                # so ``partition("--")`` sees only the adapter segment in every case.
                adapter_segment = mid.removeprefix(f"{base_ws}/")
                adapter_ws, sep, adapter_name = adapter_segment.partition("--")
                if sep != "--":
                    # Legacy id: no ``--``. Anchor on the base model's workspace, not the
                    # provider's. Pre-AALGO-129 sidecars nested adapters under the model
                    # entity (so they shared the model's workspace), and the new sidecar's
                    # ``_resolve_adapter_workspace`` fallback also collapses onto the
                    # base model's workspace when the SDK doesn't expose
                    # ``Adapter.workspace``. Using ``provider.workspace`` here would
                    # silently mis-route whenever the provider lives in a different
                    # workspace than the base model.
                    logger.warning(
                        "Deployment-backed provider %s: discovered LoRA adapter id %r has no "
                        "'--' delimiter; assuming adapter lives in base model workspace %r "
                        "(backward compatibility — please upgrade the model sidecar)",
                        provider_id,
                        mid,
                        base_ws,
                    )
                    adapter_ws = base_ws
                    adapter_name = adapter_segment

                if not _is_valid_model_entity_name(adapter_ws) or not _is_valid_model_entity_name(adapter_name):
                    logger.warning(
                        "Deployment-backed provider %s: skipping LoRA adapter id %r — recovered "
                        "(workspace=%r, name=%r) fails NAME_PATTERN validation",
                        provider_id,
                        mid,
                        adapter_ws,
                        adapter_name,
                    )
                    continue

                model_entity_id = f"{base_id}&adapters/{adapter_ws}/{adapter_name}"
                served.append(ServedModelMapping(model_entity_id=model_entity_id, served_model_name=mid))
                continue

            # Prompt-tuned: parent is null, root equals base, id != base (the
            # ``mid == base_id`` case was already handled by the base branch above).
            if parent is None and root == base_id:
                model_entity_id = f"{workspace}/{mid.removeprefix(f'{workspace}/')}"
                served.append(ServedModelMapping(model_entity_id=model_entity_id, served_model_name=mid))
                continue

            # Unmatched: skip (do not add to served_models)
            logger.debug(
                f"Deployment-backed provider {provider_id}: skipping unmatched model "
                f"id={mid!r} root={root!r} parent={parent!r}"
            )

        return served

    # -------------------------------------------------------------------------
    # Entity management
    # -------------------------------------------------------------------------

    async def _ensure_model_entity_for_provider(
        self,
        model_workspace: str,
        model_name: str,
        provider: ModelProvider,
        provider_id: str,
        ctx: ModelContext,
    ) -> None:
        """Ensure a Model Entity exists for an autodiscovered model and link it to the provider.

        If the Model Entity doesn't exist, creates it.
        If it exists, updates it to add the provider to model_providers list (if not already present).
        Populates artifact or api_endpoint based on the model's weights location.
        Uses pre-fetched data from ModelContext to avoid repeated retrieval calls.

        Args:
            model_workspace: The workspace for the model entity (same as provider workspace)
            model_name: The name for the model entity (the discovered model ID)
            provider_id: The provider ID in format "workspace/name"
            ctx: The ModelContext containing pre-fetched provider, deployment, config.
                Note: ctx.model_entity is the entity from the provider's config, not necessarily
                the entity for this specific model_name (since we may be creating entities for
                multiple discovered models like LoRAs)
        """
        # Read through the cache so the entity behind every served model does not
        # cost a round trip, and so several providers contributing to the same
        # entity are merged into a single write.
        existing_model_entity = self._entity_cache.get(model_workspace, model_name)

        details = await self._build_artifact_details(
            model_name,
            provider_id,
            provider,
            existing_model_entity,
            ctx.model_deployment,
            ctx.model_deployment_config,
        )

        fileset = None
        if details.fileset_url:
            fileset = details.fileset_url.removeprefix("hf://").removeprefix("fileset://")

        if existing_model_entity is None:
            logger.debug(f"Creating Model Entity {model_workspace}/{model_name} for provider {provider_id}")
            self._entity_cache.stage_create(
                model_workspace,
                model_name,
                description=f"Auto-discovered model from provider {provider_id}",
                backend_format=_infer_backend_format(model_name),
            )
            self._entity_cache.stage_provider_link(model_workspace, model_name, provider_id)
            self._entity_cache.stage_field_updates(
                model_workspace,
                model_name,
                fileset=fileset,
                api_endpoint=details.api_endpoint,
            )
            return

        self._entity_cache.stage_provider_link(model_workspace, model_name, provider_id)

        # Only fill in what the entity is missing, so a value a user corrected is
        # never overwritten. ``backend_format`` treats None as missing so entities
        # registered before it existed get backfilled.
        updates: dict[str, object] = {}
        if fileset and not existing_model_entity.fileset:
            updates["fileset"] = fileset
        if details.api_endpoint and not existing_model_entity.api_endpoint:
            updates["api_endpoint"] = details.api_endpoint
        if not _has_backend_format(existing_model_entity):
            updates["backend_format"] = _infer_backend_format(model_name)
        if updates:
            self._entity_cache.stage_field_updates(model_workspace, model_name, **updates)

    async def _ensure_passthrough_virtual_model(
        self, workspace: str, model_name: str, existing_vm_names: set[tuple[str, str]] | None
    ) -> None:
        """Auto-create a passthrough VirtualModel for a model entity if one does not exist.

        A passthrough VirtualModel has an empty middleware pipeline and sets
        ``default_model_entity`` to ``"{workspace}/{model_name}"``, so inference
        requests addressed to ``"workspace/model_name"`` resolve directly to
        the underlying model entity without any plugin intervention.

        ``existing_vm_names`` short-circuits names that are already taken, and is
        updated on creation so the same name is not attempted twice in one pass.
        A name held by a user-managed VirtualModel is left as it is.

        This is idempotent: a :class:`~nemo_platform_plugin.client.errors.ConflictError` (409) means
        the VirtualModel already exists and is silently ignored.  Any other
        exception is logged as a warning and does not propagate — VirtualModel
        creation failures must not block provider reconciliation.

        Args:
            workspace: Workspace of the model entity.
            model_name: Name of the model entity (also used as the VirtualModel name).
            existing_vm_names: ``(workspace, name)`` of every known VirtualModel, or
                ``None`` when the VirtualModel state could not be read this pass.
        """
        if existing_vm_names is None:
            return

        if (workspace, model_name) in existing_vm_names:
            # The name may belong to a user-managed VirtualModel pointing somewhere
            # else, so do not claim the passthrough itself exists.
            logger.debug(
                "Skipping passthrough VirtualModel for %s/%s: the name is already in use",
                workspace,
                model_name,
            )
            return

        try:
            await self._virtual_models_client.create_virtual_model(
                workspace=workspace,
                body=CreateVirtualModelRequest(
                    name=model_name,
                    default_model_entity=f"{workspace}/{model_name}",
                    autoprovisioned=True,
                ),
            )
            logger.info(
                "Auto-created passthrough VirtualModel %s/%s",
                workspace,
                model_name,
            )
        except ConflictError:
            pass  # Already exists — nothing to do
        except Exception:
            logger.warning(
                "Failed to ensure passthrough VirtualModel for %s/%s",
                workspace,
                model_name,
                exc_info=True,
            )
        finally:
            # Record the name either way: on success it now exists, and on failure
            # retrying it within the same pass would fail the same way.
            existing_vm_names.add((workspace, model_name))

    async def _cleanup_orphaned_virtual_models(
        self,
        provider_contexts: list[ModelContext],
        vm_snapshot: list[VirtualModel],
        snapshot_taken_at: datetime | None = None,
    ) -> None:
        """Delete autoprovisioned VirtualModels whose backing entities are no longer served.

        Only VirtualModels present when the pass began are considered, so anything
        created during the pass -- by this reconciler or anyone else -- is evaluated
        on a later pass rather than raced against.
        """
        snapshot_taken_at = ensure_utc(snapshot_taken_at)
        active_model_entity_ids: set[str] = set()
        for ctx in provider_contexts:
            provider = ctx.model_provider
            if provider is None:
                continue
            if provider.status == ModelProviderStatus.LOST:
                continue
            served_models = ctx.served_models if ctx.served_models is not None else provider.served_models
            for served_model in served_models or ():
                model_entity_id = served_model.model_entity_id
                if model_entity_id:
                    active_model_entity_ids.add(model_entity_id)

        for virtual_model in vm_snapshot:
            self._emit_heartbeat()
            if not virtual_model.autoprovisioned:
                continue

            if (
                virtual_model.default_model_entity is None
                or virtual_model.default_model_entity in active_model_entity_ids
            ):
                continue

            observed_at = _get_virtual_model_observed_at(virtual_model)
            if snapshot_taken_at is not None and observed_at is not None and observed_at > snapshot_taken_at:
                logger.debug(
                    "Skipping orphaned autoprovisioned VirtualModel %s/%s because it changed after the reconciliation snapshot",
                    virtual_model.workspace,
                    virtual_model.name,
                )
                continue

            expected_db_version = _get_virtual_model_db_version(virtual_model)
            if expected_db_version is None:
                logger.warning(
                    "Skipping orphaned autoprovisioned VirtualModel %s/%s because it has no database version for conditional deletion",
                    virtual_model.workspace,
                    virtual_model.name,
                )
                continue

            try:
                await self._virtual_models_client.delete_virtual_model(
                    name=virtual_model.name,
                    workspace=virtual_model.workspace,
                    query_params={"expected_db_version": expected_db_version},
                )
                logger.info(
                    "Deleted orphaned autoprovisioned VirtualModel %s/%s",
                    virtual_model.workspace,
                    virtual_model.name,
                )
            except NotFoundError:
                # Already gone; the snapshot predates the deletion.
                logger.debug(
                    "Orphaned autoprovisioned VirtualModel %s/%s no longer exists",
                    virtual_model.workspace,
                    virtual_model.name,
                )
            except Exception:
                # Best-effort cleanup intentionally catches all exceptions to keep reconciliation running.
                logger.warning(
                    "Failed to delete orphaned autoprovisioned VirtualModel %s/%s",
                    virtual_model.workspace,
                    virtual_model.name,
                    exc_info=True,
                )

    async def _build_artifact_details(
        self,
        model_name: str,
        provider_id: str,
        provider: ModelProvider,
        model_entity: ModelEntity | None,
        deployment: ModelDeployment | None,
        config: ModelDeploymentConfig | None,
    ) -> ArtifactDetails:
        """Determine artifact and API endpoint details for a model based on its weights location.

        Args:
            model_name: The name for the model entity
            provider_id: The provider ID in format "workspace/name"
            provider: The ModelProvider object
            model_entity: Optional existing model entity
            deployment: Optional ModelDeployment object
            config: Optional ModelDeploymentConfig object

        Returns:
            ArtifactDetails with fileset_url and/or api_endpoint populated, or both None
            if the weights type could not be determined or does not require either.
        """
        details = ArtifactDetails()

        try:
            weights_type = get_model_weights_type(
                model_provider=provider,
                model_deployment=deployment,
                model_deployment_config=config,
                model_entity=model_entity,
            )

            if weights_type == ModelWeightsType.EXTERNAL_PROVIDER:
                details.api_endpoint = {
                    "url": provider.host_url,
                    "model_id": model_name,
                    "format": "openai",
                }
                logger.debug(f"Built api_endpoint for external provider: {provider.host_url}")

            elif weights_type == ModelWeightsType.HUGGINGFACE and config:
                model_spec = config.model_spec
                if model_spec is None:
                    logger.warning("Missing model_spec for HuggingFace weights; skipping fileset_url build")
                    return details
                parsed_namespace, parsed_name, parsed_revision = parse_model_name_revision(
                    model_namespace=model_spec.model_namespace,
                    model_name=model_spec.model_name,
                    model_revision=model_spec.model_revision,
                )
                if not parsed_namespace or not parsed_name:
                    logger.warning(
                        "Incomplete model_spec namespace/name for HuggingFace weights; skipping fileset_url build"
                    )
                    return details
                details.fileset_url = f"hf://{parsed_namespace}/{parsed_name}"
                if parsed_revision:
                    details.fileset_url = f"{details.fileset_url}@{parsed_revision}"
                logger.debug(f"Built HuggingFace artifact: {details.fileset_url}")

            elif weights_type == ModelWeightsType.FILES_SERVICE and config:
                # Files service models (including SFT) use hf:// prefix since Files service exposes
                # models via HuggingFace-compatible API
                model_spec = config.model_spec
                if model_spec is None:
                    logger.warning("Missing model_spec for Files service weights; skipping fileset_url build")
                    return details
                parsed_namespace, parsed_name, parsed_revision = parse_model_name_revision(
                    model_namespace=model_spec.model_namespace,
                    model_name=model_spec.model_name,
                    model_revision=model_spec.model_revision,
                )
                if not parsed_namespace or not parsed_name:
                    logger.warning(
                        "Incomplete model_spec namespace/name for Files service weights; skipping fileset_url build"
                    )
                    return details
                details.fileset_url = f"hf://{parsed_namespace}/{parsed_name}"
                if parsed_revision:
                    details.fileset_url = f"{details.fileset_url}@{parsed_revision}"
                logger.debug(f"Built Files service artifact: {details.fileset_url}")

        except Exception as e:
            logger.warning(
                f"Failed to determine weights location for provider {provider_id}: {e}. "
                "Model Entity will be created without artifact/api_endpoint data."
            )

        return details
