# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metric-reference resolution for evaluator jobs.

Both ``EvaluateJob`` (row/model eval) and ``AgentEvalJob`` accept metrics as a
mix of inline bundles and references to stored metrics. During ``to_spec`` those
must be resolved into canonical inline metrics — stored refs loaded from the
entity store, and any model references carried by ``MetricWithModels`` resolved
through the platform. This module is the one place that logic lives.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote, urlsplit

from fastapi import HTTPException, status
from nemo_agents_plugin.jobs.gateway_proxy import platform_auth_proxy
from nemo_evals.api.schemas import MetricInline
from nemo_evals.metric_refs import MetricRef, MetricRefOrInline, resolve_metric_specs
from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundle,
    bundle_metric,
    metric_bundle_packager_for_payload,
    unbundle_metric,
)
from nemo_evals.source_authorization import SourceReadAuthorizer, authorize_source_read, remember_allowed
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.models.client import AsyncModelsClient, UnroutableModelError
from nemo_helix_plugin.models.refs import parse_workspace_name_ref
from nemo_helix_plugin.refs import parse_entity_ref
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.virtual_models.client import AsyncVirtualModelsClient
from nhx_evals_sdk.execution.metric_execution import run_sync
from nhx_evals_sdk.metrics.protocol import Metric, MetricWithModels
from nhx_evals_sdk.metrics.resolution import model_ref_fields
from nhx_evals_sdk.resolver_protocols import ModelResolver, SecretResolver
from nhx_evals_sdk.values import AgentBase, Model, ModelRef
from nhx_evals_sdk.values.common import SecretRef
from pydantic import BaseModel

#: The path every Inference Gateway route a ``ModelRef`` resolves to starts with.
_GATEWAY_PATH_PREFIX = "/apis/inference-gateway/"
#: The path every agent route Studio targets (``/apis/agents/v2/workspaces/{ws}/agents/{name}/-/...``) starts with.
_AGENTS_PATH_PREFIX = "/apis/agents/"
#: Workspace segment of the placeholder refs `_rebind_models` routes a metric's models through.
_REBIND_REF_WORKSPACE = "job-auth-proxy"
#: The read paths ``authorize_source_read`` checks for a model and a VirtualModel.
_MODEL_READ_PATHS = {
    "model": re.compile(r"/apis/models/v2/workspaces/(?P<workspace>[^/]+)/models/(?P<name>[^/]+)"),
    "virtual_model": re.compile(
        r"/apis/inference-gateway/v2/workspaces/(?P<workspace>[^/]+)/virtual-models/(?P<name>[^/]+)"
    ),
}


def unresolved_model_refs(metrics: list[Metric]) -> list[str]:
    """Return the sorted model references still unresolved across the given metrics."""
    refs = [
        model_ref.root
        for item in metrics
        if isinstance(item, MetricWithModels)
        for model_ref in item.model_refs().values()
    ]
    return sorted(refs)


def to_inline(bundle: MetricBundle) -> MetricInline:
    """Project a runtime bundle onto the wire DTO (JSON round-trip keeps base64 consistent)."""
    return MetricInline.model_validate_json(bundle.model_dump_json())


def to_runtime_bundle(metric: MetricInline) -> MetricBundle:
    """Reconstruct the runtime bundle from a wire DTO for execution."""
    return MetricBundle.model_validate_json(metric.model_dump_json())


def to_runtime_metrics(metrics: Sequence[MetricInline]) -> list[Metric]:
    """Return runtime metrics reconstructed from the ordered inline bundle DTOs.

    Args:
        metrics: Inline metric bundles to decode without changing their order.

    Returns:
        One runtime metric instance per supplied bundle.
    """
    return [unbundle_metric(to_runtime_bundle(metric)) for metric in metrics]


def require_resolved_model_refs(metrics: list[Metric], *, subject: str) -> None:
    """Require all models in the supplied runtime metrics to be resolved.

    Args:
        metrics: Runtime metrics whose model references are inspected.
        subject: Diagnostic prefix identifying the task or scoring kind.

    Returns:
        None when no unresolved model references remain.

    Raises:
        ValueError: A metric still carries an unresolved model reference.
    """
    if unresolved := unresolved_model_refs(metrics):
        raise ValueError(f"{subject} metric models must be resolved before run: {', '.join(unresolved)}")


def _calls_gateway_unkeyed(model: Model) -> bool:
    """Whether ``model`` calls an Inference Gateway route with no credential of its own.

    A model with ``api_key_secret`` is called as configured: its owner chose the credential.
    """
    return model.api_key_secret is None and urlsplit(model.url).path.startswith(_GATEWAY_PATH_PREFIX)


def _calls_platform_unkeyed(target: Model | AgentBase) -> bool:
    """Whether ``target`` calls a platform gateway or agent route with no credential of its own."""
    prefixes = (_GATEWAY_PATH_PREFIX, _AGENTS_PATH_PREFIX) if isinstance(target, AgentBase) else (_GATEWAY_PATH_PREFIX,)
    return target.api_key_secret is None and urlsplit(target.url).path.startswith(prefixes)


def _bundle_resolved_metric(metric: Metric, source_bundle: MetricBundle) -> MetricBundle:
    packager = metric_bundle_packager_for_payload(source_bundle.payload)
    resolved_bundle = bundle_metric(metric, packager)
    return resolved_bundle.model_copy(update={"metadata": source_bundle.metadata})


async def _readable_kind(
    authorize: SourceReadAuthorizer | None, workspace: str, name: str
) -> Literal["model", "virtual_model"]:
    """Which of the model or the VirtualModel named ``name`` the caller may read; either lets them use it.

    Every model is served through a VirtualModel of the same name, so reading either one is enough.

    Raises:
        HTTPException: The caller may read neither (the model's denial).
    """
    try:
        await authorize_source_read(authorize, "model", workspace, name)
    except HTTPException as model_denied:
        if model_denied.status_code != status.HTTP_403_FORBIDDEN:
            raise
        try:
            await authorize_source_read(authorize, "virtual_model", workspace, name)
        except HTTPException:
            raise model_denied from None
        return "virtual_model"
    return "model"


def _model_not_found_error(model_ref: ModelRef, workspace: str, name: str) -> ValueError:
    return ValueError(
        f"Model reference '{model_ref.root}' not found. "
        f"Ensure a model or VirtualModel named '{name}' exists in workspace '{workspace}', "
        "or use an inline model definition instead."
    )


def _job_reachable_url(url: str, client_base_url: str) -> str:
    """Re-anchor a URL minted on a loopback client base URL on the in-cluster service URL.

    Under embedded auth the API process calls itself over loopback, so a model reference it resolves
    would carry an address no job container can reach. Deployments without embedded auth already
    resolve against the service URL, which jobs reach today.
    """
    internal_base_url = os.environ.get("NEMO_INTERNAL_BASE_URL") or os.environ.get("NHX_INTERNAL_BASE_URL")
    if not internal_base_url or urlsplit(client_base_url).hostname not in LOOPBACK_ADDRESSES:
        return url
    return internal_base_url.rstrip("/") + url.removeprefix(client_base_url.rstrip("/"))


@dataclass(frozen=True)
class HelixMetricModelResolver(ModelResolver):
    """Resolve evaluator metric ``ModelRef`` values through the typed Models client."""

    models_client: AsyncModelsClient
    # Checks the submitting user may read the model. Needed when ``models_client`` is ``service:evals`` acting
    # on behalf of the user: policy allows a read if either principal may which is always true for ``service:evals``.
    # so we cannot rely on "on_behalf_of=caller" to guard the read.
    # ``None`` skips the check.
    authorize: SourceReadAuthorizer | None = None

    async def resolve_model(self, model_ref: ModelRef) -> Model:
        workspace, name = parse_workspace_name_ref(
            model_ref.root, label="ModelRef", expected_format="workspace/model_name"
        )
        readable = await _readable_kind(self.authorize, workspace, name)
        try:
            resolved = await self.models_client.resolve_model_reference(model_ref.root)
        except UnroutableModelError as exc:
            raise ValueError(str(exc.detail)) from exc
        except NotFoundError as exc:
            raise _model_not_found_error(model_ref, workspace, name) from exc
        except PermissionDeniedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Not allowed to read model '{model_ref.root}'."
            ) from exc
        if resolved.virtual_model and readable == "model":
            await authorize_source_read(self.authorize, "virtual_model", workspace, name)
        entity_readable = readable == "model"
        return Model(
            url=_job_reachable_url(resolved.url, self.models_client.base_url),
            name=resolved.name,
            host_url=resolved.host_url if entity_readable else None,
            served_model_name=resolved.served_model_name if entity_readable else None,
        )


def caller_model_read_authorizer(async_client: AsyncHelixClient) -> SourceReadAuthorizer:
    """Check model and VirtualModel reads as the caller, by reading them with the caller's own client.

    The Models client resolves a ref to a VirtualModel's route even when the caller can read only the
    model entity of that name, so resolution alone does not show the caller may use what the ref routes
    to. ``HelixMetricModelResolver`` closes that gap when it has an authorizer; this one needs no
    platform policy hook, only the request-scoped client. A missing source counts as allowed: whether
    it exists is resolution's question, answered with its own error.
    """
    models_client = client_from_platform(async_client, AsyncModelsClient)
    virtual_models_client = client_from_platform(async_client, AsyncVirtualModelsClient)

    async def check(method: str, path: str) -> None:
        del method
        for kind, pattern in _MODEL_READ_PATHS.items():
            if match := pattern.fullmatch(path):
                break
        else:
            raise ValueError(f"caller model read checks cover a model or VirtualModel read, not {path!r}")
        workspace, name = unquote(match["workspace"]), unquote(match["name"])
        read = models_client.get_model if kind == "model" else virtual_models_client.get_virtual_model
        try:
            await read(name=name, workspace=workspace)
        except NotFoundError:
            return
        except PermissionDeniedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Not allowed to read {workspace}/{name}."
            ) from exc

    return remember_allowed(check)


async def resolve_model_ref(
    model_ref: ModelRef,
    *,
    async_client: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None = None,
) -> Model:
    """Resolve one ``ModelRef`` the way metric model refs are resolved, for an evaluation target."""
    if async_client is None:
        raise ValueError("resolving a model reference requires a platform client")
    models_client = client_from_platform(async_client, AsyncModelsClient)
    return await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(model_ref)


@dataclass(frozen=True)
class HelixMetricSecretResolver(SecretResolver):
    """Resolve a metric's secret references through the Secrets service, on behalf of the caller.

    The SDK's default ``LocalSecretResolver`` reads ``os.environ``, which a job populates through
    ``build_task_environment`` and an in-process request handler cannot. Pass a service client that
    delegates the calling principal: an undelegated one would let any caller read another
    workspace's key and send it to an endpoint of their choosing. An unqualified ref resolves in
    ``workspace``.
    """

    secrets_client: AsyncSecretsClient
    workspace: str

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        """Return the secret's value, or ``None`` when the caller cannot see one by that name."""
        ref = parse_entity_ref(secret_ref.root, self.workspace)
        try:
            response = await self.secrets_client.access_secret(name=ref.name, workspace=ref.workspace)
        except NotFoundError:
            return None
        return response.data().value


async def resolve_metrics_to_inline(
    metrics: list[MetricRefOrInline],
    *,
    workspace: str,
    entity_client: EntityClient | None,
    async_client: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None = None,
    model_authorize: SourceReadAuthorizer | None = None,
    model_client: AsyncHelixClient | None = None,
) -> list[MetricInline]:
    """Resolve a wire metric list (inline + stored refs) into canonical inline metrics.

    Stored references are loaded from the entity store; any ``MetricWithModels``
    model references are resolved through the platform.

    Stored-ref loading awaits real file I/O, so it uses the typed Files client
    derived from the platform client. Model-ref resolution uses the typed Models client.
    Both need ``async_client``; it is only optional so callers can pass through the
    job context's handle, and it must be set when any reference needs resolving.
    When ``authorize`` is set, each stored metric and model is checked as the caller before it is read.
    ``model_authorize`` checks model refs instead of ``authorize``, for a caller that has no check for
    stored metrics; when it is unset, ``authorize`` covers models too.
    ``model_client`` resolves model refs instead of ``async_client``, which still reads stored metrics.
    """
    has_metric_ref = any(isinstance(metric, MetricRef) for metric in metrics)
    files_client = None
    if has_metric_ref:
        if async_client is None:
            raise ValueError("resolving stored metric references requires a platform client")
        files_client = client_from_platform(async_client, AsyncFilesClient)
    resolved_bundles = await resolve_metric_specs(
        metrics,
        workspace=workspace,
        entity_client=entity_client,
        files_client=files_client,
        authorize=authorize,
    )
    runtime_metrics = [unbundle_metric(bundle) for bundle in resolved_bundles]
    final_bundles = resolved_bundles
    unresolved = unresolved_model_refs(runtime_metrics)
    if unresolved:
        resolving_client = model_client if model_client is not None else async_client
        if resolving_client is None:
            raise ValueError("resolving metric model references requires a platform client")
        models_client = client_from_platform(resolving_client, AsyncModelsClient)
        resolver: ModelResolver = HelixMetricModelResolver(
            models_client, authorize=model_authorize if model_authorize is not None else authorize
        )
        await asyncio.gather(
            *(metric.resolve_models(resolver) for metric in runtime_metrics if isinstance(metric, MetricWithModels))
        )
        final_bundles = [
            _bundle_resolved_metric(metric, bundle)
            for metric, bundle in zip(runtime_metrics, resolved_bundles, strict=True)
        ]
    return [to_inline(bundle) for bundle in final_bundles]


@dataclass(frozen=True)
class _BoundModels(ModelResolver):
    """Hand back the models `_rebind_models` parked behind placeholder refs."""

    models: dict[str, Model]

    async def resolve_model(self, model_ref: ModelRef) -> Model:
        return self.models[model_ref.root]


def _gateway_models(metric: MetricWithModels) -> dict[str, Model]:
    """The metric's model fields that call Inference Gateway with no credential of their own."""
    if not isinstance(metric, BaseModel):
        return {}
    return {
        field: model
        for field in model_ref_fields(metric)
        if isinstance(model := getattr(metric, field, None), Model) and _calls_gateway_unkeyed(model)
    }


def _rebind_models(metric: MetricWithModels, models: dict[str, Model]) -> None:
    """Replace the metric's model fields, through ``resolve_models`` so it rebuilds what it derived from them.

    Metrics build their inference client from their models when constructed (RAGAS) or lazily, and
    ``resolve_models`` is their one hook for being handed a model after that. Each field is parked
    behind a placeholder ref that resolves to its replacement.
    """
    parked: dict[str, Model] = {}
    for field, model in models.items():
        ref = f"{_REBIND_REF_WORKSPACE}/{field}"
        setattr(metric, field, ModelRef(ref))
        parked[ref] = model
    run_sync(lambda: metric.resolve_models(_BoundModels(parked)))


def _via_proxy[EndpointT: Model | AgentBase](model: EndpointT, origin: str) -> EndpointT:
    """A copy of ``model`` that calls the same platform path on the proxy ``origin``."""
    proxy = urlsplit(origin)
    return model.model_copy(
        update={"url": urlsplit(model.url)._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()}
    )


@contextmanager
def gateway_models_as_job[TargetT: Model | AgentBase](
    metrics: Sequence[Metric], target: TargetT | None = None
) -> Generator[TargetT | None]:
    """Call Inference Gateway from the metrics' models and the target as this job, for the duration of the block.

    A ``ModelRef`` is resolved at submit to a gateway route on the API's own platform URL, with no
    credential: the job's identity is not known yet, and the job may reach the platform at a different
    address. Inside the block those models point at the job's loopback auth proxy, which forwards to
    the job's platform URL with its identity and refreshes workload tokens. Afterwards the metrics'
    original models are restored, so results record the resolved route.

    Models are matched by path, so an inline key-less gateway URL is proxied too, as judges have been
    since they were first proxied: the proxy calls as the job on behalf of its submitter, and the
    gateway holds that to what the submitter may call.

    An agent target on the platform's agents route is proxied the same way. Studio builds its URL from
    the browser's origin, which a job pod may not reach (the public hostname can hairpin to a plain-HTTP
    listener), and the agents service needs the caller's identity anyway.

    Yields:
        The target to call: a proxied copy of ``target`` when it calls the gateway or an agent route
        with no key of its own, else ``target`` itself, which is never changed.

    Does nothing when no model calls the gateway, or when the job has no identity to forward
    (an auth-disabled platform), which is how those models were called before.
    """
    bound = [
        (metric, models)
        for metric in metrics
        if isinstance(metric, MetricWithModels) and (models := _gateway_models(metric))
    ]
    gateway_target = target if target is not None and _calls_platform_unkeyed(target) else None
    if not bound and gateway_target is None:
        yield target
        return
    with platform_auth_proxy() as origin:
        if origin is None:
            yield target
            return
        try:
            for metric, models in bound:
                _rebind_models(metric, {field: _via_proxy(model, origin) for field, model in models.items()})
            yield _via_proxy(gateway_target, origin) if gateway_target is not None else target
        finally:
            for metric, models in bound:
                _rebind_models(metric, models)
