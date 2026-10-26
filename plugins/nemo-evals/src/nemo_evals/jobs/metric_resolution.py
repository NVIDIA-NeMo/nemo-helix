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
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

from nemo_agents_plugin.jobs.gateway_proxy import platform_auth_proxy
from nemo_evals.api.schemas import MetricInline
from nemo_evals.metric_refs import MetricRef, MetricRefOrInline, resolve_metric_specs
from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundle,
    bundle_metric,
    metric_bundle_packager_for_payload,
    unbundle_metric,
)
from nemo_evals.source_authorization import SourceReadAuthorizer, authorize_source_read
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.refs import parse_workspace_name_ref
from nemo_helix_plugin.refs import parse_entity_ref
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx_evals_sdk.execution.metric_execution import run_sync
from nhx_evals_sdk.metrics.protocol import Metric, MetricWithModels
from nhx_evals_sdk.metrics.resolution import model_ref_fields
from nhx_evals_sdk.resolver_protocols import ModelResolver, SecretResolver
from nhx_evals_sdk.values import Model, ModelRef
from nhx_evals_sdk.values.common import SecretRef
from pydantic import BaseModel

#: The path every Inference Gateway route a ``ModelRef`` resolves to starts with.
_GATEWAY_PATH_PREFIX = "/apis/inference-gateway/"
#: Workspace segment of the placeholder refs `_rebind_models` routes a metric's models through.
_REBIND_REF_WORKSPACE = "job-auth-proxy"


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


def _bundle_resolved_metric(metric: Metric, source_bundle: MetricBundle) -> MetricBundle:
    packager = metric_bundle_packager_for_payload(source_bundle.payload)
    resolved_bundle = bundle_metric(metric, packager)
    return resolved_bundle.model_copy(update={"metadata": source_bundle.metadata})


def _model_not_found_error(model_ref: ModelRef, workspace: str, name: str) -> ValueError:
    return ValueError(
        f"Model reference '{model_ref.root}' not found. "
        f"Ensure the model entity '{name}' exists in workspace '{workspace}', "
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
        await authorize_source_read(self.authorize, "model", workspace, name)
        try:
            resolved = await self.models_client.resolve_model_reference(model_ref.root)
        except NotFoundError as exc:
            raise _model_not_found_error(model_ref, workspace, name) from exc
        return Model(
            url=_job_reachable_url(resolved.url, self.models_client.base_url),
            name=resolved.name,
            host_url=resolved.host_url,
            served_model_name=resolved.served_model_name,
        )


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
) -> list[MetricInline]:
    """Resolve a wire metric list (inline + stored refs) into canonical inline metrics.

    Stored references are loaded from the entity store; any ``MetricWithModels``
    model references are resolved through the platform.

    Stored-ref loading awaits real file I/O, so it uses the typed Files client
    derived from the platform client. Model-ref resolution uses the typed Models client.
    Both need ``async_client``; it is only optional so callers can pass through the
    job context's handle, and it must be set when any reference needs resolving.
    When ``authorize`` is set, each stored metric and model is checked as the caller before it is read.
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
        if async_client is None:
            raise ValueError("resolving metric model references requires a platform client")
        models_client = client_from_platform(async_client, AsyncModelsClient)
        resolver: ModelResolver = HelixMetricModelResolver(models_client, authorize=authorize)
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


def _gateway_models(metric: Metric) -> dict[str, Model]:
    """The metric's model fields that call Inference Gateway with no credential of their own.

    A model with ``api_key_secret`` is called as configured: its owner chose the credential.
    """
    if not (isinstance(metric, MetricWithModels) and isinstance(metric, BaseModel)):
        return {}
    return {
        field: model
        for field in model_ref_fields(metric)
        if isinstance(model := getattr(metric, field, None), Model)
        and model.api_key_secret is None
        and urlsplit(model.url).path.startswith(_GATEWAY_PATH_PREFIX)
    }


def _rebind_models(metric: Metric, models: dict[str, Model]) -> None:
    """Replace the metric's model fields, through ``resolve_models`` so it rebuilds what it derived from them.

    Metrics build their inference client from their models when constructed (RAGAS) or lazily, and
    ``resolve_models`` is their one hook for being handed a model after that. Each field is parked
    behind a placeholder ref that resolves to its replacement.
    """
    assert isinstance(metric, MetricWithModels)
    parked: dict[str, Model] = {}
    for field, model in models.items():
        ref = f"{_REBIND_REF_WORKSPACE}/{field}"
        setattr(metric, field, ModelRef(ref))
        parked[ref] = model
    run_sync(lambda: metric.resolve_models(_BoundModels(parked)))


@contextmanager
def gateway_metric_models_as_job(metrics: Sequence[Metric]) -> Generator[None]:
    """Call Inference Gateway from the metrics' models as this job, for the duration of the block.

    A stored metric's ``ModelRef`` is resolved at submit to a gateway route on the API's own
    platform URL, with no credential: the job's identity is not known yet, and the job may reach
    the platform at a different address. Inside the block those models point at the job's loopback
    auth proxy, which forwards to the job's platform URL with its identity and refreshes workload
    tokens. Afterwards the original models are restored, so results record the resolved route.

    Does nothing when no metric calls the gateway, or when the job has no identity to forward
    (an auth-disabled platform), which is how those models were called before.
    """
    bound = [(metric, models) for metric in metrics if (models := _gateway_models(metric))]
    if not bound:
        yield
        return
    with platform_auth_proxy() as origin:
        if origin is None:
            yield
            return
        proxy = urlsplit(origin)
        for metric, models in bound:
            _rebind_models(
                metric,
                {
                    field: model.model_copy(
                        update={"url": urlsplit(model.url)._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()}
                    )
                    for field, model in models.items()
                },
            )
        try:
            yield
        finally:
            for metric, models in bound:
                _rebind_models(metric, models)
