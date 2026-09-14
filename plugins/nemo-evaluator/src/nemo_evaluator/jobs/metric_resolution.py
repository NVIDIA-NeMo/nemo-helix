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
from dataclasses import dataclass

from models import parse_workspace_name_ref
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.metric_refs import MetricRef, MetricRefOrInline, resolve_metric_specs
from nemo_evaluator.shared.metric_bundles.bundles import (
    MetricBundle,
    bundle_metric,
    metric_bundle_packager_for_payload,
    unbundle_metric,
)
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricWithModels
from nemo_evaluator_sdk.resolver_protocols import ModelResolver
from nemo_evaluator_sdk.values import Model, ModelRef
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.sdk import AsyncNeMoPlatform


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


@dataclass(frozen=True)
class PlatformMetricModelResolver(ModelResolver):
    """Resolve evaluator metric ``ModelRef`` values through the typed Models client."""

    models_client: AsyncModelsClient

    async def resolve_model(self, model_ref: ModelRef) -> Model:
        workspace, name = parse_workspace_name_ref(
            model_ref.root, label="ModelRef", expected_format="workspace/model_name"
        )
        try:
            resolved = await self.models_client.resolve_model_reference(model_ref.root)
        except NotFoundError as exc:
            raise _model_not_found_error(model_ref, workspace, name) from exc
        return Model(
            url=resolved.url,
            name=resolved.name,
            host_url=resolved.host_url,
            served_model_name=resolved.served_model_name,
        )


async def resolve_metrics_to_inline(
    metrics: list[MetricRefOrInline],
    *,
    workspace: str,
    entity_client: EntityClient | None,
    async_sdk: AsyncNeMoPlatform | None,
) -> list[MetricInline]:
    """Resolve a wire metric list (inline + stored refs) into canonical inline metrics.

    Stored references are loaded from the entity store; any ``MetricWithModels``
    model references are resolved through the platform. Raises if a model
    reference is present without a usable ``async_sdk`` connection.

    Stored-ref loading awaits real file I/O, so it uses the typed Files client
    derived from the public SDK. Model-ref resolution uses the typed Models client.
    """
    has_metric_ref = any(isinstance(metric, MetricRef) for metric in metrics)
    files_client = (
        client_from_platform(async_sdk, AsyncFilesClient) if has_metric_ref and async_sdk is not None else None
    )
    resolved_bundles = await resolve_metric_specs(
        metrics,
        workspace=workspace,
        entity_client=entity_client,
        files_client=files_client,
    )
    runtime_metrics = [unbundle_metric(bundle) for bundle in resolved_bundles]
    final_bundles = resolved_bundles
    unresolved = unresolved_model_refs(runtime_metrics)
    if unresolved:
        if async_sdk is None:
            raise ValueError(
                "ModelRef metrics require a platform connection (models + inference) to resolve: "
                + ", ".join(unresolved)
            )
        models_client = client_from_platform(async_sdk, AsyncModelsClient)
        resolver: ModelResolver = PlatformMetricModelResolver(models_client)
        await asyncio.gather(
            *(metric.resolve_models(resolver) for metric in runtime_metrics if isinstance(metric, MetricWithModels))
        )
        final_bundles = [
            _bundle_resolved_metric(metric, bundle)
            for metric, bundle in zip(runtime_metrics, resolved_bundles, strict=True)
        ]
    return [to_inline(bundle) for bundle in final_bundles]
