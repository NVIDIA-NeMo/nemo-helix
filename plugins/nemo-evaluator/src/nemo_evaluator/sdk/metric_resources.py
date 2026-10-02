# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for managing stored metrics (``client.evaluator.metrics``).

These resources package a runtime metric into a metric-create wire DTO and
call the evaluator service's ``/metrics`` create/get/list/delete API (metrics are
immutable). The service owns the payload's Files storage, so the SDK only ever
sends/receives the metric and its metadata.
"""

from __future__ import annotations

from nemo_evaluator.api.schemas import Metric
from nemo_evaluator.sdk.query_params import QueryParams, list_params, project_params
from nemo_evaluator.shared.metric_bundles.bundles import (
    MetricBundle,
    MetricBundlePackager,
    bundle_metric,
)
from nemo_evaluator.shared.metric_bundles.defaults import resolve_default_metric_bundle_packager
from nemo_evaluator_sdk.metrics.protocol import Metric as RuntimeMetric
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.evaluator.types import CreateMetricRequest
from nemo_helix_plugin.schema import Page


def _metric_query_params(
    page: int, page_size: int, sort: str | None, metric_type: str | None, include_derived: bool
) -> QueryParams:
    """Build the list query string: paging/sort + the route's ``filter[metric_type]`` trait filter."""
    params = list_params(page, page_size, sort)
    if metric_type is not None:
        params["filter[metric_type]"] = metric_type
    if include_derived:
        params["include_derived"] = True
    return params


def _metric_create_request(
    metric: RuntimeMetric | MetricBundle,
    metric_bundle_packager: MetricBundlePackager | None,
) -> CreateMetricRequest:
    """Package a runtime metric (or accept a pre-built bundle) as the wire DTO."""
    if isinstance(metric, MetricBundle):
        bundle = metric
    else:
        packager = resolve_default_metric_bundle_packager(
            metric, metric_bundle_packager, allow_cloudpickle_fallback=False, action="Storing"
        )
        bundle = bundle_metric(metric, packager)
    # JSON round-trip keeps the base64 payload encoding consistent with the runtime model.
    return CreateMetricRequest.model_validate_json(bundle.model_dump_json())


class EvaluatorMetricsResource:
    """Sync resource mounted as ``client.evaluator.metrics``."""

    def __init__(self, client: EvaluatorClient) -> None:
        self._client = client

    def create(
        self,
        name: str,
        *,
        metric: RuntimeMetric | MetricBundle,
        metric_bundle_packager: MetricBundlePackager | None = None,
        project: str | None = None,
        workspace: str | None = None,
    ) -> Metric:
        """Store a new metric (addressed by workspace/name), packaging a runtime metric when needed."""
        body = _metric_create_request(metric, metric_bundle_packager)
        response = self._client.create_metric(
            name=name,
            workspace=workspace,
            body=body,
            query_params=project_params(project),
        )
        return Metric.model_validate(response.data().model_dump(mode="json"))

    def retrieve(self, name: str, *, workspace: str | None = None) -> Metric:
        """Get a stored metric by name."""
        response = self._client.get_metric(name=name, workspace=workspace)
        return Metric.model_validate(response.data().model_dump(mode="json"))

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        metric_type: str | None = None,
        include_derived: bool = False,
    ) -> Page[Metric]:
        """List stored metrics in a workspace, optionally filtered by metric type.

        Derived (task-internal) metrics are hidden unless ``include_derived`` is set.
        """
        response = self._client.list_metrics(
            workspace=workspace,
            query_params=_metric_query_params(page, page_size, sort, metric_type, include_derived),
        )
        page_result = response.page()
        return Page[Metric].model_validate(
            {
                "data": [metric.model_dump(mode="json") for metric in page_result.items],
                "pagination": page_result.metadata,
                "sort": sort,
                "filter": None,
            }
        )

    def delete(self, name: str, *, workspace: str | None = None) -> None:
        """Delete a stored metric and its backing bundle."""
        self._client.delete_metric(name=name, workspace=workspace).data()


class AsyncEvaluatorMetricsResource:
    """Async resource mounted as ``client.evaluator.metrics``."""

    def __init__(self, client: AsyncEvaluatorClient) -> None:
        self._client = client

    async def create(
        self,
        name: str,
        *,
        metric: RuntimeMetric | MetricBundle,
        metric_bundle_packager: MetricBundlePackager | None = None,
        project: str | None = None,
        workspace: str | None = None,
    ) -> Metric:
        """Store a new metric (addressed by workspace/name), packaging a runtime metric when needed."""
        body = _metric_create_request(metric, metric_bundle_packager)
        response = await self._client.create_metric(
            name=name,
            workspace=workspace,
            body=body,
            query_params=project_params(project),
        )
        return Metric.model_validate(response.data().model_dump(mode="json"))

    async def retrieve(self, name: str, *, workspace: str | None = None) -> Metric:
        """Get a stored metric by name."""
        response = await self._client.get_metric(name=name, workspace=workspace)
        return Metric.model_validate(response.data().model_dump(mode="json"))

    async def list(
        self,
        *,
        workspace: str | None = None,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        metric_type: str | None = None,
        include_derived: bool = False,
    ) -> Page[Metric]:
        """List stored metrics in a workspace, optionally filtered by metric type.

        Derived (task-internal) metrics are hidden unless ``include_derived`` is set.
        """
        response = await self._client.list_metrics(
            workspace=workspace,
            query_params=_metric_query_params(page, page_size, sort, metric_type, include_derived),
        )
        page_result = response.page()
        return Page[Metric].model_validate(
            {
                "data": [metric.model_dump(mode="json") for metric in page_result.items],
                "pagination": page_result.metadata,
                "sort": sort,
                "filter": None,
            }
        )

    async def delete(self, name: str, *, workspace: str | None = None) -> None:
        """Delete a stored metric and its backing bundle."""
        response = await self._client.delete_metric(name=name, workspace=workspace)
        response.data()
