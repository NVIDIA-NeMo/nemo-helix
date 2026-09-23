# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.entities import MetricBundleEntity
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec, EvaluateSpec
from nemo_evaluator.metric_refs import (
    MetricRef,
    parse_metric_ref,
    resolve_metric_specs,
)
from nemo_evaluator.metric_storage import store_bundle
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundle, bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from pydantic import ValidationError

# ---- in-memory fakes (mirror the storage round-trip) -----------------------


class _FakeDownloadResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _FakeFiles(AsyncFilesClient):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, bytes]] = {}

    async def create_fileset(
        self,
        *,
        workspace: str | None = None,
        body: CreateFilesetRequest,
        exist_ok: bool = False,
    ) -> object:
        del exist_ok
        self._store.setdefault((workspace or "default", body.name), {})
        return object()

    async def delete_fileset(self, *, workspace: str | None = None, name: str) -> object:
        self._store.pop((workspace or "default", name), None)
        return object()

    async def upload_file(
        self,
        *,
        content: bytes,
        path: str,
        name: str,
        workspace: str | None = None,
    ) -> object:
        self._store.setdefault((workspace or "default", name), {})[path] = bytes(content)
        return object()

    async def download_file(
        self,
        *,
        path: str,
        name: str,
        workspace: str | None = None,
    ) -> "_FakeDownloadResponse":
        return _FakeDownloadResponse(self._store[(workspace or "default", name)][path])


class _FakeEntityClient:
    def __init__(self) -> None:
        self.entities: dict[tuple[str, str], MetricBundleEntity] = {}

    async def get(
        self, entity_type: type[MetricBundleEntity], *, workspace: str, name: str, parent: str | None = None
    ) -> MetricBundleEntity:
        try:
            return self.entities[(workspace, name)]
        except KeyError:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")


def _bundle():
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    return bundle_metric(metric, CloudpickleMetricBundlePackager())


def _metric_inline() -> MetricInline:
    """An inline metric as carried on the wire (MetricInline DTO)."""
    return MetricInline.model_validate_json(_bundle().model_dump_json())


async def _stored(fake_files: _FakeFiles, entity_client: _FakeEntityClient, workspace: str, name: str) -> MetricBundle:
    bundle = _bundle()
    ref = await store_bundle(fake_files, workspace, name, bundle)
    entity_client.entities[(workspace, name)] = MetricBundleEntity(
        name=name,
        workspace=workspace,
        metric_type=bundle.metric_type,
        outputs=bundle.outputs,
        payload_kind=bundle.payload.kind,
        payload_digest=bundle.payload.digest,
        bundle_ref=ref,
    )
    return bundle


# ---- ref parsing -----------------------------------------------------------


def test_parse_metric_ref_qualified() -> None:
    assert parse_metric_ref("ws/my-metric", "default") == ("ws", "my-metric")


def test_parse_metric_ref_bare_name_uses_default_workspace() -> None:
    assert parse_metric_ref("my-metric", "default") == ("default", "my-metric")


@pytest.mark.parametrize("ref", ["", "ws/", "/name", "ws/a/b", "bad name"])
def test_metric_ref_field_rejects_malformed(ref: str) -> None:
    with pytest.raises(ValidationError):
        MetricRef(root=ref)


# ---- resolution ------------------------------------------------------------


async def test_resolve_converts_inline_metric_to_runtime_bundle() -> None:
    inline = _metric_inline()
    result = await resolve_metric_specs([inline], workspace="default", entity_client=None, files_client=None)
    assert len(result) == 1
    assert result[0].metric_type == inline.metric_type
    assert result[0].payload.digest == inline.payload.digest


async def test_resolve_loads_referenced_bundle() -> None:
    fake_files = _FakeFiles()
    entity_client = _FakeEntityClient()
    stored = await _stored(fake_files, entity_client, "default", "exact")

    result = await resolve_metric_specs(
        [MetricRef(root="default/exact")],
        workspace="default",
        entity_client=entity_client,
        files_client=fake_files,
    )

    assert len(result) == 1
    assert result[0].payload.digest == stored.payload.digest


async def test_resolve_mixes_refs_and_inline_preserving_order() -> None:
    fake_files = _FakeFiles()
    entity_client = _FakeEntityClient()
    await _stored(fake_files, entity_client, "default", "exact")
    inline = _metric_inline()

    result = await resolve_metric_specs(
        [MetricRef(root="exact"), inline],
        workspace="default",
        entity_client=entity_client,
        files_client=fake_files,
    )

    assert len(result) == 2
    assert result[1].payload.digest == inline.payload.digest


async def test_resolve_ref_without_sdk_raises() -> None:
    with pytest.raises(ValueError, match="require a platform connection"):
        await resolve_metric_specs(
            [MetricRef(root="default/exact")],
            workspace="default",
            entity_client=_FakeEntityClient(),
            files_client=None,
        )


async def test_resolve_missing_metric_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="not found"):
        await resolve_metric_specs(
            [MetricRef(root="default/no-such-metric")],
            workspace="default",
            entity_client=_FakeEntityClient(),
            files_client=_FakeFiles(),
        )


async def test_resolve_ref_without_entity_client_raises() -> None:
    with pytest.raises(ValueError, match="require a platform connection"):
        await resolve_metric_specs(
            [MetricRef(root="default/exact")],
            workspace="default",
            entity_client=None,
            files_client=_FakeFiles(),
        )


# ---- spec-level union behavior ---------------------------------------------


def test_input_spec_accepts_ref_and_inline() -> None:
    spec = EvaluateInputSpec.model_validate(
        {
            "metrics": ["default/stored-metric", _metric_inline()],
            "dataset": [{"expected": "a", "output": "a"}],
        }
    )
    assert isinstance(spec.metrics[0], MetricRef)
    assert spec.metrics[0].root == "default/stored-metric"


def test_canonical_spec_rejects_unresolved_ref() -> None:
    with pytest.raises(ValidationError):
        EvaluateSpec.model_validate(
            {
                "metrics": ["default/stored-metric"],
                "dataset": [{"expected": "a", "output": "a"}],
            }
        )
