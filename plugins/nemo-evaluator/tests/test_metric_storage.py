# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_evaluator.metric_storage import (
    BUNDLE_FILENAME,
    FILESET_PREFIX,
    MetricBundleStorageError,
    delete_bundle_by_ref,
    load_bundle,
    parse_bundle_ref,
    store_bundle,
)
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundle, bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest


class _FakeDownloadResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _FakeFiles(AsyncFilesClient):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, bytes]] = {}
        self.upload_error: Exception | None = None

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
        if self.upload_error is not None:
            raise self.upload_error
        self._store.setdefault((workspace or "default", name), {})[path] = bytes(content)
        return object()

    async def download_file(
        self,
        *,
        path: str,
        name: str,
        workspace: str | None = None,
    ) -> _FakeDownloadResponse:
        return _FakeDownloadResponse(self._store[(workspace or "default", name)][path])


def _sample_bundle() -> MetricBundle:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    return bundle_metric(metric, CloudpickleMetricBundlePackager())


def test_parse_bundle_ref_splits_parts() -> None:
    assert parse_bundle_ref("default/metric-bundle.m.abc#bundle.json") == (
        "default",
        "metric-bundle.m.abc",
        "bundle.json",
    )


@pytest.mark.parametrize("ref", ["no-fragment", "missing-workspace#bundle.json", "ws/fs#"])
def test_parse_bundle_ref_rejects_malformed(ref: str) -> None:
    with pytest.raises(MetricBundleStorageError):
        parse_bundle_ref(ref)


async def test_store_returns_unique_per_metric_ref() -> None:
    fake_files = _FakeFiles()
    bundle = _sample_bundle()

    ref1 = await store_bundle(fake_files, "default", "my-metric", bundle)
    ref2 = await store_bundle(fake_files, "default", "my-metric", bundle)

    assert ref1.startswith("default/metric-bundle.")
    assert ref1.endswith(f"#{BUNDLE_FILENAME}")
    assert ref1 != ref2


async def test_store_fileset_name_stays_within_limit_for_long_metric_name() -> None:
    fake_files = _FakeFiles()
    bundle = _sample_bundle()
    long_name = "m" * 255

    ref = await store_bundle(fake_files, "default", long_name, bundle)

    _, fileset, _ = parse_bundle_ref(ref)
    assert len(fileset) <= 255


async def test_store_cleans_up_fileset_on_upload_failure() -> None:
    fake_files = _FakeFiles()
    bundle = _sample_bundle()

    fake_files.upload_error = RuntimeError("network blip during upload")

    with pytest.raises(MetricBundleStorageError):
        await store_bundle(fake_files, "default", "my-metric", bundle)

    assert [key for key in fake_files._store if key[1].startswith(FILESET_PREFIX)] == []


async def test_store_then_load_round_trips_bundle() -> None:
    fake_files = _FakeFiles()
    bundle = _sample_bundle()

    ref = await store_bundle(fake_files, "default", "my-metric", bundle)
    loaded = await load_bundle(fake_files, ref, expected_digest=bundle.payload.digest)

    assert loaded.metric_type == bundle.metric_type
    assert loaded.payload.digest == bundle.payload.digest


async def test_load_rejects_digest_mismatch() -> None:
    fake_files = _FakeFiles()
    bundle = _sample_bundle()

    ref = await store_bundle(fake_files, "default", "my-metric", bundle)
    with pytest.raises(MetricBundleStorageError, match="digest mismatch"):
        await load_bundle(fake_files, ref, expected_digest="deadbeef")


async def test_load_rejects_corrupt_bundle() -> None:
    fake_files = _FakeFiles()
    fake_files._store[("default", "metric-bundle.deadbeef")] = {"bundle.json": b"not a bundle"}

    with pytest.raises(MetricBundleStorageError, match="corrupt or unreadable"):
        await load_bundle(fake_files, "default/metric-bundle.deadbeef#bundle.json")


async def test_load_wraps_download_failure() -> None:
    fake_files = _FakeFiles()

    with pytest.raises(MetricBundleStorageError, match="failed to download metric bundle"):
        await load_bundle(fake_files, "default/metric-bundle.missing#bundle.json")


async def test_delete_by_ref_removes_only_that_fileset() -> None:
    fake_files = _FakeFiles()
    bundle = _sample_bundle()

    ref1 = await store_bundle(fake_files, "default", "my-metric", bundle)
    ref2 = await store_bundle(fake_files, "default", "my-metric", bundle)
    await delete_bundle_by_ref(fake_files, ref1)

    _, fileset1, _ = parse_bundle_ref(ref1)
    _, fileset2, _ = parse_bundle_ref(ref2)
    assert ("default", fileset1) not in fake_files._store
    assert ("default", fileset2) in fake_files._store
