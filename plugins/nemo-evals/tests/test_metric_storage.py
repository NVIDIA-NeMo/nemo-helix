# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_evals.metric_storage import (
    BUNDLE_FILENAME,
    FILESET_PREFIX,
    MetricBundleStorageError,
    delete_bundle_by_ref,
    load_bundle,
    parse_bundle_ref,
    store_bundle,
)
from nemo_evals.shared.metric_bundles.bundles import MetricBundle, bundle_metric
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nhx.testing.files import InMemoryFilesService
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric


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
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    bundle = _sample_bundle()

    ref1 = await store_bundle(fake_files, "default", "my-metric", bundle)
    ref2 = await store_bundle(fake_files, "default", "my-metric", bundle)

    assert ref1.startswith("default/metric-bundle.")
    assert ref1.endswith(f"#{BUNDLE_FILENAME}")
    assert ref1 != ref2


async def test_store_fileset_name_stays_within_limit_for_long_metric_name() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    bundle = _sample_bundle()
    long_name = "m" * 255

    ref = await store_bundle(fake_files, "default", long_name, bundle)

    _, fileset, _ = parse_bundle_ref(ref)
    assert len(fileset) <= 255


async def test_store_cleans_up_fileset_on_upload_failure() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    bundle = _sample_bundle()

    files_service.failures["upload"] = 500

    with pytest.raises(MetricBundleStorageError):
        await store_bundle(fake_files, "default", "my-metric", bundle)

    assert [key for key in files_service.filesets if key[1].startswith(FILESET_PREFIX)] == []


async def test_store_then_load_round_trips_bundle() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    bundle = _sample_bundle()

    ref = await store_bundle(fake_files, "default", "my-metric", bundle)
    loaded = await load_bundle(fake_files, ref, expected_digest=bundle.payload.digest)

    assert loaded.metric_type == bundle.metric_type
    assert loaded.payload.digest == bundle.payload.digest


async def test_load_rejects_digest_mismatch() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    bundle = _sample_bundle()

    ref = await store_bundle(fake_files, "default", "my-metric", bundle)
    with pytest.raises(MetricBundleStorageError, match="digest mismatch"):
        await load_bundle(fake_files, ref, expected_digest="deadbeef")


async def test_load_rejects_corrupt_bundle() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    files_service.filesets[("default", "metric-bundle.deadbeef")] = {"bundle.json": b"not a bundle"}

    with pytest.raises(MetricBundleStorageError, match="corrupt or unreadable"):
        await load_bundle(fake_files, "default/metric-bundle.deadbeef#bundle.json")


async def test_load_wraps_download_failure() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()

    with pytest.raises(MetricBundleStorageError, match="failed to download metric bundle"):
        await load_bundle(fake_files, "default/metric-bundle.missing#bundle.json")


async def test_delete_by_ref_removes_only_that_fileset() -> None:
    files_service = InMemoryFilesService()
    fake_files = files_service.client()
    bundle = _sample_bundle()

    ref1 = await store_bundle(fake_files, "default", "my-metric", bundle)
    ref2 = await store_bundle(fake_files, "default", "my-metric", bundle)
    await delete_bundle_by_ref(fake_files, ref1)

    _, fileset1, _ = parse_bundle_ref(ref1)
    _, fileset2, _ = parse_bundle_ref(ref2)
    assert ("default", fileset1) not in files_service.filesets
    assert ("default", fileset2) in files_service.filesets
