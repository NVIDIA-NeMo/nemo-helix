# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

try:
    from scaled_evals.api import s3
    from scaled_evals.api.settings import settings
except ModuleNotFoundError as exc:
    if exc.name != "scaled_evals":
        raise
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_gcs_media_url_encodes_valid_object_key_as_one_path_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gcs_api_base_url", "https://storage.googleapis.com")
    monkeypatch.setattr(s3, "_bucket", lambda: "eval bucket")

    assert s3._gcs_media_url("evaluations/run 1/artifacts/café.json") == (
        "https://storage.googleapis.com/download/storage/v1/b/eval%20bucket/o/"
        "evaluations%2Frun%201%2Fartifacts%2Fcaf%C3%A9.json"
    )


@pytest.mark.parametrize(
    "object_key",
    [
        "",
        "/absolute/path",
        "//metadata.google.internal/computeMetadata/v1",
        "https://metadata.google.internal/computeMetadata/v1",
        "evaluations/../private",
        "evaluations/./artifact",
        "evaluations//artifact",
        "evaluations/artifact?alt=media",
        "evaluations/artifact#fragment",
        "evaluations\\artifact",
        "evaluations/artifact\r\nX-Header: injected",
        "x" * 1025,
    ],
)
def test_gcs_urls_reject_unsafe_object_keys(object_key: str) -> None:
    with pytest.raises(ValueError, match="GCS object key"):
        s3._gcs_media_url(object_key)
    with pytest.raises(ValueError, match="GCS object key"):
        s3._gcs_object_metadata_url(object_key)
    with pytest.raises(ValueError, match="GCS object key"):
        list(s3._gcs_stream_object(object_key))
