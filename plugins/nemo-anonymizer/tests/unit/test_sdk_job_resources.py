# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from nemo_anonymizer_plugin.sdk.errors import AnonymizerJobError
from nemo_anonymizer_plugin.sdk.job_resources import _safe_extract_tar
from nemo_anonymizer_plugin.sdk.job_results import AnonymizerJobResults


def _tar_with_file(name: str, content: bytes = b"data") -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    stream.seek(0)
    return stream


def _tar_with_symlink(name: str, target: str) -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.type = tarfile.SYMTYPE
        info.linkname = target
        tar.addfile(info)
    stream.seek(0)
    return stream


def test_safe_extract_tar_rejects_parent_directory_member(tmp_path: Path) -> None:
    output_path = tmp_path / "out"
    stream = _tar_with_file("../escape.txt")

    with tarfile.open(fileobj=stream, mode="r:*") as tar:
        with pytest.raises(AnonymizerJobError, match="unsafe tar member"):
            _safe_extract_tar(tar, output_path)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    stream = _tar_with_symlink("artifacts/link", "/etc/passwd")

    with tarfile.open(fileobj=stream, mode="r:*") as tar:
        with pytest.raises(AnonymizerJobError, match="tar link member"):
            _safe_extract_tar(tar, tmp_path / "out")


def test_safe_extract_tar_allows_regular_files(tmp_path: Path) -> None:
    output_path = tmp_path / "out"
    stream = _tar_with_file("artifacts/dataset.txt", b"ok")

    with tarfile.open(fileobj=stream, mode="r:*") as tar:
        _safe_extract_tar(tar, output_path)

    assert (output_path / "artifacts" / "dataset.txt").read_bytes() == b"ok"


def test_job_results_load_trace_restores_original_text_column_metadata(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    pd.DataFrame([{"body": "Alice"}]).to_parquet(artifacts_dir / "trace.parquet", index=False)
    (artifacts_dir / "metadata.json").write_text('{"original_text_column":"body"}')

    trace = AnonymizerJobResults(artifacts_dir).load_trace()

    assert trace.attrs["original_text_column"] == "body"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("output_path", [None, "chosen-output"])
async def test_download_uses_id_indexed_job_name_and_preserves_paths(
    tmp_path,
    monkeypatch,
    asynchronous,
    output_path,
):
    import httpx
    from nemo_anonymizer_plugin.sdk.job_resources import AnonymizerJobResource, AsyncAnonymizerJobResource
    from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient

    monkeypatch.chdir(tmp_path)
    dataset = io.BytesIO()
    pd.DataFrame({"text": ["sanitized"]}).to_parquet(dataset, index=False)
    archive = _tar_with_file("artifacts/dataset.parquet", dataset.getvalue()).getvalue()
    base = "/apis/anonymizer/v2/workspaces/default/jobs/run/platform-job-original"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == base + "/status":
            return httpx.Response(
                200,
                json={
                    "id": "platform-job-original",
                    "name": "readable-job-name",
                    "status": "completed",
                },
            )
        assert request.url.path == base + "/results/artifacts/download"
        return httpx.Response(200, content=archive)

    if asynchronous:
        async with AsyncNemoClient(
            base_url="http://test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            handle = AsyncAnonymizerJobResource(job_id="platform-job-original", workspace="default", platform=client)
            result = await handle.download_artifacts(path=output_path)
    else:
        with NemoClient(
            base_url="http://test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ) as client:
            handle = AnonymizerJobResource(job_id="platform-job-original", workspace="default", platform=client)
            result = handle.download_artifacts(path=output_path)

    expected = tmp_path / (output_path or "readable-job-name") / "artifacts" / "dataset.parquet"
    assert expected.is_file()
    assert not (tmp_path / "platform-job-original").exists()
    assert result.load_dataset()["text"].tolist() == ["sanitized"]
    assert seen == [base + "/status", base + "/results/artifacts/download"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unfinished_job_creates_no_download_directory(tmp_path, monkeypatch, asynchronous):
    import httpx
    from nemo_anonymizer_plugin.sdk.job_resources import AnonymizerJobResource, AsyncAnonymizerJobResource
    from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient

    monkeypatch.chdir(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/jobs/run/platform-job-original/status")
        return httpx.Response(
            200,
            json={
                "id": "platform-job-original",
                "name": "readable-job-name",
                "status": "active",
            },
        )

    if asynchronous:
        async with AsyncNemoClient(
            base_url="http://test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            handle = AsyncAnonymizerJobResource(job_id="platform-job-original", workspace="default", platform=client)
            with pytest.raises(AnonymizerJobError, match="not available"):
                await handle.download_artifacts()
    else:
        with NemoClient(
            base_url="http://test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ) as client:
            handle = AnonymizerJobResource(job_id="platform-job-original", workspace="default", platform=client)
            with pytest.raises(AnonymizerJobError, match="not available"):
                handle.download_artifacts()
    assert not (tmp_path / "readable-job-name").exists()
