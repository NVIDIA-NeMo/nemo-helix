# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nhx.customization_common import image_refs
from nhx.customization_common.image_refs import (
    FilesetImageRef,
    FilesetImageResolver,
    ImageReferenceError,
    check_image_url,
    parse_fileset_image_ref,
)


@pytest.mark.parametrize(
    ("url", "error"),
    [
        ("data:image/png;base64,aa", None),
        ("fileset://ws/images#cats/1.png", None),
        ("fileset://images#cats/1.JPG", None),
        ("images/1.png", None),
        ("http://example.com/a.png", "does not download http URLs"),
        ("HTTPS://example.com/a.png", "does not download https URLs"),
        ("s3://bucket/a.png", "is not accepted"),
        ("file:///etc/a.png", "host path"),
        ("/etc/a.png", "host path"),
        ("C:\\images\\a.png", "host path"),
        ("fileset://ws/images#../secret.png", "inside the fileset"),
        ("fileset://ws/images#cats/1.bmp", "Unsupported image file type"),
        ("fileset://ws/images", "Invalid fileset image reference"),
    ],
)
def test_check_image_url(url: str, error: str | None) -> None:
    if error is None:
        check_image_url(url)
    else:
        with pytest.raises(ImageReferenceError, match=error):
            check_image_url(url)


def test_parse_fileset_image_ref_leaves_the_workspace_to_the_job() -> None:
    assert parse_fileset_image_ref("fileset://images#a/b.webp") == FilesetImageRef(None, "images", "a/b.webp")


def _files_client(responses: dict[str, bytes], seen: list[str]) -> NemoClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        body = responses.get(request.url.path)
        if body is None:
            return httpx.Response(404, request=request, json={"detail": "not found"})
        return httpx.Response(200, request=request, stream=httpx.ByteStream(body))

    return NemoClient(base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _resolver(tmp_path: Path, client: NemoClient, dataset_fileset: str = "train-data") -> FilesetImageResolver:
    return FilesetImageResolver(
        job_workspace="default",
        dataset_fileset=dataset_fileset,
        dataset_root=tmp_path / "dataset",
        cache_dir=tmp_path / "images",
        client_factory=lambda: client,
    )


@pytest.mark.parametrize(
    ("dataset_fileset", "url"),
    [
        ("train-data", "fileset://train-data#scans/a.png"),
        ("default/train-data", "fileset://default/train-data#scans/a.png"),
        ("train-data#rows/", "fileset://default/train-data#rows/scans/a.png"),
    ],
)
def test_a_ref_into_the_dataset_fileset_maps_onto_the_download(tmp_path: Path, dataset_fileset: str, url: str) -> None:
    image = tmp_path / "dataset" / "scans" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    seen: list[str] = []

    with _resolver(tmp_path, _files_client({}, seen), dataset_fileset) as resolver:
        assert resolver.local_path(url) == image.resolve()
    assert seen == []


def test_a_ref_into_another_fileset_is_downloaded_once(tmp_path: Path) -> None:
    seen: list[str] = []
    client = _files_client({"/apis/files/v2/workspaces/other/filesets/pics/-/cats/1.jpg": b"jpg"}, seen)

    with _resolver(tmp_path, client) as resolver:
        first = resolver.local_path("fileset://other/pics#cats/1.jpg")
        assert resolver.local_path("fileset://other/pics#cats/1.jpg") == first
    assert first.parent == tmp_path / "images"
    assert first.read_bytes() == b"jpg"
    assert len(seen) == 1


def test_a_missing_file_in_the_dataset_fileset_is_downloaded(tmp_path: Path) -> None:
    seen: list[str] = []
    client = _files_client({"/apis/files/v2/workspaces/default/filesets/train-data/-/a.png": b"png"}, seen)

    with _resolver(tmp_path, client) as resolver:
        assert resolver.local_path("fileset://train-data#a.png").read_bytes() == b"png"


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with _resolver(tmp_path, _files_client({}, [])) as resolver:
        with pytest.raises(ImageReferenceError, match="other/pics#a.png not found"):
            resolver.local_path("fileset://other/pics#a.png")


def test_a_download_is_size_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_refs, "MAX_IMAGE_BYTES", 2)
    client = _files_client({"/apis/files/v2/workspaces/other/filesets/pics/-/a.png": b"too big"}, [])

    with _resolver(tmp_path, client) as resolver:
        with pytest.raises(ImageReferenceError, match="larger than"):
            resolver.local_path("fileset://other/pics#a.png")
    assert list((tmp_path / "images").iterdir()) == []


def test_a_dataset_ref_cannot_escape_through_a_symlink(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "outside").symlink_to(tmp_path, target_is_directory=True)

    with _resolver(tmp_path, _files_client({}, [])) as resolver:
        with pytest.raises(ImageReferenceError, match="outside the dataset"):
            resolver.local_path("fileset://train-data#outside/a.png")
