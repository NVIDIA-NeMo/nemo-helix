# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout checks: what crane will read of a sandbox's untrusted output, before the push step reads it."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from nemo_builder_plugin.run import push
from nemo_builder_plugin.run.push import LayoutRejected, validate_layout


def _blob(layout: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / digest).write_bytes(content)
    return digest


def _index(layout: Path, manifest_digest: str) -> None:
    (layout / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [{"digest": manifest_digest}]}))


def _good_layout(tmp_path: Path) -> tuple[Path, str, str]:
    """A layout with one image of one layer: returns it, its manifest digest and its layer's blob name."""
    layout = tmp_path / "img"
    layout.mkdir()
    config = _blob(layout, b'{"architecture": "arm64"}')
    layer = _blob(layout, b"a layer")
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": f"sha256:{config}"},
            "layers": [{"digest": f"sha256:{layer}"}],
        }
    ).encode()
    manifest_digest = f"sha256:{_blob(layout, manifest)}"
    _index(layout, manifest_digest)
    return layout, manifest_digest, layer


class TestAcceptsAWellFormedLayout:
    def test_returns_the_manifest_digest(self, tmp_path: Path) -> None:
        layout, expected, _ = _good_layout(tmp_path)
        assert validate_layout(layout) == expected

    def test_a_symlinked_parent_directory_is_not_mistaken_for_a_symlink_in_the_layout(self, tmp_path: Path) -> None:
        """The work volume mounts under `/var/run`, a symlink to `/run` on mainstream base images."""
        real = tmp_path / "real"
        real.mkdir()
        _, expected, _ = _good_layout(real)
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        assert validate_layout(link / "img") == expected


class TestWhatCraneWillReadMustBeRealFiles:
    def test_a_symlinked_index_is_refused_rather_than_followed(self, tmp_path: Path) -> None:
        layout, _, _ = _good_layout(tmp_path)
        secret = tmp_path / "environ"
        secret.write_text("NHX_BUILD_REGISTRY_PASSWORD=x")
        (layout / "index.json").unlink()
        (layout / "index.json").symlink_to(secret)
        with pytest.raises(LayoutRejected, match="index.json is a symlink"):
            validate_layout(layout)

    def test_a_fifo_index_is_refused_without_blocking(self, tmp_path: Path) -> None:
        layout, _, _ = _good_layout(tmp_path)
        (layout / "index.json").unlink()
        os.mkfifo(layout / "index.json")
        with pytest.raises(LayoutRejected, match="index.json is not a regular file"):
            validate_layout(layout)

    def test_a_symlinked_blob_directory_is_refused(self, tmp_path: Path) -> None:
        """crane checks only a blob path's last part, so a symlink above it would point it elsewhere."""
        layout, _, _ = _good_layout(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        (layout / "blobs").rename(elsewhere)
        (layout / "blobs").symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(LayoutRejected, match="blobs is a symlink"):
            validate_layout(layout)

    @pytest.mark.parametrize("kind", ["symlink", "fifo"])
    def test_a_layer_that_is_not_a_regular_file_is_refused(self, tmp_path: Path, kind: str) -> None:
        layout, _, layer = _good_layout(tmp_path)
        path = layout / "blobs" / "sha256" / layer
        path.unlink()
        if kind == "symlink":
            path.symlink_to(tmp_path / "anything")
        else:
            os.mkfifo(path)
        with pytest.raises(LayoutRejected, match=f"{layer} is (a symlink|not a regular file)"):
            validate_layout(layout)

    def test_a_layer_the_manifest_names_but_the_layout_lacks_is_refused(self, tmp_path: Path) -> None:
        layout, _, layer = _good_layout(tmp_path)
        (layout / "blobs" / "sha256" / layer).unlink()
        with pytest.raises(LayoutRejected, match=f"layout is missing blobs/sha256/{layer}"):
            validate_layout(layout)


class TestOnlyWhatCraneWillReadIsChecked:
    def test_files_the_image_does_not_name_are_not_read(self, tmp_path: Path) -> None:
        """crane never opens them, so a FIFO or a symlink among them can't block or redirect it."""
        layout, expected, _ = _good_layout(tmp_path)
        os.mkfifo(layout / "blobs" / "sha256" / ("f" * 64))
        (layout / "blobs" / "sha256" / ("e" * 64)).symlink_to("/etc/passwd")
        (layout / "run-me.sh").write_text("#!/bin/sh\n")
        assert validate_layout(layout) == expected

    def test_blob_contents_are_left_to_the_registry(self, tmp_path: Path) -> None:
        """The registry checks each blob against its digest as it's uploaded; the push step doesn't hash them."""
        layout, expected, layer = _good_layout(tmp_path)
        (layout / "blobs" / "sha256" / layer).write_bytes(b"not what its name says")
        assert validate_layout(layout) == expected


class TestTheIndexAndManifest:
    def test_an_index_naming_an_absent_manifest_is_refused(self, tmp_path: Path) -> None:
        layout, _, _ = _good_layout(tmp_path)
        _index(layout, "sha256:" + "0" * 64)
        with pytest.raises(LayoutRejected, match="layout is missing blobs/sha256/0+"):
            validate_layout(layout)

    @pytest.mark.parametrize("digest", ["md5:whatever", "sha256:../../index.json", 42])
    def test_anything_but_a_sha256_digest_is_refused(self, tmp_path: Path, digest: object) -> None:
        layout, _, _ = _good_layout(tmp_path)
        (layout / "index.json").write_text(json.dumps({"manifests": [{"digest": digest}]}))
        with pytest.raises(LayoutRejected, match="sha256 digest"):
            validate_layout(layout)

    @pytest.mark.parametrize("index", ["not json", "[]", '{"manifests": {}}', '{"manifests": ["x"]}'])
    def test_a_malformed_index_is_refused_not_raised(self, tmp_path: Path, index: str) -> None:
        layout, _, _ = _good_layout(tmp_path)
        (layout / "index.json").write_text(index)
        with pytest.raises(LayoutRejected):
            validate_layout(layout)

    def test_a_multi_manifest_index_is_refused(self, tmp_path: Path) -> None:
        layout, digest, _ = _good_layout(tmp_path)
        (layout / "index.json").write_text(json.dumps({"manifests": [{"digest": digest}, {"digest": digest}]}))
        with pytest.raises(LayoutRejected, match="exactly one manifest"):
            validate_layout(layout)

    def test_a_manifest_that_is_not_an_images_is_refused(self, tmp_path: Path) -> None:
        layout, _, _ = _good_layout(tmp_path)
        _index(layout, f"sha256:{_blob(layout, json.dumps({'schemaVersion': 2, 'manifests': []}).encode())}")
        with pytest.raises(LayoutRejected, match="not an image manifest"):
            validate_layout(layout)

    def test_an_oversized_index_is_refused_before_it_is_parsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layout, _, _ = _good_layout(tmp_path)
        monkeypatch.setattr(push, "_MAX_JSON_BYTES", 16)
        with pytest.raises(LayoutRejected, match="index.json is larger than 16 bytes"):
            validate_layout(layout)


class TestTheLayoutDirectory:
    def test_a_symlinked_layout_directory_is_refused(self, tmp_path: Path) -> None:
        real, _, _ = _good_layout(tmp_path)
        link = tmp_path / "out-img"
        link.symlink_to(real)
        with pytest.raises(LayoutRejected, match="is a symlink"):
            validate_layout(link)

    def test_a_missing_layout_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LayoutRejected, match="no layout"):
            validate_layout(tmp_path / "never-built")
