# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout validation -- the only thing between untrusted bytes and the registry credential.

Every case here is something a sandbox could actually produce: it ran caller-authored ``RUN`` and
it had write access to this directory. The push step mounts the work volume *whole*, so a
followed symlink genuinely reaches another job's slice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from nemo_builder_plugin.run.push import LayoutRejected, validate_layout


def _blob(layout: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / digest).write_bytes(content)
    return digest


def _good_layout(tmp_path: Path) -> tuple[Path, str]:
    layout = tmp_path / "img"
    layout.mkdir()
    (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))

    manifest = json.dumps({"schemaVersion": 2, "layers": []}).encode()
    manifest_digest = _blob(layout, manifest)
    (layout / "index.json").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "manifests": [
                    {
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": f"sha256:{manifest_digest}",
                        "size": len(manifest),
                    }
                ],
            }
        )
    )
    return layout, f"sha256:{manifest_digest}"


class TestAcceptsAWellFormedLayout:
    def test_returns_the_manifest_digest(self, tmp_path: Path) -> None:
        layout, expected = _good_layout(tmp_path)
        assert validate_layout(layout) == expected


class TestRefusesHostileLayouts:
    def test_a_symlink_is_refused_rather_than_resolved(self, tmp_path: Path) -> None:
        """There is no legitimate symlink in an OCI layout, which makes refusal both safe and
        simple. Resolving one and checking where it points is a race and a bigger surface."""
        layout, _ = _good_layout(tmp_path)
        secret = tmp_path / "another-jobs-file"
        secret.write_text("not yours")
        (layout / "blobs" / "sha256" / "escape").symlink_to(secret)

        with pytest.raises(LayoutRejected, match="symlink"):
            validate_layout(layout)

    def test_a_blob_that_misdescribes_itself_is_refused(self, tmp_path: Path) -> None:
        """Filename IS the content hash. A mismatch means the layout is lying about what it holds."""
        layout, _ = _good_layout(tmp_path)
        blobs = layout / "blobs" / "sha256"
        victim = next(p for p in blobs.iterdir())
        victim.write_bytes(b"swapped after the fact")

        with pytest.raises(LayoutRejected, match="hashes to"):
            validate_layout(layout)

    def test_an_index_naming_an_absent_manifest_is_refused(self, tmp_path: Path) -> None:
        layout, _ = _good_layout(tmp_path)
        (layout / "index.json").write_text(
            json.dumps({"schemaVersion": 2, "manifests": [{"digest": "sha256:" + "0" * 64}]})
        )
        with pytest.raises(LayoutRejected, match="not in the layout"):
            validate_layout(layout)

    def test_an_unexpected_file_is_refused(self, tmp_path: Path) -> None:
        """ "Unexpected" in a directory written by untrusted code is a refusal, not a warning."""
        layout, _ = _good_layout(tmp_path)
        (layout / "run-me.sh").write_text("#!/bin/sh\n")
        with pytest.raises(LayoutRejected, match="unexpected file"):
            validate_layout(layout)

    def test_a_non_sha256_digest_is_refused(self, tmp_path: Path) -> None:
        layout, _ = _good_layout(tmp_path)
        (layout / "index.json").write_text(json.dumps({"manifests": [{"digest": "md5:whatever"}]}))
        with pytest.raises(LayoutRejected, match="non-sha256"):
            validate_layout(layout)

    def test_a_multi_manifest_index_is_refused(self, tmp_path: Path) -> None:
        """This execution mode builds one platform per spec, so an index with two entries is not
        something the design produces -- and publishing an unexpected shape is how a caller gets
        to influence what the digest means."""
        layout, digest = _good_layout(tmp_path)
        (layout / "index.json").write_text(json.dumps({"manifests": [{"digest": digest}, {"digest": digest}]}))
        with pytest.raises(LayoutRejected, match="exactly one manifest"):
            validate_layout(layout)

    def test_a_missing_marker_file_is_refused(self, tmp_path: Path) -> None:
        layout, _ = _good_layout(tmp_path)
        (layout / "oci-layout").unlink()
        with pytest.raises(LayoutRejected, match="oci-layout"):
            validate_layout(layout)

    def test_a_missing_layout_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LayoutRejected, match="no layout"):
            validate_layout(tmp_path / "never-built")
