# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-pinned-sources.py"
ARCHIVE = b"busybox source"


def load_module():
    spec = spec_from_file_location("collect_pinned_sources", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, *, sha256: str = hashlib.sha256(ARCHIVE).hexdigest(), url: str | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "busybox",
                        "version": "1.38.0",
                        "license": "GPL-2.0-only",
                        "url": url or "https://busybox.net/downloads/busybox-1.38.0.tar.bz2",
                        "sha256": sha256,
                        "covers": ["binary:busybox@1.38.0"],
                    }
                ]
            }
        )
    )
    return path


class Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


def run(module, monkeypatch: pytest.MonkeyPatch, manifest: Path, output: Path) -> int:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--manifest", str(manifest), "--output", str(output)])
    return module.main()


def test_downloads_verified_archive_and_records_what_it_covers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url, timeout: Response(ARCHIVE))
    output = tmp_path / "out"

    assert run(module, monkeypatch, write_manifest(tmp_path / "m.json"), output) == 0

    assert (output / "pinned" / "busybox-1.38.0.tar.bz2").read_bytes() == ARCHIVE
    assert (output / "manifests" / "downloaded-pinned-sources.txt").read_text() == (
        "busybox==1.38.0\tbusybox-1.38.0.tar.bz2\thttps://busybox.net/downloads/busybox-1.38.0.tar.bz2"
        "\tGPL-2.0-only\tbinary:busybox@1.38.0\n"
    )


def test_checksum_mismatch_fails_without_writing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url, timeout: Response(b"tampered"))
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        run(module, monkeypatch, write_manifest(tmp_path / "m.json"), output)

    assert not (output / "pinned" / "busybox-1.38.0.tar.bz2").exists()


def test_download_failure_fails_the_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(module, "sleep", lambda seconds: None)

    def unreachable(url: str, timeout: int) -> Response:
        raise module.urllib.error.URLError("TLS connect error")

    monkeypatch.setattr(module.urllib.request, "urlopen", unreachable)

    with pytest.raises(RuntimeError, match="could not download busybox 1.38.0"):
        run(module, monkeypatch, write_manifest(tmp_path / "m.json"), tmp_path / "out")


def test_disabled_collection_downloads_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    monkeypatch.delenv("NHX_COLLECT_SOURCES", raising=False)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url, timeout: pytest.fail("downloaded"))
    output = tmp_path / "out"

    assert run(module, monkeypatch, write_manifest(tmp_path / "m.json"), output) == 0

    assert (output / "manifests" / "source-collection-disabled.txt").exists()
    assert not (output / "pinned").exists()


@pytest.mark.parametrize(
    ("sha256", "url", "error"),
    [
        ("not-a-digest", None, "invalid sha256"),
        (hashlib.sha256(ARCHIVE).hexdigest(), "http://busybox.net/downloads/busybox-1.38.0.tar.bz2", "https"),
    ],
)
def test_manifest_is_validated_even_when_collection_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sha256: str, url: str | None, error: str
) -> None:
    module = load_module()
    monkeypatch.delenv("NHX_COLLECT_SOURCES", raising=False)

    with pytest.raises(ValueError, match=error):
        run(module, monkeypatch, write_manifest(tmp_path / "m.json", sha256=sha256, url=url), tmp_path / "out")


def test_release_archives_are_named_after_the_component() -> None:
    module = load_module()
    source = module.PinnedSource(
        name="kaniko",
        version="1.28.5",
        license="Apache-2.0",
        url="https://github.com/osscontainertools/kaniko/archive/refs/tags/v1.28.5.tar.gz",
        sha256="0" * 64,
        covers=(),
    )

    assert source.filename == "kaniko-1.28.5.tar.gz"


def pinned(name: str, covers: list[str], bundled_in: str | None = None):
    return load_module().PinnedSource(
        name, "1", "MIT", "https://example.com/a.tar.gz", "0" * 64, tuple(covers), bundled_in
    )


def test_a_python_pin_behind_the_installed_version_is_stale() -> None:
    module = load_module()
    sources = [pinned("ray", ["python:ray@2.57.0"]), pinned("launcher", ["dotnet:Simple Launcher@1.1.0.14"])]

    assert module.stale_python_pins(sources, {"ray": {"2.58.0"}, "simple-launcher": {"2"}}) == [
        "ray pins ray 2.57.0, but 2.58.0 is installed"
    ]


def test_a_python_pin_that_is_installed_absent_or_bundled_elsewhere_is_not_stale() -> None:
    module = load_module()
    sources = [
        pinned("ray", ["python:Ray@2.58.0"]),
        pinned("setuptools", ["python:setuptools@84.0.0"]),
        pinned("colorama", ["python:colorama@0.4.6"], bundled_in="ray"),
    ]

    assert module.stale_python_pins(sources, {"ray": {"2.57.0", "2.58.0"}, "colorama": {"0.4.7"}}) == []


def test_a_stale_pin_fails_the_build_even_when_collection_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "ray",
                        "version": "2.57.0",
                        "license": "Apache-2.0",
                        "url": "https://example.com/ray.tar.gz",
                        "sha256": "0" * 64,
                        "covers": ["python:ray@2.57.0"],
                    }
                ]
            }
        )
    )
    monkeypatch.delenv("NHX_COLLECT_SOURCES", raising=False)
    monkeypatch.setattr(module, "installed_versions", lambda python: {"ray": {"2.58.0"}})
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--manifest", str(manifest), "--output", str(tmp_path / "out"), "--installed-in", "python3"],
    )

    with pytest.raises(RuntimeError, match="ray pins ray 2.57.0, but 2.58.0 is installed"):
        module.main()
