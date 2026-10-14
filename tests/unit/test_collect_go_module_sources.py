# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-go-module-sources.py"
SENTINEL = b"\x30\x77\xaf\x0c\x92\x74\x08\x02\x41\xe1\xc1\x07\xe6\xd6\x18\xe6"


def load_module():
    spec = spec_from_file_location("collect_go_module_sources", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def uvarint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def go_binary(modinfo: str, go_version: str = "go1.26.8") -> bytes:
    """Bytes laid out like a Go 1.18+ binary's build information, at a 16-byte-aligned offset."""
    header = b"\xff Go buildinf:" + bytes([8, 2]) + bytes(16)
    body = SENTINEL + modinfo.encode() + SENTINEL
    return b"\x00" * 48 + header + uvarint(len(go_version)) + go_version.encode() + uvarint(len(body)) + body


def module_zip(prefix: str, files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"{prefix}/{name}", content)
    return buffer.getvalue()


MODINFO = "\n".join(
    [
        "path\tgithub.com/osscontainertools/kaniko/cmd/executor",
        "mod\tgithub.com/osscontainertools/kaniko\t(devel)\t",
        "dep\tgithub.com/Azure/go-autorest\tv14.2.0+incompatible\th1:recorded=",
        "dep\tgolang.org/x/sys\tv0.47.0\t",
        "dep\tgithub.com/google/go-containerregistry\tv0.0.0-20260904000721-8a72a424fdec+dirty\t",
        "dep\tgithub.com/docker/docker\tv28.0.0+incompatible\th1:old=",
        "=>\tgithub.com/moby/moby\tv28.1.0+incompatible\th1:replaced=",
        "build\t-ldflags=-w -s",
    ]
)


def test_reads_go_version_and_proxy_modules_with_replacements_applied(tmp_path: Path) -> None:
    """The (devel) main module and a +dirty build are not on the proxy; they are collected as pinned archives."""
    module = load_module()
    binary = tmp_path / "executor"
    binary.write_bytes(go_binary(MODINFO))

    modules, used_by, go_versions = module.linked_modules([binary])

    assert go_versions == {"go1.26.8"}
    assert [(m.path, m.version, m.sum) for m in modules] == [
        ("github.com/Azure/go-autorest", "v14.2.0+incompatible", "h1:recorded="),
        ("github.com/moby/moby", "v28.1.0+incompatible", "h1:replaced="),
        ("golang.org/x/sys", "v0.47.0", ""),
    ]
    assert all(binaries == [str(binary)] for binaries in used_by.values())


def test_a_module_used_by_two_binaries_is_collected_once_with_the_recorded_hash(tmp_path: Path) -> None:
    module = load_module()
    unhashed = tmp_path / "executor"
    unhashed.write_bytes(go_binary("dep\tgolang.org/x/sys\tv0.47.0\t"))
    hashed = tmp_path / "crane"
    hashed.write_bytes(go_binary("dep\tgolang.org/x/sys\tv0.47.0\th1:abc="))

    modules, used_by, _ = module.linked_modules([unhashed, hashed])

    assert [(m.path, m.sum) for m in modules] == [("golang.org/x/sys", "h1:abc=")]
    assert used_by[modules[0]] == [str(unhashed), str(hashed)]


def test_a_module_replaced_by_a_local_directory_is_refused(tmp_path: Path) -> None:
    module = load_module()
    binary = tmp_path / "launcher"
    binary.write_bytes(go_binary("dep\texample.com/lib\tv1.0.0\th1:x=\n=>\t../lib\t\t"))

    with pytest.raises(RuntimeError, match="replaced by a local directory"):
        module.linked_modules([binary])


def test_module_paths_are_case_escaped_for_the_proxy() -> None:
    assert load_module().escape_module_path("github.com/Azure/go-autorest") == "github.com/!azure/go-autorest"


def test_module_zip_hash_covers_every_file_name_and_content() -> None:
    module = load_module()
    original = module_zip("example.com/lib@v1.0.0", {"go.mod": b"module example.com/lib\n", "lib.go": b"package lib\n"})
    edited = module_zip("example.com/lib@v1.0.0", {"go.mod": b"module example.com/lib\n", "lib.go": b"package lib2\n"})

    assert module.module_zip_hash(original).startswith("h1:")
    assert module.module_zip_hash(original) != module.module_zip_hash(edited)


def run_collector(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, modinfo: str, responses: dict) -> Path:
    binary = tmp_path / "bin"
    binary.write_bytes(go_binary(modinfo))
    output = tmp_path / "out"
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")

    def fake_fetch(url: str, timeout: int = 120) -> bytes:
        if url not in responses:
            pytest.fail(f"unexpected download {url}")
        return responses[url]

    monkeypatch.setattr(module, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--binary", str(binary), "--output", str(output)])
    module.main()
    return output


def go_release_index(module) -> dict:
    source = b"go source"
    return {
        f"{module.GO_DOWNLOADS}?mode=json&include=all": json.dumps(
            [{"files": [{"filename": "go1.26.8.src.tar.gz", "sha256": hashlib.sha256(source).hexdigest()}]}]
        ).encode(),
        f"{module.GO_DOWNLOADS}go1.26.8.src.tar.gz": source,
    }


def test_collects_modules_verified_against_recorded_and_checksum_database_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    recorded_zip = module_zip("example.com/a@v1.0.0", {"a.go": b"package a\n"})
    unrecorded_zip = module_zip("example.com/b@v2.0.0", {"b.go": b"package b\n"})
    responses = {
        f"{module.GO_PROXY}/example.com/a/@v/v1.0.0.zip": recorded_zip,
        f"{module.GO_PROXY}/example.com/b/@v/v2.0.0.zip": unrecorded_zip,
        f"{module.GO_SUMDB}/lookup/example.com/b@v2.0.0": (
            f"123\nexample.com/b v2.0.0 {module.module_zip_hash(unrecorded_zip)}\nexample.com/b v2.0.0/go.mod h1:mod=\n"
        ).encode(),
        **go_release_index(module),
    }
    modinfo = f"dep\texample.com/a\tv1.0.0\t{module.module_zip_hash(recorded_zip)}\ndep\texample.com/b\tv2.0.0\t"

    output = run_collector(module, monkeypatch, tmp_path, modinfo, responses)

    assert (output / "go-modules" / "example.com_a@v1.0.0.zip").read_bytes() == recorded_zip
    assert (output / "go-modules" / "example.com_b@v2.0.0.zip").read_bytes() == unrecorded_zip
    assert (output / "go-toolchain" / "go1.26.8.src.tar.gz").read_bytes() == b"go source"
    assert (output / "manifests" / "downloaded-go-modules.txt").read_text().splitlines() == [
        f"example.com/a@v1.0.0\texample.com_a@v1.0.0.zip\t{tmp_path / 'bin'}",
        f"example.com/b@v2.0.0\texample.com_b@v2.0.0.zip\t{tmp_path / 'bin'}",
    ]


def test_a_module_zip_that_does_not_match_its_hash_fails_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    responses = {
        f"{module.GO_PROXY}/example.com/a/@v/v1.0.0.zip": module_zip("example.com/a@v1.0.0", {"a.go": b"tampered\n"}),
        **go_release_index(module),
    }

    with pytest.raises(RuntimeError, match="example.com/a@v1.0.0: expected h1:recorded="):
        run_collector(module, monkeypatch, tmp_path, "dep\texample.com/a\tv1.0.0\th1:recorded=", responses)


def test_disabled_collection_reads_and_downloads_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    monkeypatch.delenv("NHX_COLLECT_SOURCES", raising=False)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--binary", str(tmp_path / "absent"), "--output", str(output)])

    assert module.main() == 0

    assert (output / "manifests" / "source-collection-disabled.txt").exists()
