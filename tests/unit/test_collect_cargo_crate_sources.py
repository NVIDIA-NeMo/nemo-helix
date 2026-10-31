# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import struct
import sys
import zlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-cargo-crate-sources.py"
LOCK_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


def load_module():
    spec = spec_from_file_location("collect_cargo_crate_sources", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def elf_with_sections(sections: dict[str, bytes]) -> bytes:
    """A 64-bit little-endian ELF file holding just the named sections and a section-name table."""
    names = b"\x00" + b"".join(name.encode() + b"\x00" for name in [*sections, ".shstrtab"])
    payload = bytearray(b"\x00" * 64)
    offsets = []
    for content in [*sections.values(), names]:
        offsets.append(len(payload))
        payload += content
    section_offset = len(payload)
    headers = bytearray(b"\x00" * 64)  # the null section
    name_offset = 1
    for index, (name, content) in enumerate([*sections.items(), (".shstrtab", names)]):
        headers += struct.pack("<IIQQQQIIQQ", name_offset, 1, 0, 0, offsets[index], len(content), 0, 0, 1, 0)
        name_offset += len(name) + 1
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header += struct.pack(
        "<HHIQQQIHHHHHH", 2, 62, 1, 0, 0, section_offset, 0, 64, 0, 0, 64, len(sections) + 2, len(sections) + 1
    )
    payload[:64] = header
    return bytes(payload + headers)


def audit_section(packages: list[dict]) -> bytes:
    return zlib.compress(json.dumps({"packages": packages}).encode())


PACKAGES = [
    {"name": "uv", "version": "0.13.4", "source": "local", "root": True},
    {"name": "anyhow", "version": "1.0.102", "source": "crates.io"},
    {"name": "serde", "version": "1.0.228", "source": "crates.io"},
]


def cargo_lock(checksums: dict[str, str]) -> str:
    entries = ['[[package]]\nname = "uv"\nversion = "0.13.4"\n']
    for name, version in [("anyhow", "1.0.102"), ("serde", "1.0.228")]:
        if name in checksums:
            entries.append(
                f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = "{LOCK_SOURCE}"\n'
                f'checksum = "{checksums[name]}"\n'
            )
    return "version = 4\n\n" + "\n".join(entries)


CRATES = {"anyhow": b"anyhow crate", "serde": b"serde crate"}


def run(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lock: str,
    bodies: dict[str, bytes],
    packages: list[dict] = PACKAGES,
) -> Path:
    binary = tmp_path / "uv"
    binary.write_bytes(elf_with_sections({".text": b"\x90" * 8, ".dep-v0": audit_section(packages)}))
    (tmp_path / "Cargo.lock").write_text(lock)
    output = tmp_path / "out"
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(url: str, timeout: int) -> Response:
        return Response(bodies[url.rsplit("/", 2)[1]])

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--binary", str(binary), "--cargo-lock", str(tmp_path / "Cargo.lock"), "--output", str(output)],
    )
    module.main()
    return output


def test_collects_each_audited_crates_io_crate_verified_against_cargo_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    lock = cargo_lock({name: hashlib.sha256(body).hexdigest() for name, body in CRATES.items()})

    output = run(module, monkeypatch, tmp_path, lock, CRATES)

    assert (output / "crates" / "anyhow-1.0.102.crate").read_bytes() == b"anyhow crate"
    assert (output / "crates" / "serde-1.0.228.crate").read_bytes() == b"serde crate"
    assert not (output / "crates" / "uv-0.13.4.crate").exists()
    assert (output / "manifests" / "local-cargo-crates.txt").read_text() == f"uv==0.13.4\t\t{tmp_path / 'uv'}\n"
    assert (output / "manifests" / "downloaded-cargo-crates.txt").read_text().splitlines() == [
        f"anyhow==1.0.102\tanyhow-1.0.102.crate\t{tmp_path / 'uv'}",
        f"serde==1.0.228\tserde-1.0.228.crate\t{tmp_path / 'uv'}",
    ]


def test_a_crate_missing_from_cargo_lock_fails_the_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    lock = cargo_lock({"anyhow": hashlib.sha256(CRATES["anyhow"]).hexdigest()})

    with pytest.raises(RuntimeError, match="no checksum for crates the binary uses: serde 1.0.228"):
        run(module, monkeypatch, tmp_path, lock, CRATES)


def test_a_crate_that_does_not_match_cargo_lock_fails_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    lock = cargo_lock({name: hashlib.sha256(body).hexdigest() for name, body in CRATES.items()})

    with pytest.raises(RuntimeError, match="sha256 mismatch for anyhow-1.0.102.crate"):
        run(module, monkeypatch, tmp_path, lock, {**CRATES, "anyhow": b"tampered"})


def test_a_binary_without_audit_data_is_refused() -> None:
    module = load_module()

    with pytest.raises(RuntimeError, match="no .dep-v0 section"):
        module.audited_crates(elf_with_sections({".text": b"\x90"}))


@pytest.mark.parametrize("source", ["git", "registry"])
def test_a_crate_from_git_or_another_registry_fails_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    module = load_module()
    lock = cargo_lock({name: hashlib.sha256(body).hexdigest() for name, body in CRATES.items()})
    packages = [*PACKAGES, {"name": "forked", "version": "0.1.0", "source": source}]

    with pytest.raises(RuntimeError, match=rf"can't fetch: forked 0.1.0 \({source}\)"):
        run(module, monkeypatch, tmp_path, lock, CRATES, packages)


def test_a_cargo_lock_from_another_release_of_the_project_fails_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    lock = cargo_lock({name: hashlib.sha256(body).hexdigest() for name, body in CRATES.items()})

    with pytest.raises(RuntimeError, match="not the source the binary was built from: uv 0.11.26"):
        run(module, monkeypatch, tmp_path, lock.replace('version = "0.11.26"', 'version = "0.11.27"'), CRATES)


def test_cargo_lock_is_read_from_a_project_source_archive(tmp_path: Path) -> None:
    import io
    import tarfile

    module = load_module()
    lock = cargo_lock({"anyhow": "a" * 64})
    archive = tmp_path / "uv-0.11.26.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in [("uv-0.11.26/crates/uv/Cargo.lock", "nested"), ("uv-0.11.26/Cargo.lock", lock)]:
            info = tarfile.TarInfo(name)
            info.size = len(content.encode())
            tar.addfile(info, io.BytesIO(content.encode()))

    assert module.read_cargo_lock(archive) == lock
