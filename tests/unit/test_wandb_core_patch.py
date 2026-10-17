# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import platform
import shutil
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def installer():
    spec = spec_from_file_location("install_wandb_core", ROOT / "docker/scripts/install-wandb-core.py")
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "artifact"
    path.mkdir()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}[platform.machine()]
    binary = bytearray(64)
    binary[:6] = b"\x7fELF\x02\x01"
    binary[18:20] = {"amd64": 62, "arm64": 183}[arch].to_bytes(2, "little")
    (path / "wandb-core").write_bytes(binary)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "arch": arch,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "wandb_version": "0.30.0",
                "go_version": "go1.27.2",
                "x_net_version": "v0.60.0",
            }
        )
    )
    return path


def package(parent: Path, version: str = "0.30.0") -> Path:
    binary = parent / "wandb/bin/wandb-core"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"original core")
    info = parent / f"wandb-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: wandb\nVersion: {version}\n")
    (info / "RECORD").write_text("wandb/bin/wandb-core,sha256=old,13\nother,,\n")
    return binary


def test_installs_all_copies_and_retained_cache_without_modifying_build_cache(tmp_path: Path) -> None:
    module = installer()
    patch = artifact(tmp_path)
    root = tmp_path / "image"
    active = package(root / "opt/venv/site-packages")
    cached = package(root / "opt/uv_cache/archive-v0/entry")
    venv = root / "opt/ray_venvs/vllm/site-packages"
    venv.mkdir(parents=True)
    (venv / "wandb").symlink_to(cached.parent.parent, target_is_directory=True)
    (venv / "wandb-0.30.0.dist-info").symlink_to(cached.parents[2] / "wandb-0.30.0.dist-info", target_is_directory=True)
    external_cache = tmp_path / "build-cache-core"
    os.link(active, external_cache)
    wheel = root / "opt/uv_cache/wandb-0.30.0-py3-none-any.whl"
    wheel.write_bytes(b"original compressed wheel")
    assert module.install(patch, [root]) == 3
    assert active.read_bytes() == cached.read_bytes() == (patch / "wandb-core").read_bytes()
    assert external_cache.read_bytes() == b"original core"
    assert not wheel.exists()
    assert module.install(patch, [root], verify_only=True) == 3
    assert module.install(patch, [root]) == 3  # idempotent


def test_failed_atomic_replace_keeps_original_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "wandb-core"
    target.write_bytes(b"original core")

    def fail_replace(*args) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        installer().replace(target, b"patched core", 0o755)
    assert target.read_bytes() == b"original core"
    assert list(tmp_path.iterdir()) == [target]


def test_mismatched_retained_version_fails_before_any_changes(tmp_path: Path) -> None:
    module = installer()
    patch = artifact(tmp_path)
    root = tmp_path / "image"
    active = package(root / "a")
    package(root / "cache/archive-v0/old", "0.29.0")
    with pytest.raises(ValueError, match="version mismatch"):
        module.install(patch, [root])
    assert active.read_bytes() == b"original core"


@pytest.mark.parametrize(
    "field,value", [("sha256", "bad"), ("arch", "wrong"), ("go_version", "go1.27.1"), ("x_net_version", "v0.59.0")]
)
def test_rejects_invalid_artifact(tmp_path: Path, field: str, value: str) -> None:
    patch = artifact(tmp_path)
    manifest = json.loads((patch / "manifest.json").read_text())
    manifest[field] = value
    (patch / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        installer().install(patch, [], optional=True)


def test_no_wandb_requires_explicit_optional(tmp_path: Path) -> None:
    patch = artifact(tmp_path)
    with pytest.raises(ValueError, match="No installed"):
        installer().install(patch, [tmp_path / "missing"])
    assert installer().install(patch, [tmp_path / "missing"], optional=True) == 0


def test_verify_only_detects_original_binary(tmp_path: Path) -> None:
    patch = artifact(tmp_path)
    root = tmp_path / "image"
    package(root)
    with pytest.raises(ValueError, match="Unpatched"):
        installer().install(patch, [root], verify_only=True)


def test_missing_installed_core_is_not_silently_skipped(tmp_path: Path) -> None:
    patch = artifact(tmp_path)
    root = tmp_path / "image"
    package(root).unlink()
    with pytest.raises(ValueError, match="missing or a broken symlink"):
        installer().install(patch, [root], optional=True)


def test_symlink_to_non_retained_cache_is_rejected(tmp_path: Path) -> None:
    patch = artifact(tmp_path)
    root = tmp_path / "image"
    binary = package(root)
    outside = tmp_path / "build-only-cache"
    outside.write_bytes(binary.read_bytes())
    binary.unlink()
    binary.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the retained image roots"):
        installer().install(patch, [root])
    assert outside.read_bytes() == b"original core"


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is required to resolve Bake definitions")
def test_all_bake_consumers_have_the_shared_named_context() -> None:
    result = subprocess.run(
        ["docker", "buildx", "bake", "--print", "*"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    targets = json.loads(result.stdout)["target"]
    consumers = set()
    for name, target in targets.items():
        dockerfile = ROOT / target.get("dockerfile", "Dockerfile")
        if name == "wandb-core-artifacts":
            continue  # Its Dockerfile also contains the consuming smoke stage.
        if dockerfile.is_file() and "--mount=from=wandb-core-artifacts," in dockerfile.read_text():
            consumers.add(name)
            assert target["contexts"]["wandb-core-artifacts"] == "target:wandb-core-artifacts", name
    assert len(consumers) >= 20
    assert targets["wandb-core-artifacts"]["target"] == "artifacts"
