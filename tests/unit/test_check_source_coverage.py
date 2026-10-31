# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "check-source-coverage.py"
SITE = "/app/.venv/lib/python3.13/site-packages"


def load_module():
    spec = spec_from_file_location("check_source_coverage", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def artifact(kind: str, name: str, version: str, path: str, **metadata) -> dict:
    return {"type": kind, "name": name, "version": version, "locations": [{"path": path}], "metadata": metadata}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    root = tmp_path / "nhx-tasks"
    write(root / "manifests" / "downloaded-pypi-sdists-app-venv.txt", "requests==2.32.5\trequests-2.32.5.tar.gz\n")
    write(root / "manifests" / "downloaded-pypi-sdists-gym.txt", "setuptools==80.9.0\tsetuptools-80.9.0.tar.gz\n")
    write(root / "workspace" / "nemo_evals_plugin-0.0.0.tar.gz", "")
    write(root / "apt" / "manifests" / "downloaded-apt-sources.txt", "glibc=2.41-12\n")
    write(
        root / "manifests" / "downloaded-go-modules.txt",
        "golang.org/x/sys@v0.47.0\tgolang.org_x_sys@v0.47.0.zip\t/bin/crane\n",
    )
    write(root / "manifests" / "downloaded-go-toolchain-sources.txt", "go1.26.8\tgo1.26.8.src.tar.gz\n")
    write(root / "manifests" / "downloaded-cargo-crates.txt", "anyhow==1.0.102\tanyhow-1.0.102.crate\t/usr/bin/uv\n")
    write(root / "manifests" / "local-cargo-crates.txt", "uv-cache==0.0.1\t\t/usr/bin/uv\n")
    write(root / "manifests" / "downloaded-cpython-source.txt", "Python-3.13.15.tgz\thttps://python.org\n")
    write(
        root / "manifests" / "downloaded-pinned-sources.txt",
        "busybox==1.37.0\tbusybox-1.37.0.tar.bz2\thttps://busybox.net\tGPL-2.0-only\tbinary:busybox@1.37.0\n"
        "codex==0.144.4\tcodex-0.144.4.tar.gz\thttps://github.com\tApache-2.0\tpython:openai-codex-cli-bin@0.144.4\n",
    )
    return root


@pytest.fixture
def policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "first_party": {"python": [{"name": "nemo-fabric*"}], "npm": [{"name": "nemo-oo-agents-viewer"}]},
                "vendored": [{"path": "/setuptools/_vendor/", "covered_by": "setuptools"}],
                "embedded": [{"type": "go-module", "name": "github.com/wandb/wandb/core", "covered_by": "setuptools"}],
            }
        )
    )
    return path


COVERED = [
    artifact("python", "requests", "2.32.5", f"{SITE}/requests-2.32.5.dist-info/METADATA"),
    artifact("python", "nemo-evals-plugin", "0.0.0", f"{SITE}/nemo_evals_plugin-0.0.0.dist-info/METADATA"),
    artifact("python", "openai_codex_cli_bin", "0.144.4", f"{SITE}/openai_codex_cli_bin-0.144.4.dist-info/METADATA"),
    artifact("python", "nemo-fabric-adapters-claude", "0.4.0", f"{SITE}/nemo_fabric_adapters_claude/METADATA"),
    artifact("python", "more-itertools", "10.8.0", f"{SITE}/setuptools/_vendor/more_itertools/METADATA"),
    artifact("npm", "nemo-oo-agents-viewer", "1.0.0", f"{SITE}/nooa/viewer/package.json"),
    artifact("deb", "libc6", "2.41-12", "/var/lib/dpkg/status", source="glibc"),
    artifact("go-module", "golang.org/x/sys", "v0.47.0", "/bin/crane"),
    artifact("go-module", "stdlib", "go1.26.8", "/bin/crane"),
    artifact("go-module", "github.com/wandb/wandb/core", "(devel)", f"{SITE}/wandb/bin/wandb-core"),
    artifact("rust-crate", "anyhow", "1.0.102", "/usr/bin/uv"),
    artifact("rust-crate", "uv-cache", "0.0.1", "/usr/bin/uv"),
    artifact("binary", "busybox", "1.37.0", "/usr/bin/sh"),
    artifact("binary", "python", "3.13.15", "/usr/local/bin/python3.13"),
    artifact("python", "flask", "3.1.3", "/usr/share/nemo-helix/source-distributions/nhx-tasks/pypi/flask/METADATA"),
]


def run(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifacts: list[dict], sources: Path, policy: Path
) -> int:
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"artifacts": artifacts}))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--sbom", str(sbom), "--sources", str(sources), "--policy", str(policy)],
    )
    return module.main()


def test_every_collected_first_party_vendored_or_pinned_component_is_covered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sources: Path, policy: Path, capsys
) -> None:
    module = load_module()

    assert run(module, monkeypatch, tmp_path, COVERED, sources, policy) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "gap",
    [
        artifact("python", "requests", "2.32.4", f"{SITE}/requests-2.32.4.dist-info/METADATA"),
        artifact("python", "packaging", "26.0", f"{SITE}/pip/_vendor/packaging/METADATA"),
        artifact("deb", "libssl3", "3.5.4", "/var/lib/dpkg/status", source="openssl"),
        artifact("go-module", "golang.org/x/net", "v0.50.0", "/bin/crane"),
        artifact("go-module", "stdlib", "go1.26.7", "/tools/jobs-launcher"),
        artifact("go-module", "github.com/ray-project/ray/core", "(devel)", f"{SITE}/ray/core/raylet"),
        artifact("rust-crate", "serde", "1.0.228", "/usr/bin/uv"),
        artifact("binary", "busybox", "1.38.0", "/busybox/sh"),
        artifact("binary", "python", "3.12.11", "/usr/local/bin/python3.12"),
        artifact("dotnet", "Simple Launcher", "1.1.0.14", f"{SITE}/pip/_vendor/distlib/t64.exe"),
    ],
)
def test_a_component_without_collected_source_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sources: Path, policy: Path, capsys, gap: dict
) -> None:
    module = load_module()

    assert run(module, monkeypatch, tmp_path, [*COVERED, gap], sources, policy) == 1
    assert f"{gap['name']} {gap['version']}" in capsys.readouterr().out


def test_a_build_with_collection_disabled_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sources: Path, policy: Path, capsys
) -> None:
    module = load_module()
    write(sources / "manifests" / "source-collection-disabled.txt", "disabled\n")

    assert run(module, monkeypatch, tmp_path, COVERED, sources, policy) == 1
    assert "disabled" in capsys.readouterr().err
