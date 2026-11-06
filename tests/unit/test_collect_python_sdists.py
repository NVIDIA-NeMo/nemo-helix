# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-python-sdists.py"


def load_module():
    spec = spec_from_file_location("collect_python_sdists", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unpack(cache: Path, archive: str, name: str, version: str) -> None:
    dist_info = cache / "archive-v0" / archive / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\nA package.\n")


def test_lists_each_distribution_unpacked_in_a_uv_cache(tmp_path: Path) -> None:
    module = load_module()
    unpack(tmp_path, "AbC", "setuptools", "84.2.1")
    unpack(tmp_path, "gHi", "setuptools", "80.9.0")
    unpack(tmp_path, "dEf", "Jaraco.Text", "4.0.0")
    (tmp_path / "wheels-v5" / "pypi").mkdir(parents=True)

    assert module.uv_cache_packages(tmp_path) == {
        ("jaraco-text", "4.0.0"): ("Jaraco.Text", "4.0.0"),
        ("setuptools", "80.9.0"): ("setuptools", "80.9.0"),
        ("setuptools", "84.2.1"): ("setuptools", "84.2.1"),
    }


def test_collects_the_sdists_of_a_uv_caches_distributions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    cache = tmp_path / "cache"
    unpack(cache, "AbC", "setuptools", "84.2.1")
    unpack(cache, "gHi", "setuptools", "80.9.0")
    output = tmp_path / "out"
    downloads: list[str] = []
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(
        module, "pypi_sdist", lambda name, version: module.Sdist(f"https://files.example/{name}-{version}.tar.gz")
    )
    monkeypatch.setattr(module, "download", lambda url, destination, *hashes: downloads.append(url))
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--uv-cache", str(cache), "--output", str(output), "--label", "uv-cache"]
    )

    assert module.main() == 0

    assert downloads == [
        "https://files.example/setuptools-80.9.0.tar.gz",
        "https://files.example/setuptools-84.2.1.tar.gz",
    ]
    manifest = (output / "manifests" / "downloaded-pypi-sdists-uv-cache.txt").read_text().splitlines()
    assert [line.split("\t")[:3] for line in manifest] == [
        ["setuptools==80.9.0", "setuptools-80.9.0.tar.gz", "pypi-json"],
        ["setuptools==84.2.1", "setuptools-84.2.1.tar.gz", "pypi-json"],
    ]


def test_a_uv_cache_skips_baseline_packages_however_their_names_are_spelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    cache = tmp_path / "cache"
    unpack(cache, "AbC", "Jaraco.Text", "4.0.0")
    unpack(cache, "dEf", "setuptools", "84.2.1")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("jaraco-text==4.0.0\nsetuptools==80.9.0\n")
    downloads: list[str] = []
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(
        module, "pypi_sdist", lambda name, version: module.Sdist(f"https://files.example/{name}-{version}.tar.gz")
    )
    monkeypatch.setattr(module, "download", lambda url, destination, *hashes: downloads.append(url))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--uv-cache", str(cache), "--output", str(tmp_path / "out"), "--baseline-freeze", str(baseline)],
    )

    assert module.main() == 0

    assert downloads == ["https://files.example/setuptools-84.2.1.tar.gz"]


@pytest.mark.parametrize("args", [[], ["--python", "python3", "--uv-cache", "cache"]])
def test_exactly_one_of_an_environment_or_a_uv_cache_is_collected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args, "--output", str(tmp_path)])

    with pytest.raises(SystemExit):
        module.main()


def test_an_environment_skips_local_skipped_and_baseline_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    purelib = tmp_path / "site-packages"
    local = purelib / "workspace_pkg-0.1.0.dist-info"
    local.mkdir(parents=True)
    (local / "direct_url.json").write_text('{"url": "file:///app/packages/workspace_pkg"}')
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("pip==26.2.2\n")
    frozen = "workspace-pkg==0.1.0\npip==26.2.2\nray==2.58.0\nidna==3.19\n"
    downloads: list[str] = []
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(module, "purelib_for", lambda python: purelib)
    monkeypatch.setattr(module, "freeze_packages", lambda python, path: module.parse_freeze(frozen))
    monkeypatch.setattr(
        module, "pypi_sdist", lambda name, version: module.Sdist(f"https://files.example/{name}-{version}.tar.gz")
    )
    monkeypatch.setattr(module, "download", lambda url, destination, *hashes: downloads.append(url))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            *("--python", "python3", "--output", str(tmp_path / "out")),
            *("--skip-package", "Ray", "--baseline-freeze", str(baseline)),
        ],
    )

    assert module.main() == 0

    assert downloads == ["https://files.example/idna-3.19.tar.gz"]
