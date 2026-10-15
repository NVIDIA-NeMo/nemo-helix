# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-npm-sources.py"
TARBALLS = {"react": b"react tarball", "scheduler": b"scheduler tarball", "@radix-ui/react-slot": b"slot tarball"}


def load_module():
    spec = spec_from_file_location("collect_npm_sources", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def integrity(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


def pnpm_list() -> list[dict]:
    react = {"version": "19.2.0", "resolved": "https://registry.npmjs.org/react/-/react-19.2.0.tgz"}
    return [
        {
            "name": "nemo-studio-ui",
            "dependencies": {
                "@nemo/sdk": {"version": "link:../sdk", "dependencies": {"react": react}},
                "@nvidia/foundations-react-core": {
                    "version": "1.13.0",
                    "dependencies": {
                        "@radix-ui/react-slot": {"version": "1.2.3"},
                        "react": react,
                    },
                },
                "react": react,
                "scheduler": {
                    "version": "0.27.0",
                    "resolved": "https://registry.npmjs.org/scheduler/-/scheduler-0.27.0.tgz",
                },
            },
        }
    ]


def pnpm_lock(integrities: dict[str, str]) -> str:
    return f"""lockfileVersion: '9.0'

importers:

  .:
    dependencies:
      react:
        specifier: 19.2.0
        version: 19.2.0

packages:

  '@nvidia/foundations-react-core@1.13.0':
    resolution: {{integrity: sha512-bnZpZGlh}}

  '@radix-ui/react-slot@1.2.3':
    resolution: {{integrity: {integrities["@radix-ui/react-slot"]}}}
    peerDependencies:
      react: ^19

  react@19.2.0:
    resolution: {{integrity: {integrities["react"]}}}
    engines: {{node: '>=0.10.0'}}

  scheduler@0.27.0:
    resolution: {{integrity: {integrities["scheduler"]}}}

snapshots:

  '@radix-ui/react-slot@1.2.3(react@19.2.0)':
    dependencies:
      react: 19.2.0

  react@19.2.0: {{}}
"""


def test_collects_third_party_packages_reached_through_workspace_and_excluded_packages() -> None:
    module = load_module()

    packages = module.production_packages(pnpm_list(), exclude=["@nvidia/*"])

    assert [(p.name, p.version, p.resolved) for p in packages] == [
        ("@radix-ui/react-slot", "1.2.3", "https://registry.npmjs.org/@radix-ui/react-slot/-/react-slot-1.2.3.tgz"),
        ("react", "19.2.0", "https://registry.npmjs.org/react/-/react-19.2.0.tgz"),
        ("scheduler", "0.27.0", "https://registry.npmjs.org/scheduler/-/scheduler-0.27.0.tgz"),
    ]


def test_a_package_first_listed_as_a_deduped_stub_still_has_its_dependencies_collected() -> None:
    module = load_module()
    # pnpm breaks the cycle core -> utilities -> core with a nested full copy, as it does for vitest.
    core = {
        "version": "6.3.1",
        "dependencies": {
            "@dnd-kit/accessibility": {"version": "3.1.1"},
            "@dnd-kit/utilities": {"version": "3.2.2", "deduped": True},
        },
    }
    listed = [
        {
            "name": "studio",
            "dependencies": {
                "@dnd-kit/sortable": {
                    "version": "10.0.0",
                    "dependencies": {"@dnd-kit/core": {"version": "6.3.1", "deduped": True}},
                },
                "@dnd-kit/utilities": {"version": "3.2.2", "dependencies": {"@dnd-kit/core": core}},
            },
        }
    ]

    packages = module.production_packages(listed, exclude=[])

    assert [(p.name, p.version) for p in packages] == [
        ("@dnd-kit/accessibility", "3.1.1"),
        ("@dnd-kit/core", "6.3.1"),
        ("@dnd-kit/sortable", "10.0.0"),
        ("@dnd-kit/utilities", "3.2.2"),
    ]


def test_an_aliased_dependency_is_collected_as_the_package_it_names() -> None:
    module = load_module()
    listed = [
        {
            "name": "studio",
            "dependencies": {
                "typescript": {"from": "@typescript/typescript6", "version": "6.0.2"},
                "@typescript/old": {"from": "typescript", "version": "6.0.3"},
            },
        }
    ]

    packages = module.production_packages(listed, exclude=[])

    assert [(p.name, p.version, p.resolved) for p in packages] == [
        (
            "@typescript/typescript6",
            "6.0.2",
            "https://registry.npmjs.org/@typescript/typescript6/-/typescript6-6.0.2.tgz",
        ),
        ("typescript", "6.0.3", "https://registry.npmjs.org/typescript/-/typescript-6.0.3.tgz"),
    ]


def test_reads_integrities_from_the_lock_packages_section_only() -> None:
    module = load_module()
    integrities = {name: integrity(body) for name, body in TARBALLS.items()}

    assert module.lock_integrities(pnpm_lock(integrities)) == {
        ("@nvidia/foundations-react-core", "1.13.0"): "sha512-bnZpZGlh",
        ("@radix-ui/react-slot", "1.2.3"): integrities["@radix-ui/react-slot"],
        ("react", "19.2.0"): integrities["react"],
        ("scheduler", "0.27.0"): integrities["scheduler"],
    }


class Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


def run(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bodies: dict[str, bytes]) -> Path:
    (tmp_path / "list.json").write_text(json.dumps(pnpm_list()))
    (tmp_path / "pnpm-lock.yaml").write_text(pnpm_lock({name: integrity(body) for name, body in TARBALLS.items()}))
    output = tmp_path / "out"
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    def fake_urlopen(url: str, timeout: int) -> Response:
        name = url.removeprefix("https://registry.npmjs.org/").split("/-/")[0]
        return Response(bodies[name])

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--pnpm-list",
            str(tmp_path / "list.json"),
            "--pnpm-lock",
            str(tmp_path / "pnpm-lock.yaml"),
            "--exclude",
            "@nvidia/*",
            "--label",
            "studio",
            "--output",
            str(output),
        ],
    )
    module.main()
    return output


def test_downloads_each_tarball_verified_against_the_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()

    output = run(module, monkeypatch, tmp_path, TARBALLS)

    assert (output / "npm" / "radix-ui_react-slot-1.2.3.tgz").read_bytes() == b"slot tarball"
    assert (output / "npm" / "react-19.2.0.tgz").read_bytes() == b"react tarball"
    assert (output / "manifests" / "downloaded-npm-packages-studio.txt").read_text().splitlines() == [
        "@radix-ui/react-slot@1.2.3\tradix-ui_react-slot-1.2.3.tgz",
        "react@19.2.0\treact-19.2.0.tgz",
        "scheduler@0.27.0\tscheduler-0.27.0.tgz",
    ]


def test_a_tarball_that_does_not_match_the_lock_fails_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()

    with pytest.raises(RuntimeError, match="react@19.2.0: expected sha512-"):
        run(module, monkeypatch, tmp_path, {**TARBALLS, "react": b"tampered"})


def test_disabled_collection_downloads_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    monkeypatch.delenv("NHX_COLLECT_SOURCES", raising=False)
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--pnpm-list", "absent", "--pnpm-lock", "absent", "--label", "studio", "--output", str(output)],
    )

    assert module.main() == 0

    assert (output / "manifests" / "source-collection-disabled.txt").exists()


def test_collects_the_non_dev_packages_of_an_npm_package_lock() -> None:
    module = load_module()
    lock = {
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "nemo-oo-agents-viewer", "version": "1.0.0"},
            "node_modules/react": {
                "version": "19.2.7",
                "resolved": "https://registry.npmjs.org/react/-/react-19.2.7.tgz",
                "integrity": "sha512-cmVhY3Q=",
            },
            "node_modules/react-router/node_modules/cookie-es": {
                "version": "3.1.1",
                "resolved": "https://registry.npmjs.org/cookie-es/-/cookie-es-3.1.1.tgz",
                "integrity": "sha512-Y29va2ll",
            },
            "node_modules/vite": {
                "version": "8.0.0",
                "resolved": "https://registry.npmjs.org/vite/-/vite-8.0.0.tgz",
                "integrity": "sha512-dml0ZQ==",
                "dev": True,
            },
        },
    }

    packages, integrities = module.package_lock_packages(lock, exclude=[])

    assert [(p.name, p.version) for p in packages] == [("cookie-es", "3.1.1"), ("react", "19.2.7")]
    assert integrities == {("cookie-es", "3.1.1"): "sha512-Y29va2ll", ("react", "19.2.7"): "sha512-cmVhY3Q="}


def test_a_package_lock_entry_without_integrity_is_reported_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    lock = tmp_path / "package-lock.json"
    lock.write_text(
        json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"name": "viewer", "version": "1.0.0"},
                    "node_modules/from-git": {"version": "1.0.0", "resolved": "git+ssh://git@github.com/o/r.git#abc"},
                },
            }
        )
    )
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--package-lock", str(lock), "--label", "viewer", "--output", str(tmp_path / "out")],
    )

    with pytest.raises(RuntimeError, match="has no integrity for: from-git@1.0.0"):
        module.main()
