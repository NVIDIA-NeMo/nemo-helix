# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import tomlkit
from nemo_helix_tools.vendor import vendor_package


def test_create_core_local_extra_prepends_services_self_reference(tmp_path: Path, monkeypatch) -> None:
    """core-service and services extras are written to the wrapper only."""
    wrapper_path = tmp_path / "packages/nemo_helix"
    wrapper_path.mkdir(parents=True)

    wrapper_doc = tomlkit.parse(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]
services = ["fastapi>=1"]

[tool.bundle-package]
nhx-auth = { source = "../../services/core/auth/src/nhx/core/auth", module = "nhx/core/auth", deps_group = "auth-service" }
nhx-files = { source = "../../services/core/files/src/nhx/core/files", module = "nhx/core/files", deps_group = "files-service" }
nhx-garak = { source = "../../services/garak/src/nhx/garak", module = "nhx/garak", deps_group = "garak-service" }
nhx-platform-seed = { source = "../../services/platform-seed/src/nhx/platform_seed", module = "nhx/platform_seed", deps_group = "platform-seed-service" }
nhx-safe-synthesizer = { source = "../../services/safe-synthesizer/src/nhx/safe_synthesizer", module = "nhx/safe_synthesizer", deps_group = "safe-synthesizer-service", include_in_services = false }
nhx-platform-runner = { source = "../../packages/nhx_platform_runner/src/nhx/platform_runner", module = "nhx/platform_runner", deps_group = "services" }
nemo-garak-plugin = { source = "../../plugins/nemo-garak/src/nemo_garak", module = "nemo_garak" }
nemo-evaluator-plugin = { source = "../../plugins/nemo-evaluator/src/nemo_evaluator", module = "nemo_evaluator" }
nemo-switchyard-plugin = { source = "../../plugins/nemo-switchyard/src/nemo_switchyard", module = "nemo_switchyard", deps_group = "nemo-switchyard" }
nemo-helix-plugin = { source = "../../packages/nemo_helix_plugin/src/nemo_helix_plugin", module = "nemo_helix_plugin" }
"""
    )
    (wrapper_path / "pyproject.toml").write_text(tomlkit.dumps(wrapper_doc), encoding="utf-8")

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    monkeypatch.setattr(vendor_package, "WRAPPER_PATH", wrapper_path)

    vendor_package._create_core_local_extra()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    wrapper_optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(wrapper_optional["core-service"]) == [
        "nemo-helix[auth-service]",
        "nemo-helix[files-service]",
    ]
    assert list(wrapper_optional["plugins"]) == [
        "nemo-helix[nemo-evaluator-plugin]",
        "nemo-helix[nemo-garak-plugin]",
        "nemo-helix[nemo-switchyard]",
    ]
    assert list(wrapper_optional["services"]) == [
        "nemo-helix[core-service]",
        "nemo-helix[platform-seed-service]",
        "nemo-helix[garak-service]",
        "nemo-helix[plugins]",
        "fastapi>=1",
    ]


def test_refresh_bundle_owned_optional_dependencies_preserves_hand_written_extras() -> None:
    """Extras without the generator marker are preserved untouched, regardless of name."""
    content = """
[project]
name = "example"

[project.optional-dependencies]
docs = ["mkdocs"]
# Generated from [tool.bundle-package]; do not edit by hand.
bundled-sdk = [
  "requests",
]
services = ["manual"]

[tool.bundle-package]
""".lstrip()

    updated = vendor_package._refresh_bundle_owned_optional_dependencies(content, {"bundled-sdk"})

    # Hand-written extras (no marker): preserved as-is.
    assert "docs =" in updated
    assert 'services = ["manual"]' in updated
    # Vendor-owned extra still claimed: kept with its marker.
    assert f"{vendor_package.GENERATED_BUNDLE_GROUP_COMMENT}\nbundled-sdk" in updated


def test_refresh_bundle_owned_optional_dependencies_drops_stale_generated_extras() -> None:
    """Extras with the marker but no longer in `bundle_owned_names` are dropped as stale."""
    content = """
[project.optional-dependencies]
docs = ["mkdocs"]

# Generated from [tool.bundle-package]; do not edit by hand.
stale = ["httpx"]

# Generated from [tool.bundle-package]; do not edit by hand.
still-owned = ["pydantic"]
""".lstrip()

    updated = vendor_package._refresh_bundle_owned_optional_dependencies(content, {"still-owned"})

    assert "docs =" in updated
    assert "stale =" not in updated
    assert f"{vendor_package.GENERATED_BUNDLE_GROUP_COMMENT}\nstill-owned" in updated
    # Exactly one marker should remain (for `still-owned`).
    assert updated.count(vendor_package.GENERATED_BUNDLE_GROUP_COMMENT) == 1


def test_refresh_bundle_owned_optional_dependencies_emits_marker_for_unmarked_owned_keys() -> None:
    """Newly-added bundle-owned extras (no marker yet) get the marker on rewrite."""
    content = """
[project.optional-dependencies]
just-added = ["httpx"]
""".lstrip()

    updated = vendor_package._refresh_bundle_owned_optional_dependencies(content, {"just-added"})

    assert f"{vendor_package.GENERATED_BUNDLE_GROUP_COMMENT}\njust-added" in updated


def test_refresh_bundle_owned_optional_dependencies_alphabetizes_vendor_owned() -> None:
    """Hand-written extras keep their position; vendor-owned extras are sorted below."""
    content = """
[project.optional-dependencies]
all = ["example[services]"]

# Generated from [tool.bundle-package]; do not edit by hand.
zeta = ["z"]

# Generated from [tool.bundle-package]; do not edit by hand.
alpha = ["a"]
""".lstrip()

    updated = vendor_package._refresh_bundle_owned_optional_dependencies(content, {"alpha", "zeta"})

    # `all` (hand-written) stays first; vendor-owned sorted alphabetically below.
    assert updated.index("all =") < updated.index("alpha =")
    assert updated.index("alpha =") < updated.index("zeta =")


def test_normalize_static_force_include_spacing(tmp_path: Path) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "# force-include mappings generated by hatch_build.py from [tool.bundle-package].\n\n\n\n"
        "[tool.hatch.build.targets.wheel.force-include]\n"
        '"source" = "target"\n',
        encoding="utf-8",
    )

    vendor_package._normalize_static_force_include_spacing(pyproject_path)

    assert "\n\n\n[tool.hatch.build.targets.wheel.force-include]" not in pyproject_path.read_text(encoding="utf-8")
    assert "\n\n[tool.hatch.build.targets.wheel.force-include]" in pyproject_path.read_text(encoding="utf-8")


def test_process_bundle_packages_keeps_external_dependency_names_without_inheriting_metadata_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    plugin_path = tmp_path / "plugins/nemo-switchyard"
    (plugin_path / "src/nemo_switchyard").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_helix"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]

[project.entry-points]

[tool.bundle-package]
nemo-switchyard-plugin = { source = "../../plugins/nemo-switchyard/src/nemo_switchyard", module = "nemo_switchyard", deps_group = "nemo-switchyard" }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-switchyard-plugin"
dependencies = ["nemo-helix", "nemo-switchyard==0.3.0", "httpx>=0.28"]

[project.scripts]
switchyard-cli = "nemo_switchyard.cli:main"

[project.optional-dependencies]
aiohttp = ["aiohttp"]
test = ["pytest>=8"]

[project.entry-points."nemo.inference_middleware"]
nemo-switchyard = "nemo_switchyard.middleware:SwitchyardMiddleware"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-switchyard"]) == [
        "nemo-switchyard==0.3.0",
        "httpx>=0.28",
    ]
    assert "aiohttp" not in optional
    assert "test" not in optional
    assert "scripts" not in wrapper_updated["project"]
    assert not wrapper_updated["project"]["entry-points"]


def test_process_bundle_packages_keeps_published_workspace_dependencies(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    evaluator_path = tmp_path / "packages/nemo_evaluator_sdk"
    gym_path = tmp_path / "packages/sandboxed_gym"
    unpublished_path = tmp_path / "packages/nhx_testing"
    for path in (wrapper_path, evaluator_path, gym_path, unpublished_path):
        path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_helix", "packages/nemo_evaluator_sdk", "packages/sandboxed_gym", "packages/nhx_testing"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]

[tool.bundle-package]
nemo-evaluator-sdk = { source = "../../packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk", module = "nemo_evaluator_sdk" }

[tool.bundle-package-published]
packages = ["nemo-sandboxed-gym"]
""".lstrip(),
        encoding="utf-8",
    )
    (evaluator_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-evaluator-sdk"
dependencies = ["pydantic>=2.10.6", "nemo-sandboxed-gym", "nhx-testing"]
""".lstrip(),
        encoding="utf-8",
    )
    (gym_path / "pyproject.toml").write_text('[project]\nname = "nemo-sandboxed-gym"\n', encoding="utf-8")
    (unpublished_path / "pyproject.toml").write_text('[project]\nname = "nhx-testing"\n', encoding="utf-8")

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    assert list(wrapper_updated["project"]["optional-dependencies"]["nemo-evaluator-sdk"]) == [
        "pydantic>=2.10.6",
        "nemo-sandboxed-gym",
    ]


def test_process_bundle_packages_rebuilds_generated_dependency_groups(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    plugin_path = tmp_path / "plugins/nemo-evaluator"
    runner_path = tmp_path / "packages/nhx_platform_runner"
    (plugin_path / "src/nemo_evaluator").mkdir(parents=True)
    (runner_path / "src/nhx/platform_runner").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_helix"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]
# Generated from [tool.bundle-package]; do not edit by hand.
nemo-evaluator-plugin = ["nemo-evaluator-sdk", "stale-plugin-dep"]

# Generated from [tool.bundle-package]; do not edit by hand.
services = ["nemo-helix[core-service]", "old-service"]

[tool.bundle-package]
nemo-evaluator-plugin = { source = "../../plugins/nemo-evaluator/src/nemo_evaluator", module = "nemo_evaluator" }
nhx-platform-runner = { source = "../../packages/nhx_platform_runner/src/nhx/platform_runner", module = "nhx/platform_runner", deps_group = "services" }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-evaluator-plugin"
dependencies = ["nemo-evaluator-sdk", "nemo-helix-sdk", "nhx-common", "pydantic>=2.10.6"]
""".lstrip(),
        encoding="utf-8",
    )
    (runner_path / "pyproject.toml").write_text(
        """
[project]
name = "nhx-platform-runner"
dependencies = ["rich>=14.1.0"]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-evaluator-plugin"]) == [
        "nemo-evaluator-sdk",
        "nemo-helix-sdk",
        "nhx-common",
        "pydantic>=2.10.6",
    ]
    assert list(optional["services"]) == ["rich>=14.1.0", "nemo-helix[core-service]"]


def test_process_bundle_packages_clears_generated_group_when_dependencies_empty(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    plugin_path = tmp_path / "plugins/empty"
    (plugin_path / "src/empty_plugin").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_helix"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]
# Generated from [tool.bundle-package]; do not edit by hand.
empty-plugin = ["stale-plugin-dep"]

[tool.bundle-package]
empty-plugin = { source = "../../plugins/empty/src/empty_plugin", module = "empty_plugin" }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "empty-plugin"
dependencies = ["nemo-helix"]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_content = (wrapper_path / "pyproject.toml").read_text(encoding="utf-8")
    wrapper_updated = tomlkit.parse(wrapper_content)
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["empty-plugin"]) == []
    assert "stale-plugin-dep" not in wrapper_content


def test_process_bundle_packages_merges_shared_generated_dependency_group(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix_plugin"
    sdk_path = tmp_path / "sdk/python/nemo-helix"
    ext_path = tmp_path / "packages/nemo_helix_ext"
    models_path = tmp_path / "packages/models"
    (sdk_path / "src/nemo_helix").mkdir(parents=True)
    (ext_path / "src/nemo_helix_ext").mkdir(parents=True)
    (models_path / "src/models").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_helix_plugin"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix-plugin"

[project.optional-dependencies]
# Generated from [tool.bundle-package]; do not edit by hand.
nemo-helix-sdk = ["stale-sdk-dep"]

[tool.bundle-package]
nemo-helix-sdk = { source = "../../sdk/python/nemo-helix/src/nemo_helix", module = "nemo_helix" }
nemo-helix-ext = { source = "../nemo_helix_ext/src/nemo_helix_ext", module = "nemo_helix_ext", deps_group = "nemo-helix-sdk" }
models = { source = "../models/src/models", module = "models", deps_group = "nemo-helix-sdk" }
""".lstrip(),
        encoding="utf-8",
    )
    (sdk_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix-sdk"
dependencies = ["docker>=7.0.0", "httpx>=0.23.0"]
""".lstrip(),
        encoding="utf-8",
    )
    (ext_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix-ext"
dependencies = ["nemo-helix-sdk", "rich>=13.7.1"]
""".lstrip(),
        encoding="utf-8",
    )
    (models_path / "pyproject.toml").write_text(
        """
[project]
name = "models"
dependencies = ["pydantic>=2.10.6"]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-helix-sdk"]) == [
        "docker>=7.0.0",
        "httpx>=0.23.0",
        "rich>=13.7.1",
        "pydantic>=2.10.6",
    ]


def test_process_bundle_packages_rebuilds_platform_seed_service_group(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    seed_path = tmp_path / "services/platform-seed"
    wrapper_path.mkdir(parents=True)
    seed_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = [
    "packages/nemo_helix",
    "packages/nhx_common",
    "services/core/auth",
    "services/guardrails",
    "services/platform-seed",
]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]

[tool.bundle-package]
nhx-common = { source = "../../packages/nhx_common/src/nhx/common", module = "nhx/common" }
nhx-auth = { source = "../../services/core/auth/src/nhx/core/auth", module = "nhx/core/auth", deps_group = "auth-service" }
nhx-guardrails = { source = "../../services/guardrails/src/nhx/guardrails", module = "nhx/guardrails", deps_group = "guardrails-service" }
nhx-platform-seed = { source = "../../services/platform-seed/src/nhx/platform_seed", module = "nhx/platform_seed", deps_group = "platform-seed-service" }
""".lstrip(),
        encoding="utf-8",
    )
    (seed_path / "pyproject.toml").write_text(
        """
[project]
name = "nhx-platform-seed"
dependencies = ["nhx-common", "nhx-auth", "nhx-guardrails"]
""".lstrip(),
        encoding="utf-8",
    )
    for package_path, package_name in [
        ("packages/nhx_common", "nhx-common"),
        ("services/core/auth", "nhx-auth"),
        ("services/guardrails", "nhx-guardrails"),
    ]:
        pyproject_path = tmp_path / package_path / "pyproject.toml"
        pyproject_path.parent.mkdir(parents=True)
        pyproject_path.write_text(
            f"""
[project]
name = "{package_name}"
dependencies = []
""".lstrip(),
            encoding="utf-8",
        )

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["platform-seed-service"]) == [
        "nhx-common",
        "nhx-auth",
        "nhx-guardrails",
    ]


def test_process_bundle_packages_inherits_requested_metadata(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    plugin_path = tmp_path / "plugins/nemo-switchyard"
    (plugin_path / "src/nemo_switchyard").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_helix"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-helix"

[project.optional-dependencies]
manual = ["keep-me"]

# Generated from [tool.bundle-package]; do not edit by hand.
nemo-switchyard-aiohttp = ["stale-dependency"]

[tool.bundle-package]
nemo-switchyard-plugin = { source = "../../plugins/nemo-switchyard/src/nemo_switchyard", module = "nemo_switchyard", optional-dependencies-prefix = "nemo-switchyard-", inherit = { "entry-points" = ["nemo.*"], "optional-dependencies" = ["aio*"], scripts = ["switchyard-*"] }, deps_group = "nemo-switchyard" }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-switchyard-plugin"
dependencies = ["httpx>=0.28"]

[project.scripts]
switchyard-cli = "nemo_switchyard.cli:main"

[project.optional-dependencies]
aiohttp = ["aiohttp"]
safe-synthesizer = ["pandas"]
test = ["pytest>=8"]

[project.entry-points."nemo.inference_middleware"]
nemo-switchyard = "nemo_switchyard.middleware:SwitchyardMiddleware"

[project.entry-points."data_designer.plugins"]
switchyard = "nemo_switchyard.plugins:plugin"
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-switchyard"]) == ["httpx>=0.28"]
    assert list(optional["nemo-switchyard-aiohttp"]) == ["nemo-helix[nemo-switchyard]", "aiohttp"]
    assert list(optional["manual"]) == ["keep-me"]
    assert "aiohttp" not in optional
    assert "safe-synthesizer" not in optional
    assert "test" not in optional
    assert wrapper_updated["project"]["scripts"]["switchyard-cli"] == "nemo_switchyard.cli:main"
    assert (
        wrapper_updated["project"]["entry-points"]["nemo.inference_middleware"]["nemo-switchyard"]
        == "nemo_switchyard.middleware:SwitchyardMiddleware"
    )
    assert "data_designer.plugins" not in wrapper_updated["project"]["entry-points"]


def test_find_package_dir_continues_past_non_matching_pyproject(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_helix"
    source_path = tmp_path / "vendor/container/src/switchyard"
    source_path.mkdir(parents=True)
    wrapper_path.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[tool.uv.workspace]\nmembers = []\n", encoding="utf-8")
    (tmp_path / "vendor/container/pyproject.toml").write_text(
        '[project]\nname = "switchyard"\n',
        encoding="utf-8",
    )
    (tmp_path / "vendor/container/src/pyproject.toml").write_text(
        '[project]\nname = "not-switchyard"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NHX_ROOT_PATH", tmp_path)

    assert (
        vendor_package._find_package_dir(
            "switchyard",
            {"source": "../../vendor/container/src/switchyard"},
            wrapper_path,
        )
        == tmp_path / "vendor/container"
    )


def test_annotate_generated_project_entries_marks_only_wholly_generated_tables() -> None:
    """Wholly-generated scripts/entry-point tables get a table-level marker.

    Mixed tables (containing any hand-written entry) are left unannotated —
    we no longer emit per-key markers in scripts/entry-points tables, so a
    table is either wholly generated (and gets one header) or not.
    """
    content = """
[project]
name = "example"

[project.scripts]
nemo = "nemo_helix.cli.app:cli"
manual-script = "example:main"

[project.entry-points."nemo.cli"]
garak = "nemo_garak.cli:GarakPluginCLI"
manual = "example:manual"

[project.entry-points."nemo.docs"]
garak = "nemo_garak.docs:get_docs_path"

[project.entry-points."manual"]
manual = "example:manual"
""".lstrip()

    updated = vendor_package._annotate_generated_project_entries(
        content,
        {"nemo"},
        {"nemo.cli": {"garak"}, "nemo.docs": {"garak"}},
    )

    # Wholly-generated table gets a header marker.
    assert (
        f'{vendor_package.GENERATED_BUNDLE_TABLE_COMMENT}\n[project.entry-points."nemo.docs"]\n'
        'garak = "nemo_garak.docs:get_docs_path"'
    ) in updated
    # Mixed tables (`[project.scripts]`, `[project.entry-points."nemo.cli"]`)
    # are left unannotated — no per-key markers.
    assert vendor_package.GENERATED_BUNDLE_GROUP_COMMENT not in updated
    # Hand-written entries survive untouched.
    assert 'manual-script = "example:main"' in updated
    assert 'manual = "example:manual"' in updated


def test_remove_marked_generated_project_tables_removes_marked_entries() -> None:
    content = """
[project]
name = "example"

[project.scripts]
# Generated from [tool.bundle-package]; do not edit by hand.
nemo = "nemo_helix.cli.app:cli"
manual = "example:main"

[project.entry-points."manual"]
manual = "example:manual"

[project.entry-points."nemo.cli"]
# Generated from [tool.bundle-package]; do not edit by hand.
garak = "nemo_garak.cli:GarakPluginCLI"
manual = "example:manual"

[tool.example]
""".lstrip()

    updated = vendor_package._remove_marked_generated_project_tables(content)

    assert "[project.scripts]" in updated
    assert 'nemo = "nemo_helix.cli.app:cli"' not in updated
    assert 'manual = "example:main"' in updated
    assert '[project.entry-points."nemo.cli"]' in updated
    assert 'garak = "nemo_garak.cli:GarakPluginCLI"' not in updated
    assert '[project.entry-points."manual"]' in updated
    assert "[tool.example]" in updated


def test_remove_marked_generated_project_tables_removes_marked_whole_tables() -> None:
    content = """
[project]
name = "example"

# Generated from [tool.bundle-package]; do not edit this table by hand.
[project.scripts]
nemo = "nemo_helix.cli.app:cli"

[project.entry-points."manual"]
manual = "example:manual"
""".lstrip()

    updated = vendor_package._remove_marked_generated_project_tables(content)

    assert "[project.scripts]" not in updated
    assert "nemo_helix.cli.app" not in updated
    assert '[project.entry-points."manual"]' in updated


def test_annotate_generated_project_entries_trims_generated_table_edge_whitespace() -> None:
    content = """
[project]
name = "example"

[project.entry-points."nemo.skills"]

agents = "nemo_agents_plugin.skills:skills_dir"



[tool.uv.sources]
nemo-helix-sdk = { workspace = true }
""".lstrip()

    updated = vendor_package._annotate_generated_project_entries(
        content,
        set(),
        {"nemo.skills": {"agents"}},
    )

    assert "\n\n\n[tool.uv.sources]" not in updated
