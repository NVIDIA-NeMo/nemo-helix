# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from importlib import import_module, resources, util
from pathlib import Path

import tomlkit
from nemo_platform_sdk_tools.sdk.vendor import vendor_package


def test_load_package_config_finds_service_config(tmp_path: Path, monkeypatch) -> None:
    services_root = tmp_path / "services/core/auth"
    services_root.mkdir(parents=True)
    pyproject_path = services_root / "pyproject.toml"
    pyproject_path.write_text(
        """
[tool.vendor-package]
package = "nmp_auth"
package_root = "services/core/auth"
source_module = "nmp.core.auth"
target_sdk_module = "nmp.core.auth"
top_level = true
included_paths = ["**/*.py"]
""".strip()
        + "\n"
    )

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)

    config = vendor_package._load_package_config("nmp_auth")

    assert config["package"] == "nmp_auth"
    assert config["package_root"] == "services/core/auth"


def test_build_and_validate_package_path_uses_repo_relative_root(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "services/core/auth/src/nmp/core/auth"
    source_path.mkdir(parents=True)

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)

    package_root_path, package_path = vendor_package._build_and_validate_package_path(
        package="nmp_auth",
        package_root="services/core/auth",
        source_module="nmp.core.auth",
        with_src=True,
    )

    assert package_root_path == tmp_path / "services/core/auth"
    assert package_path == source_path


def test_build_and_validate_target_paths_supports_top_level_targets(tmp_path: Path) -> None:
    sdk_path = tmp_path / "sdk/python/nemo-platform"
    sdk_path.mkdir(parents=True)

    top_level_path = vendor_package._build_and_validate_target_paths(sdk_path, "nmp.core.auth", top_level=True)
    nested_path = vendor_package._build_and_validate_target_paths(sdk_path, "services.runner")

    assert top_level_path == sdk_path / "src/nmp/core/auth"
    assert nested_path == sdk_path / "src/nemo_platform/services/runner"


def test_copy_included_paths_preserves_generated_header_for_empty_init(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    destination_path = tmp_path / "destination"
    source_path.mkdir()
    (source_path / "__init__.py").write_text("", encoding="utf-8")

    vendor_package._copy_included_paths(source_path, destination_path, ["**/*.py"])

    assert (destination_path / "__init__.py").read_text(encoding="utf-8") == vendor_package.GENERATED_FILE_HEADER


def test_alias_package_imports_submodules_from_source_module(tmp_path: Path, monkeypatch) -> None:
    source_package = tmp_path / "source_pkg"
    source_package.mkdir()
    (source_package / "__init__.py").write_text("VALUE = 'source'\n", encoding="utf-8")
    (source_package / "child.py").write_text("VALUE = 7\n", encoding="utf-8")

    alias_package = tmp_path / "alias_pkg"
    alias_package.mkdir()
    (alias_package / "__init__.py").write_text(
        "from nemo_platform._alias import alias_package\n\nalias_package('source_pkg', globals())\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(vendor_package.NMP_ROOT_PATH / "sdk/python/nemo-platform/src"))
    for module_name in (
        "alias_pkg",
        "alias_pkg.child",
        "source_pkg",
        "source_pkg.child",
        "nemo_platform._alias",
        "nemo_platform",
    ):
        sys.modules.pop(module_name, None)

    try:
        source_child = import_module("source_pkg.child")
        alias = import_module("alias_pkg")
        alias_child = import_module("alias_pkg.child")
        fromlist_child = getattr(__import__("alias_pkg", fromlist=["child"]), "child")

        assert alias.VALUE == "source"
        assert alias_child is fromlist_child
        assert alias_child.__name__ == "alias_pkg.child"
        assert source_child.__name__ == "source_pkg.child"
        assert alias_child.VALUE == source_child.VALUE == 7
    finally:
        for module_name in (
            "alias_pkg",
            "alias_pkg.child",
            "source_pkg",
            "source_pkg.child",
            "nemo_platform._alias",
            "nemo_platform",
        ):
            sys.modules.pop(module_name, None)


def test_alias_package_exposes_source_resource_metadata(tmp_path: Path, monkeypatch) -> None:
    source_package = tmp_path / "source_pkg"
    source_package.mkdir()
    (source_package / "__init__.py").write_text(
        "from pathlib import Path\n\n\ndef package_dir() -> Path:\n    return Path(__file__).parent\n",
        encoding="utf-8",
    )
    (source_package / "data.txt").write_text("source-data\n", encoding="utf-8")

    alias_package = tmp_path / "alias_pkg"
    alias_package.mkdir()
    (alias_package / "__init__.py").write_text(
        "from nemo_platform._alias import alias_package\n\nalias_package('source_pkg', globals())\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(vendor_package.NMP_ROOT_PATH / "sdk/python/nemo-platform/src"))
    for module_name in ("alias_pkg", "source_pkg", "nemo_platform._alias", "nemo_platform"):
        sys.modules.pop(module_name, None)

    try:
        alias = import_module("alias_pkg")

        assert alias.__file__ == str(source_package / "__init__.py")
        assert alias.package_dir() == source_package
        assert resources.files(alias).joinpath("data.txt").read_text(encoding="utf-8") == "source-data\n"
    finally:
        for module_name in ("alias_pkg", "source_pkg", "nemo_platform._alias", "nemo_platform"):
            sys.modules.pop(module_name, None)


def test_alias_package_loader_provides_runpy_get_code(tmp_path: Path, monkeypatch) -> None:
    source_package = tmp_path / "source_pkg"
    source_package.mkdir()
    (source_package / "__init__.py").write_text("VALUE = 'source'\n", encoding="utf-8")
    (source_package / "child.py").write_text("VALUE = 7\n", encoding="utf-8")

    alias_package = tmp_path / "alias_pkg"
    alias_package.mkdir()
    (alias_package / "__init__.py").write_text(
        "from nemo_platform._alias import alias_package\n\nalias_package('source_pkg', globals())\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(vendor_package.NMP_ROOT_PATH / "sdk/python/nemo-platform/src"))
    for module_name in (
        "alias_pkg",
        "alias_pkg.child",
        "source_pkg",
        "source_pkg.child",
        "nemo_platform._alias",
        "nemo_platform",
    ):
        sys.modules.pop(module_name, None)

    try:
        import_module("alias_pkg")
        spec = util.find_spec("alias_pkg.child")
        assert spec is not None
        assert spec.loader is not None
        get_code = getattr(spec.loader, "get_code", None)
        assert get_code is not None
        code = get_code("alias_pkg.child")
        assert code is not None
        assert code.co_filename == str(source_package / "child.py")
    finally:
        for module_name in (
            "alias_pkg",
            "alias_pkg.child",
            "source_pkg",
            "source_pkg.child",
            "nemo_platform._alias",
            "nemo_platform",
        ):
            sys.modules.pop(module_name, None)


def test_vendor_package_files_source_package_mode_writes_target_alias(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "packages/models/src/models"
    source_path.mkdir(parents=True)
    (source_path / "resources.py").write_text("VALUE = 1\n", encoding="utf-8")

    stale_target = tmp_path / "sdk/python/nemo-platform/src/nemo_platform/models"
    stale_target.mkdir(parents=True)
    (stale_target / "resources.py").write_text("STALE = True\n", encoding="utf-8")

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)

    vendor_package._vendor_package_files(
        {
            "package": "models",
            "package_root": "packages/models",
            "sdk_include_mode": "source-package",
            "target_sdk_module": "models",
        }
    )

    alias_init = stale_target / "__init__.py"
    assert alias_init.read_text(encoding="utf-8") == vendor_package.GENERATED_ALIAS_INIT_TEMPLATE.format(
        source_module="models"
    )
    assert not (stale_target / "resources.py").exists()


def test_vendor_package_files_source_package_mode_writes_top_level_aliases(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "packages/nemo_platform_ext/src/nemo_platform_ext"
    (source_path / "cli").mkdir(parents=True)
    (source_path / "quickstart").mkdir()
    (source_path / "__pycache__").mkdir()

    sdk_path = tmp_path / "sdk/python/nemo-platform/src/nemo_platform"
    sdk_path.mkdir(parents=True)
    stale_tests = tmp_path / "sdk/python/nemo-platform/tests/vendored/nemo_platform_ext"
    stale_tests.mkdir(parents=True)
    (stale_tests / "test_stale.py").write_text("STALE = True\n", encoding="utf-8")

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)

    vendor_package._vendor_package_files(
        {
            "package": "nemo_platform_ext",
            "package_root": "packages/nemo_platform_ext",
            "sdk_include_mode": "source-package",
        }
    )

    assert (sdk_path / "cli/__init__.py").read_text(encoding="utf-8") == (
        vendor_package.GENERATED_ALIAS_INIT_TEMPLATE.format(source_module="nemo_platform_ext.cli")
    )
    assert (sdk_path / "quickstart/__init__.py").read_text(encoding="utf-8") == (
        vendor_package.GENERATED_ALIAS_INIT_TEMPLATE.format(source_module="nemo_platform_ext.quickstart")
    )
    assert not (sdk_path / "__pycache__").exists()
    assert not stale_tests.exists()


def test_vendor_package_files_source_package_mode_creates_beta_parent_init(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk"
    source_path.mkdir(parents=True)

    sdk_path = tmp_path / "sdk/python/nemo-platform/src/nemo_platform"
    sdk_path.mkdir(parents=True)

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)

    vendor_package._vendor_package_files(
        {
            "package": "nemo_evaluator_sdk",
            "package_root": "packages/nemo_evaluator_sdk",
            "sdk_include_mode": "source-package",
            "source_module": "nemo_evaluator_sdk",
            "target_sdk_module": "beta.evaluator",
        }
    )

    assert (sdk_path / "beta/__init__.py").read_text(encoding="utf-8") == vendor_package.GENERATED_FILE_HEADER
    assert (sdk_path / "beta/evaluator/__init__.py").read_text(encoding="utf-8") == (
        vendor_package.GENERATED_ALIAS_INIT_TEMPLATE.format(source_module="nemo_evaluator_sdk")
    )


def test_update_dependencies_of_sdk_pyproject_merges_optional_dependency_groups(tmp_path: Path, monkeypatch) -> None:
    """SDK client extension deps are written to the SDK pyproject."""
    sdk_path = tmp_path / "sdk/python/nemo-platform"
    sdk_path.mkdir(parents=True)
    package_root = tmp_path / "packages/nemo_evaluator_sdk"
    package_root.mkdir(parents=True)

    sdk_doc = tomlkit.document()
    sdk_doc["project"] = tomlkit.table()
    sdk_doc["project"]["name"] = "nemo-platform-sdk"
    sdk_doc["project"]["dependencies"] = ["typer>=0.20.0"]
    sdk_doc["project"]["optional-dependencies"] = tomlkit.table()
    sdk_doc["project"]["optional-dependencies"]["evaluator"] = ["requests>=2.0.0"]

    package_doc = tomlkit.document()
    package_doc["project"] = tomlkit.table()
    package_doc["project"]["dependencies"] = ["httpx>=0.27.0", "requests>=2.5.0"]

    with open(sdk_path / "pyproject.toml", "w") as f:
        tomlkit.dump(sdk_doc, f)

    with open(package_root / "pyproject.toml", "w") as f:
        tomlkit.dump(package_doc, f)

    vendor_package._update_dependencies_of_sdk_pyproject(
        sdk_path=sdk_path,
        package_root_path=package_root,
        excluded_dependencies=[],
        optional_deps_name="evaluator",
    )

    with open(sdk_path / "pyproject.toml", "rb") as f:
        sdk_updated = tomlkit.load(f)

    sdk_deps = list(sdk_updated["project"]["optional-dependencies"]["evaluator"])
    assert any(dep.startswith("requests") and ">=2.5.0" in dep for dep in sdk_deps)
    assert "httpx>=0.27.0" in sdk_deps


def test_create_core_local_extra_prepends_services_self_reference(tmp_path: Path, monkeypatch) -> None:
    """core-service and services extras are written to the wrapper only."""
    wrapper_path = tmp_path / "packages/nemo_platform"
    wrapper_path.mkdir(parents=True)

    wrapper_doc = tomlkit.parse(
        """
[project]
name = "nemo-platform"

[project.optional-dependencies]
services = ["fastapi>=1"]

[tool.bundle-package]
nmp-auth = { source = "../../services/core/auth/src/nmp/core/auth", module = "nmp/core/auth", deps_group = "auth-service" }
nmp-files = { source = "../../services/core/files/src/nmp/core/files", module = "nmp/core/files", deps_group = "files-service" }
nmp-auditor = { source = "../../services/auditor/src/nmp/auditor", module = "nmp/auditor", deps_group = "auditor-service" }
nmp-platform-seed = { source = "../../services/platform-seed/src/nmp/platform_seed", module = "nmp/platform_seed", deps_group = "platform-seed-service" }
nmp-safe-synthesizer = { source = "../../services/safe-synthesizer/src/nmp/safe_synthesizer", module = "nmp/safe_synthesizer", deps_group = "safe-synthesizer-service", include_in_services = false }
nmp-platform-runner = { source = "../../packages/nmp_platform_runner/src/nmp/platform_runner", module = "nmp/platform_runner", deps_group = "services" }
nemo-auditor-plugin = { source = "../../plugins/nemo-auditor/src/nemo_auditor", module = "nemo_auditor" }
nemo-evaluator-plugin = { source = "../../plugins/nemo-evaluator/src/nemo_evaluator", module = "nemo_evaluator" }
nemo-switchyard = { source = "../../plugins/nemo-switchyard/src/nemo_switchyard", module = "nemo_switchyard" }
nemo-platform-plugin = { source = "../../packages/nemo_platform_plugin/src/nemo_platform_plugin", module = "nemo_platform_plugin" }
switchyard = { source = "../../plugins/nemo-switchyard/vendor/switchyard/switchyard", module = "switchyard" }
"""
    )
    (wrapper_path / "pyproject.toml").write_text(tomlkit.dumps(wrapper_doc), encoding="utf-8")

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    monkeypatch.setattr(vendor_package, "WRAPPER_PATH", wrapper_path)

    vendor_package._create_core_local_extra([])

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    wrapper_optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(wrapper_optional["core-service"]) == [
        "nemo-platform[auth-service]",
        "nemo-platform[files-service]",
    ]
    assert list(wrapper_optional["plugins"]) == [
        "nemo-platform[nemo-auditor-plugin]",
        "nemo-platform[nemo-evaluator-plugin]",
        "nemo-platform[nemo-switchyard]",
    ]
    assert list(wrapper_optional["services"]) == [
        "nemo-platform[core-service]",
        "nemo-platform[platform-seed-service]",
        "nemo-platform[auditor-service]",
        "nemo-platform[plugins]",
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


def test_process_bundle_packages_keeps_bundled_dependency_names_without_inheriting_metadata_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform"
    plugin_path = tmp_path / "plugins/nemo-switchyard"
    switchyard_path = plugin_path / "vendor/switchyard"
    (plugin_path / "src/nemo_switchyard").mkdir(parents=True)
    (switchyard_path / "switchyard").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_platform"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform"

[project.optional-dependencies]

[project.entry-points]

[tool.bundle-package]
nemo-switchyard = { source = "../../plugins/nemo-switchyard/src/nemo_switchyard", module = "nemo_switchyard" }
switchyard = { source = "../../plugins/nemo-switchyard/vendor/switchyard/switchyard", module = "switchyard" }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-switchyard"
dependencies = ["nemo-platform", "switchyard", "httpx>=0.28"]

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
    (switchyard_path / "pyproject.toml").write_text(
        """
[project]
name = "switchyard"
dependencies = ["openai>=2"]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-switchyard"]) == [
        "switchyard",
        "httpx>=0.28",
    ]
    assert list(optional["switchyard"]) == ["openai>=2"]
    assert "aiohttp" not in optional
    assert "test" not in optional
    assert "scripts" not in wrapper_updated["project"]
    assert not wrapper_updated["project"]["entry-points"]


def test_process_bundle_packages_rebuilds_generated_dependency_groups(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform"
    plugin_path = tmp_path / "plugins/nemo-evaluator"
    runner_path = tmp_path / "packages/nmp_platform_runner"
    (plugin_path / "src/nemo_evaluator").mkdir(parents=True)
    (runner_path / "src/nmp/platform_runner").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_platform"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform"

[project.optional-dependencies]
# Generated from [tool.bundle-package]; do not edit by hand.
nemo-evaluator-plugin = ["nemo-evaluator-sdk", "stale-plugin-dep"]

# Generated from [tool.bundle-package]; do not edit by hand.
services = ["nemo-platform[core-service]", "old-service"]

[tool.bundle-package]
nemo-evaluator-plugin = { source = "../../plugins/nemo-evaluator/src/nemo_evaluator", module = "nemo_evaluator" }
nmp-platform-runner = { source = "../../packages/nmp_platform_runner/src/nmp/platform_runner", module = "nmp/platform_runner", deps_group = "services" }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-evaluator-plugin"
dependencies = ["nemo-evaluator-sdk", "nemo-platform-sdk", "nmp-common", "pydantic>=2.10.6"]
""".lstrip(),
        encoding="utf-8",
    )
    (runner_path / "pyproject.toml").write_text(
        """
[project]
name = "nmp-platform-runner"
dependencies = ["rich>=14.1.0"]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-evaluator-plugin"]) == [
        "nemo-evaluator-sdk",
        "nemo-platform-sdk",
        "nmp-common",
        "pydantic>=2.10.6",
    ]
    assert list(optional["services"]) == ["rich>=14.1.0", "nemo-platform[core-service]"]


def test_process_bundle_packages_clears_generated_group_when_dependencies_empty(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform"
    plugin_path = tmp_path / "plugins/empty"
    (plugin_path / "src/empty_plugin").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_platform"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform"

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
dependencies = ["nemo-platform"]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_content = (wrapper_path / "pyproject.toml").read_text(encoding="utf-8")
    wrapper_updated = tomlkit.parse(wrapper_content)
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["empty-plugin"]) == []
    assert "stale-plugin-dep" not in wrapper_content


def test_process_bundle_packages_merges_shared_generated_dependency_group(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform_plugin"
    sdk_path = tmp_path / "sdk/python/nemo-platform"
    ext_path = tmp_path / "packages/nemo_platform_ext"
    models_path = tmp_path / "packages/models"
    (sdk_path / "src/nemo_platform").mkdir(parents=True)
    (ext_path / "src/nemo_platform_ext").mkdir(parents=True)
    (models_path / "src/models").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_platform_plugin"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform-plugin"

[project.optional-dependencies]
# Generated from [tool.bundle-package]; do not edit by hand.
nemo-platform-sdk = ["stale-sdk-dep"]

[tool.bundle-package]
nemo-platform-sdk = { source = "../../sdk/python/nemo-platform/src/nemo_platform", module = "nemo_platform" }
nemo-platform-ext = { source = "../nemo_platform_ext/src/nemo_platform_ext", module = "nemo_platform_ext", deps_group = "nemo-platform-sdk" }
models = { source = "../models/src/models", module = "models", deps_group = "nemo-platform-sdk" }
""".lstrip(),
        encoding="utf-8",
    )
    (sdk_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform-sdk"
dependencies = ["docker>=7.0.0", "httpx>=0.23.0"]
""".lstrip(),
        encoding="utf-8",
    )
    (ext_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform-ext"
dependencies = ["nemo-platform-sdk", "rich>=13.7.1"]
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

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-platform-sdk"]) == [
        "docker>=7.0.0",
        "httpx>=0.23.0",
        "rich>=13.7.1",
        "pydantic>=2.10.6",
    ]


def test_process_bundle_packages_rebuilds_platform_seed_service_group(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform"
    seed_path = tmp_path / "services/platform-seed"
    wrapper_path.mkdir(parents=True)
    seed_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = [
    "packages/nemo_platform",
    "packages/nmp_common",
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
name = "nemo-platform"

[project.optional-dependencies]

[tool.bundle-package]
nmp-common = { source = "../../packages/nmp_common/src/nmp/common", module = "nmp/common" }
nmp-auth = { source = "../../services/core/auth/src/nmp/core/auth", module = "nmp/core/auth", deps_group = "auth-service" }
nmp-guardrails = { source = "../../services/guardrails/src/nmp/guardrails", module = "nmp/guardrails", deps_group = "guardrails-service" }
nmp-platform-seed = { source = "../../services/platform-seed/src/nmp/platform_seed", module = "nmp/platform_seed", deps_group = "platform-seed-service" }
""".lstrip(),
        encoding="utf-8",
    )
    (seed_path / "pyproject.toml").write_text(
        """
[project]
name = "nmp-platform-seed"
dependencies = ["nmp-common", "nmp-auth", "nmp-guardrails"]
""".lstrip(),
        encoding="utf-8",
    )
    for package_path, package_name in [
        ("packages/nmp_common", "nmp-common"),
        ("services/core/auth", "nmp-auth"),
        ("services/guardrails", "nmp-guardrails"),
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

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["platform-seed-service"]) == [
        "nmp-common",
        "nmp-auth",
        "nmp-guardrails",
    ]


def test_process_bundle_packages_inherits_requested_metadata(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform"
    plugin_path = tmp_path / "plugins/nemo-switchyard"
    (plugin_path / "src/nemo_switchyard").mkdir(parents=True)
    wrapper_path.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["packages/nemo_platform"]
""".lstrip(),
        encoding="utf-8",
    )
    (wrapper_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-platform"

[project.optional-dependencies]

[tool.bundle-package]
nemo-switchyard = { source = "../../plugins/nemo-switchyard/src/nemo_switchyard", module = "nemo_switchyard", inherit = { "entry-points" = ["nemo.*"], "optional-dependencies" = ["aio*"], scripts = ["switchyard-*"] } }
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_path / "pyproject.toml").write_text(
        """
[project]
name = "nemo-switchyard"
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

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)
    vendor_package._process_bundle_packages()

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    optional = wrapper_updated["project"]["optional-dependencies"]

    assert list(optional["nemo-switchyard"]) == ["httpx>=0.28"]
    assert list(optional["aiohttp"]) == ["aiohttp"]
    assert "safe-synthesizer" not in optional
    assert "test" not in optional
    assert wrapper_updated["project"]["scripts"]["switchyard-cli"] == "nemo_switchyard.cli:main"
    assert (
        wrapper_updated["project"]["entry-points"]["nemo.inference_middleware"]["nemo-switchyard"]
        == "nemo_switchyard.middleware:SwitchyardMiddleware"
    )
    assert "data_designer.plugins" not in wrapper_updated["project"]["entry-points"]


def test_find_package_dir_continues_past_non_matching_pyproject(tmp_path: Path, monkeypatch) -> None:
    wrapper_path = tmp_path / "packages/nemo_platform"
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

    monkeypatch.setattr(vendor_package, "NMP_ROOT_PATH", tmp_path)

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
nemo = "nemo_platform.cli.app:cli"
manual-script = "example:main"

[project.entry-points."nemo.cli"]
auditor = "nemo_auditor.cli:AuditorPluginCLI"
manual = "example:manual"

[project.entry-points."nemo.docs"]
auditor = "nemo_auditor.docs:get_docs_path"

[project.entry-points."manual"]
manual = "example:manual"
""".lstrip()

    updated = vendor_package._annotate_generated_project_entries(
        content,
        {"nemo"},
        {"nemo.cli": {"auditor"}, "nemo.docs": {"auditor"}},
    )

    # Wholly-generated table gets a header marker.
    assert (
        f'{vendor_package.GENERATED_BUNDLE_TABLE_COMMENT}\n[project.entry-points."nemo.docs"]\n'
        'auditor = "nemo_auditor.docs:get_docs_path"'
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
nemo = "nemo_platform.cli.app:cli"
manual = "example:main"

[project.entry-points."manual"]
manual = "example:manual"

[project.entry-points."nemo.cli"]
# Generated from [tool.bundle-package]; do not edit by hand.
auditor = "nemo_auditor.cli:AuditorPluginCLI"
manual = "example:manual"

[tool.example]
""".lstrip()

    updated = vendor_package._remove_marked_generated_project_tables(content)

    assert "[project.scripts]" in updated
    assert 'nemo = "nemo_platform.cli.app:cli"' not in updated
    assert 'manual = "example:main"' in updated
    assert '[project.entry-points."nemo.cli"]' in updated
    assert 'auditor = "nemo_auditor.cli:AuditorPluginCLI"' not in updated
    assert '[project.entry-points."manual"]' in updated
    assert "[tool.example]" in updated


def test_remove_marked_generated_project_tables_removes_marked_whole_tables() -> None:
    content = """
[project]
name = "example"

# Generated from [tool.bundle-package]; do not edit this table by hand.
[project.scripts]
nemo = "nemo_platform.cli.app:cli"

[project.entry-points."manual"]
manual = "example:manual"
""".lstrip()

    updated = vendor_package._remove_marked_generated_project_tables(content)

    assert "[project.scripts]" not in updated
    assert "nemo_platform.cli.app" not in updated
    assert '[project.entry-points."manual"]' in updated


def test_annotate_generated_project_entries_trims_generated_table_edge_whitespace() -> None:
    content = """
[project]
name = "example"

[project.entry-points."nemo.skills"]

agents = "nemo_agents_plugin.skills:skills_dir"



[tool.uv.sources]
nemo-platform-sdk = { workspace = true }
""".lstrip()

    updated = vendor_package._annotate_generated_project_entries(
        content,
        set(),
        {"nemo.skills": {"agents"}},
    )

    assert "\n\n\n[tool.uv.sources]" not in updated


def test_vendor_scripts_writes_to_sdk(tmp_path: Path, monkeypatch) -> None:
    """Scripts from SDK client extensions are written to the SDK pyproject."""
    sdk_path = tmp_path / "sdk/python/nemo-platform"
    sdk_path.mkdir(parents=True)

    doc = tomlkit.parse(
        """
[project]
name = "example"
"""
    )
    (sdk_path / "pyproject.toml").write_text(tomlkit.dumps(doc), encoding="utf-8")

    vendor_package._vendor_scripts(
        sdk_path=sdk_path,
        scripts=[{"name": "nemo", "value": "nemo_platform.cli.app:cli"}],
    )

    sdk_updated = tomlkit.parse((sdk_path / "pyproject.toml").read_text(encoding="utf-8"))
    assert sdk_updated["project"]["scripts"]["nemo"] == "nemo_platform.cli.app:cli"


def test_vendor_entrypoints_writes_to_sdk(tmp_path: Path, monkeypatch) -> None:
    """Entrypoints from SDK client extensions are written to the SDK pyproject."""
    sdk_path = tmp_path / "sdk/python/nemo-platform"
    sdk_path.mkdir(parents=True)
    wrapper_path = tmp_path / "packages/nemo_platform"
    wrapper_path.mkdir(parents=True)

    doc = tomlkit.parse(
        """
[project]
name = "example"
"""
    )
    (sdk_path / "pyproject.toml").write_text(tomlkit.dumps(doc), encoding="utf-8")
    (wrapper_path / "pyproject.toml").write_text(tomlkit.dumps(doc), encoding="utf-8")

    monkeypatch.setattr(vendor_package, "WRAPPER_PATH", wrapper_path)

    vendor_package._vendor_entrypoints(
        sdk_path=sdk_path,
        entrypoints=[
            {
                "group": "data_designer.plugins",
                "entrypoints": [{"name": "seed", "value": "pkg.module:func"}],
            }
        ],
    )

    sdk_updated = tomlkit.parse((sdk_path / "pyproject.toml").read_text(encoding="utf-8"))
    assert sdk_updated["project"]["entry-points"]["data_designer.plugins"]["seed"] == "pkg.module:func"

    wrapper_updated = tomlkit.parse((wrapper_path / "pyproject.toml").read_text(encoding="utf-8"))
    assert "entry-points" not in wrapper_updated["project"]


def test_replace_client_methods_updates_init_and_getattr(tmp_path: Path, monkeypatch) -> None:
    sdk_path = tmp_path / "sdk/python/nemo-platform"
    client_path = sdk_path / "src/nemo_platform/_client.py"
    source_path = tmp_path / "packages/nemo_platform_ext/src/nemo_platform_ext/client/enhanced.py"
    client_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_client_path = tmp_path / "nemo_platform_plugin/client"
    plugin_client_path.mkdir(parents=True)
    plugin_jobs_path = tmp_path / "nemo_platform_plugin/jobs"
    plugin_jobs_path.mkdir(parents=True)
    plugin_secrets_path = tmp_path / "nemo_platform_plugin/secrets"
    plugin_secrets_path.mkdir(parents=True)
    (client_path.parent / "__init__.py").write_text("", encoding="utf-8")
    (client_path.parent / "_base_client.py").write_text(
        """
class DefaultAsyncHttpxClient:
    pass


class DefaultHttpxClient:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "nemo_platform_plugin/__init__.py").write_text("", encoding="utf-8")
    (plugin_client_path / "__init__.py").write_text("", encoding="utf-8")
    (plugin_client_path / "constants.py").write_text(
        'WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR = "NMP_WORKLOAD_IDENTITY_TOKEN_FILE"\n',
        encoding="utf-8",
    )
    (plugin_client_path / "tls.py").write_text(
        "def client_verify_from_env() -> bool:\n    return True\n",
        encoding="utf-8",
    )
    (plugin_jobs_path / "__init__.py").write_text("", encoding="utf-8")
    (plugin_jobs_path / "client.py").write_text(
        """
class AsyncJobsClient:
    pass


class JobsClient:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    (plugin_secrets_path / "__init__.py").write_text("", encoding="utf-8")
    (plugin_secrets_path / "compat.py").write_text(
        """
class AsyncSecretsResource:
    pass


class SecretsResource:
    pass
""".lstrip(),
        encoding="utf-8",
    )

    client_path.write_text(
        """
from typing import Any
from nemo_platform_ext.client.tls import client_verify_from_env


def _should_bootstrap_config(config_path: object | None = None) -> bool:
    return False


class NeMoPlatform:
    def __init__(self) -> None:
        self.value = 1


class AsyncNeMoPlatform:
    def __init__(self) -> None:
        self.value = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    source_path.write_text(
        """
from pathlib import Path
from typing import Any


def _should_bootstrap_config(config_path: Path | None = None) -> bool:
    return config_path is not None


class NeMoPlatform:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path
        self.should_bootstrap = _should_bootstrap_config(config_path)

    def __getattr__(self, name: str) -> Any:
        return name

    @property
    def jobs(self) -> JobsClient:
        return JobsClient()

    @property
    def secrets(self) -> SecretsResource:
        return SecretsResource()


class AsyncNeMoPlatform:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path
        self.should_bootstrap = _should_bootstrap_config(config_path)

    def __getattr__(self, name: str) -> Any:
        return name

    @property
    def jobs(self) -> AsyncJobsClient:
        return AsyncJobsClient()

    @property
    def secrets(self) -> AsyncSecretsResource:
        return AsyncSecretsResource()
""".strip()
        + "\n",
        encoding="utf-8",
    )

    vendor_package._replace_client_methods(
        sdk_path=sdk_path,
        source_path=source_path,
    )

    updated = client_path.read_text(encoding="utf-8")

    assert "from pathlib import Path" in updated
    assert "from nemo_platform._base_client import DefaultAsyncHttpxClient, DefaultHttpxClient" in updated
    assert "from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR" in updated
    assert "from nemo_platform_plugin.client.tls import client_verify_from_env" in updated
    assert "from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient" in updated
    assert "from nemo_platform_plugin.secrets.compat import AsyncSecretsResource, SecretsResource" in updated
    assert "from nemo_platform_ext.client.tls import client_verify_from_env" not in updated
    assert "def _should_bootstrap_config(config_path: Path | None = None) -> bool:" in updated
    assert "return config_path is not None" in updated
    assert "return False" not in updated
    assert "def __init__(self, config_path: Path | None = None) -> None:" in updated
    assert updated.count("self.should_bootstrap = _should_bootstrap_config(config_path)") == 2
    assert updated.count("def __getattr__(self, name: str) -> Any:") == 2
    assert "def jobs(self) -> JobsClient:" in updated
    assert "def jobs(self) -> AsyncJobsClient:" in updated
    assert "def secrets(self) -> SecretsResource:" in updated
    assert "def secrets(self) -> AsyncSecretsResource:" in updated
    assert "self.value = 1" not in updated
    assert "self.value = 2" not in updated

    class BlockNemoPlatformExt:
        def find_spec(
            self,
            fullname: str,
            path: object | None = None,
            target: object | None = None,
        ) -> None:
            del path, target
            if fullname == "nemo_platform_ext" or fullname.startswith("nemo_platform_ext."):
                raise ModuleNotFoundError("nemo_platform_ext must not be imported")
            return None

    blocked_finder = BlockNemoPlatformExt()
    module_names = (
        "nemo_platform",
        "nemo_platform._base_client",
        "nemo_platform._client",
        "nemo_platform_plugin",
        "nemo_platform_plugin.client",
        "nemo_platform_plugin.client.constants",
        "nemo_platform_plugin.client.tls",
        "nemo_platform_plugin.secrets",
        "nemo_platform_plugin.secrets.compat",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(sdk_path / "src"))
    sys.meta_path.insert(0, blocked_finder)
    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        generated_client = import_module("nemo_platform._client")

        assert generated_client.NeMoPlatform(config_path=Path("config.yaml")).should_bootstrap is True
    finally:
        if blocked_finder in sys.meta_path:
            sys.meta_path.remove(blocked_finder)
        for module_name in module_names:
            sys.modules.pop(module_name, None)
