# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from sandboxed_gym.environment_package import (
    IMAGE_ADAPTER_ALLOWLIST,
    AdapterRef,
    AdapterWheelsV1Manifest,
    AdapterWheelsV1Package,
    ComponentNamespaces,
    EnvironmentFormat,
    EnvironmentMetadata,
    EnvironmentPackageError,
    NativeV1Manifest,
    NativeV1Package,
    WheelsV1Manifest,
    WheelsV1Package,
    duplicate_wheel_distributions,
    inspect_environment_components,
    inspect_environment_namespaces,
    load_environment_manifest,
    load_environment_package,
    parse_environment_manifest,
    validate_environment_manifest_against_listing,
    validate_environment_namespaces,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
WHEELS_V1_ENVIRONMENT = FIXTURES_DIR / "wheels_v1_environment"
NATIVE_V1_CUSTOM_ENVIRONMENT = FIXTURES_DIR / "native_v1_custom_environment"


def _write_manifest(root: Path, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "nemo-environment.yaml").write_text(text, encoding="utf-8")


def _complete_manifest(format_name: str, config_path: str) -> str:
    return f"format: {format_name}\nconfig_paths:\n  - {config_path}\nmetadata:\n  name: test-environment\n"


def _write_config(root: Path, relative_path: str) -> Path:
    config = root / relative_path
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("test: {}\n", encoding="utf-8")
    # Gym only discovers a server directory that carries an install marker.
    server_dir = config.parent.parent
    if server_dir.parent.name in ("resources_servers", "responses_api_agents"):
        (server_dir / "requirements.txt").write_text("", encoding="utf-8")
    return config


def test_loads_complete_wheels_v1_fixture() -> None:
    package = load_environment_package(WHEELS_V1_ENVIRONMENT)

    assert isinstance(package, WheelsV1Package)
    assert package.manifest == WheelsV1Manifest(
        format=EnvironmentFormat.WHEELS_V1,
        config_paths=("resources_servers/wheels_v1_example/configs/wheels_v1_example.yaml",),
        metadata=EnvironmentMetadata(name="wheels-v1-environment"),
    )
    assert package.config_paths == (
        (WHEELS_V1_ENVIRONMENT / "resources_servers/wheels_v1_example/configs/wheels_v1_example.yaml").resolve(),
    )
    assert package.wheelhouse_path == (WHEELS_V1_ENVIRONMENT / "wheels").resolve()
    assert package.wheel_files == (
        (WHEELS_V1_ENVIRONMENT / "wheels/example_dependency-1.0-py3-none-any.whl").resolve(),
    )


def test_loads_adapter_wheels_v1_package(tmp_path: Path) -> None:
    """The format the platform writes for a hub environment. The host must accept it."""
    config = "configs/verifiers_agent.yaml"
    _write_config(tmp_path, config)
    wheel = tmp_path / "wheels" / "example_dependency-1.0.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"PK\x03\x04")
    _write_manifest(
        tmp_path,
        "format: adapter-wheels-v1\n"
        "adapter:\n"
        "  agent: verifiers_agent\n"
        "  agent_type: responses_api_agents\n"
        "  image_config_root: responses_api_agents/verifiers_agent\n"
        "config_paths:\n"
        "  - configs/verifiers_agent.yaml\n"
        "metadata:\n"
        "  name: ascii-tree\n"
        "  hub_id: primeintellect/ascii-tree\n"
        "  vf_env_id: ascii-tree\n"
        "  adapter_agent: verifiers_agent\n",
    )

    package = load_environment_package(tmp_path)

    assert isinstance(package, AdapterWheelsV1Package)
    assert package.manifest == AdapterWheelsV1Manifest(
        format=EnvironmentFormat.ADAPTER_WHEELS_V1,
        adapter=AdapterRef(
            agent="verifiers_agent",
            image_config_root=IMAGE_ADAPTER_ALLOWLIST["verifiers_agent"],
        ),
        config_paths=(config,),
        metadata=EnvironmentMetadata(
            name="ascii-tree",
            hub_id="primeintellect/ascii-tree",
            vf_env_id="ascii-tree",
            adapter_agent="verifiers_agent",
        ),
    )
    assert package.wheel_files == (wheel.resolve(),)


def test_adapter_wheels_rejects_an_agent_the_image_does_not_ship(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "format: adapter-wheels-v1\n"
        "adapter:\n"
        "  agent: not_in_the_image\n"
        "config_paths:\n"
        "  - configs/verifiers_agent.yaml\n"
        "metadata:\n"
        "  name: ascii-tree\n",
    )

    with pytest.raises(EnvironmentPackageError, match="not built into the training image"):
        load_environment_manifest(tmp_path)


def test_adapter_wheels_requires_a_config_path(tmp_path: Path) -> None:
    """Only wheels-v1 may ship configs-free; an adapter package is nothing without the config
    that selects its image agent."""
    _write_manifest(
        tmp_path,
        "format: adapter-wheels-v1\n"
        "adapter:\n"
        "  agent: verifiers_agent\n"
        "config_paths: []\n"
        "metadata:\n"
        "  name: ascii-tree\n",
    )

    with pytest.raises(EnvironmentPackageError, match="at least one config"):
        load_environment_manifest(tmp_path)


def test_loads_native_v1_fixture() -> None:
    package = load_environment_package(NATIVE_V1_CUSTOM_ENVIRONMENT)

    assert isinstance(package, NativeV1Package)
    assert package.manifest == NativeV1Manifest(
        format=EnvironmentFormat.NATIVE_V1,
        config_paths=("resources_servers/native_v1_custom_greeting/configs/native_v1_custom_greeting.yaml",),
        metadata=EnvironmentMetadata(name="native-v1-custom-greeting"),
    )
    assert package.config_paths == (
        (
            NATIVE_V1_CUSTOM_ENVIRONMENT
            / "resources_servers/native_v1_custom_greeting/configs/native_v1_custom_greeting.yaml"
        ).resolve(),
    )
    components = inspect_environment_components(package)
    assert components.agents == frozenset()
    assert components.resources_servers == frozenset({"native_v1_custom_greeting"})


def test_native_validation_does_not_import_customer_code(tmp_path: Path) -> None:
    marker = tmp_path / "customer-code-ran"
    config = tmp_path / "resources_servers/custom/configs/custom.yaml"
    config.parent.mkdir(parents=True)
    (config.parent.parent / "requirements.txt").write_text("", encoding="utf-8")
    config.write_text("custom: {}\n", encoding="utf-8")
    app = config.parent.parent / "app.py"
    app.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf-8")
    _write_manifest(
        tmp_path, _complete_manifest("native-v1", config_path="resources_servers/custom/configs/custom.yaml")
    )

    package = load_environment_package(tmp_path)

    assert isinstance(package, NativeV1Package)
    assert not marker.exists()


@pytest.mark.parametrize(
    "manifest",
    [
        "format: native-v1\n",
        _complete_manifest("unknown-v1", "resources_servers/test/configs/test.yaml"),
        _complete_manifest("native-v1", "resources_servers/test/configs/test.yaml") + "unexpected: true\n",
        _complete_manifest("[native-v1, wheels-v1]", "resources_servers/test/configs/test.yaml"),
    ],
)
def test_rejects_invalid_or_combined_manifest_contracts(tmp_path: Path, manifest: str) -> None:
    _write_manifest(tmp_path, manifest)

    with pytest.raises(EnvironmentPackageError, match="is invalid"):
        load_environment_manifest(tmp_path)


@pytest.mark.parametrize(
    "config_path",
    [
        "/etc/passwd",
        "../outside.yaml",
        "configs/../../outside.yaml",
        r"configs\windows.yaml",
        " configs/test.yaml",
        "configs/test.yaml ",
    ],
)
def test_rejects_unsafe_native_config_paths(tmp_path: Path, config_path: str) -> None:
    _write_manifest(
        tmp_path,
        f"format: native-v1\nconfig_paths:\n  - {config_path!r}\nmetadata:\n  name: test-environment\n",
    )

    with pytest.raises(EnvironmentPackageError, match="is invalid"):
        load_environment_manifest(tmp_path)


def test_rejects_duplicate_native_config_paths(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "format: native-v1\n"
        "config_paths:\n"
        "  - resources_servers/test/configs/test.yaml\n"
        "  - resources_servers/test/configs/test.yaml\n"
        "metadata:\n"
        "  name: test-environment\n",
    )

    with pytest.raises(EnvironmentPackageError, match="duplicates"):
        load_environment_manifest(tmp_path)


def test_rejects_missing_native_config(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        _complete_manifest("native-v1", "resources_servers/missing/configs/missing.yaml"),
    )

    with pytest.raises(EnvironmentPackageError, match="does not exist or is not a file"):
        load_environment_package(tmp_path)


def test_rejects_native_config_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    outside.write_text("outside: true\n", encoding="utf-8")
    config_path = "resources_servers/test/configs/escape.yaml"
    config = tmp_path / config_path
    config.parent.mkdir(parents=True)
    (config.parent.parent / "requirements.txt").write_text("", encoding="utf-8")
    config.symlink_to(outside)
    _write_manifest(tmp_path, _complete_manifest("native-v1", config_path))

    with pytest.raises(EnvironmentPackageError, match="config symlinks are not allowed"):
        load_environment_package(tmp_path)


def test_rejects_native_config_symlink_within_root(tmp_path: Path) -> None:
    target = tmp_path / "resources_servers/test/configs/actual.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("actual: true\n", encoding="utf-8")
    alias_path = "resources_servers/test/configs/alias.yaml"
    (tmp_path / alias_path).symlink_to(target)
    _write_manifest(tmp_path, _complete_manifest("native-v1", alias_path))

    with pytest.raises(EnvironmentPackageError, match="config symlinks are not allowed"):
        load_environment_package(tmp_path)


def test_rejects_wheelhouse_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-wheels"
    outside.mkdir()
    (outside / "dependency-1.0-py3-none-any.whl").write_bytes(b"fixture")
    (tmp_path / "wheels").symlink_to(outside, target_is_directory=True)
    config_path = "resources_servers/test/configs/test.yaml"
    _write_config(tmp_path, config_path)
    _write_manifest(tmp_path, _complete_manifest("wheels-v1", config_path))

    with pytest.raises(EnvironmentPackageError, match="wheelhouse symlinks are not allowed"):
        load_environment_package(tmp_path)


def test_rejects_wheel_file_symlink(tmp_path: Path) -> None:
    config_path = "resources_servers/test/configs/test.yaml"
    _write_config(tmp_path, config_path)
    _write_manifest(tmp_path, _complete_manifest("wheels-v1", config_path))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-dependency-1.0-py3-none-any.whl"
    outside.write_bytes(b"fixture")
    (wheels / "dependency-1.0-py3-none-any.whl").symlink_to(outside)

    with pytest.raises(EnvironmentPackageError, match="wheel symlinks are not allowed"):
        load_environment_package(tmp_path)


def test_rejects_empty_wheels_v1_package(tmp_path: Path) -> None:
    config_path = "resources_servers/test/configs/test.yaml"
    _write_config(tmp_path, config_path)
    _write_manifest(tmp_path, _complete_manifest("wheels-v1", config_path))
    (tmp_path / "wheels").mkdir()

    with pytest.raises(EnvironmentPackageError, match="non-empty wheels/ directory"):
        load_environment_package(tmp_path)


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "format: [native-v1\n")

    with pytest.raises(EnvironmentPackageError, match="not valid YAML"):
        load_environment_manifest(tmp_path)


def test_parse_manifest_bytes_without_filesystem_access() -> None:
    manifest = parse_environment_manifest(
        _complete_manifest("native-v1", "resources_servers/test/configs/test.yaml").encode()
    )

    assert manifest == NativeV1Manifest(
        format=EnvironmentFormat.NATIVE_V1,
        config_paths=("resources_servers/test/configs/test.yaml",),
        metadata=EnvironmentMetadata(name="test-environment"),
    )


def test_listing_rejects_prompt_jsonl() -> None:
    manifest = parse_environment_manifest(_complete_manifest("native-v1", "resources_servers/test/configs/test.yaml"))

    with pytest.raises(EnvironmentPackageError, match="JSONL files must not live in an environment package"):
        validate_environment_manifest_against_listing(
            manifest,
            ["resources_servers/test/configs/test.yaml", "training.jsonl"],
        )


@pytest.mark.parametrize(
    ("wheel_entries", "error"),
    [
        ([], "non-empty wheels/ directory"),
        (["wheels/nested/dependency-1.0-py3-none-any.whl"], "must be flat"),
        (["wheels/requirements.txt"], "non-wheel files"),
    ],
)
def test_listing_rejects_invalid_wheelhouse_entries(wheel_entries: list[str], error: str) -> None:
    config_path = "resources_servers/test/configs/test.yaml"
    manifest = parse_environment_manifest(_complete_manifest("wheels-v1", config_path))

    with pytest.raises(EnvironmentPackageError, match=error):
        validate_environment_manifest_against_listing(
            manifest, [config_path, "resources_servers/test/requirements.txt", *wheel_entries]
        )


@pytest.mark.parametrize("format_name", ["native-v1", "wheels-v1"])
@pytest.mark.parametrize(
    ("markers", "detail"),
    [
        ([], "no marker"),
        (["requirements.txt", "pyproject.toml"], "both markers"),
    ],
)
def test_listing_requires_exactly_one_install_marker_per_server(
    format_name: str, markers: list[str], detail: str
) -> None:
    """Gym discovers a server only by requirements.txt or pyproject.toml, and refuses a directory with both."""
    config_path = "resources_servers/test/configs/test.yaml"
    manifest = parse_environment_manifest(_complete_manifest(format_name, config_path))
    listing = [config_path, *(f"resources_servers/test/{m}" for m in markers), "wheels/dep-1.0-py3-none-any.whl"]

    with pytest.raises(
        EnvironmentPackageError, match="exactly one of requirements.txt or pyproject.toml.*resources_servers/test"
    ):
        validate_environment_manifest_against_listing(manifest, listing)


def test_listing_accepts_a_pyproject_marker() -> None:
    config_path = "responses_api_agents/my_agent/configs/my_agent.yaml"
    manifest = parse_environment_manifest(_complete_manifest("wheels-v1", config_path))

    validate_environment_manifest_against_listing(
        manifest, [config_path, "responses_api_agents/my_agent/pyproject.toml", "wheels/dep-1.0-py3-none-any.whl"]
    )


def test_duplicate_wheel_distributions_are_rejected(tmp_path: Path) -> None:
    config_path = "resources_servers/test/configs/test.yaml"
    _write_config(tmp_path, config_path)
    _write_manifest(tmp_path, _complete_manifest("wheels-v1", config_path))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "dependency-1.0-py3-none-any.whl").write_bytes(b"fixture")
    (wheels / "dependency-2.0-py3-none-any.whl").write_bytes(b"fixture")

    assert duplicate_wheel_distributions(wheels) == {"dependency": ["1.0", "2.0"]}
    with pytest.raises(EnvironmentPackageError, match="multiple versions"):
        load_environment_package(tmp_path)


def test_native_v1_accepts_a_declared_model_config(tmp_path: Path) -> None:
    config_path = "responses_api_models/vllm_model/configs/policy_model.yaml"
    _write_config(tmp_path, config_path)
    _write_manifest(tmp_path, _complete_manifest("native-v1", config_path))

    package = load_environment_package(tmp_path)

    assert package.manifest.config_paths == (config_path,)


def test_rejects_native_config_outside_gym_server_dirs(tmp_path: Path) -> None:
    _write_manifest(tmp_path, _complete_manifest("native-v1", "configs/policy_model.yaml"))

    with pytest.raises(EnvironmentPackageError, match="responses_api_models/"):
        load_environment_manifest(tmp_path)


def test_listing_rejects_undeclared_model_files() -> None:
    config_path = "responses_api_models/vllm_model/configs/policy_model.yaml"
    manifest = parse_environment_manifest(_complete_manifest("native-v1", config_path))

    with pytest.raises(EnvironmentPackageError, match="model configuration is operator-owned"):
        validate_environment_manifest_against_listing(
            manifest,
            [config_path, "responses_api_models/vllm_model/app.py"],
        )


def test_listing_rejects_customer_model_configuration() -> None:
    config_path = "resources_servers/custom/configs/custom.yaml"
    manifest = parse_environment_manifest(_complete_manifest("wheels-v1", config_path))

    with pytest.raises(EnvironmentPackageError, match="model configuration is operator-owned"):
        validate_environment_manifest_against_listing(
            manifest,
            [
                config_path,
                "responses_api_models/customer/configs/customer.yaml",
                "wheels/dependency-1.0-py3-none-any.whl",
            ],
        )


def test_inspects_top_level_agent_and_resources_server_instance_names(tmp_path: Path) -> None:
    config_path = "responses_api_agents/custom/configs/custom.yaml"
    config = tmp_path / config_path
    config.parent.mkdir(parents=True)
    (config.parent.parent / "requirements.txt").write_text("", encoding="utf-8")
    config.write_text(
        "custom_agent_instance:\n"
        "  responses_api_agents:\n"
        "    custom_implementation: {entrypoint: app.py}\n"
        "custom_resources_instance:\n"
        "  resources_servers:\n"
        "    custom_resources: {entrypoint: app.py}\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path, _complete_manifest("native-v1", config_path))

    components = inspect_environment_components(load_environment_package(tmp_path))

    assert components.agents == frozenset({"custom_agent_instance"})
    assert components.resources_servers == frozenset({"custom_resources_instance"})


def test_rejects_duplicate_component_instances_across_configs(tmp_path: Path) -> None:
    config_paths = (
        "responses_api_agents/first/configs/first.yaml",
        "responses_api_agents/second/configs/second.yaml",
    )
    for config_path in config_paths:
        config = tmp_path / config_path
        config.parent.mkdir(parents=True)
        (config.parent.parent / "requirements.txt").write_text("", encoding="utf-8")
        config.write_text(
            "duplicate_instance:\n  responses_api_agents:\n    implementation: {entrypoint: app.py}\n",
            encoding="utf-8",
        )
    _write_manifest(
        tmp_path,
        "format: native-v1\n"
        "config_paths:\n"
        f"  - {config_paths[0]}\n"
        f"  - {config_paths[1]}\n"
        "metadata:\n"
        "  name: duplicate-components\n",
    )

    with pytest.raises(EnvironmentPackageError, match="Duplicate responses_api_agents instance"):
        inspect_environment_components(load_environment_package(tmp_path))


def test_package_namespace_inventory_includes_source_directory_names(tmp_path: Path) -> None:
    config_path = "responses_api_agents/directory_name/configs/variant.yaml"
    config = tmp_path / config_path
    config.parent.mkdir(parents=True)
    (config.parent.parent / "requirements.txt").write_text("", encoding="utf-8")
    config.write_text(
        "declared_instance:\n  responses_api_agents:\n    implementation: {entrypoint: app.py}\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path, _complete_manifest("native-v1", config_path))
    package = load_environment_package(tmp_path)

    namespaces = inspect_environment_namespaces(package)

    assert namespaces.agents == frozenset({"directory_name", "declared_instance"})


def test_rejects_cross_type_namespace_collisions() -> None:
    package_namespaces = ComponentNamespaces(
        agents=frozenset({"shared_name"}),
        resources_servers=frozenset({"shared_name"}),
    )

    with pytest.raises(EnvironmentPackageError, match="both agents and resources servers"):
        validate_environment_namespaces(package_namespaces)
