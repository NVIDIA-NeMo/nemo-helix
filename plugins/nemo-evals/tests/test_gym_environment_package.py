# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-time Gym environment FileSet contract."""

import pytest
from nemo_evals.jobs.gym_environment_package import (
    GymEnvironmentPackageError,
    parse_environment_manifest,
    validate_environment_manifest_against_listing,
)

NATIVE_MANIFEST = """
format: native-v1
config_paths:
  - resources_servers/custom/configs/custom.yaml
metadata:
  name: custom
"""

WHEELS_MANIFEST = """
format: wheels-v1
config_paths:
  - resources_servers/custom/configs/custom.yaml
metadata:
  name: custom
"""

WHEELS_LISTING = [
    "nemo-environment.yaml",
    "resources_servers/custom/configs/custom.yaml",
    "resources_servers/custom/requirements.txt",
    "wheels/custom_dependency-1.0-py3-none-any.whl",
]


def test_wheels_v1_manifest_is_accepted_against_a_valid_listing() -> None:
    manifest = parse_environment_manifest(WHEELS_MANIFEST)
    validate_environment_manifest_against_listing(manifest, WHEELS_LISTING)


def test_native_v1_accepts_a_declared_model_config() -> None:
    config_path = "responses_api_models/vllm_model/configs/policy_model.yaml"
    manifest = parse_environment_manifest(
        f"format: native-v1\nconfig_paths:\n  - {config_path}\nmetadata:\n  name: custom\n"
    )

    validate_environment_manifest_against_listing(manifest, ["nemo-environment.yaml", config_path])
    assert manifest.config_paths == (config_path,)


def test_native_v1_rejects_undeclared_model_files() -> None:
    config_path = "responses_api_models/vllm_model/configs/policy_model.yaml"
    manifest = parse_environment_manifest(
        f"format: native-v1\nconfig_paths:\n  - {config_path}\nmetadata:\n  name: custom\n"
    )

    with pytest.raises(GymEnvironmentPackageError, match="model configuration is operator-owned"):
        validate_environment_manifest_against_listing(
            manifest,
            [config_path, "responses_api_models/vllm_model/app.py"],
        )


def test_native_v1_manifest_is_accepted_against_a_valid_listing() -> None:
    manifest = parse_environment_manifest(NATIVE_MANIFEST)
    validate_environment_manifest_against_listing(
        manifest,
        [
            "nemo-environment.yaml",
            "resources_servers/custom/configs/custom.yaml",
            "resources_servers/custom/requirements.txt",
        ],
    )


@pytest.mark.parametrize(
    "raw_manifest",
    [
        "format: adapter-wheels-v1\nconfig_paths: [configs/test.yaml]\nmetadata: {name: test}\n",
        "format: native-v1\nconfig_paths: []\nmetadata: {name: test}\n",
        "format: native-v1\nconfig_paths: [../test.yaml]\nmetadata: {name: test}\n",
        "format: wheels-v1\nconfig_paths: [configs/test.yaml]\n",
    ],
)
def test_invalid_manifests_are_rejected(raw_manifest: str) -> None:
    with pytest.raises(GymEnvironmentPackageError):
        parse_environment_manifest(raw_manifest)


@pytest.mark.parametrize(
    ("listing", "error"),
    [
        (
            [
                "nemo-environment.yaml",
                "resources_servers/custom/configs/custom.yaml",
                "resources_servers/custom/requirements.txt",
            ],
            "non-empty wheels/ directory",
        ),
        (
            [
                "nemo-environment.yaml",
                "resources_servers/custom/configs/custom.yaml",
                "resources_servers/custom/requirements.txt",
                "wheels/requirements.txt",
            ],
            "non-wheel files",
        ),
        (
            [
                "nemo-environment.yaml",
                "resources_servers/custom/configs/custom.yaml",
                "wheels/custom_dependency-1.0-py3-none-any.whl",
            ],
            "exactly one of requirements.txt or pyproject.toml.*resources_servers/custom",
        ),
        (
            [
                "nemo-environment.yaml",
                "resources_servers/custom/configs/custom.yaml",
                "resources_servers/custom/requirements.txt",
                "resources_servers/custom/pyproject.toml",
                "wheels/custom_dependency-1.0-py3-none-any.whl",
            ],
            "exactly one of requirements.txt or pyproject.toml.*resources_servers/custom",
        ),
        (
            [
                "nemo-environment.yaml",
                "resources_servers/custom/configs/custom.yaml",
                "responses_api_models/customer/configs/customer.yaml",
                "wheels/custom_dependency-1.0-py3-none-any.whl",
            ],
            "model configuration is operator-owned",
        ),
        (["nemo-environment.yaml"], "config_paths reference files that are not in the package"),
        (
            [
                "nemo-environment.yaml",
                "resources_servers/custom/configs/custom.yaml",
                "training.jsonl",
                "wheels/custom_dependency-1.0-py3-none-any.whl",
            ],
            "prompt JSONL",
        ),
    ],
)
def test_wheels_listing_rejections(listing: list[str], error: str) -> None:
    manifest = parse_environment_manifest(WHEELS_MANIFEST)
    with pytest.raises(GymEnvironmentPackageError, match=error):
        validate_environment_manifest_against_listing(manifest, listing)


def test_native_v1_requires_an_install_marker_too() -> None:
    """Gym discovers servers the same way for both formats."""
    manifest = parse_environment_manifest(NATIVE_MANIFEST)
    with pytest.raises(GymEnvironmentPackageError, match="exactly one of requirements.txt or pyproject.toml"):
        validate_environment_manifest_against_listing(
            manifest, ["nemo-environment.yaml", "resources_servers/custom/configs/custom.yaml"]
        )


def test_an_extras_only_wheels_v1_listing_is_accepted() -> None:
    """A FileSet may carry just the extra wheels a built-in environment needs, and no component."""
    manifest = parse_environment_manifest("format: wheels-v1\nconfig_paths: []\nmetadata:\n  name: extras\n")
    validate_environment_manifest_against_listing(
        manifest, ["nemo-environment.yaml", "wheels/extra_dep-1.0-py3-none-any.whl"]
    )


def test_native_v1_without_config_paths_is_rejected() -> None:
    with pytest.raises(GymEnvironmentPackageError, match="at least one config"):
        parse_environment_manifest("format: native-v1\nconfig_paths: []\nmetadata:\n  name: x\n")


def test_a_shipped_server_directory_needs_its_marker_even_when_the_config_lives_elsewhere() -> None:
    """Gym runs the directory the YAML names, not the directory the YAML sits in."""
    manifest = parse_environment_manifest(
        "format: wheels-v1\nconfig_paths:\n  - configs/custom.yaml\nmetadata:\n  name: custom\n"
    )
    listing = [
        "nemo-environment.yaml",
        "configs/custom.yaml",
        "resources_servers/custom/app.py",
        "wheels/d-1.0-py3-none-any.whl",
    ]

    with pytest.raises(GymEnvironmentPackageError, match="resources_servers/custom"):
        validate_environment_manifest_against_listing(manifest, listing)

    validate_environment_manifest_against_listing(manifest, [*listing, "resources_servers/custom/requirements.txt"])
