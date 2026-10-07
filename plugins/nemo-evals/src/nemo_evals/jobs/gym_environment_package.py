# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-time validation for Gym environment FileSets.

This intentionally mirrors ``sandboxed_gym.environment_package`` without importing it.
Evaluator needs these checks in the service process, while the complete filesystem validator
belongs to the sandbox runtime. Extract the shared contract once both consumers are stable.

Submit accepts ``native-v1`` and ``wheels-v1``. Listing validation does not import customer code.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

#: Package identity file at the FileSet root. Submit and the Gym host both key off this name.
ENVIRONMENT_MANIFEST_FILENAME = "nemo-environment.yaml"
#: Flat directory of ``.whl`` files that wheels-v1 installs with ``--no-index --find-links``.
WHEELS_V1_SUBDIR = "wheels"
#: Subdirectory for a custom Gym agent.
CUSTOM_AGENT_SUBDIR = "responses_api_agents"
#: Subdirectory for a custom Gym resources server.
CUSTOM_RESOURCES_SERVER_SUBDIR = "resources_servers"
#: Gym model configs. A native-v1 ``config_paths`` entry may name one; other files in this tree are rejected.
OPERATOR_MODEL_SUBDIR = "responses_api_models"


class GymEnvironmentPackageError(ValueError):
    """Raised when an environment FileSet violates its submit-time contract."""


class EnvironmentFormat(StrEnum):
    """Supported environment dependency-resolution strategies."""

    NATIVE_V1 = "native-v1"
    WHEELS_V1 = "wheels-v1"


class EnvironmentMetadata(BaseModel):
    """Identity and optional provenance carried by an environment package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2048)
    hub_id: str | None = Field(default=None, max_length=512)
    vf_env_id: str | None = Field(default=None, max_length=512)
    adapter_agent: str | None = Field(default=None, max_length=255)


def _validate_relative_config_path(value: str) -> str:
    """Reject absolute, escaped, or traversing config paths before any filesystem access."""
    if not value or value != value.strip():
        raise ValueError("config paths must be non-empty and cannot have surrounding whitespace")
    if "\\" in value:
        raise ValueError("config paths must use POSIX '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("config paths must be relative to the environment root")
    if path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError("config paths cannot contain '.' or '..' traversal")

    return value


class _ManifestBase(BaseModel):
    """Fields shared by every complete Gym environment package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_paths: tuple[str, ...] = Field(min_length=1)
    metadata: EnvironmentMetadata

    @field_validator("config_paths")
    @classmethod
    def _validate_config_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize relative config paths and reject duplicates."""
        validated = tuple(_validate_relative_config_path(value) for value in values)
        if len(set(validated)) != len(validated):
            raise ValueError("config_paths cannot contain duplicates")
        return validated


class NativeV1Manifest(_ManifestBase):
    """A complete environment whose dependencies resolve through a package index."""

    format: Literal[EnvironmentFormat.NATIVE_V1] = EnvironmentFormat.NATIVE_V1

    @field_validator("config_paths")
    @classmethod
    def _under_gym_component_directories(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Keep native-v1 configs under a Gym server tree."""
        allowed = (
            f"{CUSTOM_AGENT_SUBDIR}/",
            f"{CUSTOM_RESOURCES_SERVER_SUBDIR}/",
            f"{OPERATOR_MODEL_SUBDIR}/",
        )
        for value in values:
            if not value.startswith(allowed):
                raise ValueError(f"native-v1 config_paths must be under {allowed}: {value!r}")
        return values


class WheelsV1Manifest(_ManifestBase):
    """A complete environment whose dependencies resolve from its wheelhouse."""

    format: Literal[EnvironmentFormat.WHEELS_V1] = EnvironmentFormat.WHEELS_V1


EnvironmentManifest = Annotated[NativeV1Manifest | WheelsV1Manifest, Field(discriminator="format")]
_ENVIRONMENT_MANIFEST_ADAPTER = TypeAdapter(EnvironmentManifest)


def parse_environment_manifest(raw_yaml: bytes | str) -> EnvironmentManifest:
    """Parse manifest content without filesystem access or customer-code imports."""
    try:
        raw_manifest = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise GymEnvironmentPackageError(f"{ENVIRONMENT_MANIFEST_FILENAME} is not valid YAML") from exc

    if not isinstance(raw_manifest, dict):
        raise GymEnvironmentPackageError(f"{ENVIRONMENT_MANIFEST_FILENAME} must be a mapping")

    try:
        return _ENVIRONMENT_MANIFEST_ADAPTER.validate_python(raw_manifest)
    except ValidationError as exc:
        raise GymEnvironmentPackageError(f"{ENVIRONMENT_MANIFEST_FILENAME} is invalid: {exc}") from exc


def validate_environment_manifest_against_listing(
    manifest: EnvironmentManifest,
    paths: Iterable[str],
) -> None:
    """Validate package rules decidable from a remote FileSet listing."""
    entries = {path.removeprefix("./") for path in paths}

    # A declared config_path may name a model config. Any other file in that tree is rejected.
    declared_configs = set(manifest.config_paths)
    customer_model_files = sorted(
        path for path in entries if path.startswith(f"{OPERATOR_MODEL_SUBDIR}/") and path not in declared_configs
    )
    if customer_model_files:
        raise GymEnvironmentPackageError(
            f"customer-provided {OPERATOR_MODEL_SUBDIR} are not supported; model configuration is operator-owned: "
            f"{', '.join(customer_model_files)}"
        )

    # Prompts are a dataset, not an environment. Mixing them here would stage eval data into
    # the read-only Gym mount and skip the dataset FileSet path.
    prompt_files = sorted(path for path in entries if path.endswith(".jsonl"))
    if prompt_files:
        raise GymEnvironmentPackageError(
            "prompt JSONL must not live in an environment package: "
            f"{', '.join(prompt_files)}; use a separate dataset FileSet"
        )

    # Fail at submit if the manifest names configs the FileSet does not actually contain.
    missing_configs = sorted(path for path in manifest.config_paths if path not in entries)
    if missing_configs:
        raise GymEnvironmentPackageError(
            f"config_paths reference files that are not in the package: {', '.join(missing_configs)}"
        )

    if not isinstance(manifest, WheelsV1Manifest):
        return

    # wheels-v1 installs with ``--find-links wheels/``. Nested dirs and non-wheels would be
    # skipped by that installer and look like a successful empty install.
    wheel_entries = sorted(path for path in entries if path.startswith(f"{WHEELS_V1_SUBDIR}/"))
    nested_entries = [path for path in wheel_entries if "/" in path.removeprefix(f"{WHEELS_V1_SUBDIR}/")]
    if nested_entries:
        raise GymEnvironmentPackageError(
            f"{WHEELS_V1_SUBDIR}/ must be flat; nested entries are not supported: {', '.join(nested_entries)}"
        )
    if not wheel_entries:
        raise GymEnvironmentPackageError(
            f"{EnvironmentFormat.WHEELS_V1.value} installs environment dependencies from a non-empty "
            f"{WHEELS_V1_SUBDIR}/ directory"
        )

    non_wheels = [path for path in wheel_entries if not path.endswith(".whl")]
    if non_wheels:
        raise GymEnvironmentPackageError(f"non-wheel files in {WHEELS_V1_SUBDIR}/: {', '.join(non_wheels)}")
