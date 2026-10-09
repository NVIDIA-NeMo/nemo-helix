# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts and layered validation for Gym environment FileSets."""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

#: Package identity file at the FileSet root. Submit and the Gym host both key off this name.
ENVIRONMENT_MANIFEST_FILENAME = "nemo-environment.yaml"
NATIVE_V1_FORMAT = "native-v1"
WHEELS_V1_FORMAT = "wheels-v1"
ADAPTER_WHEELS_V1_FORMAT = "adapter-wheels-v1"
#: Harnesses the training image ships. An adapter package selects one; it does not vendor the agent.
IMAGE_ADAPTER_ALLOWLIST: dict[str, str] = {
    "verifiers_agent": "responses_api_agents/verifiers_agent",
}
#: Flat directory of ``.whl`` files that wheels-v1 installs with ``--no-index --find-links``.
WHEELS_V1_SUBDIR = "wheels"
#: Subdirectory for a custom Gym agent.
CUSTOM_AGENT_SUBDIR = "responses_api_agents"
#: Subdirectory for a custom Gym resources server.
CUSTOM_RESOURCES_SERVER_SUBDIR = "resources_servers"
#: Gym model configs. A native-v1 ``config_paths`` entry may name one; other files in this tree are rejected.
OPERATOR_MODEL_SUBDIR = "responses_api_models"
SERVER_INSTALL_MARKERS = ("requirements.txt", "pyproject.toml")


class EnvironmentPackageError(ValueError):
    """Raised when an environment FileSet violates its packaging contract."""


class EnvironmentFormat(StrEnum):
    """Supported environment dependency-resolution strategies."""

    NATIVE_V1 = NATIVE_V1_FORMAT
    WHEELS_V1 = WHEELS_V1_FORMAT
    ADAPTER_WHEELS_V1 = ADAPTER_WHEELS_V1_FORMAT


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

    config_paths: tuple[str, ...] = Field(default=(), validate_default=True)
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
        if not values:
            raise ValueError("native-v1 config_paths must name at least one config")
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
    """A complete environment whose dependencies resolve from its wheelhouse.

    ``config_paths`` may be empty: the package then adds only wheels, and the job runs the
    built-in agent and resources server it selected.
    """

    format: Literal[EnvironmentFormat.WHEELS_V1] = EnvironmentFormat.WHEELS_V1


class AdapterRef(BaseModel):
    """Agent harness an adapter-wheels-v1 package runs on, supplied by the training image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str = Field(min_length=1, max_length=255)
    agent_type: Literal["responses_api_agents"] = "responses_api_agents"
    image_config_root: str | None = Field(default=None, max_length=512)


class AdapterWheelsV1Manifest(_ManifestBase):
    """A wheelhouse plus a config that selects an agent already built into the image."""

    format: Literal[EnvironmentFormat.ADAPTER_WHEELS_V1] = EnvironmentFormat.ADAPTER_WHEELS_V1
    adapter: AdapterRef

    @model_validator(mode="after")
    def _adapter_contract(self) -> "AdapterWheelsV1Manifest":
        if not self.config_paths:
            raise ValueError("adapter-wheels-v1 config_paths must name at least one config")
        for value in self.config_paths:
            if not value.startswith("configs/"):
                raise ValueError(f"adapter-wheels-v1 config_paths should live under configs/: {value!r}")
        if self.adapter.agent not in IMAGE_ADAPTER_ALLOWLIST:
            supported = ", ".join(sorted(IMAGE_ADAPTER_ALLOWLIST))
            raise ValueError(
                f"adapter.agent {self.adapter.agent!r} is not built into the training image. Supported: {supported}"
            )
        return self


EnvironmentManifest = Annotated[
    NativeV1Manifest | WheelsV1Manifest | AdapterWheelsV1Manifest,
    Field(discriminator="format"),
]
_ENVIRONMENT_MANIFEST_ADAPTER = TypeAdapter(EnvironmentManifest)


@dataclass(frozen=True)
class NativeV1Package:
    """Validated source package whose dependencies resolve from a package index.

    Validation checks its manifest and layout without importing customer code.
    """

    root: Path
    manifest: NativeV1Manifest
    config_paths: tuple[Path, ...]


@dataclass(frozen=True)
class WheelsV1Package:
    """Validated environment plus wheelhouse ready for host installation."""

    root: Path
    manifest: WheelsV1Manifest
    config_paths: tuple[Path, ...]
    wheelhouse_path: Path
    wheel_files: tuple[Path, ...]


@dataclass(frozen=True)
class AdapterWheelsV1Package:
    """Validated adapter package: image agent, package config, and a vendored wheelhouse."""

    root: Path
    manifest: AdapterWheelsV1Manifest
    config_paths: tuple[Path, ...]
    wheelhouse_path: Path
    wheel_files: tuple[Path, ...]


EnvironmentPackage = NativeV1Package | WheelsV1Package | AdapterWheelsV1Package


def _vendors_wheels(manifest: EnvironmentManifest) -> bool:
    return isinstance(manifest, (WheelsV1Manifest, AdapterWheelsV1Manifest))


def parse_environment_manifest(raw_yaml: bytes | str) -> EnvironmentManifest:
    """Parse manifest content without inspecting files or importing customer code."""
    import yaml

    try:
        raw_manifest = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise EnvironmentPackageError(f"{ENVIRONMENT_MANIFEST_FILENAME} is not valid YAML") from exc

    if not isinstance(raw_manifest, dict):
        raise EnvironmentPackageError(f"{ENVIRONMENT_MANIFEST_FILENAME} must be a mapping")

    try:
        return _ENVIRONMENT_MANIFEST_ADAPTER.validate_python(raw_manifest)
    except ValidationError as exc:
        raise EnvironmentPackageError(f"{ENVIRONMENT_MANIFEST_FILENAME} is invalid: {exc}") from exc


def load_environment_manifest(environment_root: str | Path) -> EnvironmentManifest:
    """Load and type-check ``nemo-environment.yaml`` without importing customer code."""
    root = _validated_root(environment_root)
    manifest_path = root / ENVIRONMENT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise EnvironmentPackageError(f"environment manifest does not exist or is not a file: {manifest_path}")
    return parse_environment_manifest(manifest_path.read_bytes())


def shipped_server_directories(entries: Iterable[str]) -> list[str]:
    """``<kind>/<name>`` directories the package ships under the agent and resources-server trees.

    Gym starts a server from the directory its config YAML names, which need not be where the
    YAML itself lives, so the package's directories are what count, not the config paths.
    """
    directories: list[str] = []
    for entry in entries:
        parts = PurePosixPath(entry).parts
        if len(parts) >= 3 and parts[0] in (CUSTOM_RESOURCES_SERVER_SUBDIR, CUSTOM_AGENT_SUBDIR):
            directory = f"{parts[0]}/{parts[1]}"
            if directory not in directories:
                directories.append(directory)
    return directories


def missing_server_install_markers(entries: set[str]) -> list[str]:
    """Shipped server directories with no ``requirements.txt`` or ``pyproject.toml``, or with both.

    Gym recognises a server directory only by one of those files and refuses a directory that has
    both. A directory without one is invisible: Gym silently runs a same-named built-in instead, or
    fails at startup with its own error.
    """
    problems: list[str] = []
    for directory in shipped_server_directories(entries):
        present = [marker for marker in SERVER_INSTALL_MARKERS if f"{directory}/{marker}" in entries]
        if len(present) != 1:
            problems.append(directory)
    return problems


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
        raise EnvironmentPackageError(
            f"Custom {OPERATOR_MODEL_SUBDIR} directory is not supported; model configuration is operator-owned: "
            f"{', '.join(customer_model_files)}"
        )

    # Prompts are a dataset, not an environment. Mixing them here would stage eval data into
    # the read-only Gym mount and skip the dataset FileSet path.
    prompt_files = sorted(path for path in entries if path.endswith(".jsonl"))
    if prompt_files:
        raise EnvironmentPackageError(
            "JSONL files must not live in an environment package: "
            f"{', '.join(prompt_files)}; use a separate dataset FileSet"
        )

    # Fail at submit if the manifest names configs the FileSet does not actually contain.
    missing_configs = sorted(path for path in manifest.config_paths if path not in entries)
    if missing_configs:
        raise EnvironmentPackageError(
            f"The `config_paths` field references files that are not in the package: {', '.join(missing_configs)}"
        )

    bad_server_dirs = missing_server_install_markers(entries)
    if bad_server_dirs:
        raise EnvironmentPackageError(
            "each agent and resources-server directory needs exactly one of requirements.txt or pyproject.toml "
            f"(Gym discovers servers by that file): {', '.join(bad_server_dirs)}"
        )

    if not _vendors_wheels(manifest):
        return

    # wheels-v1 installs with ``--find-links wheels/``. Nested dirs and non-wheels would be
    # skipped by that installer and look like a successful empty install.
    wheel_entries = sorted(path for path in entries if path.startswith(f"{WHEELS_V1_SUBDIR}/"))
    nested_entries = [path for path in wheel_entries if "/" in path.removeprefix(f"{WHEELS_V1_SUBDIR}/")]
    if nested_entries:
        raise EnvironmentPackageError(
            f"{WHEELS_V1_SUBDIR}/ must be flat; nested entries are not supported: {', '.join(nested_entries)}"
        )

    if not wheel_entries:
        raise EnvironmentPackageError(
            f"A {manifest.format.value} package installs environment dependencies from a non-empty "
            f"{WHEELS_V1_SUBDIR}/ directory"
        )

    non_wheels = [path for path in wheel_entries if not path.endswith(".whl")]
    if non_wheels:
        raise EnvironmentPackageError(f"non-wheel files in {WHEELS_V1_SUBDIR}/: {', '.join(non_wheels)}")


def duplicate_wheel_distributions(wheelhouse_path: Path) -> dict[str, list[str]]:
    """Return distributions represented by more than one wheel version."""
    versions: dict[str, list[str]] = defaultdict(list)
    for wheel in sorted(wheelhouse_path.glob("*.whl")):
        parts = wheel.name.split("-")
        if len(parts) < 5:
            continue
        normalized_name = re.sub(r"[-_.]+", "-", parts[0]).lower()
        versions[normalized_name].append(parts[1])
    return {name: found for name, found in versions.items() if len(found) > 1}


@dataclass(frozen=True)
class EnvironmentComponents:
    """Selectable instance names declared by top-level keys in package config files.

    These names come only from config YAML; source-directory names are tracked
    separately for collision detection.
    """

    agents: frozenset[str]
    resources_servers: frozenset[str]


@dataclass(frozen=True)
class ComponentNamespaces:
    """All names occupied during Gym discovery and config merging.

    Each namespace includes config instance names and immediate source-directory
    names because Gym can encounter both while resolving a component.
    """

    agents: frozenset[str]
    resources_servers: frozenset[str]


def inspect_environment_components(package: EnvironmentPackage) -> EnvironmentComponents:
    """Return the component instances declared by the package's config files.

    This reads only the YAML files named by the manifest and does not import customer
    code. Duplicate instance names are rejected because Gym could not reliably choose
    which config should define that instance. Customer model entries are also rejected
    because model configuration belongs to the runtime image.
    """
    agents: set[str] = set()
    resources_servers: set[str] = set()
    for config_path in package.config_paths:
        # A manifest may split its agent and resources-server declarations across several
        # YAML files, so collect names across the whole package before checking uniqueness.
        config_agents, config_resources_servers = _read_component_instances(
            config_path,
            reject_customer_models=True,
        )
        for name in config_agents:
            _add_unique_component(agents, name, CUSTOM_AGENT_SUBDIR, config_path)
        for name in config_resources_servers:
            _add_unique_component(resources_servers, name, CUSTOM_RESOURCES_SERVER_SUBDIR, config_path)

    return EnvironmentComponents(
        agents=frozenset(agents),
        resources_servers=frozenset(resources_servers),
    )


def inspect_environment_namespaces(
    package: EnvironmentPackage,
    *,
    components: EnvironmentComponents | None = None,
) -> ComponentNamespaces:
    """Return every agent and resources-server name occupied by the package.

    Gym uses both config instance names and component directory names during lookup.
    Combining them lets callers detect cross-type collisions before Gym starts. Pass
    ``components`` when the config files have already been inspected.
    """
    declared = components or inspect_environment_components(package)
    # Directory names affect collision detection, but eval selection still uses the
    # config-declared names held in EnvironmentComponents.
    return ComponentNamespaces(
        agents=frozenset({*declared.agents, *_component_directory_names(package.root, CUSTOM_AGENT_SUBDIR)}),
        resources_servers=frozenset(
            {
                *declared.resources_servers,
                *_component_directory_names(package.root, CUSTOM_RESOURCES_SERVER_SUBDIR),
            }
        ),
    )


def validate_environment_namespaces(package_namespaces: ComponentNamespaces) -> None:
    """Reject a name that the package uses for both component types.

    Gym merges these namespaces while discovering components. Allowing the same name
    on both sides could load an agent when a resources server was requested, or vice
    versa.
    """
    cross_type = sorted(package_namespaces.agents & package_namespaces.resources_servers)
    if cross_type:
        raise EnvironmentPackageError(
            f"Component names cannot be used as both agents and resources servers: {', '.join(cross_type)}"
        )


def _component_directory_names(root: Path, component_type: str) -> set[str]:
    """Return immediate child directories below a Gym component root.

    Files, nested paths, and an absent component root do not occupy discovery names.
    """
    component_root = root / component_type
    if not component_root.is_dir():
        return set()
    return {child.name for child in component_root.iterdir() if child.is_dir()}


def _read_component_instances(
    config_path: Path,
    *,
    reject_customer_models: bool,
) -> tuple[set[str], set[str]]:
    """Extract component instance names from one Gym config without importing it.

    Each top-level mapping key is an instance name. Its nested keys identify whether
    that instance configures an agent, a resources server, or another Gym component.
    """
    import yaml

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EnvironmentPackageError(f"Environment config is not valid YAML: {config_path}") from exc
    if not isinstance(raw_config, dict):
        raise EnvironmentPackageError(f"Environment config must be a mapping: {config_path}")

    agents: set[str] = set()
    resources_servers: set[str] = set()
    for raw_instance_name, raw_instance in raw_config.items():
        # Ignore unrelated YAML sections here. Pydantic/Gym performs full config-shape
        # validation later; this pass only needs enough structure to detect namespaces.
        if not isinstance(raw_instance_name, str) or not isinstance(raw_instance, dict):
            continue
        # The runtime image and selected VirtualModel own model configuration. Rejecting
        # package model entries prevents customer YAML from overriding that trusted setup.
        if reject_customer_models and OPERATOR_MODEL_SUBDIR in raw_instance:
            raise EnvironmentPackageError(f"Customer-provided {OPERATOR_MODEL_SUBDIR} are not supported: {config_path}")
        if CUSTOM_AGENT_SUBDIR in raw_instance:
            agents.add(raw_instance_name)
        if CUSTOM_RESOURCES_SERVER_SUBDIR in raw_instance:
            # An entry may contain both component keys. Keeping it in both sets lets
            # validate_environment_namespaces report the cross-type collision.
            resources_servers.add(raw_instance_name)
    return agents, resources_servers


def _add_unique_component(components: set[str], name: str, component_type: str, config_path: Path) -> None:
    """Register one instance name across all manifest config files.

    A second declaration for the same component type is ambiguous even when it
    appears in a different YAML file.
    """
    if name in components:
        raise EnvironmentPackageError(
            f"Duplicate {component_type} instance {name!r} declared by environment config: {config_path}"
        )
    components.add(name)


def validate_environment_package_layout(
    environment_root: str | Path,
    manifest: EnvironmentManifest,
) -> None:
    """Validate filesystem-specific package rules after FileSet staging."""
    root = _validated_root(environment_root)
    for relative_path in manifest.config_paths:
        unresolved = root / relative_path
        if unresolved.is_symlink():
            raise EnvironmentPackageError(f"config symlinks are not allowed: {relative_path}")
        _resolve_contained_file(root, relative_path)

    if _vendors_wheels(manifest):
        unresolved_wheelhouse = root / WHEELS_V1_SUBDIR
        if unresolved_wheelhouse.is_symlink():
            raise EnvironmentPackageError(f"wheelhouse symlinks are not allowed: {WHEELS_V1_SUBDIR}")
        wheelhouse = _resolve_contained_directory(root, WHEELS_V1_SUBDIR)
        for wheel in wheelhouse.glob("*.whl"):
            if wheel.is_symlink():
                raise EnvironmentPackageError(f"wheel symlinks are not allowed: {wheel.relative_to(root).as_posix()}")
            _require_contained(root, wheel.resolve(), wheel.relative_to(root).as_posix())

    validate_environment_manifest_against_listing(
        manifest,
        (path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()),
    )

    if _vendors_wheels(manifest):
        duplicates = duplicate_wheel_distributions(root / WHEELS_V1_SUBDIR)
        if duplicates:
            details = ", ".join(f"{name} ({', '.join(versions)})" for name, versions in sorted(duplicates.items()))
            raise EnvironmentPackageError(
                f"The {WHEELS_V1_SUBDIR}/ directory vendors multiple versions of the same distribution: {details}; "
                "regenerate the package with one resolved version per distribution"
            )


def load_environment_package(environment_root: str | Path) -> EnvironmentPackage:
    """Validate one native-v1, wheels-v1, or adapter-wheels-v1 package."""
    root = _validated_root(environment_root)
    manifest = load_environment_manifest(root)
    validate_environment_package_layout(root, manifest)
    config_paths = tuple((root / relative).resolve() for relative in manifest.config_paths)
    if isinstance(manifest, NativeV1Manifest):
        return NativeV1Package(root=root, manifest=manifest, config_paths=config_paths)

    wheelhouse_path = (root / WHEELS_V1_SUBDIR).resolve()
    wheel_files = tuple(path.resolve() for path in sorted(wheelhouse_path.glob("*.whl")))
    if isinstance(manifest, AdapterWheelsV1Manifest):
        return AdapterWheelsV1Package(
            root=root,
            manifest=manifest,
            config_paths=config_paths,
            wheelhouse_path=wheelhouse_path,
            wheel_files=wheel_files,
        )
    return WheelsV1Package(
        root=root,
        manifest=manifest,
        config_paths=config_paths,
        wheelhouse_path=wheelhouse_path,
        wheel_files=wheel_files,
    )


def _validated_root(environment_root: str | Path) -> Path:
    """Resolve the package root and refuse a missing or non-directory path."""
    root = Path(environment_root)
    if not root.is_dir():
        raise EnvironmentPackageError(f"environment root does not exist or is not a directory: {root}")
    return root.resolve()


def _resolve_contained_file(root: Path, relative_path: str) -> Path:
    """Resolve a package file and refuse symlink/path escape outside ``root``."""
    candidate = (root / relative_path).resolve()
    _require_contained(root, candidate, relative_path)
    if not candidate.is_file():
        raise EnvironmentPackageError(f"environment path does not exist or is not a file: {relative_path}")
    return candidate


def _resolve_contained_directory(root: Path, relative_path: str) -> Path:
    """Resolve a package directory and refuse symlink/path escape outside ``root``."""
    candidate = (root / relative_path).resolve()
    _require_contained(root, candidate, relative_path)
    if not candidate.is_dir():
        raise EnvironmentPackageError(f"environment path does not exist or is not a directory: {relative_path}")
    return candidate


def _require_contained(root: Path, candidate: Path, relative_path: str) -> None:
    """Raise if ``candidate`` resolved outside the environment root."""
    if not candidate.is_relative_to(root):
        raise EnvironmentPackageError(f"environment path escapes its root: {relative_path}")
