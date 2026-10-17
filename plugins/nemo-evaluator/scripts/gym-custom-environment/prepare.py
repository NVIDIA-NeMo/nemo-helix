# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and stage complete custom Gym environment inputs.

The shared workflow calls this module after either a developer supplies a
complete ``wheels-v1`` directory and Gym JSONL dataset or the default example
adapter produces those same inputs. This module copies caller-owned inputs into
the run directory, validates the package and dataset without importing customer
code, discovers selectable resources servers, and extracts package metadata
used by generic runtime verification.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.runtimes.gym import discover_gym_tasks
from sandboxed_gym.environment_package import (
    WheelsV1Package,
    inspect_environment_components,
    load_environment_package,
)


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    """Carry validated paths and manifest details into upload and verification."""

    root: Path
    dataset: Path
    name: str
    resources_server: str
    resources_servers: tuple[str, ...]
    wheel_names: tuple[str, ...]
    task_count: int
    input_mode: str

    def evidence(self) -> dict[str, Any]:
        """Return JSON-serializable preparation evidence."""
        return asdict(self)


def _require_safe_copy(source: Path, destination: Path, *, label: str) -> None:
    """Reject copy layouts that could overwrite or recursively copy an input."""
    resolved_source = source.resolve()
    resolved_destination = destination.resolve()
    if (
        resolved_source == resolved_destination
        or resolved_source in resolved_destination.parents
        or resolved_destination in resolved_source.parents
    ):
        raise ValueError(f"{label} source and run destination must not contain one another")


def _copy_custom_inputs(
    environment_source: Path,
    dataset_source: Path,
    *,
    environment_output: Path,
    dataset_output: Path,
) -> None:
    """Copy caller-owned inputs without modifying their original files."""
    if not environment_source.is_dir():
        raise ValueError(f"environment directory does not exist: {environment_source}")
    if not dataset_source.is_file():
        raise ValueError(f"dataset does not exist or is not a file: {dataset_source}")

    # A nested source or destination could erase inputs when an old run
    # directory is removed, or recursively copy the output into itself.
    _require_safe_copy(environment_source, environment_output, label="environment")
    _require_safe_copy(dataset_source, dataset_output, label="dataset")

    if environment_output.exists():
        shutil.rmtree(environment_output)

    environment_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(environment_source, environment_output)

    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_source, dataset_output)


def _select_resources_server(
    available_servers: frozenset[str],
    requested_server: str | None,
) -> str:
    """Select a declared resources server or explain why selection is ambiguous."""
    if requested_server is not None:
        if requested_server not in available_servers:
            raise ValueError(
                f"resources server {requested_server!r} is not declared; "
                f"available servers: {', '.join(sorted(available_servers))}"
            )
        return requested_server

    if len(available_servers) != 1:
        raise ValueError(
            "environment must declare exactly one resources server or select one with "
            f"--resources-server; available: {', '.join(sorted(available_servers)) or 'none'}"
        )
    return next(iter(available_servers))


def inspect_prepared_inputs(
    environment: Path,
    dataset: Path,
    *,
    requested_resources_server: str | None,
    input_mode: str,
) -> PreparedEnvironment:
    """Validate staged inputs and derive runtime choices from their contents."""
    environment_package = load_environment_package(environment)
    if not isinstance(environment_package, WheelsV1Package):
        actual_format = type(environment_package).__name__
        raise ValueError(f"expected a wheels-v1 package, got {actual_format}")

    # Resource-server names come from the package itself; the workflow contains
    # no knowledge of the bundled example's ``ascii_tree`` server.
    components = inspect_environment_components(environment_package)
    resources_server = _select_resources_server(
        components.resources_servers,
        requested_resources_server,
    )
    tasks = discover_gym_tasks(dataset)
    if not tasks:
        raise ValueError(f"Gym discovered no tasks in {dataset}")

    # Keep this summary independent of SDK/Pydantic models so it can be logged,
    # serialized as evidence, and passed between workflow stages predictably.
    return PreparedEnvironment(
        root=environment_package.root,
        dataset=dataset.resolve(),
        name=environment_package.manifest.metadata.name,
        resources_server=resources_server,
        resources_servers=tuple(sorted(components.resources_servers)),
        wheel_names=tuple(wheel.name for wheel in environment_package.wheel_files),
        task_count=len(tasks),
        input_mode=input_mode,
    )


def prepare_custom_inputs(
    environment_source: Path,
    dataset_source: Path,
    *,
    environment_output: Path,
    dataset_output: Path,
    requested_resources_server: str | None,
) -> PreparedEnvironment:
    """Stage and inspect a developer-provided wheels-v1 package and Gym dataset.

    Validation happens once against the source tree to preserve symlink checks,
    then again against the exact copy that later stages upload.
    """
    # Validate the caller-owned tree before copying so symlinks cannot be
    # dereferenced into apparently valid files inside the run directory.
    source_package = load_environment_package(environment_source)
    if not isinstance(source_package, WheelsV1Package):
        actual_format = type(source_package).__name__
        raise ValueError(f"expected a wheels-v1 package, got {actual_format}")

    _copy_custom_inputs(
        environment_source,
        dataset_source,
        environment_output=environment_output,
        dataset_output=dataset_output,
    )
    return inspect_prepared_inputs(
        environment_output,
        dataset_output,
        requested_resources_server=requested_resources_server,
        input_mode="custom",
    )
