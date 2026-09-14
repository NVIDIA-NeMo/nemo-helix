# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for an optimize bundle before it is staged into a fileset.

A submitted optimize study runs on the platform, reading its config and every asset that config
references out of a downloaded fileset.  Nothing on the submitting client's filesystem is visible,
so a config that names ``/Users/me/agents/...`` — or a sibling file the author forgot to include in
the bundle — fails minutes later inside the study loop, with a stack trace from Fabric rather than
an actionable message.

:func:`preflight_bundle` catches those cases up front, at ``prepare-fileset`` time, by walking the
path-bearing keys of the config and checking each one against the bundle directory.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_platform_plugin.refs import FILESET_REF_PATTERN

from nemo_optimization.fabric import FABRIC_AGENT_SCHEMA_VERSION, is_fabric_agent_config, looks_like_nat_config
from nemo_optimization.schemas.optimize import is_fileset_relative


class BundlePreflightError(ValueError):
    """Raised when an optimize bundle cannot be staged as-is."""


@dataclass(frozen=True)
class PathReference:
    """One path-shaped value found in the optimize config.

    ``location`` is the dotted config key it came from, used verbatim in error messages so the
    author can jump straight to the offending line.  ``must_exist`` separates inputs the study
    reads (dataset, ``base_dir``, hook modules, MCP configs) from directories it writes
    (``environment.workspace``, ``runtime.artifacts``) — the latter only have to be relative.
    """

    location: str
    value: str
    must_exist: bool
    is_dir: bool = False


def preflight_bundle(
    source: Path,
    optimize_config: str,
    *,
    agent: str | None = None,
) -> dict[str, Any]:
    """Validate the bundle rooted at *source* and return its parsed optimize config.

    Args:
        source: Directory that will be uploaded as the fileset.
        optimize_config: Path to the optimize YAML, relative to *source*.
        agent: Optional platform agent ref supplying the Agent under Test, for configs that
            carry only the optimizer/eval overlay.

    Raises:
        BundlePreflightError: with every problem found, one per line, so a bundle with several
            bad paths is fixed in one pass rather than one round trip per path.
    """
    config = _load_config(source, optimize_config)
    problems = [
        *_agent_problems(config, agent=agent),
        *_optimizer_problems(config),
        *_path_problems(source, config),
        *_symlink_problems(source),
    ]
    if problems:
        raise BundlePreflightError(
            f"{len(problems)} problem(s) in optimize bundle {str(source)!r}:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    return config


def _load_config(source: Path, optimize_config: str) -> dict[str, Any]:
    if not source.is_dir():
        raise BundlePreflightError(f"--source {str(source)!r} is not a directory.")
    if not is_fileset_relative(optimize_config):
        raise BundlePreflightError(
            f"--optimize-config must be a path relative to --source (no leading '/', no '..' "
            f"segments); got {optimize_config!r}."
        )
    config_path = source / optimize_config
    if not config_path.is_file():
        raise BundlePreflightError(f"--optimize-config {optimize_config!r} was not found under {str(source)!r}.")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BundlePreflightError(f"{optimize_config} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundlePreflightError(f"{optimize_config} must contain a mapping at the top level.")
    return raw


def _agent_problems(config: Mapping[str, Any], *, agent: str | None) -> Iterator[str]:
    """The study needs an Agent under Test: inline in the config, or a platform ``--agent``."""
    if is_fabric_agent_config(config) or bool(agent):
        return
    if looks_like_nat_config(config):
        yield (
            "the config looks like legacy NAT workflow YAML; optimize requires a Fabric-native "
            f"package (schema_version: {FABRIC_AGENT_SCHEMA_VERSION})"
        )
        return
    yield (
        "no Agent under Test: the config declares no "
        f"schema_version: {FABRIC_AGENT_SCHEMA_VERSION} package, and no --agent was given"
    )


def _optimizer_problems(config: Mapping[str, Any]) -> Iterator[str]:
    """``OptimizeRouter`` picks its backend off these flags; an unset optimizer has nothing to run."""
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        yield "optimizer section is missing or is not a mapping"
        return
    enabled = [
        name
        for name in ("numeric", "prompt")
        if isinstance(optimizer.get(name), Mapping) and optimizer[name].get("enabled")
    ]
    if not enabled:
        yield "no optimizer is enabled; set optimizer.numeric.enabled or optimizer.prompt.enabled"
    if "numeric" in enabled and not optimizer.get("search_space"):
        yield "optimizer.numeric is enabled but optimizer.search_space is empty"


def _path_problems(source: Path, config: Mapping[str, Any]) -> Iterator[str]:
    for reference in path_references(config):
        if not is_fileset_relative(reference.value):
            yield (
                f"{reference.location} is an absolute path ({reference.value!r}), which will not "
                "exist on the platform; make it relative to the bundle root"
            )
            continue
        if "${" in reference.value:
            # Relative, expanded from the task environment at run time; nothing to resolve here.
            continue
        if not reference.must_exist:
            continue
        target = source / reference.value
        if target.is_symlink():
            yield f"{reference.location} points at {reference.value!r}, which is a symlink; symlinks are not allowed in an optimize bundle"
        elif reference.is_dir and not target.is_dir():
            yield f"{reference.location} points at {reference.value!r}, which is not a directory under --source"
        elif not reference.is_dir and not target.is_file():
            yield f"{reference.location} points at {reference.value!r}, which is not a file under --source"


def _symlink_problems(source: Path) -> Iterator[str]:
    """Reject every symlink under *source*.

    Upload walks the tree and follows symlinks, so an unreferenced symlink can smuggle files
    from outside the bundle into the fileset, and a symlinked directory in a referenced path's
    ancestry can let that reference resolve outside ``--source`` even though the reference
    itself looks bundle-relative.  ``os.walk(..., followlinks=False)`` never descends into a
    symlinked directory, so this only reports the symlink itself rather than its (possibly
    unbounded or external) contents.
    """
    for root, dirnames, filenames in os.walk(source, followlinks=False):
        root_path = Path(root)
        for name in (*dirnames, *filenames):
            entry = root_path / name
            if entry.is_symlink():
                yield (f"{entry.relative_to(source)} is a symlink, which is not allowed in an optimize bundle")


def path_references(config: Mapping[str, Any]) -> Iterator[PathReference]:
    """Yield every path-shaped value the optimize config can carry.

    An explicit key walk, not a scan for path-looking strings: the config is full of strings that
    resemble paths but are not (``models.default.model: nvidia/meta/llama-3.1-8b-instruct``, and
    ``optimizer.search_space.<param>.path``, which is a dotted overlay target).  Author-supplied
    hook payloads are the one open-ended shape, so ``run_hook`` is walked by its documented keys.
    """
    runtime = config.get("runtime")
    if isinstance(runtime, Mapping):
        yield from _optional(runtime.get("artifacts"), "runtime.artifacts", must_exist=False, is_dir=True)

    environment = config.get("environment")
    if isinstance(environment, Mapping):
        yield from _optional(environment.get("workspace"), "environment.workspace", must_exist=False, is_dir=True)
        yield from _optional(environment.get("artifacts"), "environment.artifacts", must_exist=False, is_dir=True)

    eval_config = config.get("eval")
    if not isinstance(eval_config, Mapping):
        return

    general = eval_config.get("general")
    if isinstance(general, Mapping):
        yield from _dataset_reference(general.get("dataset"))

    fabric = eval_config.get("fabric")
    if isinstance(fabric, Mapping):
        yield from _optional(fabric.get("base_dir"), "eval.fabric.base_dir", must_exist=True, is_dir=True)

    run_hook = eval_config.get("run_hook")
    if isinstance(run_hook, Mapping):
        yield from _run_hook_references(run_hook)


def _dataset_reference(dataset: Any) -> Iterator[PathReference]:
    """``eval.general.dataset`` is a path string, or a mapping with ``file_path`` / ``path``."""
    value = dataset if isinstance(dataset, str) else None
    if isinstance(dataset, Mapping):
        candidate = dataset.get("file_path") or dataset.get("path")
        value = candidate if isinstance(candidate, str) else None
    if value is None or re.match(FILESET_REF_PATTERN, value):
        # A `workspace/fileset#path` dataset is staged separately by the job at run time.
        return
    yield PathReference("eval.general.dataset", value, must_exist=True)


def _run_hook_references(run_hook: Mapping[str, Any]) -> Iterator[PathReference]:
    yield from _optional(run_hook.get("path"), "eval.run_hook.path", must_exist=True)
    yield from _optional(run_hook.get("agent_src"), "eval.run_hook.agent_src", must_exist=True, is_dir=True)
    bindings = run_hook.get("bindings")
    if not isinstance(bindings, list):
        return
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            continue
        prefix = f"eval.run_hook.bindings[{index}]"
        # The executable is resolved on PATH inside the task container, so it only has to not be
        # an absolute client path; the config files it is handed must ship with the bundle.
        yield from _optional(binding.get("executable"), f"{prefix}.executable", must_exist=False)
        config_paths = binding.get("config_paths")
        if isinstance(config_paths, list):
            for path_index, config_path in enumerate(config_paths):
                yield from _optional(config_path, f"{prefix}.config_paths[{path_index}]", must_exist=True)


def _optional(value: Any, location: str, *, must_exist: bool, is_dir: bool = False) -> Iterator[PathReference]:
    if isinstance(value, str) and value:
        yield PathReference(location, value, must_exist=must_exist, is_dir=is_dir)
