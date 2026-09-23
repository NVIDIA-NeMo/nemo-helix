# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative *build* spec for agent-eval task images (example-local glue).

Resolves a task's ``environment.yaml`` (or a Dockerfile escape hatch) into a
:class:`BuildPlan` and builds the image. This is the example's prepare-task
(build) concern, paired with the SDK's run-time environment abstraction
(``nemo_evaluator_sdk.agent_eval.runtimes.environment``).

``yaml`` is imported lazily so the module stays importable without PyYAML.

Spec shape (``environment.yaml`` in the task dir)::

    environment:
      image: nemo-helix-agentic-base:2026.06
      profile: evaluator-platform
      dependencies:
        python: [pytest, nemo-evaluator-sdk]
      setup: [seed-providers, create-workspace]

Escape hatch::

    environment:
      dockerfile: environment/Dockerfile
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ENVIRONMENT_SPEC_FILENAME = "environment.yaml"
DEFAULT_DOCKERFILE_RELPATH = "environment/Dockerfile"


@dataclass(frozen=True)
class BuildSpec:
    """Declarative build inputs for one task (or a Dockerfile escape hatch)."""

    image: str | None = None
    profile: str | None = None
    python_dependencies: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)
    dockerfile: Path | None = None

    def __post_init__(self) -> None:
        if self.dockerfile is None and self.image is None:
            raise ValueError("build spec requires either 'image' or 'dockerfile'")


def load_build_spec(task_dir: str | Path) -> BuildSpec:
    """Load a task's build spec.

    Resolution order: ``environment.yaml`` (declarative spec, preferred), then
    ``environment/Dockerfile`` (backward-compatible escape hatch).
    """
    root = Path(task_dir)
    spec_path = root / ENVIRONMENT_SPEC_FILENAME
    if spec_path.is_file():
        import yaml  # optional dependency; imported only when a spec is read

        return _parse_spec(yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}, root)

    dockerfile = root / DEFAULT_DOCKERFILE_RELPATH
    if dockerfile.is_file():
        return BuildSpec(dockerfile=dockerfile)

    raise FileNotFoundError(
        f"No build spec for task {root}: expected {ENVIRONMENT_SPEC_FILENAME} or {DEFAULT_DOCKERFILE_RELPATH}"
    )


def _parse_spec(payload: dict, task_dir: Path) -> BuildSpec:
    data = payload.get("environment", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid build spec in {task_dir}: expected a mapping")

    dockerfile_value = data.get("dockerfile")
    dockerfile = None
    if dockerfile_value:
        dockerfile = Path(dockerfile_value)
        if not dockerfile.is_absolute():
            dockerfile = (task_dir / dockerfile).resolve()
        if not dockerfile.is_file():
            raise FileNotFoundError(f"environment.dockerfile not found: {dockerfile}")

    dependencies = data.get("dependencies") or {}
    python_deps = dependencies.get("python") if isinstance(dependencies, dict) else None

    return BuildSpec(
        image=data.get("image"),
        profile=data.get("profile"),
        python_dependencies=_str_list(python_deps, "dependencies.python", task_dir),
        setup=_str_list(data.get("setup"), "setup", task_dir),
        dockerfile=dockerfile,
    )


def _str_list(value: object, field_name: str, task_dir: Path) -> list[str]:
    """Coerce a YAML value to a list[str], rejecting wrong shapes loudly.

    Guards against the silent ``list("pytest") -> ['p', 'y', ...]`` trap: a bare
    string (or any non-list) for a list-valued field is a spec error, not a
    character sequence.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Invalid build spec in {task_dir}: '{field_name}' must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Invalid build spec in {task_dir}: '{field_name}' must be a list of strings")
        result.append(item)
    return result


@dataclass(frozen=True)
class BuildPlan:
    """A resolved, executable Docker build for one task."""

    image_tag: str
    dockerfile: Path
    context_dir: Path
    generated: bool
    base_image: str | None = None
    setup: list[str] = field(default_factory=list)


def plan_task_build(
    task_dir: str | Path,
    image_tag: str,
    *,
    spec: BuildSpec | None = None,
    generated_dir: Path | None = None,
) -> BuildPlan:
    """Resolve a task's build spec into a concrete :class:`BuildPlan`.

    For the Dockerfile escape hatch the existing Dockerfile/context is used; for
    an ``image``-based spec a minimal derived Dockerfile is written under
    ``generated_dir`` (defaults to ``<task_dir>/.agentic-build``).
    """
    root = Path(task_dir)
    spec = spec or load_build_spec(root)

    if spec.dockerfile is not None:
        return BuildPlan(
            image_tag=image_tag,
            dockerfile=spec.dockerfile,
            context_dir=spec.dockerfile.parent,
            generated=False,
            setup=list(spec.setup),
        )

    # image-based spec: generate a tiny derived Dockerfile.
    context_dir = generated_dir if generated_dir is not None else (root / ".agentic-build")
    context_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = context_dir / "Dockerfile"
    dockerfile.write_text(render_derived_dockerfile(spec), encoding="utf-8")
    return BuildPlan(
        image_tag=image_tag,
        dockerfile=dockerfile,
        context_dir=context_dir,
        generated=True,
        base_image=spec.image,
        setup=list(spec.setup),
    )


def execute_build_plan(plan: BuildPlan) -> None:
    """Build the Docker image described by ``plan`` via the ``docker`` CLI."""
    cmd = ["docker", "build", "-f", str(plan.dockerfile), "-t", plan.image_tag, str(plan.context_dir)]
    print(f"[agent-eval-runtime] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# Note: '<'/'>' are intentionally allowed — pip version pins (``numpy<2``,
# ``pytest>=8``) use them and a Dockerfile ``RUN`` arg doesn't shell-redirect.
_DOCKERFILE_UNSAFE_CHARS = set('\n\r"`$;&|')


def _reject_unsafe(values: list[str], field_name: str) -> None:
    """Reject tokens that would break out of the generated Dockerfile line.

    Inputs come from a task's ``environment.yaml``. Even though that file is
    author-controlled, embedding shell metacharacters into a generated ``RUN``/
    ``LABEL`` line is almost always a mistake; fail loudly instead of silently
    emitting an injectable Dockerfile.
    """
    for value in values:
        if _DOCKERFILE_UNSAFE_CHARS & set(value):
            raise ValueError(f"Unsafe character in build spec '{field_name}' entry: {value!r}")


def render_derived_dockerfile(spec: BuildSpec) -> str:
    """Render a minimal derived Dockerfile from an image-based spec."""
    if spec.image is None:
        raise ValueError("cannot render a derived Dockerfile without a base image")
    _reject_unsafe(spec.python_dependencies, "dependencies.python")
    _reject_unsafe(spec.setup, "setup")
    lines = [f"FROM {spec.image}"]
    if spec.profile:
        lines.append(f"LABEL com.nvidia.agentic.profile={spec.profile}")
    if spec.python_dependencies:
        # Quote each dep so version pins (numpy<2, pytest>=8) aren't treated as
        # shell redirection by the shell-form RUN instruction.
        deps = " ".join(shlex.quote(dep) for dep in spec.python_dependencies)
        lines.append(f"RUN pip install --no-cache-dir {deps}")
    if spec.setup:
        # Setup steps are recorded for traceability only (not executed here).
        lines.append(f'LABEL com.nvidia.agentic.setup="{",".join(spec.setup)}"')
    return "\n".join(lines) + "\n"
