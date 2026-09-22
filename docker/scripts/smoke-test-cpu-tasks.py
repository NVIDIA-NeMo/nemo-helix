# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the built `nmp-cpu-tasks` image.

Every job that declares `container = "cpu-tasks"` launches its task module in
this image, and the failure mode when a plugin is missing from the `cpu-tasks`
dependency group is an import error at job runtime, not at build time. A unit
test asserting `pyproject.toml` group membership passes while the built image
is broken; these checks do not.

Each check returns its failures instead of raising, so one run reports
everything that is wrong. The image takes ~20 minutes to build, which makes
"fix one, rebuild, discover the next" an expensive loop.

Run inside the image:

    python smoke-test-cpu-tasks.py
"""

import importlib
import importlib.util
import pathlib
import sys
import sysconfig

# Nothing this script validates may be imported at module scope. A package that
# is missing because its plugin fell out of the `cpu-tasks` dependency group is
# the exact failure being tested for, and importing it here would abort the
# whole script with a bare traceback before any check reports anything.

#: Modules and the attribute each must expose, covering the Evaluator and Gym
#: task entrypoints plus the server deps they import at runtime.
EVALUATOR_IMPORTS: dict[str, str | None] = {
    "nemo_evaluator.tasks.agent_evaluate": None,
    "nemo_evaluator.tasks.evaluate": None,
    "nemo_evaluator.tasks.stage_environment": None,
    "uvicorn": "__version__",
    "yaml": "__version__",
    "opensandbox.config": "ConnectionConfig",
    "sandboxed_gym": "SandboxedGymOrchestrator",
}

#: Task modules launched via `python -m`, so the `__main__` submodule is what
#: has to resolve -- not merely the containing package.
TASK_ENTRYPOINTS = ("nemo_agents_plugin.tasks.execute.__main__",)

#: First-party Fabric adapters that must ship in this image. Checked as a
#: subset so a newly added adapter does not fail the build.
REQUIRED_FABRIC_ADAPTERS = frozenset({"claude", "codex", "deepagents", "hermes"})


def check_evaluator_imports() -> list[str]:
    """Import the Evaluator and Gym task modules and their runtime deps."""
    failures: list[str] = []
    resolved: dict[str, str] = {}

    for module_name, attribute in EVALUATOR_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the suite
            failures.append(f"{module_name}: import failed: {exc!r}")
            continue

        if attribute is None:
            resolved[module_name] = "ok"
            continue

        try:
            resolved[module_name] = str(getattr(module, attribute))
        except AttributeError:
            failures.append(f"{module_name}: missing attribute {attribute!r}")

    if not failures:
        print("evaluator/gym imports:", resolved)
    return failures


def check_task_entrypoints() -> list[str]:
    """Resolve the `python -m` entrypoints for agent execution."""
    failures: list[str] = []

    for name in TASK_ENTRYPOINTS:
        # find_spec imports the parent packages to locate a submodule, so it
        # raises ModuleNotFoundError when the parent is absent and returns None
        # only when the leaf is. A plugin missing from the cpu-tasks group hits
        # the raising path, which is precisely the case being tested for.
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError) as exc:
            failures.append(f"{name}: not resolvable: {exc!r}")
            continue

        if not found:
            failures.append(f"{name}: not found")

    if not failures:
        print("task entrypoints:", list(TASK_ENTRYPOINTS))
    return failures


def check_fabric_adapters() -> list[str]:
    """Verify the Fabric adapter data directory the native runtime lists.

    Fabric discovers adapters by listing this directory, so they have to survive
    into the image as data files -- being importable is not enough.
    """
    adapter_dir = pathlib.Path(sysconfig.get_path("data")) / "share" / "nemo-fabric" / "adapters"
    if not adapter_dir.is_dir():
        return [f"Fabric adapter directory not found: {adapter_dir}"]

    found = {entry.name for entry in adapter_dir.iterdir()}
    if missing := REQUIRED_FABRIC_ADAPTERS - found:
        return [f"missing Fabric adapters in {adapter_dir}: {sorted(missing)!r} (found {sorted(found)!r})"]

    print("fabric adapters:", sorted(found))
    return []


CHECKS = (
    check_evaluator_imports,
    check_task_entrypoints,
    check_fabric_adapters,
)


def main() -> int:
    failures: list[str] = []
    for check in CHECKS:
        failures.extend(check())

    if failures:
        print(f"\ncpu-tasks image smoke test: {len(failures)} check(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("\nall cpu-tasks image smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
