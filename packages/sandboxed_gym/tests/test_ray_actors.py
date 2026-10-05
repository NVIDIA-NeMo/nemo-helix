# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restart policy of the Ray actors, asserted against the source.

Read with ``ast`` rather than by importing: ``ray`` is an optional extra, so these modules are not
importable in the default test environment, and the property under test is a decorator argument
that is fixed at import time anyway.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import sandboxed_gym

RAY_DIR = Path(sandboxed_gym.__file__).parent / "ray"
RESTART_OPTIONS = ("max_restarts", "max_task_retries")


def _ray_remote_keywords(module_path: Path, class_name: str) -> dict[str, str]:
    """Keyword arguments on the ``@ray.remote`` decorator of ``class_name``."""
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "ray.remote":
                return {kw.arg: ast.unparse(kw.value) for kw in decorator.keywords if kw.arg}
            if ast.unparse(decorator) == "ray.remote":
                return {}
        pytest.fail(f"{class_name} is not decorated with ray.remote")
    pytest.fail(f"{class_name} not found in {module_path}")


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("gym_actor.py", "SandboxedGymActor"), ("broker_actor.py", "SandboxEpisodeBrokerActor")],
)
def test_actors_are_not_restartable(module_name: str, class_name: str) -> None:
    """Neither actor may be restarted: both hold state a replacement cannot recover.

    The Gym actor holds the only reference to its sandbox, and the broker actor the only handle
    map. A restart cannot name what its predecessor created, so it provisions a second sandbox and
    leaks the first until ttl_s -- silently doubling the pods a job holds.
    """
    keywords = _ray_remote_keywords(RAY_DIR / module_name, class_name)
    assert not [option for option in RESTART_OPTIONS if option in keywords]


def _method_body(module_path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.unparse(item)
    pytest.fail(f"{class_name}.{method_name} not found in {module_path}")


def test_spinup_arms_the_termination_cleanup() -> None:
    """The actor still arms its own teardown for a clean exit of this worker.

    A cancel kills the worker without running that hook. The driver installs
    ``install_job_sandbox_reaper`` for that path, sweeping every sandbox for the
    job. What is asserted here is that the actor still arms its own cleanup.
    """
    body = _method_body(RAY_DIR / "gym_actor.py", "SandboxedGymActor", "spinup")

    assert "install_job_sandbox_reaper(" in body
    assert "install_termination_cleanup(self.shutdown)" in body
