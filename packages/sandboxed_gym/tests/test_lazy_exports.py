# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy export surface of ``sandboxed_gym/__init__.py``.

The package's ``__init__`` re-exports names from ten modules. Doing that eagerly meant importing
*any* submodule pulled in FastAPI and Starlette, because importing ``sandboxed_gym.wire`` runs the
parent ``__init__`` first. A client that only speaks the broker's wire contract -- Evaluator's
planned ``BrokerSandboxProvider`` is the motivating case -- should pay for Pydantic and nothing
else. These tests hold that property in place and keep the three parallel lists in step.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import sandboxed_gym

INIT_PATH = Path(sandboxed_gym.__file__)


def _typing_block_names() -> set[str]:
    """Names imported under ``if TYPE_CHECKING:`` -- the static half of the lazy surface."""
    tree = ast.parse(INIT_PATH.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for statement in ast.walk(node):
                if isinstance(statement, ast.ImportFrom):
                    names.update(alias.name for alias in statement.names)
    return names


def test_importing_the_wire_contract_does_not_pull_in_the_server_stack() -> None:
    """The reason this module is lazy.

    Run in a subprocess: pytest has already imported FastAPI through the other test modules, so
    checking ``sys.modules`` in-process would pass regardless of what the import actually costs.
    """
    probe = (
        "import sys; import sandboxed_gym.wire; "
        "print(','.join(m for m in ('fastapi', 'starlette', 'uvicorn') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    leaked = result.stdout.strip()
    assert not leaked, f"`import sandboxed_gym.wire` pulled in the server stack: {leaked}"


def test_every_export_resolves_to_the_object_its_defining_module_holds() -> None:
    # Lazy resolution has to return the *same* object an eager import would, or callers doing
    # isinstance checks across the two import styles would silently disagree.
    for name, module_name in sandboxed_gym._LAZY_EXPORTS.items():
        module = __import__(module_name, fromlist=[name])
        assert getattr(sandboxed_gym, name) is getattr(module, name), name


@pytest.mark.parametrize(
    ("label", "getter"),
    [
        ("_LAZY_EXPORTS", lambda: set(sandboxed_gym._LAZY_EXPORTS)),
        ("TYPE_CHECKING block", _typing_block_names),
        ("__dir__()", lambda: set(dir(sandboxed_gym))),
    ],
)
def test_the_parallel_export_lists_agree_with_all(label: str, getter) -> None:
    """A name in one list and not another fails at runtime or in type checking, never both.

    ``__all__`` is the reference: it is what ``from sandboxed_gym import *`` and the docs promise.
    """
    assert getter() == set(sandboxed_gym.__all__), f"{label} disagrees with __all__"


def test_an_unknown_attribute_raises_rather_than_importing_something_unexpected() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'NotAThing'"):
        sandboxed_gym.NotAThing


def test_a_resolved_export_is_cached_in_module_globals() -> None:
    # Without the cache every attribute access re-enters `importlib.import_module`. That is correct
    # but wasteful on a hot path, so the caching is deliberate and worth pinning.
    name = "EpisodeBrokerConfig"
    globals_before = sandboxed_gym.__dict__.get(name)
    resolved = getattr(sandboxed_gym, name)

    assert sandboxed_gym.__dict__[name] is resolved
    assert globals_before is None or globals_before is resolved
