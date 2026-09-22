# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a test's environment-variable skip gate is set nowhere CI can set it.

A gate like ``skipif(not os.environ.get("RUN_X"))`` is indistinguishable, in a CI summary, from a
suite that passes: both report green. If nothing ever sets ``RUN_X``, those tests are dead weight
that still look like coverage. That is not hypothetical -- ``RUN_AGENT_EVAL_INTEGRATION`` guarded 24
evaluator integration tests for months while CI reported success.

Gates naming a variable that some workflow *does* set are fine, and so are gates that probe for
infrastructure (a Docker daemon, a kubeconfig) rather than read an opt-in flag, because those skip
visibly when the infrastructure is absent and run when it is present.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

#: Gates CI is not expected to satisfy, each because the suite behind it needs a credential or a live
#: external service. Keep this short and justified: every entry is coverage CI does not have.
ALLOWED_UNSET = {
    "RUN_EXTERNAL_STORAGE_TESTS": "needs real NGC storage credentials",
    "NVIDIA_API_KEY": "needs a real build.nvidia.com key",
    "NMP_INSIGHTS_E2E": "needs a live Insights deployment",
    "TRACE_FIXTURE_LIVE_CODEX": "regenerates fixtures against a live Codex CLI",
    "SCALED_EVALS_TEST_DATABASE_URL": "needs a live Postgres for the scaled-evals migration tests",
}

#: Directory names that are not this repository's source: installed packages and build caches. Their
#: tests are not ours to gate, and parsing them is pure noise. Matched as whole path *components*, so
#: keep these specific -- an everyday word here would silently exempt real directories.
NOT_OURS = frozenset({".venv", ".flox", "node_modules", "site-packages", ".git"})

#: Generated trees, excluded by path prefix rather than by component name. The Stainless-generated
#: SDK lives under `sdk/python`; a gate it emitted would not be ours to set, and failing the build
#: over one would leave no fix available. `tools/nemo-platform-sdk-tools/tests/sdk` is *not* this --
#: it is first-party, which is why a bare "sdk" component match would be wrong.
VENDORED_ROOTS = ("sdk/python",)

#: What pytest itself collects, per `python_files` in pytest.ini. Scanning only one of them would
#: leave gates in the other invisible to this check while pytest still skipped the tests.
TEST_FILE_GLOBS = ("test_*.py", "*_test.py")


def _module_constants(tree: ast.AST) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings.

    Naming the variable once and referring to it is ordinary style -- the repository already does it
    in ``e2e/files/test_storage_backends.py`` (``HF_TOKEN_ENV``) and the scaled-evals migration tests
    (``TEST_DSN_ENV``). Reading only string literals at the call site would miss every one of them.

    Bindings imported from another module are not resolved. The variable still surfaces from the
    module that defines it, so an orphan is never hidden outright -- only the list of files naming it
    can be short.
    """
    constants: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _env_name(node: ast.AST, constants: dict[str, str]) -> str | None:
    """The environment variable a node names, whether written inline or via a constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _env_reads(node: ast.AST, constants: dict[str, str]) -> set[str]:
    """Variables read by ``os.environ[X]``, ``os.environ.get(X)`` or ``os.getenv(X)`` under ``node``."""
    found: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Subscript):
            value = inner.value
            if isinstance(value, ast.Attribute) and value.attr == "environ":
                name = _env_name(inner.slice, constants)
                if name:
                    found.add(name)
        elif isinstance(inner, ast.Call):
            func = inner.func
            reads_env = (isinstance(func, ast.Attribute) and func.attr in {"get", "getenv"}) or (
                isinstance(func, ast.Name) and func.id == "getenv"
            )
            if reads_env and inner.args:
                name = _env_name(inner.args[0], constants)
                if name:
                    found.add(name)
    return found


def gate_variables(tree: ast.AST) -> set[str]:
    """Environment variables read inside a ``pytest.mark.skipif`` condition."""
    constants = _module_constants(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name != "skipif":
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            found.update(_env_reads(argument, constants))
    return found


def variables_set_by_ci(workflow_dir: Path) -> set[str]:
    """Every ``NAME:`` or ``NAME=`` assignment appearing anywhere under the workflow directory.

    Deliberately crude: a false *positive* here would fail a build over a gate CI really does set,
    which is worse than missing an exotic spelling.
    """
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sorted(workflow_dir.rglob("*.y*ml"))
    )
    return set(re.findall(r"([A-Z][A-Z0-9_]*)\s*[:=]", text))


def orphaned_gates(test_root: Path, workflow_dir: Path) -> dict[str, list[Path]]:
    """Gate variables no workflow sets, mapped to the test files that gate on them."""
    ci_variables = variables_set_by_ci(workflow_dir)
    orphans: dict[str, list[Path]] = {}
    candidates = sorted({path for glob in TEST_FILE_GLOBS for path in test_root.rglob(glob)})
    for path in candidates:
        if NOT_OURS.intersection(path.parts):
            continue
        posix = path.as_posix()
        if any(posix.startswith(f"{root}/") or f"/{root}/" in posix for root in VENDORED_ROOTS):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            # A test file this cannot parse cannot be carrying a gate this understands. Skipping it
            # keeps the check advisory about what it can see rather than failing the build on a
            # fixture that merely happens to be named test_*.py.
            continue
        for variable in gate_variables(tree):
            if variable in ci_variables or variable in ALLOWED_UNSET:
                continue
            orphans.setdefault(variable, []).append(path)
    return orphans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=Path("."))
    parser.add_argument("--workflow-dir", type=Path, default=Path(".github/workflows"))
    args = parser.parse_args()

    orphans = orphaned_gates(args.test_root, args.workflow_dir)
    if not orphans:
        return 0
    for variable, paths in sorted(orphans.items()):
        print(f"{variable} gates {len(paths)} test file(s) but no workflow sets it:", file=sys.stderr)
        for path in paths:
            print(f"  {path}", file=sys.stderr)
    print(
        "\nEither set the variable in the workflow that should run these tests, remove the gate, or "
        f"add the variable to ALLOWED_UNSET in {Path(__file__).name} with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
