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
}

#: Directory names that are not this repository's source: installed packages, build caches, vendored
#: trees. Their tests are not ours to gate, and parsing them is pure noise.
NOT_OURS = frozenset({".venv", ".flox", "node_modules", "site-packages", ".git", "sdk"})

_ENV_READ = re.compile(r"""os\.environ(?:\.get)?[(\[]\s*["']([A-Z][A-Z0-9_]*)["']""")


def gate_variables(tree: ast.AST) -> set[str]:
    """Environment variables read inside a ``pytest.mark.skipif`` condition."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name != "skipif":
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            found.update(_ENV_READ.findall(ast.unparse(argument)))
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
    for path in sorted(test_root.rglob("test_*.py")):
        if NOT_OURS.intersection(path.parts):
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
