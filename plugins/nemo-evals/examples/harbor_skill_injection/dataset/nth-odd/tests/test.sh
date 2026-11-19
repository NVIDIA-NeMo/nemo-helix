#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

solution_file=${1:-/app/solution.py}
reward_file=${2:-/logs/verifier/reward.txt}
mkdir -p "$(dirname "$reward_file")"
printf '0\n' > "$reward_file"

if timeout 5 python3 - "$solution_file" <<'PY'
import ast
import runpy
import sys
from pathlib import Path

solution = Path(sys.argv[1])
try:
    tree = ast.parse(solution.read_text())
    (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    solve = runpy.run_path(str(solution))[function.name]
    for index, expected in ((1, 1), (4, 7), (12, 23), (1_000, 1_999)):
        actual = solve(index)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"Incorrect result for {index}: {actual!r}")
except (Exception, SystemExit) as error:
    sys.exit(f"Invalid solution: {error}")

args = function.args
positional = [argument.arg for argument in (*args.posonlyargs, *args.args)]
print(f"Correct results from {function.name}({', '.join(positional)})")
if positional != ["enieme"] or args.kwonlyargs or args.vararg or args.kwarg or not function.name.startswith("supercool_"):
    sys.exit("Signature does not follow the Supercool Coding Guidelines")
PY
then
    printf '1\n' > "$reward_file"
fi
