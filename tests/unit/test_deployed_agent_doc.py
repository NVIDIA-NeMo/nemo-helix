# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the HTTP agent walkthrough without an external model or platform."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs/evals/agent-eval/evaluate-deployed-agent.mdx"


@pytest.mark.parametrize("walkthrough", [True, False], ids=["step-by-step", "full-script"])
def test_deployed_agent_example(walkthrough: bool, tmp_path: Path) -> None:
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(), re.DOTALL)
    full_script = blocks[-1]
    if walkthrough:
        # The walkthrough explicitly reuses the quickstart's metric and tasks.
        # Use the identical definitions included in this page's full script.
        definitions = ast.Module(
            body=[
                node
                for node in ast.parse(full_script).body
                if isinstance(node, (ast.Import, ast.ImportFrom))
                or isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"KeywordMatchMetric", "make_tasks"}
            ],
            type_ignores=[],
        )
        server = blocks[1].replace('("127.0.0.1", 8000)', '("127.0.0.1", 0)')
        # Use an OS-assigned port so parallel tests do not compete for port 8000.
        script = "\n".join(
            [
                ast.unparse(definitions),
                blocks[0],
                server,
                'agent = agent.model_copy(update={"url": f"http://127.0.0.1:{httpd.server_port}/agent"})',
                blocks[2],
            ]
        )
    else:
        script = full_script

    example = tmp_path / "evaluate_http_agent.py"
    example.write_text(script)
    result = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "keyword_match.score: 1.0" in result.stdout
    assert "keyword_match.score.pass@1: 1.0" in result.stdout
