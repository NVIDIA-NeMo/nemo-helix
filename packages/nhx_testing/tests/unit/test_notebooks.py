# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Markdown/MDX notebook execution."""

import json
from pathlib import Path

import papermill
import pytest
from nhx.testing.notebooks import execute_notebook


@pytest.mark.parametrize("suffix", [".md", ".mdx"])
@pytest.mark.parametrize("execution_fails", [False, True])
def test_execute_markdown_notebook(tmp_path, monkeypatch, suffix, execution_fails):
    source = tmp_path / f"tutorial{suffix}"
    source.write_text(
        '---\ntitle: Test\n---\n{/* @nemo-nb: process */}\n\n'
        '<Note>A note</Note>\n\n```python\nprint("hello")\n```\n'
    )
    executed = []

    def fake_execute(**kwargs):
        notebook = json.loads(Path(kwargs["input_path"]).read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code_cells) == 1
        assert 'print("hello")' in "".join(code_cells[0]["source"])
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["execution_timeout"] == 123
        executed.append(kwargs)
        Path(kwargs["output_path"]).write_text(json.dumps(notebook))
        if execution_fails:
            raise RuntimeError("execution failed")

    monkeypatch.setattr(papermill, "execute_notebook", fake_execute)
    if execution_fails:
        with pytest.raises(RuntimeError, match="execution failed"):
            execute_notebook(source, execution_timeout=123)
    else:
        assert execute_notebook(source, execution_timeout=123) == source.with_suffix(".executed.ipynb")
    assert len(executed) == 1
    assert source.with_suffix(".executed.ipynb").exists()
    assert not source.with_suffix(".tmp.ipynb").exists()
    assert not source.with_suffix(".expanded.md").exists()
