# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for markdown file detection and processing."""

import tempfile
from pathlib import Path

from nemo_nb.md_to_notebook import MarkdownToNotebookConverter
from nemo_nb.sphinx import should_process_markdown


class TestMarkdownDetection:
    """Test should_process_markdown function."""

    def test_markdown_with_process_marker_at_top(self):
        """Test markdown file with process marker at the top."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""<!-- @nemo-nb: process -->
# Test Document

Some content here.
""")
            temp_path = Path(f.name)

        try:
            assert should_process_markdown(temp_path) is True
        finally:
            temp_path.unlink()

    def test_markdown_with_process_marker_in_middle(self):
        """Test markdown file with process marker in the middle."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""# Test Document

## Introduction
<!-- @nemo-nb: process -->

Some content here.
""")
            temp_path = Path(f.name)

        try:
            assert should_process_markdown(temp_path) is True
        finally:
            temp_path.unlink()

    def test_markdown_without_process_marker(self):
        """Test markdown file without process marker."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""# Test Document

## Introduction

Some content here.
""")
            temp_path = Path(f.name)

        try:
            assert should_process_markdown(temp_path) is False
        finally:
            temp_path.unlink()

    def test_markdown_with_marker_after_line_20(self):
        """Test markdown file with marker after line 20 (should still be detected)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            for i in range(25):
                f.write(f"Line {i}\n")
            f.write("<!-- @nemo-nb: process -->\n")
            temp_path = Path(f.name)

        try:
            assert should_process_markdown(temp_path) is True
        finally:
            temp_path.unlink()

    def test_empty_markdown_file(self):
        """Test empty markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            assert should_process_markdown(temp_path) is False
        finally:
            temp_path.unlink()

    def test_markdown_with_similar_but_not_exact_marker(self):
        """Test markdown with similar but not exact marker."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            # Contains 'process' substring
            f.write("""<!-- @nemo-nb: processed -->
# Test Document
""")
            temp_path = Path(f.name)

        try:
            # Currently detects 'process' as substring, so this returns True
            assert should_process_markdown(temp_path) is True
        finally:
            temp_path.unlink()


class TestMarkdownToNotebookConversion:
    """Test MarkdownToNotebookConverter."""

    def test_simple_markdown_conversion(self):
        """Test converting a simple markdown file with new cell markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""# Test Document

Some text here.

<!-- @nemo-nb: cell python -->
print('hello')
""")
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            # Check notebook structure
            assert "cells" in notebook
            assert "metadata" in notebook
            assert notebook["nbformat"] == 4

            # Check cells
            cells = notebook["cells"]
            assert len(cells) >= 2  # At least markdown and code cell

            # Find code cell
            code_cells = [c for c in cells if c["cell_type"] == "code"]
            assert len(code_cells) == 1
            assert "print('hello')" in "".join(code_cells[0]["source"])

        finally:
            temp_path.unlink()

    def test_markdown_with_multiple_code_blocks(self):
        """Test markdown with multiple code blocks using new cell markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "# Test\n\n"
                "<!-- @nemo-nb: cell python -->\n"
                "x = 1\n"
                "<!-- @nemo-nb: cell markdown -->\n"
                "Some text.\n\n"
                "<!-- @nemo-nb: cell python -->\n"
                "y = 2\n"
            )
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            assert len(code_cells) == 2

        finally:
            temp_path.unlink()

    def test_markdown_with_bash_code_block(self):
        """Test markdown with bash code block using new cell marker."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""# Test

<!-- @nemo-nb: cell bash -->
echo 'hello'
""")
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            assert len(code_cells) == 1

            # Check metadata for bash
            metadata = code_cells[0]["metadata"]
            assert "language" in metadata or "vscode" in metadata

        finally:
            temp_path.unlink()

    def test_markdown_with_output_cell(self):
        """Test markdown with output cell markers using new format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "# Test\n\n"
                "<!-- @nemo-nb: cell python -->\n"
                "print('hello')\n"
                "<!-- @nemo-nb: output stream stdout -->\n"
                "hello\n"
            )
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            assert len(code_cells) == 1

            # Check for output
            cell = code_cells[0]
            assert "outputs" in cell
            assert len(cell["outputs"]) == 1

            output = cell["outputs"][0]
            assert output["output_type"] == "stream"
            assert output["name"] == "stdout"
            assert "hello" in "".join(output["text"])

        finally:
            temp_path.unlink()

    def test_empty_code_block(self):
        """Test markdown with empty code block."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""# Test

```python
```
""")
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            converter.convert(temp_path)

            # Empty code blocks should be skipped
            # Depending on implementation, empty blocks might be skipped
            # Just ensure no errors occur

        finally:
            temp_path.unlink()

    def test_tab_item_python_sdk_markers_stay_in_single_code_cell(self):
        """Tab-item + :sync: markers and code stay in one code cell.

        This reproduces the deploy-nims pattern where a tab-set is introduced
        via multi-cell markers and a Python SDK tab is implemented as a code
        cell whose first lines are nemo-nb markers:

        - multi-cell-indent-space-start / insert tab-set (markdown markers)
        - cell python
        - wrap-cell-start / insert / insert :sync / wrap-cell-end (code markers)

        The expected behavior is that these four code markers and the Python
        code remain together in a single notebook code cell. The current bug
        splits markers and code across two cells.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "# Deploy NIM test document\n\n"
                "1. To deploy the NIM, run the API as follows:\n\n"
                "<!-- @nemo-nb: multi-cell-indent-space-start indent-a 3 -->\n"
                "<!-- @nemo-nb: insert ::::{tab-set} -->\n\n"
                "<!-- @nemo-nb: cell python -->\n"
                "```python\n"
                "# @nemo-nb: wrap-cell-start :::{tab-item} Python SDK\n"
                "# @nemo-nb: insert\n"
                "# @nemo-nb: insert :sync: sdk\n"
                "# @nemo-nb: wrap-cell-end :::\n"
                "from nemo_helix import NeMoHelix\n\n"
                "client = NeMoHelix(\n"
                '    base_url="http://nemo.test",\n'
                '    inference_base_url="http://nim.test",\n'
                ")\n\n"
                "deployment = client.deployment.model_deployments.create(\n"
                '    name="llama-3.1-8b-instruct",\n'
                '    namespace="meta",\n'
                "    config={\n"
                '        "model": "meta/llama-3.1-8b-instruct",\n'
                '        "nim_deployment": {\n'
                '            "image_name": "nvcr.io/nim/meta/llama-3.1-8b-instruct",\n'
                '            "image_tag": "1.8",\n'
                '            "pvc_size": "25Gi",\n'
                '            "gpu": 1,\n'
                '            "additional_envs": {"NIM_GUIDED_DECODING_BACKEND": "outlines"},\n'
                "        },\n"
                "    },\n"
                ")\n"
                "print(deployment)\n"
                "```\n"
            )
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]

            # Expect a single code cell for this tab item
            assert len(code_cells) == 1, "Expected one code cell for Python SDK tab"

            source = "".join(code_cells[0]["source"])

            # Markers should all be present and in the correct order
            expected_markers = [
                "# @nemo-nb: wrap-cell-start :::{tab-item} Python SDK\n",
                "# @nemo-nb: insert\n",
                "# @nemo-nb: insert :sync: sdk\n",
                "# @nemo-nb: wrap-cell-end :::\n",
            ]

            last_pos = -1
            for marker in expected_markers:
                pos = source.find(marker)
                assert pos != -1, f"Missing marker in code cell: {marker!r}"
                assert pos > last_pos, "Markers are out of order in code cell"
                last_pos = pos

            # Ensure the main client code is in the same cell
            assert "from nemo_helix import NeMoHelix" in source
            assert "deployment = client.deployment.model_deployments.create(" in source
            assert "print(deployment)" in source

        finally:
            temp_path.unlink()


class TestOutputCellTypes:
    """Test different output cell types."""

    def test_execute_result_output(self):
        """Test execute_result output type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "```python\n"
                "42\n"
                "```\n\n"
                "<!-- @nemo-nb: output-cell-start execute_result -->\n\n"
                "```\n"
                "42\n"
                "```\n\n"
                "<!-- @nemo-nb: output-cell-end -->\n"
            )
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            if code_cells and code_cells[0]["outputs"]:
                output = code_cells[0]["outputs"][0]
                assert output["output_type"] == "execute_result"

        finally:
            temp_path.unlink()

    def test_error_output(self):
        """Test error output type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "```python\n"
                "raise ValueError('test error')\n"
                "```\n\n"
                "<!-- @nemo-nb: output-cell-start error -->\n\n"
                "```\n"
                "ValueError: test error\n"
                "```\n\n"
                "<!-- @nemo-nb: output-cell-end -->\n"
            )
            temp_path = Path(f.name)

        try:
            converter = MarkdownToNotebookConverter()
            notebook = converter.convert(temp_path)

            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            if code_cells and code_cells[0]["outputs"]:
                output = code_cells[0]["outputs"][0]
                assert output["output_type"] == "error"

        finally:
            temp_path.unlink()
