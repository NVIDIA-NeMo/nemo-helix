# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for discovery utilities."""

import tempfile
from pathlib import Path

import pytest
from nemo_nb.discovery import (
    expand_literalincludes,
    find_processable_notebooks,
    find_testable_notebooks,
    has_process_marker_markdown,
    has_process_marker_notebook,
    is_excluded_file,
    is_in_excluded_dir,
)


def test_is_excluded_file():
    """Test that excluded filenames are detected."""
    assert is_excluded_file(Path("README.md"))
    assert is_excluded_file(Path("REDIRECTS.md"))
    assert is_excluded_file(Path("LINTING_FINDINGS.md"))
    assert not is_excluded_file(Path("tutorial.md"))
    assert not is_excluded_file(Path("index.md"))


def test_is_in_excluded_dir():
    """Test that files in excluded directories are detected."""
    assert is_in_excluded_dir(Path("docs/_build/html/index.html"))
    assert is_in_excluded_dir(Path("docs/_generated/notebook.ipynb"))
    assert not is_in_excluded_dir(Path("docs/tutorials/tutorial.md"))


def test_has_process_marker_markdown():
    """Test marker detection in markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # File with marker
        with_marker = tmpdir / "with_marker.md"
        with_marker.write_text("<!-- @nemo-nb: process -->\n\n# Tutorial")
        assert has_process_marker_markdown(with_marker)

        # File without marker
        without_marker = tmpdir / "without_marker.md"
        without_marker.write_text("# Tutorial\n\nNo marker here")
        assert not has_process_marker_markdown(without_marker)


def test_has_process_marker_notebook():
    """Test marker detection in .ipynb files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Notebook with marker
        with_marker = tmpdir / "with_marker.ipynb"
        with_marker.write_text(
            """{
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["<!-- @nemo-nb: process -->\\n\\n# Tutorial"]
    }
  ]
}"""
        )
        assert has_process_marker_notebook(with_marker)

        # Notebook without marker
        without_marker = tmpdir / "without_marker.ipynb"
        without_marker.write_text(
            """{
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["# Tutorial"]
    }
  ]
}"""
        )
        assert not has_process_marker_notebook(without_marker)


def test_find_processable_notebooks():
    """Test finding notebooks with process marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some markdown files
        (tmpdir / "with_marker.md").write_text("<!-- @nemo-nb: process -->\n# Tutorial")
        (tmpdir / "without_marker.md").write_text("# Tutorial")
        (tmpdir / "README.md").write_text("<!-- @nemo-nb: process -->\n# README")  # Excluded

        # Create build directory with files (should be excluded)
        build_dir = tmpdir / "_build"
        build_dir.mkdir()
        (build_dir / "generated.md").write_text("<!-- @nemo-nb: process -->\n# Generated")

        result = find_processable_notebooks(str(tmpdir))

        # Should find only the marked file (not README, not _build)
        assert len(result.md_files) == 1
        assert result.md_files[0].name == "with_marker.md"
        assert len(result.ipynb_files) == 0
        assert len(result.conflicts) == 0


def test_find_processable_notebooks_conflicts():
    """Test conflict detection when both .md and .ipynb exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create both .md and .ipynb with same base name
        (tmpdir / "tutorial.md").write_text("<!-- @nemo-nb: process -->\n# Tutorial")
        (tmpdir / "tutorial.ipynb").write_text(
            """{
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["<!-- @nemo-nb: process -->\\n\\n# Tutorial"]
    }
  ]
}"""
        )

        result = find_processable_notebooks(str(tmpdir))

        # Should detect conflict
        assert len(result.conflicts) == 1
        assert (tmpdir / "tutorial") in result.conflicts


def test_mdx_discovery(tmp_path):
    marked = tmp_path / "tutorial.mdx"
    marked.write_text('{/* @nemo-nb: process */}\n\n```python\nprint("hello")\n```')
    skipped = tmp_path / "skipped.mdx"
    skipped.write_text('{/* @nemo-nb: process */}\n{/* @nemo-nb: skip-test */}')
    (tmp_path / "unmarked.mdx").write_text("# Not executable")
    build = tmp_path / "_build"
    build.mkdir()
    (build / "generated.mdx").write_text('{/* @nemo-nb: process */}')

    assert set(find_processable_notebooks(str(tmp_path)).md_files) == {marked, skipped}
    assert find_processable_notebooks(str(marked)).md_files == [marked]
    assert find_testable_notebooks(str(tmp_path)).md_files == [marked]
    assert find_testable_notebooks(str(skipped)).md_files == []


def test_mdx_ipynb_conflict(tmp_path):
    (tmp_path / "tutorial.mdx").write_text('{/* @nemo-nb: process */}')
    (tmp_path / "tutorial.ipynb").write_text(
        '{"cells": [{"cell_type": "markdown", "source": "<!-- @nemo-nb: process -->"}]}'
    )
    assert find_processable_notebooks(str(tmp_path)).conflicts == {tmp_path / "tutorial"}


def test_expand_literalincludes_basic():
    """Test basic literalinclude expansion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a source file
        source_file = tmpdir / "example.py"
        source_file.write_text("print('hello')\nprint('world')")

        # Create markdown with literalinclude
        content = """# Tutorial

```{literalinclude} example.py
:language: python
```

More content."""

        result = expand_literalincludes(content, tmpdir)

        # Should expand to regular code block
        assert "```python" in result
        assert "print('hello')" in result
        assert "print('world')" in result
        assert "{literalinclude}" not in result


def test_expand_literalincludes_with_slicing():
    """Test literalinclude with start-after and end-before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a source file
        source_file = tmpdir / "example.py"
        source_file.write_text(
            """# START SNIPPET
x = 1
y = 2
# END SNIPPET
z = 3"""
        )

        # Create markdown with literalinclude and slicing
        content = """```{literalinclude} example.py
:language: python
:start-after: # START SNIPPET
:end-before: # END SNIPPET
```"""

        result = expand_literalincludes(content, tmpdir)

        # Should only include sliced content
        assert "x = 1" in result
        assert "y = 2" in result
        assert "z = 3" not in result
        assert "START SNIPPET" not in result
        assert "END SNIPPET" not in result


def test_expand_literalincludes_missing_file():
    """Test that missing files are left unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        content = """```{literalinclude} nonexistent.py
:language: python
```"""

        result = expand_literalincludes(content, tmpdir)

        # Should leave directive unchanged
        assert "{literalinclude}" in result
        assert "nonexistent.py" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
