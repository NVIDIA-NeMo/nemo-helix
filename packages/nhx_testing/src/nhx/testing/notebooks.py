# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook runner utilities for NeMo Helix documentation acceptance tests.

This module provides functions for finding and running Jupyter notebooks
with the @nemo-nb: process marker, supporting both .ipynb and .md files.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from nemo_nb import (
    MarkdownToNotebookConverter,
    expand_includes,
    expand_literalincludes,
    find_processable_notebooks,
    print_conflicts_error,
)

# Configure logging so papermill's log_output=True actually prints to console
logging.basicConfig(level=logging.INFO, format="%(message)s")

TIMEOUT_SECONDS = 3600


def create_temp_venv_with_kernel(
    requirements_file: str | None = None,
    extra_pip_args: list[str] | None = None,
) -> tuple[str, str, str]:
    """
    Create a temporary virtualenv using uv with ipykernel installed and a kernel spec.

    Args:
        requirements_file: Optional path to a requirements file to install in the venv using uv.
        extra_pip_args: Optional extra arguments passed verbatim to ``uv pip install``
            alongside ``ipykernel``.  Use this to install editable packages or
            packages with extras, e.g. ``["-e", "path/to/pkg[extra1,extra2]"]``.

    Returns:
        Tuple of (kernel_name, venv_dir, kernel_spec_dir) that should be cleaned up after use
    """
    venv_dir = tempfile.mkdtemp(prefix="notebook_venv_")
    venv_path = Path(venv_dir)

    print(f"Creating temporary virtualenv at: {venv_dir}")

    try:
        subprocess.run(
            ["uv", "venv", str(venv_path), "--python", sys.executable],
            check=True,
            capture_output=True,
            text=True,
        )

        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"

        if not python_exe.exists():
            raise RuntimeError(f"Failed to create virtualenv: Python executable not found at {python_exe}")

        if requirements_file:
            requirements_path = Path(requirements_file).resolve()
            if not requirements_path.exists():
                raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

            print(f"Installing packages from {requirements_file}...")
            result = subprocess.run(
                ["uv", "pip", "install", "-r", str(requirements_path), "--python", str(python_exe)],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)

        install_cmd = ["uv", "pip", "install", "ipykernel"]
        if extra_pip_args:
            install_cmd.extend(extra_pip_args)
        install_cmd.extend(["--python", str(python_exe)])

        print("Installing ipykernel (+ extra packages) in temporary virtualenv...")
        subprocess.run(
            install_cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        kernel_name = f"temp_venv_{os.getpid()}"
        kernel_spec_dir = tempfile.mkdtemp(prefix="kernel_spec_")
        kernel_dir = Path(kernel_spec_dir) / kernel_name

        kernel_spec = {
            "argv": [str(python_exe), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": f"Temp Venv ({kernel_name})",
            "language": "python",
            "metadata": {"debugger": True},
        }

        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_json_path = kernel_dir / "kernel.json"

        with open(kernel_json_path, "w", encoding="utf-8") as f:
            json.dump(kernel_spec, f, indent=2)

        print(f"Installing temporary kernel spec: {kernel_name}")
        subprocess.run(
            ["jupyter", "kernelspec", "install", str(kernel_dir), "--user", "--name", kernel_name],
            check=True,
            capture_output=True,
            text=True,
        )

        return kernel_name, venv_dir, kernel_spec_dir

    except Exception as e:
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to create temporary virtualenv: {e}")


def cleanup_temp_venv_and_kernel(kernel_name: str, venv_dir: str, kernel_spec_dir: str) -> None:
    """
    Remove a temporary virtualenv, kernel spec, and associated directories.

    Args:
        kernel_name: Name of the kernel to remove
        venv_dir: Virtualenv directory to clean up
        kernel_spec_dir: Kernel spec directory to clean up
    """
    try:
        subprocess.run(
            ["jupyter", "kernelspec", "remove", "-f", kernel_name],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        pass

    try:
        shutil.rmtree(venv_dir, ignore_errors=True)
    except Exception:
        pass

    try:
        shutil.rmtree(kernel_spec_dir, ignore_errors=True)
    except Exception:
        pass


def make_notebook_executable(notebook: dict, language_filter: str = "all") -> dict:
    """
    Transform code cells based on language filter.

    Args:
        notebook: Notebook dictionary
        language_filter: Which cells to execute:
            - "all": Execute both Python and shell cells (default)
            - "python": Execute only Python cells, skip shell
            - "shell": Execute only shell cells, skip Python

    Returns:
        Modified notebook dictionary with cells transformed for execution
    """
    SHELL_LANGUAGES = {"sh", "bash", "shell"}
    DISPLAY_ONLY_LANGUAGES = {"json", "jsonc", "yaml", "yml", "text"}

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        metadata = cell.get("metadata", {})
        language = metadata.get("language", "python")
        source = cell.get("source", [])

        if isinstance(source, str):
            source = source.split("\n")

        is_shell = language in SHELL_LANGUAGES
        is_python = language == "python"
        is_display_only = language in DISPLAY_ONLY_LANGUAGES

        if is_shell:
            if language_filter == "python":
                cell["cell_type"] = "raw"
                cell.pop("outputs", None)
                cell.pop("execution_count", None)
            else:
                filtered_source = [line for line in source if not line.strip().startswith("# @nemo-nb:")]
                cell["source"] = ["%%bash\n", "set -o pipefail\n"] + filtered_source

        elif is_python:
            if language_filter == "shell":
                cell["cell_type"] = "raw"
                cell.pop("outputs", None)
                cell.pop("execution_count", None)

        elif is_display_only:
            cell["cell_type"] = "raw"
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

    return notebook


def convert_md_to_notebook(md_path: Path, language_filter: str = "all") -> Path:
    """
    Convert a markdown notebook to a temporary .ipynb file.

    Args:
        md_path: Path to the markdown file
        language_filter: Which cells to execute ("all", "python", "shell")

    Returns:
        Path to the generated temporary notebook file
    """
    content = md_path.read_text(encoding="utf-8")
    expanded_content = expand_includes(content, md_path.parent)
    expanded_content = expand_literalincludes(expanded_content, md_path.parent)

    temp_md_path = md_path.with_suffix(".expanded.md")
    temp_md_path.write_text(expanded_content, encoding="utf-8")

    try:
        converter = MarkdownToNotebookConverter()
        notebook = converter.convert(temp_md_path)
    finally:
        temp_md_path.unlink(missing_ok=True)

    notebook = make_notebook_executable(notebook, language_filter)

    temp_nb_path = md_path.with_suffix(".tmp.ipynb")
    with open(temp_nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    return temp_nb_path


def execute_notebook(
    notebook_path: Path,
    language_filter: str = "all",
    kernel_name: str = "python3",
    execution_timeout: int | None = None,
) -> Path:
    """Execute a single ``.md`` or ``.ipynb`` notebook via papermill.

    Handles conversion/transformation, runs the notebook, and cleans up
    intermediate temp files (``.tmp.ipynb``).  The executed output
    (``.executed.ipynb``) is **preserved** so it can be collected as a
    CI artifact or inspected after failure.

    Args:
        notebook_path: Path to the ``.md`` or ``.ipynb`` notebook.
        language_filter: Which cells to run (``"all"``, ``"python"``, ``"shell"``).
        kernel_name: Jupyter kernel to use.
        execution_timeout: Per-cell timeout in seconds (``None`` for no limit).

    Returns:
        Path to the executed ``.executed.ipynb`` output file.

    Raises:
        papermill.PapermillExecutionError: If any cell raises an error.
    """
    import papermill  # noqa: PLC0415 — lazy import to allow importing this module without papermill installed

    output_path = notebook_path.with_suffix(".executed.ipynb")

    if notebook_path.suffix == ".md":
        temp_nb = convert_md_to_notebook(notebook_path, language_filter)
        try:
            papermill.execute_notebook(
                input_path=str(temp_nb),
                output_path=str(output_path),
                kernel_name=kernel_name,
                cwd=str(notebook_path.parent),
                log_output=True,
                execution_timeout=execution_timeout,
            )
        finally:
            temp_nb.unlink(missing_ok=True)
    else:
        with open(notebook_path, encoding="utf-8") as f:
            nb_data = json.load(f)

        nb_data = make_notebook_executable(nb_data, language_filter)
        temp_nb = notebook_path.with_suffix(".tmp.ipynb")
        try:
            with open(temp_nb, "w", encoding="utf-8") as f:
                json.dump(nb_data, f, indent=1)
            papermill.execute_notebook(
                input_path=str(temp_nb),
                output_path=str(output_path),
                kernel_name=kernel_name,
                cwd=str(notebook_path.parent),
                log_output=True,
                execution_timeout=execution_timeout,
            )
        finally:
            temp_nb.unlink(missing_ok=True)

    return output_path


# TODO: Split notebook runs into smaller CI jobs scoped to
#       microservice/team-specific subdirectories under the docs tree,
#       so each job executes only the notebooks for its area instead of
#       traversing the entire docs structure in one batch.
def run_notebooks(
    notebooks_dir: str,
    language_filter: str = "all",
    keep_temp_files: bool = False,
    use_temporary_venv: bool = False,
    requirements_file: str | None = None,
) -> int:
    """
    Finds and runs notebooks with @nemo-nb: process marker in the given directory.
    Supports both .ipynb files and .md files (markdown notebooks).
    Enforces a fixed wall-clock timeout across all notebooks.

    Args:
        notebooks_dir: Directory to search for notebooks
        language_filter: Which cells to execute:
            - "all": Execute both Python and shell cells (default)
            - "python": Execute only Python cells
            - "shell": Execute only shell cells
        keep_temp_files: Whether to keep temporary files generated during execution
        use_temporary_venv: If True, create a temporary virtualenv for running notebooks.
            This allows testing library installations from scratch.
        requirements_file: Optional path to a requirements file to install in the temporary venv.
            Only used if use_temporary_venv is True.

    Returns:
        0 on success, 1 on failure
    """
    start_time = time.monotonic()

    if requirements_file and not use_temporary_venv:
        print("Warning: --requirements specified without --use-temporary-venv. Requirements will be ignored.")
        print("Use --use-temporary-venv to install requirements in a temporary virtualenv.\n")
        requirements_file = None

    kernel_name = "python3"
    temp_venv_dir: str | None = None
    temp_kernel_spec_dir: str | None = None

    if use_temporary_venv:
        print("Creating temporary virtualenv for notebook execution...")
        try:
            kernel_name, temp_venv_dir, temp_kernel_spec_dir = create_temp_venv_with_kernel(requirements_file)
            os.environ["VIRTUAL_ENV"] = temp_venv_dir
            if sys.platform == "win32":
                bin_path = os.path.join(temp_venv_dir, "Scripts")
            else:
                bin_path = os.path.join(temp_venv_dir, "bin")
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
            print(f"Created temporary kernel: {kernel_name}\n")
        except Exception as e:
            print(f"Failed to create temporary virtualenv: {e}")
            return 1

    def assert_not_timed_out() -> None:
        elapsed_seconds = time.monotonic() - start_time
        if elapsed_seconds > TIMEOUT_SECONDS:
            raise TimeoutError(
                "Timeout running notebooks after "
                f"{TIMEOUT_SECONDS} seconds. Please split notebook runs into "
                "smaller batches based on the docs folder structure "
                "(for example, per microservice or team-specific subfolder)."
            )

    result = find_processable_notebooks(notebooks_dir)

    if result.conflicts:
        print_conflicts_error(result.conflicts)
        return 1

    ipynb_to_run = result.ipynb_files
    md_to_run = result.md_files
    total_count = len(ipynb_to_run) + len(md_to_run)

    if total_count == 0:
        print(f"No notebooks with @nemo-nb: process marker found in {notebooks_dir}")
        notebooks_dir_path = Path(notebooks_dir)
        if notebooks_dir_path.is_dir():
            all_ipynb = list(notebooks_dir_path.rglob("*.ipynb"))
            all_md = list(notebooks_dir_path.rglob("*.md"))
            if all_ipynb or all_md:
                print(f"(Found {len(all_ipynb)} .ipynb and {len(all_md)} .md files total, but none had the marker)")
        return 0

    lang_desc = {"all": "all cells", "python": "Python cells only", "shell": "shell cells only"}
    print(f"Found {total_count} notebooks with @nemo-nb: process marker to run ({lang_desc[language_filter]}):")
    for nb in ipynb_to_run:
        print(f"  [ipynb] {nb}")
    for md in md_to_run:
        print(f"  [md]    {md}")

    failed_notebooks = []

    for nb in [*ipynb_to_run, *md_to_run]:
        assert_not_timed_out()
        suffix_label = "markdown notebook" if nb.suffix == ".md" else "ipynb"
        print(f"\nRunning {nb} ({suffix_label})...")
        output_path = nb.with_suffix(".executed.ipynb")
        try:
            execute_notebook(nb, language_filter, kernel_name)
            print(f"SUCCESS: {nb}")
        except Exception as e:
            print(f"FAILURE: {nb}")
            print(f"Error: {e}")
            failed_notebooks.append(str(nb))
        finally:
            if not keep_temp_files and output_path.exists():
                output_path.unlink()
        assert_not_timed_out()

    if temp_venv_dir and temp_kernel_spec_dir:
        print(f"\nCleaning up temporary virtualenv and kernel: {kernel_name}")
        cleanup_temp_venv_and_kernel(kernel_name, temp_venv_dir, temp_kernel_spec_dir)

    print("\n" + "=" * 30)
    if failed_notebooks:
        print(f"FAILED: {len(failed_notebooks)} notebook(s) failed.")
        for nb in failed_notebooks:
            print(f" - {nb}")
        return 1
    else:
        print("SUCCESS: All notebook(s) ran successfully.")
        return 0
