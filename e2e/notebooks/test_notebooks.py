# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests that execute documentation notebooks against a running quickstart.

Each processable notebook (marked with ``@nemo-nb: process``) in ``docs/``
becomes an individual pytest test case.  The quickstart backend is started
once per session via the shared ``e2e/conftest.py`` fixtures;
``nhx_base_url`` (from this package's conftest) sets ``NHX_BASE_URL`` so the
notebook kernels can reach the API.

Notebooks are tagged by their infrastructure requirements:

- **GPU_NOTEBOOKS** need a GPU-enabled quickstart (model deployment, NIM).
  Run with ``--feature gpu``.
- **BIFURCATED_NOTEBOOKS** contain both CLI and Python SDK cells and are
  automatically split into two test cases: one running only ``shell`` cells
  and one running only ``python`` cells.  The ``--notebook-language`` flag
  does **not** affect bifurcated notebooks; use ``-k "shell"`` or
  ``-k "python"`` to select a specific variant.
- Everything else should pass against a bare CPU quickstart.

Required environment variables
------------------------------
``NHX_BASE_URL`` is set automatically by the ``nhx_base_url`` fixture.
Other variables must be present in the environment (CI secrets, ``.env``,
or exported in your shell) **before** the relevant tests run:

=================  ============================================================
Variable           Notebooks that require it
=================  ============================================================
``NVIDIA_API_KEY`` evaluator tutorials, example-applications,
                   run-inference (about, deploy-models),
                   data-designer (quickstart, basics, seeding)
``NGC_API_KEY``    guardrails tutorials, audit/docker-local-nim
``HF_TOKEN``       evaluator/run-an-evaluation,
                   safe-synthesizer-101, run-inference/deploy-models
``NIM_API_KEY``    safe-synthesizer/pii-replacement,
                   safe-synthesizer/safe-synthesizer-101
``OPENAI_API_KEY`` run-inference/deploy-models
=================  ============================================================

Usage:
    # CPU notebooks only (default)
    uv run pytest e2e/notebooks -v

    # GPU notebooks only
    uv run pytest e2e/notebooks -v --feature gpu

    # Specific notebook
    uv run pytest e2e/notebooks -v -k "safe-synthesizer-101"

    # Only the shell variant of bifurcated notebooks
    uv run pytest e2e/notebooks -v -k "shell"

    # Only the Python variant of bifurcated notebooks
    uv run pytest e2e/notebooks -v -k "python"
"""

import os
from pathlib import Path

import pytest
from nemo_nb import find_processable_notebooks
from nemo_nb.discovery import (
    has_skip_test_marker_markdown,
    has_skip_test_marker_notebook,
)
from nhx.testing.notebooks import execute_notebook
from nhx.testing.pytest_outcomes import pytest_skip

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Per-notebook environment variable requirements.
#
# Keys are repo-relative paths; values are sets of env var names the
# notebook reads at runtime.  ``NHX_BASE_URL`` is omitted because the
# ``nhx_base_url`` fixture handles it for every test.
# ---------------------------------------------------------------------------
NOTEBOOK_REQUIRED_ENV: dict[str, set[str]] = {
    # Guardrails: chat completions route through the system/nvidia-build model provider,
    # which uses the system/ngc-api-key secret for authentication.
    "docs/guardrails/concepts/configurations/default-configs.md": {"NGC_API_KEY"},
    "docs/guardrails/concepts/inference.md": {"NGC_API_KEY"},
    "docs/guardrails/tutorials/content-safety.md": {"NGC_API_KEY"},
    "docs/guardrails/tutorials/injection-detection.md": {"NGC_API_KEY"},
    "docs/guardrails/tutorials/multimodal-data.md": {"NGC_API_KEY"},
    "docs/guardrails/tutorials/parallel-rails.md": {"NGC_API_KEY"},
    # Evaluator
    "docs/evaluator/tutorials/run-an-evaluation.md": {"NVIDIA_API_KEY", "HF_TOKEN"},
    "docs/evaluator/tutorials/run-llm-judge-evaluation.md": {"NVIDIA_API_KEY"},
    # Example applications
    "docs/example-applications/custom-evaluations-synthetic-data.md": {"NVIDIA_API_KEY"},
    # Run-inference
    "docs/run-inference/tutorials/deploy-models.md": {
        "NVIDIA_API_KEY",
        "HF_TOKEN",
        "OPENAI_API_KEY",
    },
    "docs/run-inference/about.md": {"NVIDIA_API_KEY"},
    # Data-designer: preview calls NIM for generation via nvidia-build provider
    "docs/data-designer/quickstart.md": {"NVIDIA_API_KEY"},
    "docs/data-designer/tutorials/basics.md": {"NVIDIA_API_KEY"},
    "docs/data-designer/tutorials/seeding.md": {"NVIDIA_API_KEY"},
    # Audit
    "docs/audit/tutorials/docker-local-nim.md": {"NGC_API_KEY"},
    # Safe-synthesizer
    "docs/safe-synthesizer/tutorials/pii-replacement.md": {"NIM_API_KEY"},
    "docs/safe-synthesizer/tutorials/safe-synthesizer-101.md": {
        "NIM_API_KEY",
        "HF_TOKEN",
    },
}

# ---------------------------------------------------------------------------
# Notebooks that require a GPU-enabled quickstart (model deployment, NIM, HF).
# Skipped unless ``--feature gpu`` is passed.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Notebooks that are split into two test cases: one for shell cells only and
# one for Python cells only.  The ``--notebook-language`` CLI flag is ignored
# for these notebooks; they always produce a ``[shell]`` and ``[python]``
# variant.
# ---------------------------------------------------------------------------
BIFURCATED_NOTEBOOKS: set[str] = {
    "docs/run-inference/tutorials/deploy-models.md",
    "docs/run-inference/tutorials/run-inference.md",
}

GPU_NOTEBOOKS: set[str] = {
    # Run-inference: need a deployed NIM
    "docs/run-inference/tutorials/deploy-models.md",
    "docs/run-inference/tutorials/run-inference.md",
    # Audit: local NIM required
    "docs/audit/tutorials/docker-local-nim.md",
    # Safe-synthesizer tutorials: run GPU jobs
    "docs/safe-synthesizer/tutorials/differential-privacy.md",
    "docs/safe-synthesizer/tutorials/pii-replacement.md",
    "docs/safe-synthesizer/tutorials/safe-synthesizer-101.md",
}

CELL_TIMEOUT_SECONDS = 1800


def _discover_notebooks() -> list:
    """Discover all testable notebooks and attach requirement marks.

    Uses ``find_processable_notebooks`` and attaches ``pytest.mark.skip`` to
    notebooks carrying the ``@nemo-nb: skip-test`` marker so they appear as
    skipped in the test output rather than being invisible.

    Bifurcated notebooks (listed in ``BIFURCATED_NOTEBOOKS``) produce two
    entries: ``(path, "shell")`` and ``(path, "python")``.  All other
    notebooks produce ``(path, None)``, which means "use the
    ``--notebook-language`` CLI option at runtime".
    """
    if not DOCS_DIR.is_dir():
        return []

    result = find_processable_notebooks(str(DOCS_DIR))
    params = []
    for nb_path in sorted([*result.ipynb_files, *result.md_files], key=str):
        rel = str(nb_path.relative_to(REPO_ROOT))
        marks: list = []

        is_skip_test = (
            has_skip_test_marker_notebook(nb_path)
            if nb_path.suffix == ".ipynb"
            else has_skip_test_marker_markdown(nb_path)
        )
        if is_skip_test:
            marks.append(pytest.mark.skip(reason="@nemo-nb: skip-test"))

        if rel in GPU_NOTEBOOKS:
            marks.append(pytest.mark.feature("gpu"))
        if rel in BIFURCATED_NOTEBOOKS:
            for lang in ("shell", "python"):
                params.append(pytest.param(nb_path, lang, id=f"{rel}[{lang}]", marks=marks))
        else:
            params.append(pytest.param(nb_path, None, id=rel, marks=marks))
    return params


def _check_required_env_vars(notebook_path: Path) -> None:
    """Skip the test early if any required env vars are missing."""
    rel = str(notebook_path.relative_to(REPO_ROOT))
    required = NOTEBOOK_REQUIRED_ENV.get(rel)
    if not required:
        return
    missing = sorted(v for v in required if not os.environ.get(v))
    if missing:
        pytest_skip(f"Missing env var(s): {', '.join(missing)}")


@pytest.mark.parametrize("notebook_path,language_override", _discover_notebooks())
def test_notebook(
    notebook_path: Path,
    language_override: str | None,
    nhx_base_url: str,
    notebook_language: str,
    notebook_kernel: str,
) -> None:
    """Execute a documentation notebook against the quickstart and assert it passes.

    The executed ``.executed.ipynb`` output is preserved next to the source
    file so CI can collect it as an artifact for debugging.

    ``language_override`` is set for bifurcated notebooks (``BIFURCATED_NOTEBOOKS``)
    so they always run as two separate test cases (``shell`` and ``python``),
    independent of the ``--notebook-language`` CLI option.  For all other
    notebooks ``language_override`` is ``None`` and the CLI option is used.
    """
    _check_required_env_vars(notebook_path)
    language = language_override if language_override is not None else notebook_language
    output_path = (
        notebook_path.with_suffix(f".{language}.executed.ipynb")
        if language_override is not None
        else notebook_path.with_suffix(".executed.ipynb")
    )
    execute_notebook(
        notebook_path,
        language_filter=language,
        kernel_name=notebook_kernel,
        execution_timeout=CELL_TIMEOUT_SECONDS,
        output_path=output_path,
    )
