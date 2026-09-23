#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Jupyter notebooks to inline Fern MDX using nemo-nb.

Wraps ``nemo_nb.converter.NotebookConverter`` (``nemo-nb to-sphinx-md``) and
post-processes the output for Fern:
  - Fern frontmatter (title, description)
  - Google Colab link instead of the nemo-nb download anchor
  - Relative doc links rewritten to canonical ``/documentation/...`` URLs

Usage:
  python ipynb-to-mdx.py input.ipynb -o output.mdx --title "Page Title"
  python ipynb-to-mdx.py --all-customizer-tutorials
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from nemo_nb.converter import NotebookConverter

COLAB_REPO = "https://colab.research.google.com/github/NVIDIA-NeMo/nemo-helix/blob/main"

DOWNLOAD_LINK_RE = re.compile(
    r'<a href="[^"]+\.ipynb" download="[^"]+\.ipynb">Download this tutorial as a Jupyter notebook</a>\s*',
    re.IGNORECASE,
)
FIRST_H1_RE = re.compile(r"\A# [^\n]+\n+")

_LINK_REWRITES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\]\(\.\./\.\./get-started/quickstart\.md?\)"),
        "](/documentation/get-started)",
    ),
    (
        re.compile(r"\]\(\.\./\.\./get-started/concepts/manage-secrets\.md?\)"),
        "](/documentation/get-started/core-concepts/manage-secrets)",
    ),
    (
        re.compile(r"\]\(\.\./manage-customization-jobs/hyperparameters\.md?\)"),
        "](/documentation/customizer-reference/manage-customization-jobs/training-configuration)",
    ),
    (
        re.compile(r"\]\(\.\./manage-customization-jobs/get-job-status\.md?\)"),
        "](/documentation/customizer-reference/manage-customization-jobs/get-job-status)",
    ),
    (
        re.compile(r"\]\(\.\./\.\./evaluator/index(?:\.md)?\)"),
        "](/documentation/evaluate-models)",
    ),
    (
        re.compile(r"\]\(\.\./\.\./evaluator/metrics/rag\.md?\)"),
        "](/documentation/evaluate-models/metrics/rag-metrics)",
    ),
    (
        re.compile(r"\]\(\./distillation-customization-job(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/distillation-customization-job)",
    ),
    (
        re.compile(r"\]\(\./embedding-customization-job(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/embedding-customization-job)",
    ),
    (
        re.compile(r"\]\(\./embedding-triplet-dataset(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/embedding-triplet-dataset)",
    ),
    (
        re.compile(r"\]\(\.\./\.\./data-designer/tutorials/retrieval-generate(?:\.ipynb)?\)"),
        "](/documentation/design-synthetic-data/tutorials/retrieval-generate)",
    ),
    (
        re.compile(r"\]\(\.\./\.\./customizer/tutorials/embedding-customization-job(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/embedding-customization-job)",
    ),
    (
        re.compile(r"\]\(\.\./retrieval-sdg(?:\.mdx)?\)"),
        "](/documentation/design-synthetic-data/retrieval-sdg)",
    ),
    (
        re.compile(r"\]\(\.\./\.\./data-designer/retrieval-sdg(?:\.mdx)?\)"),
        "](/documentation/design-synthetic-data/retrieval-sdg)",
    ),
    (
        re.compile(r"\]\(\.\./models/data-format(?:\.md)?\)"),
        "](/documentation/customizer-reference/models/dataset-format)",
    ),
    (
        re.compile(r"\]\(\./lora-customization-job(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/lora-customization-job)",
    ),
    (
        re.compile(r"\]\(\./optimize-throughput(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/optimize-throughput)",
    ),
    (
        re.compile(r"\]\(\./sft-customization-job(?:\.ipynb)?\)"),
        "](/documentation/customizer-reference/tutorials/sft-customization-job)",
    ),
    (
        re.compile(r"\]\(fine-tune-metrics\)"),
        "](/documentation/customizer-reference/tutorials/metrics)",
    ),
    (
        re.compile(r"\]\(nemo-ms-about-concepts-customization\)"),
        "](/documentation/customizer-reference/customization-concepts#nemo-ms-about-concepts-customization)",
    ),
]

CUSTOMIZER_TUTORIALS: list[tuple[str, str]] = [
    ("sft-customization-job.ipynb", "Full SFT Customization"),
    ("lora-customization-job.ipynb", "LoRA Model Customization"),
    ("distillation-customization-job.ipynb", "Knowledge Distillation Customization"),
    ("embedding-customization-job.ipynb", "Embedding Model Customization"),
    ("embedding-triplet-dataset.ipynb", "Train on SPECTER or Triplet JSONL"),
    ("optimize-throughput.ipynb", "Optimize for Tokens/GPU Throughput"),
]

DATA_DESIGNER_TUTORIALS: list[tuple[str, str]] = [
    ("retrieval-generate.ipynb", "Generate Retrieval Training Data"),
]


def rewrite_links(text: str) -> str:
    for pattern, replacement in _LINK_REWRITES:
        text = pattern.sub(replacement, text)
    return text


# Jupyter shell cells start with %%bash; published MDX is a copy-paste sh block.
_JUPYTER_SHELL_MAGIC_RE = re.compile(r"(```(?:sh|bash|shell)\n)%%bash\n")


def strip_jupyter_shell_magics(text: str) -> str:
    return _JUPYTER_SHELL_MAGIC_RE.sub(r"\1", text)


def repo_relative_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return path.resolve().relative_to(repo_root).as_posix()


def colab_link_for(ipynb_path: Path) -> str:
    return f"[Run in Google Colab]({COLAB_REPO}/{repo_relative_path(ipynb_path)})"


def convert_notebook_to_mdx(ipynb_path: Path, *, title: str) -> str:
    body = NotebookConverter().convert(ipynb_path)
    body = DOWNLOAD_LINK_RE.sub("", body).lstrip("\n")
    body = FIRST_H1_RE.sub("", body, count=1)
    body = rewrite_links(body)
    body = strip_jupyter_shell_magics(body)

    return (
        "---\n"
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f'title: "{title}"\n'
        'description: ""\n'
        "---\n"
        "\n"
        f"{colab_link_for(ipynb_path)}\n"
        "\n"
        f"{body.rstrip()}\n"
    )


def convert_all_customizer_tutorials(tutorials_dir: Path) -> int:
    rc = 0
    for notebook_name, title in CUSTOMIZER_TUTORIALS:
        ipynb = tutorials_dir / notebook_name
        mdx = tutorials_dir / notebook_name.replace(".ipynb", ".mdx")
        if not ipynb.exists():
            print(f"Error: {ipynb} not found", file=sys.stderr)
            rc = 1
            continue
        mdx.write_text(convert_notebook_to_mdx(ipynb, title=title), encoding="utf-8")
        print(f"Wrote {mdx}")
    return rc


def convert_all_data_designer_tutorials(tutorials_dir: Path) -> int:
    rc = 0
    for notebook_name, title in DATA_DESIGNER_TUTORIALS:
        ipynb = tutorials_dir / notebook_name
        mdx = tutorials_dir / notebook_name.replace(".ipynb", ".mdx")
        if not ipynb.exists():
            print(f"Error: {ipynb} not found", file=sys.stderr)
            rc = 1
            continue
        mdx.write_text(convert_notebook_to_mdx(ipynb, title=title), encoding="utf-8")
        print(f"Wrote {mdx}")
    return rc


def main() -> int:
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        return 0

    if "--all-customizer-tutorials" in args:
        repo_root = Path(__file__).resolve().parents[3]
        tutorials_dir = repo_root / "docs" / "customizer" / "tutorials"
        rc = convert_all_customizer_tutorials(tutorials_dir)
        dd_dir = repo_root / "docs" / "data-designer" / "tutorials"
        dd_rc = convert_all_data_designer_tutorials(dd_dir)
        return rc or dd_rc

    input_path = Path(args[0])
    output_path: Path | None = None
    title: str | None = None

    if "-o" in args:
        idx = args.index("-o")
        if idx + 1 < len(args):
            output_path = Path(args[idx + 1])
    if "--title" in args:
        idx = args.index("--title")
        if idx + 1 < len(args):
            title = args[idx + 1]

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1
    if output_path is None:
        output_path = input_path.with_suffix(".mdx")
    if not title:
        print("Error: --title is required", file=sys.stderr)
        return 1

    output_path.write_text(convert_notebook_to_mdx(input_path, title=title), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
