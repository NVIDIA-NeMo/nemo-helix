# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place Harbor's LangGraph virtualenv on the writable agent volume."""

from __future__ import annotations

import sys
from pathlib import Path

VENV_PATH_REPLACEMENT = (
    '_REMOTE_VENV_DIR = PurePosixPath("/opt/harbor-langgraph-venv")',
    '_REMOTE_VENV_DIR = PurePosixPath("/tmp/harbor-langgraph-venv")',
)

LEGACY_REPLACEMENTS = (
    (
        'f"python3 -m venv {venv_dir}; "',
        'f"python3 -m venv --without-pip {venv_dir}; "',
    ),
    (
        '"python -m pip install uv; "',
        '"python /tmp/pip.pyz install uv; "',
    ),
)

UV_MANAGED_VENV_ANCHOR = 'f"uv venv {venv_dir} --python {python_version} --clear; "'


def _replace_exactly_once(source: str, old: str, new: str, path: Path) -> str:
    if new in source and old not in source:
        return source
    if source.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one LangGraph venv source anchor in {path}; "
            "update this patch for the installed Harbor version"
        )
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source = path.read_text()
    source = _replace_exactly_once(source, *VENV_PATH_REPLACEMENT, path)

    if UV_MANAGED_VENV_ANCHOR not in source:
        for old, new in LEGACY_REPLACEMENTS:
            source = _replace_exactly_once(source, old, new, path)

    path.write_text(source)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} LANGGRAPH_PY")
    patch(Path(sys.argv[1]))
