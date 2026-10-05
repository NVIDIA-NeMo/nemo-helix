# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

from nemo_eval_author_plugin.runner import SKILLS_DIR

FIRST_EVAL_SKILL = SKILLS_DIR / "eval-author-first-eval" / "SKILL.md"
RELATIVE_LINK = re.compile(r"\]\((?!https?://)([^)#\s]+)(?:#[^)]*)?\)")


def _linked_files(start: Path) -> set[Path]:
    seen: set[Path] = set()
    queue = [start.resolve()]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix == ".md" and path.is_file():
            text = path.read_text(encoding="utf-8")
            queue.extend((path.parent / link).resolve() for link in RELATIVE_LINK.findall(text))
    return seen


def test_first_eval_skill_is_vendored_with_every_linked_file() -> None:
    files = _linked_files(FIRST_EVAL_SKILL)

    assert "name: eval-author-first-eval" in FIRST_EVAL_SKILL.read_text(encoding="utf-8")
    assert len(files) > 1
    assert [path for path in files if not path.is_file()] == []
