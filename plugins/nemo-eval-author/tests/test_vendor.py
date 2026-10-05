# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from nemo_eval_author_plugin.runner import SKILLS_DIR

FIRST_EVAL_SKILL = SKILLS_DIR / "eval-author-first-eval" / "SKILL.md"
RELATIVE_LINK = re.compile(r"\]\((?!https?://)([^)#\s]+)(?:#[^)]*)?\)")


def test_first_eval_skill_is_vendored_with_every_linked_file() -> None:
    text = FIRST_EVAL_SKILL.read_text(encoding="utf-8")
    links = RELATIVE_LINK.findall(text)

    assert "name: eval-author-first-eval" in text
    assert links
    assert [link for link in links if not (FIRST_EVAL_SKILL.parent / link).is_file()] == []
