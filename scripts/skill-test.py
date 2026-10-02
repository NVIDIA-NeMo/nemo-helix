#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Run four-mode invocation tests for NeMo skills.

Each skill ships a `tests.json` next to its `SKILL.md` with four test types:
- explicit: user names the skill; should fire
- implicit: user states the intent; should fire
- contextual: request with surrounding situation; should route to its stated skill
- negative-control: unrelated topic; should NOT fire

v1: keyword-overlap heuristic against skill `triggers` + `description`.
v2: route through a real classifier (LLM call against the skill catalog).

Exit codes:
    0  all tests pass
    1  one or more tests fail
    2  internal error (missing files, malformed JSON, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_GLOBS = [
    "packages/nemo_helix_ext/src/nemo_helix_ext/skills/*/SKILL.md",
    ".agents/skills/*/SKILL.md",
    "plugins/*/src/*/skills/*/SKILL.md",
    "sdk/python/nemo-helix/src/nemo_helix/cli/commands/skills/content/*/SKILL.md",
    "packages/*/src/*/.agents/skills/*/SKILL.md",
]

FRONTMATTER = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


@dataclass
class Skill:
    name: str
    path: Path
    triggers: list[str]
    description: str

    def keywords(self) -> set[str]:
        tokens = set()
        for trig in self.triggers:
            tokens.update(w.lower() for w in WORD.findall(trig))
        tokens.update(w.lower() for w in WORD.findall(self.description))
        # drop stop-ish words
        return tokens - {
            "a",
            "the",
            "an",
            "for",
            "of",
            "to",
            "with",
            "use",
            "when",
            "and",
            "or",
            "from",
            "this",
            "that",
            "as",
        }


def parse_skill(path: Path) -> Skill | None:
    text = path.read_text()
    m = FRONTMATTER.search(text)
    if not m:
        return None
    try:
        fields = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        print(f"WARNING: malformed frontmatter in {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(fields, dict):
        return None
    name = fields.get("name") or path.parent.name
    description = fields.get("description") or ""
    raw_triggers = fields.get("triggers") or []
    if isinstance(raw_triggers, str):
        # tolerate comma-separated form
        triggers = [t.strip() for t in raw_triggers.split(",") if t.strip()]
    elif isinstance(raw_triggers, list):
        triggers = [str(t).strip() for t in raw_triggers if str(t).strip()]
    else:
        triggers = []
    return Skill(name=name, path=path, triggers=triggers, description=description)


def find_skills(root: Path) -> list[Skill]:
    skills: list[Skill] = []
    for pattern in SKILL_GLOBS:
        for p in root.glob(pattern):
            skill = parse_skill(p)
            if skill:
                skills.append(skill)
    return skills


def score(prompt: str, skill: Skill) -> int:
    prompt_words = {w.lower() for w in WORD.findall(prompt)}
    return len(prompt_words & skill.keywords())


def best_match(prompt: str, skills: list[Skill]) -> Skill | None:
    scored = [(score(prompt, s), s) for s in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] == 0:
        return None
    return scored[0][1]


def run_tests(skill_dir: Path, all_skills: list[Skill]) -> tuple[int, int, list[str]]:
    tests_file = skill_dir / "tests.json"
    if not tests_file.exists():
        return 0, 0, [f"NO TESTS: {skill_dir}"]
    try:
        data = json.loads(tests_file.read_text())
    except json.JSONDecodeError as exc:
        return 0, 0, [f"MALFORMED tests.json in {skill_dir}: {exc}"]
    passed = 0
    failed: list[str] = []
    for t in data.get("tests", []):
        match = best_match(t["prompt"], all_skills)
        match_name = match.name if match else None
        expected = t.get("expected_skill")
        if expected:
            if match_name == expected:
                passed += 1
            else:
                failed.append(f"  [{t['type']}] expected `{expected}`, got `{match_name}`: {t['prompt']!r}")
        else:
            forbidden = t.get("expected_skill_not")
            if not forbidden:
                failed.append(f"  [{t['type']}] needs `expected_skill` or `expected_skill_not`: {t['prompt']!r}")
                continue
            if match_name != forbidden:
                passed += 1
            else:
                failed.append(f"  [{t['type']}] should NOT fire `{forbidden}`: {t['prompt']!r}")
    return passed, len(failed), failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--format", choices=["human", "github"], default="human")
    args = parser.parse_args()

    skills = find_skills(args.root)
    if not skills:
        print(f"No skills found under {args.root}", file=sys.stderr)
        return 2

    total_pass = 0
    total_fail = 0
    failures_by_skill: dict[str, list[str]] = {}

    for skill in skills:
        passed, failed_count, failures = run_tests(skill.path.parent, skills)
        total_pass += passed
        total_fail += failed_count
        if failures:
            failures_by_skill[skill.name] = failures

    if total_fail == 0:
        print(f"OK: {total_pass} tests passed across {len(skills)} skills.")
        return 0

    if args.format == "github":
        for skill_name, failures in failures_by_skill.items():
            for failure in failures:
                print(f"::error title=skill-test {skill_name}::{failure.strip()}")
    else:
        print(f"FAIL: {total_fail} failures, {total_pass} passes\n")
        for skill_name, failures in failures_by_skill.items():
            print(f"{skill_name}:")
            for failure in failures:
                print(failure)
            print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
