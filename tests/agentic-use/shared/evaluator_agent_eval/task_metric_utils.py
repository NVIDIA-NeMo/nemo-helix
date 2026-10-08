# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for task-local Evals SDK metrics.

These helpers keep task-specific metric modules focused on their scoring policy.
They deliberately return plain Python values so callers can use them from SDK
``Metric`` implementations today and from enriched metric outputs later.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from evaluator_agent_eval.artifacts import AgentArtifacts

CodePredicate = Callable[[str], bool]
CodeScorer = Callable[[str], int]


def contains_all(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def score_checks(checks: list[bool]) -> float:
    if not checks:
        return 0.0
    return sum(float(value) for value in checks) / len(checks)


def extract_fenced_python_code(
    text: str,
    *,
    predicate: CodePredicate,
    scorer: CodeScorer | None = None,
) -> str | None:
    best_code: str | None = None
    best_score = -1
    for match in re.finditer(r"```(?:python|py)\s*\n(?P<code>.*?)```", text, re.DOTALL | re.IGNORECASE):
        code = match.group("code").strip()
        if not predicate(code):
            continue
        if scorer is None:
            return code
        score = scorer(code)
        if score > best_score:
            best_code = code
            best_score = score
    return best_code


def find_workspace_python_code(
    artifacts: AgentArtifacts,
    *,
    preferred_names: list[str],
    predicate: CodePredicate,
) -> str | None:
    workspace_dir = artifacts.workspace_dir
    if workspace_dir is None:
        return None

    candidates = [artifacts.workspace_artifact(name) for name in preferred_names]
    candidates.extend(sorted(path for path in workspace_dir.glob("*.py") if path.is_file()))

    seen: set[Path] = set()
    for path in candidates:
        if path is None or path in seen or not path.is_file():
            continue
        seen.add(path)
        code = path.read_text(encoding="utf-8", errors="replace").strip()
        if predicate(code):
            return code
    return None


def run_python_code(
    code: str,
    *,
    filename: str,
    timeout: int,
    appended_code: str = "",
    cwd: Path | None = None,
    timeout_stderr: str,
) -> subprocess.CompletedProcess[str]:
    script_text = f"{code}\n\n{appended_code}\n" if appended_code else code
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / filename
        script_path.write_text(script_text, encoding="utf-8")
        return run_python_file(
            script_path,
            timeout=timeout,
            cwd=cwd,
            timeout_stderr=timeout_stderr,
            missing_stderr=f"missing {filename}",
        )


def run_python_file(
    path: Path | None,
    *,
    timeout: int,
    cwd: Path | None = None,
    timeout_stderr: str,
    missing_stderr: str,
) -> subprocess.CompletedProcess[str]:
    if path is None:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=missing_stderr)
    command = [sys.executable, str(path)]
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=exc.cmd or command,
            returncode=124,
            stdout=timeout_text(exc.stdout),
            stderr=timeout_text(exc.stderr, fallback=timeout_stderr),
        )


def extract_json_object_from_stdout(stdout: str) -> dict[str, object] | None:
    lines = stdout.splitlines()
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:]).strip()
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed = object_dict(value)
        if parsed:
            return parsed
    return None


def extract_marker_json_object(stdout: str, *, marker: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line.removeprefix(marker))
        except json.JSONDecodeError:
            continue
        parsed = object_dict(value)
        if parsed:
            return parsed
    return None


def timeout_text(value: str | bytes | None, *, fallback: str = "") -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return fallback
