# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrail: the agent_eval package must stay free of NeMo-Helix imports.

The SDK is consumed by NeMo-Helix adapters, never the reverse. This test
fails if any module under ``agent_eval`` imports a platform-specific package,
which keeps the promoted generics from leaking coupling into the SDK.
"""

from __future__ import annotations

import re
from pathlib import Path

import nemo_evaluator_sdk.agent_eval as agent_eval

# agent_eval is an implicit namespace package (no __init__.py), so resolve its
# directory via __path__ rather than __file__ (which is None for namespaces).
AGENT_EVAL_ROOT = Path(next(iter(agent_eval.__path__))).resolve()

# Import statements that would couple the SDK to the platform / adapter.
_FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+"
    r"(nemo_helix|nhx_[A-Za-z0-9_]+|nat_runner|runtimes(?:\.|\s|$)|evaluator_agent_eval)",
    re.MULTILINE,
)


def test_agent_eval_has_no_platform_imports() -> None:
    offenders: list[str] = []
    for path in sorted(AGENT_EVAL_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _FORBIDDEN.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(AGENT_EVAL_ROOT)}:{line_no}: {match.group(0).strip()}")

    assert not offenders, "agent_eval must not import NeMo-Helix packages:\n" + "\n".join(offenders)
