# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrail: the intake mapping module must not import the Intake service.

The mapping is pure boundary code: it reads SDK types and returns plain dicts
shaped for Intake's requests, but it must not depend on the Intake service
(``nhx.intake.*``), an HTTP client, or the platform client. This keeps the
translation isolated so D3/D4/D5 can build the wire calls on top of it without
the mapping itself pulling in the service.
"""

from __future__ import annotations

import re
from pathlib import Path

import nemo_evaluator.intake as intake

INTAKE_ROOT = Path(next(iter(intake.__path__))).resolve()

# Imports that would couple the pure mapping to the Intake service or transport.
_FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+(nhx\.intake|nhx_intake|httpx)",
    re.MULTILINE,
)


def test_intake_mapping_has_no_service_imports() -> None:
    offenders: list[str] = []
    for path in sorted(INTAKE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _FORBIDDEN.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(INTAKE_ROOT)}:{line_no}: {match.group(0).strip()}")

    assert not offenders, "nemo_evaluator.intake must not import the Intake service / transport:\n" + "\n".join(
        offenders
    )
