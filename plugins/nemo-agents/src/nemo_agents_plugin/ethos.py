# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight ETHOS.md contract.

Ethos is a human-readable markdown file stored locally at
``agents/<name>-ethos/ETHOS.md`` and canonically in Filesets as
``<workspace>/<name>-ethos#ETHOS.md``.

Only the front matter and section outline are machine-validated here. The
section bodies remain markdown for humans and agents to read directly.

The schema records intent, not just a build-time inventory: how to weigh
competing wins (``Trade-offs``), what no change may cross (``Constraints``),
how to decide when no rule covers the case (``Principles``), what the
telemetry actually means (``Metric Semantics``), and where the agent is
headed (``Vision``). Mission and the result the agent is accountable for live
together in ``Purpose & Outcomes``.

Every canonical body section is required. When a section has nothing to say,
keep the heading and write ``_(none)_`` rather than dropping it.

Additional ``##`` headings and extra YAML front-matter keys are allowed. The
parser keeps unknown body sections and does not fail on unknown front-matter
keys. The fifteen headings are a floor, not a closed schema.

Four headings from the earlier AGENT-SPEC outline are retired rather than
required:

- ``Framework`` had no reader, and the container's framework label comes from
  ``agent.yaml``. Describe how the agent runs in ``Harness``, or write
  ``_(none)_``. Do not map the implementation onto a named platform harness.
- ``Model`` restated implementation the config already carries and went stale
  on the first model swap. Which models are *allowed* is a ``Constraints``
  entry; whether the loop may swap them is a ``Change Scope`` lever.
- ``Signals`` was a single consumer's configuration wearing a schema section.
  How a consumer reads evidence belongs in that consumer, not in this file.
- ``Purpose`` merged into ``Purpose & Outcomes``.

The same rule retired a drafted ``Budget`` section before it shipped: anything
that only configures one optimization run belongs to the tool running it, or
this file becomes a dumping ground that rots between runs.

A leftover retired heading is tolerated so a file can be filled in one section
at a time. If a ``Framework`` heading is still present, it must be resolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

ETHOS_SCHEMA_VERSION = 1
"""Schema version written by current tooling."""

ETHOS_SECTION_TITLES: tuple[str, ...] = (
    "Role",
    "Purpose & Outcomes",
    "Scope",
    "Tools",
    "Harness",
    "Behavior",
    "Principles",
    "Success Criteria",
    "Trade-offs",
    "Constraints",
    "Evaluation Setup",
    "Metric Semantics",
    "Change Scope",
    "Vision",
    "Open Questions",
)
"""Every canonical ``##`` heading, in reading order. All of these are required.

Additional headings are allowed and stored in :attr:`Ethos.sections`. The
order below tells a story: who the agent is, what it is today, how to judge it,
what may change, and where it is going. ``Principles`` follows ``Behavior``
because the pair reads as the concrete rules and then what to do when the rules
run out. ``Vision`` follows ``Change Scope`` so today's permissions and
tomorrow's direction sit together.
"""

RETIRED_SECTION_TITLES: tuple[str, ...] = ("Framework", "Model", "Signals", "Purpose")
"""Earlier AGENT-SPEC headings with no equivalent here, kept for recognition only.

A file mid-upgrade may still carry these. The parser neither requires nor
rejects them, so they survive in :attr:`Ethos.sections` for a human to read
while nothing downstream is asked to interpret them.
"""

CHANGE_SCOPE_LEVER_VALUES: tuple[str, ...] = ("yes", "no", "with-approval")
"""Recognized ``Change Scope`` lever values.

``with-approval`` covers a change that is permitted but must not ship
unattended. Conditional permission used to live only in the section's prose
``Notes`` line, where no consumer could act on it.
"""

_LEVER_RE = re.compile(r"^-[ \t]*([^:\r\n]+?)[ \t]*:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def _require_supported_schema_version(version: int) -> None:
    if version < 1 or version > ETHOS_SCHEMA_VERSION:
        raise ValueError(f"schema version must be between 1 and {ETHOS_SCHEMA_VERSION}")


def required_sections(version: int) -> tuple[str, ...]:
    """Sections that must be present for ``version`` to parse at all."""
    _require_supported_schema_version(version)
    return ETHOS_SECTION_TITLES


def known_sections(version: int) -> tuple[str, ...]:
    """Every section ``version`` defines, in canonical order."""
    _require_supported_schema_version(version)
    return ETHOS_SECTION_TITLES


@dataclass(frozen=True)
class Ethos:
    """Parsed ETHOS.md document.

    ``sections`` stores raw markdown by heading title. Downstream agents should
    read that markdown rather than relying on a bespoke nested Python schema.
    """

    name: str
    created_timestamp: datetime
    author: str
    sections: dict[str, str]
    schema_version: int = 1
    updated_timestamp: datetime | None = None
    owner: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def role(self) -> str:
        return self.sections["Role"]

    @property
    def change_scope_levers(self) -> dict[str, str]:
        """``Change Scope`` levers whose value is a recognized permission.

        Lenient by design: the section body is free-form markdown in practice,
        so anything that is not a plain ``- Label: yes|no|with-approval`` line
        is skipped rather than reported. Read the raw section for the rest.
        """
        levers: dict[str, str] = {}
        for label, value in _LEVER_RE.findall(self.sections.get("Change Scope", "")):
            normalized = value.strip().strip("*`").lower()
            if normalized in CHANGE_SCOPE_LEVER_VALUES:
                levers[label.strip()] = normalized
        return levers
