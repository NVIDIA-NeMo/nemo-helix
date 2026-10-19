"""The routing question, rendered one way everywhere.

The same text is shown to every model under test (large model, untuned small
model, fine-tuned decision model) and is the `prompt` field of every training
record. data/incidents.py, scripts/split_dataset.py, eval/make_eval_specs.py and
the agent's tool server all call these functions, so change the wording here and
nowhere else. Then regenerate or re-split the data, since the stored prompts in
data/meridian-example/ were rendered with the current text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LETTERS = "ABCDEFGHIJKL"
ACTION_LETTERS = "WXYZ"

DEFAULT_TAXONOMY = Path(__file__).resolve().parents[2] / "taxonomy.json"


def load_taxonomy(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_TAXONOMY, encoding="utf-8") as fh:
        return json.load(fh)


def runbook_options(taxonomy: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """[(letter, runbook), ...] in taxonomy order. Fixed order on purpose."""
    return list(zip(LETTERS, taxonomy["runbooks"]))


def action_options(taxonomy: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return list(zip(ACTION_LETTERS, taxonomy["actions"]))


def render_runbook_question(incident_text: str, taxonomy: dict[str, Any], with_context: bool = False) -> str:
    lines = [
        "You route incidents for Meridian Compute's GPU platform operations.",
        "Read the incident and pick the runbook that owns it.",
        "",
        "Incident:",
        incident_text.strip(),
        "",
        "Runbooks:",
    ]
    for letter, rb in runbook_options(taxonomy):
        lines.append(f"{letter}. {rb['id']} {rb['name']} ({rb['team']}): {rb['definition']}")
    if with_context:
        lines += ["", "Meridian routing notes:"]
        for _, rb in runbook_options(taxonomy):
            lines.append(f"- {rb['id']}: {rb['private_context']}")
    lines += ["", "Answer with the letter only.", "Answer:"]
    return "\n".join(lines)


def render_action_question(incident_text: str, runbook_id: str, taxonomy: dict[str, Any], with_context: bool = False) -> str:
    rb = next(r for r in taxonomy["runbooks"] if r["id"] == runbook_id)
    lines = [
        "You route incidents for Meridian Compute's GPU platform operations.",
        f"This incident is owned by {rb['id']} {rb['name']} ({rb['team']}). Pick the response action.",
        "",
        "Incident:",
        incident_text.strip(),
        "",
        "Actions:",
    ]
    for letter, act in action_options(taxonomy):
        lines.append(f"{letter}. {act['id']}: {act['definition']}")
    if with_context:
        lines += ["", "Meridian routing notes:", f"- {rb['private_context']}"]
    lines += ["", "Answer with the letter only.", "Answer:"]
    return "\n".join(lines)


def letter_for_runbook(runbook_id: str, taxonomy: dict[str, Any]) -> str:
    return next(letter for letter, rb in runbook_options(taxonomy) if rb["id"] == runbook_id)


def letter_for_action(action_id: str, taxonomy: dict[str, Any]) -> str:
    return next(letter for letter, act in action_options(taxonomy) if act["id"] == action_id)
