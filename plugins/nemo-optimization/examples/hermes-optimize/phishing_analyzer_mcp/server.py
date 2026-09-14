# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A mock email phishing analyzer, exposed as an MCP stdio server, that replays canned results.

The ``optimize-mcp.yaml`` study tunes the Hermes coordinator that calls this tool, not the analyzer,
so the analyzer is a fixture: each row of ``dataset-mcp.json`` carries the ``analysis`` a real
analyzer returned for that email, and this server hands it back when the agent passes that email.
To add a task, run the real analyzer on the new email once and store its result in the row.

Because the reply is keyed on the email text, an agent that edits, truncates or paraphrases the
email gets an ``unknown`` verdict instead of a confident one. That is the input-binding check the
system prompt's "copy verbatim" instruction relies on.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

#: Overrides the dataset the canned results are read from (default: the bundle's ``dataset-mcp.json``).
DATASET_ENV = "PHISHING_ANALYZER_DATASET"
_DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "dataset-mcp.json"
_WHITESPACE = re.compile(r"\s+")

mcp = FastMCP("email-phishing-analyzer")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


def load_responses(dataset: Path | None = None) -> dict[str, dict[str, object]]:
    """Map each dataset row's normalized ``subject\\n\\nbody`` and body to its canned ``analysis``."""
    path = dataset or Path(os.environ.get(DATASET_ENV) or _DEFAULT_DATASET)
    responses: dict[str, dict[str, object]] = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        analysis = row.get("analysis")
        if not isinstance(analysis, dict):
            raise ValueError(f"dataset row {row.get('id')!r} has no analysis to replay")
        body = str(row.get("body") or "")
        subject = str(row.get("subject") or "")
        for key in (f"{subject}\n\n{body}" if subject else body, body):
            responses[_normalize(key)] = dict(analysis)
    return responses


def analyze(text: str, responses: dict[str, dict[str, object]]) -> dict[str, object]:
    """The canned analysis for ``text``: an exact match on the email, else a match on its body, else unknown."""
    normalized = _normalize(text)
    if normalized in responses:
        return {**responses[normalized], "matched": "exact"}
    # The coordinator may add framing around the email; the body still has to be intact. A row is
    # keyed both by its full text and its body, so collapse matches to distinct analyses.
    matches = {id(analysis): analysis for key, analysis in responses.items() if key and key in normalized}
    if len(matches) == 1:
        return {**next(iter(matches.values())), "matched": "body"}
    return {
        "is_likely_phishing": None,
        "label": "unknown",
        "confidence": 0.0,
        "reasons": ["email text does not match any analyzed email; pass the message verbatim"],
        "matched": "none",
    }


_RESPONSES: dict[str, dict[str, object]] | None = None


@mcp.tool()
def email_phishing_analyzer(text: str) -> dict[str, object]:
    """Analyze an email (subject and body, verbatim) and report whether it is likely phishing."""
    global _RESPONSES
    if _RESPONSES is None:
        _RESPONSES = load_responses()
    return analyze(text, _RESPONSES)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
