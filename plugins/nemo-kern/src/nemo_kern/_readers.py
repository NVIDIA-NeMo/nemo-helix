# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level data readers for kern host data sources.

Each reader function is a pure I/O function with no FastAPI or platform
dependencies, so they can be exercised directly in ``python -c`` smoke checks
against real host paths.

Sources:
- Gate decisions: SQLite at ``~/.pi/agent/pi-kern/decisions.db``
- Sleep flags: ``~/.pi/agent/kern-sleep/flags/<ulid>.json`` +
  ``core-proposals.md`` (severity, audience, reason)
- kmem memory: ``~/.pi/agent/pi-hermes-memory/entries/<ulid>.md``
- Ack ledger: a local JSON file (default ``~/.pi/agent/kern/studio/acks.json``)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate decisions
# ---------------------------------------------------------------------------

_DECISION_COLUMNS = (
    "ts",
    "rule",
    "decision",
    "action",
    "stage",
    "latency_ms",
    "reason",
    "session_id",
)


def read_decisions(db_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Return the most-recent gate decisions from the SQLite database.

    Opens in read-only URI mode so the reader never blocks the writer.
    Falls back gracefully on locked/busy errors.

    Args:
        db_path: Absolute path to ``decisions.db``.
        limit: Maximum number of rows to return (newest first).

    Returns:
        List of dicts with keys: ts (ISO-8601 str), rule, decision, action,
        stage, latency_ms, reason, session_id.
    """
    uri = f"file:{db_path}?mode=ro"
    rows: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(uri, uri=True, timeout=2.0) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT ts, rule, decision, action, stage, latency_ms, reason, session_id "
                "FROM gate_decisions ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            for row in cur.fetchall():
                ts_ms: int | None = row["ts"]
                if ts_ms is not None:
                    ts_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
                else:
                    ts_iso = None
                rows.append(
                    {
                        "ts": ts_iso,
                        "rule": row["rule"],
                        "decision": row["decision"],
                        "action": row["action"],
                        "stage": row["stage"],
                        "latency_ms": row["latency_ms"],
                        "reason": row["reason"],
                        "session_id": row["session_id"],
                    }
                )
    except sqlite3.OperationalError as exc:
        logger.warning("kern: decisions db unavailable (%s): %s", db_path, exc)
    return rows


# ---------------------------------------------------------------------------
# Sleep flags
# ---------------------------------------------------------------------------

# Parses lines like:
# - 2026-09-21 flag 01M32GQYW9SSP6821XB1SSP682 (medium, monitor): reason text
_PROPOSAL_RE = re.compile(
    r"-\s+(\d{4}-\d{2}-\d{2})\s+flag\s+([A-Z0-9]+)\s+\(([^,]+),\s*([^)]+)\):\s*(.*)"
)


def _parse_proposals(proposals_file: Path) -> dict[str, dict[str, str]]:
    """Parse core-proposals.md into a dict keyed by ULID.

    Returns: ``{ulid: {severity, audience, reason, created}}``
    """
    result: dict[str, dict[str, str]] = {}
    try:
        for line in proposals_file.read_text().splitlines():
            m = _PROPOSAL_RE.match(line.strip())
            if m:
                date_str, ulid, severity, audience, reason = m.groups()
                result[ulid] = {
                    "severity": severity.strip(),
                    "audience": audience.strip(),
                    "reason": reason.strip(),
                    "created": date_str,
                }
    except OSError as exc:
        logger.warning("kern: proposals file unreadable (%s): %s", proposals_file, exc)
    return result


def read_flags(
    flags_dir: Path,
    proposals_file: Path,
    ack_file: Path,
    limit: int = 50,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    """Return kern sleep flags, newest first.

    Merges data from:
    - ``flags/<ulid>.json`` — structured flag record (id, createdAt, session.id)
    - ``core-proposals.md`` — severity, audience, reason
    - ``acks.json`` — ack state (verdict, acked bool)

    Args:
        flags_dir: Directory containing ``<ulid>.json`` files.
        proposals_file: Path to ``core-proposals.md``.
        ack_file: Path to the ack JSON ledger.
        limit: Maximum number of flags to return.
        severity: Optional filter: ``"low"``, ``"medium"``, or ``"high"``.

    Returns:
        List of dicts: ulid, severity, audience, reason, session_id, created, verdict, acked.
    """
    proposals = _parse_proposals(proposals_file)
    acks = _load_acks(ack_file)

    flags: list[dict[str, Any]] = []
    try:
        json_files = sorted(flags_dir.glob("*.json"), key=lambda p: p.stem, reverse=True)
    except OSError as exc:
        logger.warning("kern: flags dir unreadable (%s): %s", flags_dir, exc)
        return []

    for jf in json_files:
        ulid = jf.stem
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("kern: skipping flag file %s: %s", jf, exc)
            continue

        # Merge proposal metadata (severity / audience / reason / created)
        prop = proposals.get(ulid, {})
        sev = prop.get("severity", "")
        if severity and sev != severity:
            continue

        session_id: str | None = None
        session_data = data.get("session")
        if isinstance(session_data, dict):
            session_id = session_data.get("id")

        created = prop.get("created") or data.get("createdAt", "")

        ack_entry = acks.get(ulid, {})
        flags.append(
            {
                "ulid": ulid,
                "severity": sev,
                "audience": prop.get("audience", ""),
                "reason": prop.get("reason") or data.get("reason", ""),
                "session_id": session_id,
                "created": created,
                "verdict": ack_entry.get("verdict"),
                "acked": ack_entry.get("acked", False),
            }
        )
        if len(flags) >= limit:
            break

    return flags


# ---------------------------------------------------------------------------
# Ack ledger helpers
# ---------------------------------------------------------------------------


def _load_acks(ack_file: Path) -> dict[str, dict[str, Any]]:
    """Load the ack JSON ledger, returning ``{}`` on any error."""
    try:
        return json.loads(ack_file.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("kern: ack file unreadable (%s): %s", ack_file, exc)
        return {}


def write_ack(ack_file: Path, ulid: str, verdict: str) -> None:
    """Atomically update the ack ledger for the given flag ULID.

    Creates the parent directory and the file if they do not exist.
    """
    ack_file.parent.mkdir(parents=True, exist_ok=True)
    acks = _load_acks(ack_file)
    acks[ulid] = {"verdict": verdict, "acked": True}
    # Write to a temp sibling then rename for atomicity
    tmp = ack_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(acks, indent=2))
    tmp.replace(ack_file)


# ---------------------------------------------------------------------------
# Failures (kmem memory with kind=failure)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_YAML_FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from a markdown string.

    Returns ``(fields_dict, body_text)``.  Uses a lightweight regex parser
    rather than a full YAML library to avoid adding a dependency.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_block = m.group(1)
    body = text[m.end():]
    fields: dict[str, str] = {}
    for match in _YAML_FIELD_RE.finditer(fm_block):
        key, value = match.group(1), match.group(2).strip()
        # Strip inline YAML list brackets for tags: [a, b] → "a, b"
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        fields[key] = value
    return fields, body


def _entry_title(body: str, ulid: str) -> str:
    """Extract the first heading or first non-blank line from body, else ulid."""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line:
            # Truncate long first lines
            return line[:120]
    return ulid


def read_failures(memory_dir: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Return kmem memory entries where ``kind == failure``, newest first.

    Args:
        memory_dir: Directory containing ``<ulid>.md`` entries.
        limit: Maximum number of entries to return.

    Returns:
        List of dicts: ulid, kind, category, created, summary (first line of body).
    """
    results: list[dict[str, Any]] = []
    try:
        # Sort descending by ULID (ULIDs are lexicographically time-ordered)
        md_files = sorted(memory_dir.glob("*.md"), key=lambda p: p.stem, reverse=True)
    except OSError as exc:
        logger.warning("kern: memory dir unreadable (%s): %s", memory_dir, exc)
        return []

    for mf in md_files:
        if len(results) >= limit:
            break
        try:
            text = mf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm, body = _parse_frontmatter(text)
        if fm.get("kind") != "failure":
            continue
        results.append(
            {
                "ulid": fm.get("ulid", mf.stem),
                "kind": fm.get("kind", ""),
                "category": fm.get("category", ""),
                "created": fm.get("created", ""),
                "summary": _entry_title(body, mf.stem),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Memory entries (all kinds, with optional text search)
# ---------------------------------------------------------------------------


def read_memory(
    memory_dir: Path, limit: int = 50, q: str | None = None
) -> list[dict[str, Any]]:
    """Return kmem memory entries, newest first, with optional substring search.

    Args:
        memory_dir: Directory containing ``<ulid>.md`` entries.
        limit: Maximum number of entries to return after filtering.
        q: Optional substring to match against the title (first heading/line)
           and full body text.  Case-insensitive.

    Returns:
        List of dicts: ulid, kind, tags, scope, created, title.
    """
    results: list[dict[str, Any]] = []
    q_lower = q.lower() if q else None
    try:
        md_files = sorted(memory_dir.glob("*.md"), key=lambda p: p.stem, reverse=True)
    except OSError as exc:
        logger.warning("kern: memory dir unreadable (%s): %s", memory_dir, exc)
        return []

    for mf in md_files:
        if len(results) >= limit:
            break
        try:
            text = mf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm, body = _parse_frontmatter(text)
        title = _entry_title(body, mf.stem)
        if q_lower and q_lower not in title.lower() and q_lower not in body.lower():
            continue
        tags_raw = fm.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        results.append(
            {
                "ulid": fm.get("ulid", mf.stem),
                "kind": fm.get("kind", ""),
                "tags": tags,
                "scope": fm.get("scope", ""),
                "created": fm.get("created", ""),
                "title": title,
            }
        )
    return results


def read_memory_entry(memory_dir: Path, ulid: str) -> dict[str, Any] | None:
    """Return a single kmem entry by ULID, including its full body.

    Returns ``None`` if the file does not exist.
    """
    path = memory_dir / f"{ulid}.md"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("kern: cannot read memory entry %s: %s", path, exc)
        return None
    fm, body = _parse_frontmatter(text)
    title = _entry_title(body, ulid)
    tags_raw = fm.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    return {
        "ulid": fm.get("ulid", ulid),
        "kind": fm.get("kind", ""),
        "tags": tags,
        "scope": fm.get("scope", ""),
        "created": fm.get("created", ""),
        "title": title,
        "body": body.strip(),
    }
