#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render or update ``audit.md`` from ETHOS.md metadata and reviewed audit items."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _markdown import BEGIN_MARKER, END_MARKER, AuditMarkdownError, extract_schema_block  # noqa: E402
from _schema import AUDIT_SCHEMA, AuditEnvironmentError, AuditSpecError, item_counts, validate_audit_spec  # noqa: E402

_MARKED_BLOCK_RE = re.compile(
    rf"(?ms)^(?P<begin>[ \t]*{re.escape(BEGIN_MARKER)}[ \t]*\n).*?"
    rf"^(?P<end>[ \t]*{re.escape(END_MARKER)}[ \t]*$)"
)
ETHOS_DOCS_URL = "https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ethos", type=Path, required=True, help="source ETHOS.md")
    parser.add_argument("--items", type=Path, required=True, help="YAML file containing audit items")
    parser.add_argument("--out", type=Path, required=True, help="audit.md file to write")
    parser.add_argument("--agent", help="agent name; defaults to ETHOS.md front matter name or file stem")
    parser.add_argument("--status", choices=("draft", "approved"), help="audit review status; defaults to draft")
    parser.add_argument(
        "--mode",
        choices=("reconcile", "replace", "suggest"),
        default="reconcile",
        help=(
            "reconcile existing audit.md by stable item name, replace it completely, or suggest changes without writing"
        ),
    )
    parser.add_argument(
        "--items-mode",
        choices=("partial", "full"),
        default="partial",
        help=("treat --items as incremental additions/edits, or as the full denominator for stale-item reporting"),
    )
    args = parser.parse_args(argv)

    try:
        yaml = _load_yaml()
        args.out = _validated_output_path(args.out)
        candidate = validate_audit_spec(_candidate_spec(args, yaml))
        existing = _load_existing_audit(args.out, yaml) if args.out.exists() and args.mode != "replace" else None
        spec, summary = _prepare_output(candidate, existing, args)

        if args.mode != "suggest":
            args.out.parent.mkdir(parents=True, exist_ok=True)
        spec = validate_audit_spec(spec, audit_path=args.out if args.mode != "suggest" or args.out.exists() else None)
    except AuditEnvironmentError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, UnicodeError, AuditMarkdownError, AuditSpecError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary.update(
        {
            "valid": True,
            "output": str(args.out),
            "agent": spec["agent"],
            "status": spec["status"],
            "item_counts": item_counts(spec),
        }
    )
    if args.mode != "suggest":
        rendered = _render_reconciled(args.out, spec, yaml) if existing is not None else _render_full(spec, yaml)
        args.out.write_text(rendered, encoding="utf-8")
        summary["written"] = True
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise AuditEnvironmentError("PyYAML is required to generate audit specs") from exc
    return yaml


def _load_items_file(path: Path, yaml: Any) -> list[Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AuditSpecError(f"audit items YAML does not parse: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("items")
    if not isinstance(payload, list):
        raise AuditSpecError("audit items file must be a list or a mapping with an items list")
    return payload


def _load_existing_audit(path: Path, yaml: Any) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(extract_schema_block(path))
    except yaml.YAMLError as exc:
        raise AuditSpecError(f"existing audit YAML does not parse: {exc}") from exc
    return validate_audit_spec(payload)


def _validated_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if ".eval-author" not in resolved.parts[:-1]:
        raise AuditSpecError("--out must resolve inside a .eval-author/ directory")
    return resolved


def _candidate_spec(args: argparse.Namespace, yaml: Any) -> dict[str, Any]:
    ethos_bytes = _read_ethos(args.ethos)
    ethos_text = ethos_bytes.decode("utf-8")
    agent = args.agent or _agent_name(ethos_text, args.ethos, yaml)
    return {
        "schema": AUDIT_SCHEMA,
        "agent": agent,
        "sources": [
            {
                "name": "ethos",
                "path": _source_path(args.ethos, args.out),
                "sha256": f"sha256:{hashlib.sha256(ethos_bytes).hexdigest()}",
            }
        ],
        "status": args.status or "draft",
        "items": _load_items_file(args.items, yaml),
    }


def _read_ethos(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise AuditSpecError(
            "Missing Ethos\n\n"
            "Eval Author needs a source of truth for how the agent is supposed to behave before it can "
            "generate an audit coverage report.\n\n"
            f"Missing file: {path}\n\n"
            "ETHOS.md records intended behavior, mission, constraints, success and failure criteria, "
            "and what may change.\n\n"
            f"Docs: {ETHOS_DOCS_URL}\n\n"
            "Next steps:\n"
            "- Use Eval Author's Local Ethos procedure "
            "(skills/eval-author/references/local-ethos.md), or author ETHOS.md from the documentation.\n"
            "- Save ETHOS.md in the repository and review its contents before generating audit items.\n"
            "- Then rerun this command with --ethos <path>. No platform service or upload is required."
        ) from exc
    except OSError as exc:
        raise AuditSpecError(
            "Unreadable Ethos\n\n"
            "Eval Author needs a source of truth for how the agent is supposed to behave before it can "
            "generate an audit coverage report.\n\n"
            f"Unreadable file: {path}\n"
            f"Error: {exc}\n\n"
            f"Docs: {ETHOS_DOCS_URL}\n\n"
            "Next steps:\n"
            "- Fix read access for the Ethos file, then rerun this command.\n"
            "- Or pass a readable Ethos path with --ethos <path>."
        ) from exc


def _prepare_output(
    candidate: dict[str, Any],
    existing: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {
        "mode": args.mode,
        "written": False,
        "items_mode": args.items_mode,
        "added_items": [],
        "unchanged_items": [],
        "conflicting_items": [],
        "conflicting_items_applied": True,
        "possibly_stale_items": [],
        "agent_change": None,
    }
    if existing is None:
        summary["action"] = (
            "suggest_create" if args.mode == "suggest" else "replace" if args.mode == "replace" else "create"
        )
        summary["added_items"] = _item_names(candidate["items"])
        return candidate, summary

    reconciled, reconciliation = _reconcile(
        candidate,
        existing,
        explicit_status=args.status,
        explicit_agent=args.agent,
        items_mode=args.items_mode,
    )
    summary["action"] = "suggest_reconcile" if args.mode == "suggest" else "reconcile"
    summary.update(reconciliation)
    return reconciled, summary


def _reconcile(
    candidate: dict[str, Any],
    existing: dict[str, Any],
    *,
    explicit_status: str | None,
    explicit_agent: str | None,
    items_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_items_by_name = _items_by_name(existing["items"])
    candidate_items_by_name = _items_by_name(candidate["items"])
    items: list[dict[str, Any]] = []
    unchanged_items: list[str] = []
    conflicting_items: list[str] = []
    possibly_stale_items: list[str] = []
    added_items: list[str] = []
    agent, agent_change = _reconcile_agent(candidate, existing, explicit_agent=explicit_agent)

    for item in existing["items"]:
        name = item["name"]
        candidate_item = candidate_items_by_name.get(name)
        if candidate_item is None:
            if items_mode == "full":
                possibly_stale_items.append(name)
        elif candidate_item == item:
            unchanged_items.append(name)
        else:
            conflicting_items.append(name)
        items.append(item)

    for item in candidate["items"]:
        name = item["name"]
        if name not in existing_items_by_name:
            added_items.append(name)
            items.append(item)

    status = explicit_status or existing["status"]
    if explicit_status is None and (added_items or conflicting_items or possibly_stale_items or agent_change):
        status = "draft"

    return (
        {
            "schema": candidate["schema"],
            "agent": agent,
            "sources": _reconcile_sources(candidate.get("sources", []), existing.get("sources", [])),
            "status": status,
            "items": items,
        },
        {
            "added_items": added_items,
            "unchanged_items": unchanged_items,
            "conflicting_items": conflicting_items,
            "conflicting_items_applied": not conflicting_items,
            "possibly_stale_items": possibly_stale_items,
            "agent_change": agent_change,
        },
    )


def _reconcile_agent(
    candidate: dict[str, Any],
    existing: dict[str, Any],
    *,
    explicit_agent: str | None,
) -> tuple[str, dict[str, Any] | None]:
    if candidate["agent"] == existing["agent"]:
        return existing["agent"], None

    applied = explicit_agent is not None
    return (
        candidate["agent"] if applied else existing["agent"],
        {"from": existing["agent"], "to": candidate["agent"], "applied": applied},
    )


def _reconcile_sources(
    candidate_sources: list[dict[str, Any]], existing_sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate_by_name = {source["name"]: source for source in candidate_sources}
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for source in existing_sources:
        name = source["name"]
        if name in candidate_by_name:
            sources.append({**source, **candidate_by_name[name]})
        else:
            sources.append(source)
        seen.add(name)
    sources.extend(source for source in candidate_sources if source["name"] not in seen)
    return sources


def _items_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in items}


def _item_names(items: list[dict[str, Any]]) -> list[str]:
    return [item["name"] for item in items]


def _agent_name(ethos_text: str, ethos_path: Path, yaml: Any) -> str:
    if ethos_text.startswith("---\n"):
        parts = ethos_text.split("---\n", 2)
        if len(parts) == 3:
            try:
                payload = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError as exc:
                raise AuditSpecError(f"ETHOS.md frontmatter YAML does not parse: {exc}") from exc
            if isinstance(payload, dict):
                name = payload.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return ethos_path.stem.lower()


def _source_path(source: Path, out: Path) -> str:
    return Path(os.path.relpath(source.resolve(), start=out.parent.resolve())).as_posix()


def _dump_audit_yaml(spec: dict[str, Any], yaml: Any) -> str:
    class AuditDumper(yaml.SafeDumper):
        def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
            return super().increase_indent(flow=flow, indentless=False)

    def represent_str(dumper: Any, value: str) -> Any:
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    AuditDumper.add_representer(str, represent_str)
    return yaml.dump(spec, Dumper=AuditDumper, sort_keys=False, allow_unicode=True)


def _render_full(spec: dict[str, Any], yaml: Any) -> str:
    yaml_body = _dump_audit_yaml(spec, yaml)
    source_path = spec["sources"][0]["path"]
    return (
        f"# Audit: {spec['agent']}\n\n"
        f"Generated from `{source_path}`.\n\n"
        "This file defines the finite coverage denominator for audit measurement.\n"
        "Generated and hand-edited content is allowed outside the marked block; scripts\n"
        "validate only the block between the markers.\n\n"
        f"{BEGIN_MARKER}\n"
        "```yaml\n"
        f"{yaml_body}"
        "```\n"
        f"{END_MARKER}\n"
    )


def _render_reconciled(path: Path, spec: dict[str, Any], yaml: Any) -> str:
    text = path.read_text(encoding="utf-8")
    matches = list(_MARKED_BLOCK_RE.finditer(text))
    if len(matches) != 1:
        raise AuditMarkdownError(f"{path} must contain exactly one marked audit block")
    match = matches[0]
    yaml_body = _dump_audit_yaml(spec, yaml)
    return text[: match.end("begin")] + "```yaml\n" + yaml_body + "```\n" + text[match.start("end") :]


if __name__ == "__main__":
    raise SystemExit(main())
